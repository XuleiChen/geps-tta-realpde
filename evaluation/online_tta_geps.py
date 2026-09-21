"""
online_tta_geps.py — Online TTA evaluation for GEPS U-Net, RealPDE Track 2.

Two modes (controlled by --N_adapt_blocks):
  > 0  GEPS + TTA  : adapt_code 在前 N 个 block 用 Adam 更新
  = 0  GEPS no-TTA : adapt_code 固定为 finetune codes 的均值

自回归规则：
  - block 0 输入 = 真实 frames[0:20]
  - block k 输入 = 上一块的模型预测（不用 GT）
  - 适应在预测之后（方案1）：GT 释放 → adapt → 更新 adapt_code 供下一块使用

输出 .npz:
  rel_l2 : (n_traj, n_blocks)
  tke    : (n_traj, n_blocks)
  mvpe   : (n_traj, n_blocks)

示例:
  # no-TTA baseline
  python online_tta_geps.py \\
      --checkpoint checkpoints/geps_finetune.pth \\
      --normalizer /scratch/xulei03/data/train_real/mean_std.pt \\
      --N_adapt_blocks 0 --output results/geps_noTTA.npz

  # with TTA
  python online_tta_geps.py \\
      --checkpoint checkpoints/geps_finetune.pth \\
      --normalizer /scratch/xulei03/data/train_real/mean_std.pt \\
      --N_adapt_blocks 5 --n_steps 10 --output results/geps_TTA.npz
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from datasets import load_from_disk
from realpdebench.model.unet_geps import Unet3d as GEPSUnet3d
from realpdebench.utils.metrics import eval_metrics, probe_diagnostic


# ── Foil-specific constants ───────────────────────────────────────────────────
FOIL_D          = 62
FOIL_CENTER_X   = 30
FOIL_CENTER_Y   = 64
FOIL_SUB_S_REAL = 2
BLOCK_SIZE      = 20   # in_step = out_step


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GEPS online TTA evaluation")
    p.add_argument("--checkpoint",     required=True,
                   help="GEPS finetune checkpoint (.pth)")
    p.add_argument("--normalizer",     required=True,
                   help="mean_std.pt 路径（来自 train_real）")
    p.add_argument("--data_root",
                   default="/Users/xulei/Downloads/RealpdeTrack2/data",
                   help="数据集根目录（包含 foil/hf_dataset/real/）")
    p.add_argument("--N_adapt_blocks", type=int, default=5,
                   help="前几个 block 做适应（0 = no-TTA）")
    p.add_argument("--n_steps",        type=int, default=10,
                   help="每个 block 的梯度步数")
    p.add_argument("--adapt_lr",       type=float, default=1e-3,
                   help="adapt_code 的 Adam 学习率")
    p.add_argument("--output",         required=True,
                   help="输出 .npz 路径")
    p.add_argument("--device",         default="cpu",
                   help="cpu | mps | cuda:N")
    p.add_argument("--max_frames",     type=int, default=None,
                   help="每条轨迹最多用几帧（None = 全用，868 = 对齐竞赛 h5 格式）")
    p.add_argument("--exclude_sim_ids", nargs="*", default=["3750_10.0", "3750_0.0"],
                   help="排除与 train_real 重叠的 sim_id（默认排除 Re=3750 两条）")
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GEPSUnet3d(
        dim=ckpt["dim"],
        n_env=ckpt["n_env"],
        out_channels=3,
        dim_mults=tuple(ckpt["dim_mults"]),
        channels=3,
        in_time=ckpt["in_step"],
        out_time=ckpt["out_step"],
        code_c=ckpt["code_c"],
        factor=ckpt["factor"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    logging.info(
        f"Loaded GEPS model: n_env={ckpt['n_env']}, code_c={ckpt['code_c']}, "
        f"stage={ckpt.get('stage', '?')}"
    )
    return model


# ── Normalizer ────────────────────────────────────────────────────────────────
def load_normalizer(normalizer_path, device):
    mean_inp, mean_tgt, std_inp, std_tgt = torch.load(
        normalizer_path, map_location="cpu", weights_only=False
    )
    # std=0 的通道（p 全零）替换为 1，避免除零
    std_inp = torch.where(std_inp == 0, torch.ones_like(std_inp), std_inp)
    std_tgt = torch.where(std_tgt == 0, torch.ones_like(std_tgt), std_tgt)
    return (
        mean_inp.to(device), mean_tgt.to(device),
        std_inp.to(device),  std_tgt.to(device),
    )

def norm_input(x, mean_inp, std_inp):
    """x: (..., 3) physical → normalised input space"""
    return (x - mean_inp) / std_inp

def norm_target(y, mean_tgt, std_tgt):
    """y: (..., 3) physical → normalised target space"""
    return (y - mean_tgt) / std_tgt

def denorm_output(y_hat_norm, mean_tgt, std_tgt):
    """y_hat_norm: (..., 3) normalised target space → physical"""
    return y_hat_norm * std_tgt + mean_tgt


# ── Data loading ──────────────────────────────────────────────────────────────
def load_trajectories(data_root):
    traj_path = os.path.join(data_root, "foil", "hf_dataset", "real")
    logging.info(f"Loading trajectories from: {traj_path}")
    return load_from_disk(traj_path)

def decode_trajectory(row, max_frames=None):
    """
    Arrow row → (n_frames, 64, 128, 3) float32 numpy array.
    对 u, v 做 sub_s=2 空间降采样，p 补全零通道。
    """
    shape = (row["shape_t"], row["shape_h"], row["shape_w"])
    u_full = np.frombuffer(row["u"], dtype=np.float32).reshape(shape).copy()
    v_full = np.frombuffer(row["v"], dtype=np.float32).reshape(shape).copy()

    u = u_full[:, ::FOIL_SUB_S_REAL, ::FOIL_SUB_S_REAL]   # (n_frames, 64, 128)
    v = v_full[:, ::FOIL_SUB_S_REAL, ::FOIL_SUB_S_REAL]
    p = np.zeros_like(u)

    data = np.stack([u, v, p], axis=-1)   # (n_frames, 64, 128, 3)
    if max_frames is not None:
        data = data[:max_frames]
    return data


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_block_metrics(y_hat_cpu, y_true_cpu):
    """
    y_hat_cpu, y_true_cpu : torch.Tensor (1, 20, 64, 128, 3), CPU, physical space
    Returns: (rel_l2, tke) as Python floats
    """
    _, _, rel_l2, _, tke, *_ = eval_metrics(y_hat_cpu, y_true_cpu, c=2)
    return rel_l2.item(), tke.item()


# ── TTA loop for one trajectory ───────────────────────────────────────────────
def run_trajectory(data, model, mean_inp, mean_tgt, std_inp, std_tgt,
                   N_adapt_blocks, n_steps, adapt_lr, device):
    """
    data : (n_frames, 64, 128, 3) numpy, physical space
    Returns : (rel_l2s, tkes, mvpes) — Python lists of length n_blocks
    """
    n_frames = data.shape[0]
    n_blocks = (n_frames - BLOCK_SIZE) // BLOCK_SIZE
    if n_blocks <= 0:
        logging.warning(f"Too few frames ({n_frames}) for even one block, skipping.")
        return [], [], []

    # adapt_code: 外部 Parameter，从 0 开始（不带 sim 偏置）
    adapt_code = nn.Parameter(torch.zeros(model.code_c, device=device))
    optimizer  = torch.optim.Adam([adapt_code], lr=adapt_lr)

    rel_l2s, tkes = [], []

    # 第一块输入：真实帧 frames[0:20]（physical → 归一化输入空间）
    x_phys = torch.tensor(
        data[:BLOCK_SIZE], dtype=torch.float32
    ).unsqueeze(0).to(device)                  # (1, 20, 64, 128, 3)
    x_norm = norm_input(x_phys, mean_inp, std_inp)

    for block_idx in range(n_blocks):
        gt_start = BLOCK_SIZE * (block_idx + 1)
        gt_end   = gt_start + BLOCK_SIZE
        if gt_end > n_frames:
            break

        # ── GT（归一化目标空间 + 物理空间各备一份）──────────────────────
        y_phys = torch.tensor(
            data[gt_start:gt_end], dtype=torch.float32
        ).unsqueeze(0).to(device)              # (1, 20, 64, 128, 3)
        y_norm = norm_target(y_phys, mean_tgt, std_tgt)

        # ── 预测（adapt_code 更新前，方案1）────────────────────────────
        with torch.no_grad():
            y_hat_norm = model(x_norm, adapt_code=adapt_code)

        # 反归一化到物理空间，用于计算指标
        y_hat_phys = denorm_output(y_hat_norm, mean_tgt, std_tgt)

        # ── 记录本块指标 ────────────────────────────────────────────────
        rl, tk = compute_block_metrics(y_hat_phys.cpu(), y_phys.cpu())
        rel_l2s.append(rl)
        tkes.append(tk)

        # ── 适应（仅前 N_adapt_blocks 块）──────────────────────────────
        if block_idx < N_adapt_blocks:
            for _ in range(n_steps):
                pred_norm = model(x_norm, adapt_code=adapt_code)
                loss = F.mse_loss(pred_norm[..., :2], y_norm[..., :2])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ── 下一块输入 = 当前预测（自回归）────────────────────────────
        # 物理空间的预测 → 归一化为输入空间（mean_inp/std_inp）
        x_norm = norm_input(y_hat_phys.detach(), mean_inp, std_inp)

    return rel_l2s, tkes


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    args = parse_args()
    device = torch.device(args.device)

    exclude = set(args.exclude_sim_ids or [])
    logging.info(f"Excluded sim_ids: {exclude or 'none'}")
    logging.info(f"Mode: {'TTA (N_adapt_blocks=' + str(args.N_adapt_blocks) + ', n_steps=' + str(args.n_steps) + ')' if args.N_adapt_blocks > 0 else 'no-TTA'}")

    model = load_model(args.checkpoint, device)
    mean_inp, mean_tgt, std_inp, std_tgt = load_normalizer(args.normalizer, device)
    trajectories = load_trajectories(args.data_root)

    all_rel_l2, all_tke, all_mvpe, sim_ids_used = [], [], [], []

    for i in range(len(trajectories)):
        row    = trajectories[i]
        sim_id = row.get("sim_id", str(i))

        if sim_id in exclude:
            logging.info(f"[{i+1}/{len(trajectories)}] Skipping {sim_id} (excluded)")
            continue

        logging.info(f"[{i+1}/{len(trajectories)}] {sim_id}")
        data = decode_trajectory(row, max_frames=args.max_frames)
        n_blocks = (data.shape[0] - BLOCK_SIZE) // BLOCK_SIZE
        logging.info(f"  frames={data.shape[0]}, blocks={n_blocks}")

        rel_l2s, tkes = run_trajectory(
            data, model,
            mean_inp, mean_tgt, std_inp, std_tgt,
            args.N_adapt_blocks, args.n_steps, args.adapt_lr,
            device,
        )

        if rel_l2s:
            all_rel_l2.append(rel_l2s)
            all_tke.append(tkes)
            sim_ids_used.append(sim_id)

    if not all_rel_l2:
        logging.error("No trajectories processed.")
        return

    # 不同轨迹帧数可能不同，用 NaN 补齐
    max_blocks = max(len(x) for x in all_rel_l2)
    def pad_to(lst):
        arr = np.full((len(lst), max_blocks), np.nan, dtype=np.float32)
        for i, row in enumerate(lst):
            arr[i, :len(row)] = row
        return arr

    out = {
        "rel_l2":         pad_to(all_rel_l2),   # (n_traj, n_blocks)
        "tke":            pad_to(all_tke),
        "sim_ids":        np.array(sim_ids_used),
        "N_adapt_blocks": np.array(args.N_adapt_blocks),
        "n_steps":        np.array(args.n_steps),
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(args.output, **out)
    logging.info(
        f"Saved → {args.output}  "
        f"shape={out['rel_l2'].shape}  "
        f"trajectories={len(sim_ids_used)}"
    )


if __name__ == "__main__":
    main()
