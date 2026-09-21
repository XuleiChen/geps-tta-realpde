"""
sps_calibration.py — 校准 SPS 区间宽度，生成 (channel, lead_time_bucket) 分组查找表。

用法:
  python sps_calibration.py \
      --checkpoint ../submission/model.pth \
      --normalizer /Users/xulei/Downloads/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt \
      --data_root  /Users/xulei/Downloads/RealpdeTrack2/data \
      --scoring_py /Users/xulei/Downloads/realpde_t2_starting_kit_v6/scoring.py \
      --output     ../results/sps_widths.npz \
      --device     mps
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from datasets import load_from_disk
from realpdebench.model.unet_geps import Unet3d as GEPSUnet3d

# ── Constants ────────────────────────────────────────────────────────────────
SIGMA_GLOBAL = 0.0563870259   # 官方固定常数，不随分组改变
BLOCK_SIZE   = 20             # T_in = T_out = 20
FOIL_SUB_S   = 2              # HF dataset: raw 128×256 → 64×128

# lead_time buckets: frame indices within a block (0-indexed, 0..19)
BUCKETS = {
    "early": list(range(0,  7)),   # frames 1-7  → idx 0-6
    "mid":   list(range(7,  14)),  # frames 8-14 → idx 7-13
    "late":  list(range(14, 20)),  # frames 15-20 → idx 14-19
}
BUCKET_NAMES = list(BUCKETS.keys())   # ["early", "mid", "late"]
CHANNELS     = [0, 1]                 # u, v  (p not scored)

# Held-out split (20 test trajectories total)
CALIB_END = 14    # trajectories [0:14] for grid search
# trajectories [14:20] reserved for validation

MIN_SAMPLES = 10_000   # 样本数低于此阈值时发出警告


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--normalizer",  required=True)
    p.add_argument("--data_root",   default="/Users/xulei/Downloads/RealpdeTrack2/data")
    p.add_argument("--scoring_py",  required=True,
                   help="path to starting kit scoring.py")
    p.add_argument("--output",      default="../results/sps_widths.npz")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--n_candidates", type=int, default=100)
    return p.parse_args()


# ── Load scoring module ───────────────────────────────────────────────────────
def load_scoring(scoring_py: str):
    spec = importlib.util.spec_from_file_location("scoring", scoring_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Model ────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GEPSUnet3d(
        dim       = ckpt["dim"],
        n_env     = ckpt["n_env"],
        out_channels = 3,
        dim_mults = tuple(ckpt["dim_mults"]),
        channels  = 3,
        in_time   = ckpt["in_step"],
        out_time  = ckpt["out_step"],
        code_c    = ckpt["code_c"],
        factor    = ckpt["factor"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # adapt_code = mean of finetune codes (same as submission)
    adapt_code = model.codes.data.mean(dim=0).to(device)
    logging.info(f"Loaded model: n_env={ckpt['n_env']}, code_c={ckpt['code_c']}, stage={ckpt.get('stage')}")
    return model, adapt_code


# ── Normalizer ───────────────────────────────────────────────────────────────
def load_normalizer(path: str, device):
    mean_inp, mean_tgt, std_inp, std_tgt = torch.load(
        path, map_location="cpu", weights_only=False
    )
    std_inp = torch.where(std_inp == 0, torch.ones_like(std_inp), std_inp)
    std_tgt = torch.where(std_tgt == 0, torch.ones_like(std_tgt), std_tgt)
    return (mean_inp.to(device), mean_tgt.to(device),
            std_inp.to(device),  std_tgt.to(device))


# ── Data ─────────────────────────────────────────────────────────────────────
def decode_trajectory(row, max_frames=None) -> np.ndarray:
    """Arrow row → (n_frames, 64, 128, 3) float32, physical space."""
    shape = (row["shape_t"], row["shape_h"], row["shape_w"])
    u_full = np.frombuffer(row["u"], dtype=np.float32).reshape(shape).copy()
    v_full = np.frombuffer(row["v"], dtype=np.float32).reshape(shape).copy()
    u = u_full[:, ::FOIL_SUB_S, ::FOIL_SUB_S]
    v = v_full[:, ::FOIL_SUB_S, ::FOIL_SUB_S]
    p = np.zeros_like(u)
    data = np.stack([u, v, p], axis=-1)   # (T, 64, 128, 3)
    if max_frames is not None:
        data = data[:max_frames]
    return data


def downsample_spatial(arr: np.ndarray) -> np.ndarray:
    """(T, 64, 128, C) → (T, 32, 64, C) via stride-2 subsampling."""
    return arr[:, ::2, ::2, :]


# ── Inference: collect (pred, target) at 32×64 ───────────────────────────────
def run_trajectory_collect(
    data_64: np.ndarray,
    model, adapt_code,
    mean_inp, mean_tgt, std_inp, std_tgt,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autoregressive block-by-block prediction (no TTA, fixed adapt_code).
    Returns pred_phys, target_phys each of shape (n_blocks, 20, 32, 64, 3).
    """
    n_frames = data_64.shape[0]
    n_blocks = (n_frames - BLOCK_SIZE) // BLOCK_SIZE
    if n_blocks <= 0:
        return np.empty((0, BLOCK_SIZE, 32, 64, 3)), np.empty((0, BLOCK_SIZE, 32, 64, 3))

    preds, targets = [], []

    # First block input: real frames [0:20] at 64×128
    x_phys = torch.tensor(
        data_64[:BLOCK_SIZE], dtype=torch.float32
    ).unsqueeze(0).to(device)                       # (1, 20, 64, 128, 3)
    x_norm = (x_phys - mean_inp) / std_inp

    for block_idx in range(n_blocks):
        gt_start = BLOCK_SIZE * (block_idx + 1)
        gt_end   = gt_start + BLOCK_SIZE
        if gt_end > n_frames:
            break

        y_phys = torch.tensor(
            data_64[gt_start:gt_end], dtype=torch.float32
        ).unsqueeze(0).to(device)                   # (1, 20, 64, 128, 3)

        with torch.no_grad():
            y_hat_norm = model(x_norm, adapt_code=adapt_code)

        y_hat_phys = y_hat_norm * std_tgt + mean_tgt

        # Downsample to 32×64 for scoring
        pred_32   = y_hat_phys[0].cpu().numpy()[:, ::2, ::2, :]   # (20, 32, 64, 3)
        target_32 = y_phys[0].cpu().numpy()[:, ::2, ::2, :]

        preds.append(pred_32)
        targets.append(target_32)

        # Teacher forcing: next input = GT frames (matches competition eval protocol)
        x_norm = (y_phys - mean_inp) / std_inp

    return np.stack(preds), np.stack(targets)   # (n_blocks, 20, 32, 64, 3)


# ── Residual collection ───────────────────────────────────────────────────────
def collect_residuals(
    trajectories, indices, model, adapt_code,
    mean_inp, mean_tgt, std_inp, std_tgt, device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on selected trajectory indices.
    Returns all_pred, all_target of shape (N_blocks_total, 20, 32, 64, 3).
    """
    all_pred, all_tgt = [], []
    for idx in indices:
        row  = trajectories[idx]
        sim_id = row.get("sim_id", str(idx))
        logging.info(f"  trajectory {idx}: {sim_id}")
        data_64 = decode_trajectory(row)
        pred, tgt = run_trajectory_collect(
            data_64, model, adapt_code,
            mean_inp, mean_tgt, std_inp, std_tgt, device,
        )
        if pred.shape[0] > 0:
            all_pred.append(pred)
            all_tgt.append(tgt)

    return np.concatenate(all_pred), np.concatenate(all_tgt)


# ── Group residuals ───────────────────────────────────────────────────────────
GroupKey = Tuple[int, str]   # (channel_idx, bucket_name)

def build_group_residuals(
    pred: np.ndarray,      # (N, 20, 32, 64, 3)
    target: np.ndarray,    # (N, 20, 32, 64, 3)
) -> Dict[GroupKey, np.ndarray]:
    """
    Split |pred - target| into 6 groups: (channel∈{0,1}) × (bucket∈{early,mid,late}).
    Excludes points where target == 0 (airfoil body / outside PIV FOV).
    """
    residuals = np.abs(pred - target)   # (N, 20, 32, 64, 3)
    groups = {}
    for ch in CHANNELS:
        for bname, fidxs in BUCKETS.items():
            # residuals for this (channel, bucket): (N, len(fidxs), 32, 64)
            r = residuals[:, fidxs, :, :, ch]
            t = target[:, fidxs, :, :, ch]
            mask = (t != 0.0)
            flat_r = r[mask]
            groups[(ch, bname)] = flat_r
            n = len(flat_r)
            if n < MIN_SAMPLES:
                logging.warning(
                    f"Group (ch={ch}, {bname}): only {n} samples — estimate may be unstable"
                )
    return groups


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search_width(
    residuals: np.ndarray,
    n_candidates: int = 100,
) -> Tuple[float, float, float]:
    """
    Find optimal symmetric half-width w/2 for this group.
    Optimizes: score = coverage * exp(-w / SIGMA_GLOBAL)
    Returns (w_opt, coverage_opt, score_opt).
    """
    r_max = float(np.percentile(residuals, 99.5))
    if r_max == 0:
        r_max = float(np.max(residuals)) + 1e-8

    candidates = np.linspace(0.02 * r_max, 3.0 * r_max, n_candidates)
    best_w, best_score, best_cov = candidates[0], -1.0, 0.0

    for w in candidates:
        half_w   = w / 2.0
        coverage = float(np.mean(residuals <= half_w))
        score    = coverage * np.exp(-w / SIGMA_GLOBAL)
        if score > best_score:
            best_w, best_score, best_cov = w, score, coverage

    # Check if w_opt is at the boundary (indicates range too narrow/wide)
    if best_w == candidates[0]:
        logging.warning(f"  w_opt at LOWER boundary — consider widening search range")
    if best_w == candidates[-1]:
        logging.warning(f"  w_opt at UPPER boundary — consider widening search range")

    return best_w, best_cov, best_score


# ── Build lower/upper arrays ──────────────────────────────────────────────────
def make_interval_arrays(
    pred: np.ndarray,      # (N, 20, 32, 64, 3)
    w_dict: Dict[GroupKey, float],
    bias_dict: Dict[GroupKey, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct per-element lower/upper from group widths (and optional bias)."""
    lower = pred.copy()
    upper = pred.copy()

    for ch in CHANNELS:
        for bname, fidxs in BUCKETS.items():
            w    = w_dict[(ch, bname)]
            bias = bias_dict[(ch, bname)] if bias_dict else 0.0
            # interval center = pred + bias; width = w
            lower[:, fidxs, :, :, ch] = pred[:, fidxs, :, :, ch] + bias - w / 2.0
            upper[:, fidxs, :, :, ch] = pred[:, fidxs, :, :, ch] + bias + w / 2.0

    # pressure channel: use default ±5% (not scored but must be valid)
    lower[:, :, :, :, 2] = pred[:, :, :, :, 2] - 0.05 * np.abs(pred[:, :, :, :, 2])
    upper[:, :, :, :, 2] = pred[:, :, :, :, 2] + 0.05 * np.abs(pred[:, :, :, :, 2])

    return lower, upper


# ── SPS validation ────────────────────────────────────────────────────────────
def validate(
    scoring,
    pred: np.ndarray,     # (N, 20, 32, 64, 3)
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    c: int = 2,
) -> None:
    """Compare calibrated vs default SPS on held-out data using scoring.py."""
    # Calibrated
    sps_cal, cov_cal = scoring.aggregate_sps(pred, target, c, lower=lower, upper=upper)
    score_cal = scoring.score_sps(sps_cal)

    # Default (0.1 * |pred|)
    sps_def, cov_def = scoring.aggregate_sps(pred, target, c, lower=None, upper=None)
    score_def = scoring.score_sps(sps_def)

    print("\n" + "="*55)
    print("SPS Validation on held-out set")
    print("="*55)
    print(f"  Calibrated : sps_score={score_cal:.3f}  coverage={cov_cal:.4f}  raw={sps_cal:.6f}")
    print(f"  Default    : sps_score={score_def:.3f}  coverage={cov_def:.4f}  raw={sps_def:.6f}")
    delta = score_cal - score_def
    print(f"  Δ sps_score: {delta:+.3f} ({'better' if delta > 0 else 'worse'})")
    print("="*55)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args   = parse_args()
    device = torch.device(args.device)

    scoring = load_scoring(args.scoring_py)
    model, adapt_code = load_model(args.checkpoint, device)
    mean_inp, mean_tgt, std_inp, std_tgt = load_normalizer(args.normalizer, device)

    traj_path    = os.path.join(args.data_root, "foil", "hf_dataset", "real")
    trajectories = load_from_disk(traj_path)
    n_total      = len(trajectories)
    logging.info(f"Dataset: {n_total} trajectories")

    calib_idx = list(range(0, CALIB_END))
    val_idx   = list(range(CALIB_END, n_total))
    logging.info(f"Calibration: {calib_idx}  |  Validation: {val_idx}")

    # ── Collect calibration predictions ──────────────────────────────────────
    logging.info("Running calibration trajectories...")
    pred_cal, tgt_cal = collect_residuals(
        trajectories, calib_idx, model, adapt_code,
        mean_inp, mean_tgt, std_inp, std_tgt, device,
    )
    logging.info(f"Calibration data shape: {pred_cal.shape}")

    # ── Group residuals and grid search ──────────────────────────────────────
    groups = build_group_residuals(pred_cal, tgt_cal)

    print("\n" + "="*65)
    print(f"{'Group':<20} {'r_max(99.5%)':<14} {'w_opt':<12} {'coverage':<10} {'score'}")
    print("="*65)

    w_dict   = {}
    bias_dict = {}
    for (ch, bname), residuals in groups.items():
        r_max = float(np.percentile(residuals, 99.5))
        w_opt, cov, score = grid_search_width(residuals, args.n_candidates)
        w_dict[(ch, bname)] = w_opt

        # Bias check
        signed = (pred_cal[:, BUCKETS[bname], :, :, ch] - tgt_cal[:, BUCKETS[bname], :, :, ch])
        t_mask = (tgt_cal[:, BUCKETS[bname], :, :, ch] != 0.0)
        bias = float(np.mean(signed[t_mask])) if t_mask.any() else 0.0
        bias_dict[(ch, bname)] = bias

        ch_name = "u" if ch == 0 else "v"
        print(f"  ch={ch_name} {bname:<10}   {r_max:<14.5f} {w_opt:<12.5f} {cov:<10.4f} {score:.6f}")
        if abs(bias) > w_opt * 0.1:
            print(f"    ⚠ bias={bias:+.5f} (>10% of w_opt, applying correction)")
        else:
            bias_dict[(ch, bname)] = 0.0   # negligible bias, skip correction

    print("="*65)

    # ── Validate on held-out trajectories ────────────────────────────────────
    logging.info("Running validation trajectories...")
    pred_val, tgt_val = collect_residuals(
        trajectories, val_idx, model, adapt_code,
        mean_inp, mean_tgt, std_inp, std_tgt, device,
    )
    logging.info(f"Validation data shape: {pred_val.shape}")

    lower_val, upper_val = make_interval_arrays(pred_val, w_dict, bias_dict)
    validate(scoring, pred_val, tgt_val, lower_val, upper_val)

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Flatten dict keys for npz
    channels  = [k[0] for k in w_dict]
    buckets   = [k[1] for k in w_dict]
    widths    = [w_dict[k] for k in w_dict]
    biases    = [bias_dict.get(k, 0.0) for k in w_dict]

    np.savez(args.output,
             channels=channels, buckets=buckets,
             widths=widths, biases=biases,
             sigma_global=SIGMA_GLOBAL)
    logging.info(f"Saved → {args.output}")

    # Print lookup table for submission.py
    print("\nLookup table for submission.py:")
    print("W_TABLE = {")
    for (ch, bname), w in w_dict.items():
        bias = bias_dict.get((ch, bname), 0.0)
        print(f"    ({ch}, '{bname}'): ({w:.6f}, {bias:.6f}),  # width, bias")
    print("}")


if __name__ == "__main__":
    main()
