"""
online_tta.py  –  Track2 Online Ridge-TTA, rollout mode

Evaluation protocol (aligned with the competition's LTTTA setting):
  - For each test trajectory, start from the first 20 ground-truth frames.
  - Predict the next 20 frames; feed that prediction (NOT the GT) as input
    to the next block.  This is autoregressive rollout: errors compound.
  - After predicting block b, reveal the GT for block b and ridge-update
    the TTA model's last layer.  The baseline model is never updated.
  - Metrics are computed in PHYSICAL units (identical to eval.py convention).

Runs on all 20 real test trajectories (from test_index_real.json).
Fresh model weights are loaded for each trajectory (TTA does not carry
weight state across trajectories).

Outputs
-------
  results/online_tta/error_curve.png          mean ± 1-std envelope
  results/online_tta/weight_update_norm.png   mean ± 1-std envelope
  results/online_tta/per_trajectory.png       individual curves
  results/online_tta/summary.txt              numerical summary
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_from_disk
from realpdebench.model.unet import Unet3d
from ridge_tta import RidgeLastLayerAdapter


# ── Config ──────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "/Users/xulei/Downloads/RealpdeTrack2/checkpoints/foil/unet/finetune.pth"
DATA_ROOT       = "/Users/xulei/Downloads/RealpdeTrack2/data"
HF_DATASET_DIR  = os.path.join(DATA_ROOT, "foil", "hf_dataset", "real")
INDEX_PATH      = os.path.join(DATA_ROOT, "foil", "hf_dataset", "test_index_real.json")
MEAN_STD_PATH   = os.path.join(DATA_ROOT, "foil", "mean_std.pt")

SUB_S           = 2      # sub_s_real for foil
N_EVAL_CHANNELS = 2      # evaluate u, v only (p is unmeasured / all-zero)
WINDOW_SIZE     = 20     # in_step == out_step == 20
LAM             = 1e-2   # ridge regularisation

DEVICE     = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
OUTPUT_DIR = "./results/online_tta"


# ── Model ───────────────────────────────────────────────────────────────────

def build_model():
    model = Unet3d(
        dim=64, out_channels=3, dim_mults=[1, 2, 4],
        channels=3, in_time=WINDOW_SIZE, out_time=WINDOW_SIZE,
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


# ── Data ────────────────────────────────────────────────────────────────────

def get_test_sim_ids():
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    return sorted(set(e["sim_id"] for e in idx))


def load_full_trajectory(dataset, sim_id_to_idx, sim_id):
    """
    Return [T, H, W, 3] float32 tensor in physical units.
    Applies sub_s=2 spatial downsampling (128x256 → 64x128).
    Channel order: [u, v, p=0].
    """
    row = dataset[sim_id_to_idx[sim_id]]
    full_shape = (row["shape_t"], row["shape_h"], row["shape_w"])
    u = np.frombuffer(row["u"], dtype=np.float32).reshape(full_shape)
    v = np.frombuffer(row["v"], dtype=np.float32).reshape(full_shape)
    u = u[:, ::SUB_S, ::SUB_S]
    v = v[:, ::SUB_S, ::SUB_S]
    p = np.zeros_like(u)
    data = np.stack([u, v, p], axis=-1)
    return torch.tensor(data, dtype=torch.float32)


def load_normalizer():
    mi, mt, si, st = torch.load(MEAN_STD_PATH, weights_only=True)
    si = torch.where(si == 0, torch.ones_like(si), si)
    st = torch.where(st == 0, torch.ones_like(st), st)
    return (mi.to(DEVICE), mt.to(DEVICE), si.to(DEVICE), st.to(DEVICE))


# ── Metric ──────────────────────────────────────────────────────────────────

def rel_l2_physical(pred_phys, gt_phys):
    """Relative L2 in physical space, u+v channels only (matches eval.py)."""
    p = pred_phys[..., :N_EVAL_CHANNELS]
    t = gt_phys[..., :N_EVAL_CHANNELS]
    return (torch.linalg.norm((p - t).reshape(-1)) /
            (torch.linalg.norm(t.reshape(-1)) + 1e-12)).item()


def weight_delta_norm(w_old, b_old, w_new, b_new):
    dw = torch.linalg.norm((w_new - w_old).reshape(-1))
    db = (torch.linalg.norm((b_new - b_old).reshape(-1))
          if b_old is not None else torch.tensor(0., device=w_old.device))
    return torch.sqrt(dw**2 + db**2).item()


# ── Per-trajectory rollout ──────────────────────────────────────────────────

def run_one_trajectory(trajectory, mi, mt, si, st, sim_id):
    """
    Autoregressive rollout on one trajectory.

    Protocol:
      Block 0:   input = GT frames [0:W]         (only this one uses GT as input)
      Block b>0: input = model's own pred_{b-1}  (rollout)

    Baseline: frozen weights, rollout input = baseline's own predictions.
    TTA:      ridge-updates last layer after each block using revealed GT,
              rollout input = TTA's own predictions (pre-update).

    Metrics computed in physical space (pred_phys vs raw_gt).
    """
    n_frames = trajectory.shape[0]
    n_blocks = (n_frames - 2 * WINDOW_SIZE) // WINDOW_SIZE
    if n_blocks < 1:
        return None

    tta_model      = build_model()
    baseline_model = build_model()

    assert torch.allclose(
        tta_model.final_conv[1].weight.cpu(),
        baseline_model.final_conv[1].weight.cpu(),
    ), "Weight identity check failed"

    adapter = RidgeLastLayerAdapter(tta_model, lam=LAM)
    adapter.freeze_backbone()

    # Initial rollout inputs: GT first window, physical units
    tta_input_phys      = trajectory[:WINDOW_SIZE]          # [W, H, W, 3]
    baseline_input_phys = trajectory[:WINDOW_SIZE].clone()

    tta_errs, bl_errs, dw_norms = [], [], []

    for b in range(n_blocks):
        # Ground truth for this block (physical)
        raw_gt = trajectory[(b+1)*WINDOW_SIZE : (b+2)*WINDOW_SIZE].unsqueeze(0).to(DEVICE)
        norm_gt = (raw_gt - mt) / st  # out-space, for ridge update

        # ── Baseline ──────────────────────────────────────────────────────
        norm_bl_in = (baseline_input_phys.unsqueeze(0).to(DEVICE) - mi) / si
        with torch.no_grad():
            bl_pred_norm = baseline_model(norm_bl_in)
        bl_pred_phys = bl_pred_norm * st + mt
        bl_errs.append(rel_l2_physical(bl_pred_phys, raw_gt))

        # ── TTA ───────────────────────────────────────────────────────────
        norm_tta_in = (tta_input_phys.unsqueeze(0).to(DEVICE) - mi) / si
        adapter.enable_capture()
        with torch.no_grad():
            tta_pred_norm = tta_model(norm_tta_in)
        adapter.disable_capture()
        tta_pred_phys = tta_pred_norm * st + mt
        tta_errs.append(rel_l2_physical(tta_pred_phys, raw_gt))

        # Ridge update with revealed GT
        w_old = adapter.last_layer.weight.detach().clone()
        b_old = (adapter.last_layer.bias.detach().clone()
                 if adapter.last_layer.bias is not None else None)
        adapter.ridge_update(norm_gt)
        w_new = adapter.last_layer.weight.detach().clone()
        b_new = (adapter.last_layer.bias.detach().clone()
                 if adapter.last_layer.bias is not None else None)
        dw_norms.append(weight_delta_norm(w_old, b_old, w_new, b_new))

        # ── Rollout: next input = this block's prediction (physical) ──────
        baseline_input_phys = bl_pred_phys.squeeze(0).cpu()
        tta_input_phys      = tta_pred_phys.squeeze(0).cpu()  # pre-update pred

        if (b + 1) % 20 == 0 or b == n_blocks - 1:
            print(f"  [{sim_id}] block {b+1:3d}/{n_blocks}  "
                  f"bl={bl_errs[-1]:.4f}  tta={tta_errs[-1]:.4f}  "
                  f"||dw||={dw_norms[-1]:.4f}")

    return {
        "tta_errors":          tta_errs,
        "baseline_errors":     bl_errs,
        "weight_update_norms": dw_norms,
        "n_blocks":            n_blocks,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_envelope(ax, xs, curves, color, label):
    arr  = np.array(curves)                  # [n_traj, n_blocks]
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    ax.plot(xs, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)


def plot_results(all_results, sim_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    min_blocks = min(r["n_blocks"] for r in all_results)
    xs = list(range(min_blocks))

    bl_curves  = [r["baseline_errors"][:min_blocks]     for r in all_results]
    tta_curves = [r["tta_errors"][:min_blocks]           for r in all_results]
    dw_curves  = [r["weight_update_norms"][:min_blocks]  for r in all_results]

    # Error envelope
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_envelope(ax, xs, bl_curves,  color="steelblue",  label="No TTA (frozen)")
    plot_envelope(ax, xs, tta_curves, color="darkorange", label="Ridge TTA")
    ax.set_xlabel("Block index (autoregressive step along trajectory)")
    ax.set_ylabel("Relative L2 error  (u, v  –  physical units)")
    ax.set_title(f"Online TTA rollout  –  {len(all_results)} test trajectories  "
                 f"(mean ± 1σ)")
    ax.legend(); ax.grid(True, alpha=0.3)
    p1 = os.path.join(output_dir, "error_curve.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

    # Weight update envelope
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_envelope(ax, xs, dw_curves, color="darkred", label="||Δw||_F")
    ax.set_xlabel("Block index"); ax.set_ylabel("||w_new − w_old||_F")
    ax.set_title("Ridge update magnitude per block  (mean ± 1σ)")
    ax.legend(); ax.grid(True, alpha=0.3)
    p2 = os.path.join(output_dir, "weight_update_norm.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()

    # Per-trajectory curves
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    cmap = plt.cm.tab20
    for i, (r, sid) in enumerate(zip(all_results, sim_ids)):
        c = cmap(i / len(all_results))
        n = r["n_blocks"]
        axes[0].plot(range(n), r["baseline_errors"], color=c, alpha=0.6,
                     linewidth=1, linestyle="--")
        axes[0].plot(range(n), r["tta_errors"], color=c, alpha=0.9,
                     linewidth=1, label=sid)
        axes[1].plot(range(n), r["weight_update_norms"], color=c,
                     linewidth=1, alpha=0.8)
    axes[0].set_ylabel("Rel L2  (physical)")
    axes[0].set_title("Per-trajectory: dashed=baseline, solid=TTA")
    axes[0].legend(fontsize=6, ncol=4, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Block index"); axes[1].set_ylabel("||Δw||")
    axes[1].set_title("Weight update norm per trajectory")
    axes[1].grid(True, alpha=0.3)
    p3 = os.path.join(output_dir, "per_trajectory.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()

    print(f"\nSaved: {p1}\nSaved: {p2}\nSaved: {p3}")
    return p1, p2, p3


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"LAM    : {LAM}")

    mi, mt, si, st = load_normalizer()
    sim_ids = get_test_sim_ids()
    print(f"\nTest trajectories ({len(sim_ids)}): {sim_ids}\n")

    print("Loading Arrow dataset ...")
    dataset = load_from_disk(HF_DATASET_DIR)
    sim_id_to_idx = {dataset[i]["sim_id"]: i for i in range(len(dataset))}

    all_results = []
    for tid, sim_id in enumerate(sim_ids):
        print(f"\n── Trajectory {tid+1}/{len(sim_ids)}: {sim_id} ──")
        traj = load_full_trajectory(dataset, sim_id_to_idx, sim_id)
        T, H, W, C = traj.shape
        print(f"   Shape: {T} x {H}x{W} x {C}")
        result = run_one_trajectory(traj, mi, mt, si, st, sim_id)
        if result is not None:
            all_results.append(result)

    # ── Aggregate summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("=== Aggregate Summary ===")
    print(f"Trajectories   : {len(all_results)}")

    all_bl  = [e for r in all_results for e in r["baseline_errors"]]
    all_tta = [e for r in all_results for e in r["tta_errors"]]
    all_dw  = [e for r in all_results for e in r["weight_update_norms"]]

    print(f"Baseline rel-L2: mean={np.mean(all_bl):.5f}  "
          f"std={np.std(all_bl):.5f}  "
          f"min={np.min(all_bl):.5f}  max={np.max(all_bl):.5f}")
    print(f"TTA      rel-L2: mean={np.mean(all_tta):.5f}  "
          f"std={np.std(all_tta):.5f}  "
          f"min={np.min(all_tta):.5f}  max={np.max(all_tta):.5f}")
    print(f"||dw||         : mean={np.mean(all_dw):.5f}  max={np.max(all_dw):.5f}")

    # Per-trajectory means
    print("\nPer-trajectory means:")
    print(f"{'sim_id':25s}  {'bl_mean':>9s}  {'tta_mean':>9s}  {'delta':>9s}")
    for r, sid in zip(all_results, sim_ids):
        bm  = np.mean(r["baseline_errors"])
        tm  = np.mean(r["tta_errors"])
        print(f"{sid:25s}  {bm:9.5f}  {tm:9.5f}  {tm-bm:+9.5f}")

    # Save summary text
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Trajectories: {len(all_results)}\n")
        f.write(f"Baseline rel-L2: mean={np.mean(all_bl):.5f} std={np.std(all_bl):.5f}\n")
        f.write(f"TTA      rel-L2: mean={np.mean(all_tta):.5f} std={np.std(all_tta):.5f}\n")
        f.write(f"||dw||:          mean={np.mean(all_dw):.5f} max={np.max(all_dw):.5f}\n\n")
        f.write(f"{'sim_id':25s}  {'bl_mean':>9s}  {'tta_mean':>9s}  {'delta':>9s}\n")
        for r, sid in zip(all_results, sim_ids):
            bm = np.mean(r["baseline_errors"])
            tm = np.mean(r["tta_errors"])
            f.write(f"{sid:25s}  {bm:9.5f}  {tm:9.5f}  {tm-bm:+9.5f}\n")
    print(f"\nSummary saved: {summary_path}")

    plot_results(all_results, sim_ids, OUTPUT_DIR)
