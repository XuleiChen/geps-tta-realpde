"""
compare_methods.py — 4-way autoregressive rollout comparison on real foil trajectories.

Conditions
----------
  C1: GEPS U-Net + adapt_code TTA           (gradient update c^e every block, 1 step)
  C2: Vanilla U-Net + no TTA                (pure autoregressive, frozen weights)
  C3: GEPS U-Net + adapt_code TTA + Ridge   (C1 + closed-form ridge on final_conv[1])
  C4: Vanilla U-Net + Ridge TTA             (closed-form ridge on final_conv[1])

Protocol
--------
  - Block 0 input: GT frames [0:W] (warm start, not scored)
  - Block b≥1 input: model's own prediction from block b-1 (autoregressive)
  - After predicting block b, GT for block b is revealed → ridge/adapt update
  - Metrics per block: rel_l2 (u+v, physical) and TKE error (physical)
  - Fresh model weights loaded per trajectory; no state bleeds across trajectories

Outputs
-------
  results/compare_methods/rel_l2_curve.png
  results/compare_methods/tke_curve.png
  results/compare_methods/per_trajectory_rel_l2.png
  results/compare_methods/summary.txt
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMISSION_DIR = os.path.join(os.path.dirname(_BENCH_DIR), "submission")
for p in [_BENCH_DIR, _SUBMISSION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import load_from_disk
from realpdebench.model.unet import Unet3d
from realpdebench.model.unet_geps import Unet3d as GEPSUnet3d
from ridge_tta import RidgeLastLayerAdapter
from geps_adapter import GEPSAdapter

# ── Config ────────────────────────────────────────────────────────────────────
VANILLA_CKPT = "/Users/xulei/Downloads/RealpdeTrack2/checkpoints/foil/unet/finetune.pth"
GEPS_CKPT    = "/Users/xulei/Downloads/RealpdeTrack2/submission/model.pth"

DATA_ROOT      = "/Users/xulei/Downloads/RealpdeTrack2/data"
HF_DATASET_DIR = os.path.join(DATA_ROOT, "foil", "hf_dataset", "real")
INDEX_PATH     = os.path.join(DATA_ROOT, "foil", "hf_dataset", "test_index_real.json")
MEAN_STD_PATH  = os.path.join(DATA_ROOT, "foil", "mean_std.pt")

SUB_S           = 2      # spatial downsampling (128×256 → 64×128)
N_EVAL_CHANNELS = 2      # u, v only
WINDOW_SIZE     = 20
RIDGE_LAM       = 1e-2
ADAPT_LR        = 1e-3
N_ADAPT_STEPS   = 1      # gradient steps per block for adapt_code
MAX_BLOCKS      = 15     # cap autoregressive rollout length

DEVICE     = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
OUTPUT_DIR = os.path.join(_BENCH_DIR, "results", "compare_methods")

CONDITION_NAMES = {
    "c1": "GEPS + adapt_code TTA",
    "c2": "Vanilla (no TTA)",
    "c3": "GEPS + adapt_code + Ridge",
    "c4": "Vanilla + Ridge TTA",
}
COLORS = {"c1": "darkorange", "c2": "steelblue", "c3": "crimson", "c4": "seagreen"}

# ── Model builders ────────────────────────────────────────────────────────────

def build_vanilla():
    ckpt = torch.load(VANILLA_CKPT, map_location="cpu", weights_only=False)
    model = Unet3d(
        dim=ckpt["dim"], out_channels=3,
        dim_mults=tuple(ckpt["dim_mults"]),
        channels=3, in_time=ckpt["in_step"], out_time=ckpt["out_step"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model


def build_geps():
    ckpt = torch.load(GEPS_CKPT, map_location="cpu", weights_only=False)
    model = GEPSUnet3d(
        dim=ckpt["dim"], n_env=ckpt["n_env"], out_channels=3,
        dim_mults=tuple(ckpt["dim_mults"]),
        channels=3, in_time=ckpt["in_step"], out_time=ckpt["out_step"],
        code_c=ckpt["code_c"], factor=ckpt["factor"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model

# ── Data helpers ──────────────────────────────────────────────────────────────

def get_test_sim_ids():
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    return sorted(set(e["sim_id"] for e in idx))


def load_full_trajectory(dataset, sim_id_to_idx, sim_id):
    row = dataset[sim_id_to_idx[sim_id]]
    shape = (row["shape_t"], row["shape_h"], row["shape_w"])
    u = np.frombuffer(row["u"], dtype=np.float32).reshape(shape)
    v = np.frombuffer(row["v"], dtype=np.float32).reshape(shape)
    u = u[:, ::SUB_S, ::SUB_S]
    v = v[:, ::SUB_S, ::SUB_S]
    p = np.zeros_like(u)
    return torch.tensor(np.stack([u, v, p], axis=-1), dtype=torch.float32)


def load_normalizer():
    mi, mt, si, st = torch.load(MEAN_STD_PATH, weights_only=True)
    si = torch.where(si == 0, torch.ones_like(si), si)
    st = torch.where(st == 0, torch.ones_like(st), st)
    return mi.to(DEVICE), mt.to(DEVICE), si.to(DEVICE), st.to(DEVICE)

# ── Metrics ───────────────────────────────────────────────────────────────────

def rel_l2(pred_phys, gt_phys):
    """Relative L2 on u+v channels in physical space."""
    p = pred_phys[..., :N_EVAL_CHANNELS]
    t = gt_phys[..., :N_EVAL_CHANNELS]
    return (torch.linalg.norm((p - t).reshape(-1)) /
            (torch.linalg.norm(t.reshape(-1)) + 1e-12)).item()


def tke_error(pred_phys, gt_phys):
    """Mean absolute TKE error (physical space).  pred/gt: (1, T, H, W, C)."""
    def tke(x):
        u_prime = (x[..., 0] - x[..., 0].mean(dim=1, keepdim=True)) ** 2
        v_prime = (x[..., 1] - x[..., 1].mean(dim=1, keepdim=True)) ** 2
        return 0.5 * (u_prime + v_prime).mean(dim=1)   # (1, H, W)
    return (tke(pred_phys) - tke(gt_phys)).abs().mean().item()

# ── Per-trajectory rollout ────────────────────────────────────────────────────

def run_one_trajectory(trajectory, mi, mt, si, st, sim_id):
    """
    Returns dict of lists, one entry per block:
        { "c1_rl2", "c2_rl2", "c3_rl2", "c4_rl2",
          "c1_tke", "c2_tke", "c3_tke", "c4_tke" }

    Bug fixes applied
    -----------------
    Fix 1 (model isolation): each condition gets its OWN model instance so that
    ridge updates to C3/C4's final_conv[1] cannot contaminate C1/C2.

    Fix 2 (adapt_code update semantics): geps_adapter.update(x, y) re-runs a
    forward pass on x and computes MSE(pred, y).  x must therefore be the
    normalised MODEL INPUT for this block (mi/si space), NOT the prediction
    output, and NOT renormalised with mt/st.  We save c1_norm_in / c3_norm_in
    before the variable name is reused.
    """
    n_frames = trajectory.shape[0]
    n_blocks = min((n_frames - 2 * WINDOW_SIZE) // WINDOW_SIZE, MAX_BLOCKS)
    if n_blocks < 1:
        print(f"  [{sim_id}] too short ({n_frames} frames), skipping")
        return None

    # ── FIX 1: one independent model instance per condition ───────────────────
    model_c1 = build_geps()     # GEPS, adapt_code only
    model_c2 = build_vanilla()  # Vanilla, fully frozen
    model_c3 = build_geps()     # GEPS, adapt_code + ridge
    model_c4 = build_vanilla()  # Vanilla, ridge only

    for model in (model_c1, model_c2, model_c3, model_c4):
        for p in model.parameters():
            p.requires_grad_(False)

    # Adapters — each wraps its own dedicated model
    adapter_c1   = GEPSAdapter(model_c1, adapt_lr=ADAPT_LR, n_steps=N_ADAPT_STEPS)
    adapter_c3   = GEPSAdapter(model_c3, adapt_lr=ADAPT_LR, n_steps=N_ADAPT_STEPS)
    ridge_c3     = RidgeLastLayerAdapter(model_c3, lam=RIDGE_LAM)
    ridge_c4     = RidgeLastLayerAdapter(model_c4, lam=RIDGE_LAM)
    adapter_c1.reset()
    adapter_c3.reset()

    # Sanity check: snapshot C2 final_conv[1] to verify it never changes
    _c2_w0 = model_c2.final_conv[1].weight.detach().clone()

    # Per-condition rollout inputs (physical), shape [W, H, W_spatial, 3]
    inp = {k: trajectory[:WINDOW_SIZE].clone() for k in ("c1", "c2", "c3", "c4")}

    results = {k: [] for k in
               ["c1_rl2", "c2_rl2", "c3_rl2", "c4_rl2",
                "c1_tke", "c2_tke", "c3_tke", "c4_tke"]}
    adapt_losses_c1, adapt_losses_c3 = [], []

    for b in range(n_blocks):
        raw_gt  = trajectory[(b+1)*WINDOW_SIZE : (b+2)*WINDOW_SIZE].unsqueeze(0).to(DEVICE)
        norm_gt = (raw_gt - mt) / st    # normalised GT (mt/st space)

        preds_phys = {}

        # ── C2: vanilla, fully frozen ──────────────────────────────────────────
        c2_norm_in = (inp["c2"].unsqueeze(0).to(DEVICE) - mi) / si
        with torch.no_grad():
            preds_phys["c2"] = model_c2(c2_norm_in) * st + mt

        # ── C4: vanilla + Ridge TTA ────────────────────────────────────────────
        c4_norm_in = (inp["c4"].unsqueeze(0).to(DEVICE) - mi) / si
        ridge_c4.enable_capture()
        with torch.no_grad():
            pred_norm_c4 = model_c4(c4_norm_in)
        ridge_c4.disable_capture()
        preds_phys["c4"] = pred_norm_c4 * st + mt
        ridge_c4.ridge_update(norm_gt)

        # ── C1: GEPS + adapt_code TTA (no ridge) ──────────────────────────────
        # Save norm_in NOW — variable will be reused for C3 below
        c1_norm_in = (inp["c1"].unsqueeze(0).to(DEVICE) - mi) / si
        preds_phys["c1"] = adapter_c1.predict(c1_norm_in) * st + mt

        # ── C3: GEPS + adapt_code + Ridge TTA ─────────────────────────────────
        c3_norm_in = (inp["c3"].unsqueeze(0).to(DEVICE) - mi) / si
        ridge_c3.enable_capture()
        pred_norm_c3 = adapter_c3.predict(c3_norm_in)
        ridge_c3.disable_capture()
        preds_phys["c3"] = pred_norm_c3 * st + mt
        ridge_c3.ridge_update(norm_gt)

        # ── Record metrics ─────────────────────────────────────────────────────
        for cond in ("c1", "c2", "c3", "c4"):
            results[f"{cond}_rl2"].append(rel_l2(preds_phys[cond], raw_gt))
            results[f"{cond}_tke"].append(tke_error(preds_phys[cond], raw_gt))

        # ── Rollout: next block input = this block's prediction (physical) ─────
        for cond in ("c1", "c2", "c3", "c4"):
            inp[cond] = preds_phys[cond].squeeze(0).cpu()

        # ── FIX 2: adapt_code update — use this block's normalised INPUT ───────
        # geps_adapter.update(x, y) does: pred = model(x, adapt_code); loss = MSE(pred,y)
        # so x must be the mi/si-normalised model input, not the prediction output.
        # C1 and C3 now have independent adapt_codes via separate adapter instances.
        adapter_c1.update(c1_norm_in, norm_gt)
        adapter_c3.update(c3_norm_in, norm_gt)

        # Track adapt loss (post-update forward, no grad) for sanity logging
        with torch.no_grad():
            loss_c1 = F.mse_loss(
                adapter_c1.predict(c1_norm_in)[..., :2], norm_gt[..., :2]
            ).item()
            loss_c3 = F.mse_loss(
                adapter_c3.predict(c3_norm_in)[..., :2], norm_gt[..., :2]
            ).item()
        adapt_losses_c1.append(loss_c1)
        adapt_losses_c3.append(loss_c3)

        if (b + 1) % 5 == 0 or b == n_blocks - 1:
            # Weight norms to confirm C1/C3 final_conv[1] evolve independently
            w1 = model_c1.final_conv[1].weight.norm().item()
            w3 = model_c3.final_conv[1].weight.norm().item()
            print(f"  [{sim_id}] b={b+1:2d}/{n_blocks} "
                  f"| rl2 c1={results['c1_rl2'][-1]:.4f} c2={results['c2_rl2'][-1]:.4f} "
                  f"c3={results['c3_rl2'][-1]:.4f} c4={results['c4_rl2'][-1]:.4f} "
                  f"| adapt_loss c1={loss_c1:.4f} c3={loss_c3:.4f} "
                  f"| ||w_fc1|| c1={w1:.4f} c3={w3:.4f}")

    # ── Sanity check: C2 weights must be bit-for-bit unchanged ────────────────
    _c2_w_final = model_c2.final_conv[1].weight.detach()
    assert torch.equal(_c2_w0, _c2_w_final), \
        f"[{sim_id}] SANITY FAIL: C2 final_conv[1] weights changed — model isolation broken!"
    print(f"  [{sim_id}] Sanity OK: C2 final_conv[1] unchanged across {n_blocks} blocks.")

    results["n_blocks"] = n_blocks
    results["adapt_losses_c1"] = adapt_losses_c1
    results["adapt_losses_c3"] = adapt_losses_c3
    return results

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_envelope(ax, xs, curves, color, label):
    arr  = np.array(curves)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    ax.plot(xs, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)


def plot_results(all_results, sim_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    min_blocks = min(r["n_blocks"] for r in all_results)
    xs = list(range(1, min_blocks + 1))   # 1-indexed (block 1 = first prediction)

    for metric, ylabel, fname in [
        ("rl2", "Relative L2 error  (u, v  —  physical)", "rel_l2_curve.png"),
        ("tke", "TKE error  (physical)",                   "tke_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for cond in ("c1", "c2", "c3", "c4"):
            curves = [r[f"{cond}_{metric}"][:min_blocks] for r in all_results]
            plot_envelope(ax, xs, curves, COLORS[cond], CONDITION_NAMES[cond])
        ax.set_xlabel("Block index (autoregressive step)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Autoregressive rollout comparison  —  {len(all_results)} trajectories  (mean ± 1σ)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        p = os.path.join(output_dir, fname)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

    # Per-trajectory rel_l2 (one subplot per trajectory)
    n_traj = len(all_results)
    ncols = min(4, n_traj)
    nrows = (n_traj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    for i, (r, sid) in enumerate(zip(all_results, sim_ids)):
        ax = axes[i // ncols][i % ncols]
        n = r["n_blocks"]
        x = list(range(1, n + 1))
        for cond in ("c1", "c2", "c3", "c4"):
            ax.plot(x, r[f"{cond}_rl2"], color=COLORS[cond],
                    label=CONDITION_NAMES[cond], linewidth=1)
        ax.set_title(sid, fontsize=8)
        ax.set_xlabel("Block"); ax.set_ylabel("Rel L2")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)
    # hide unused subplots
    for j in range(n_traj, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.tight_layout()
    p = os.path.join(output_dir, "per_trajectory_rel_l2.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(all_results, sim_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.txt")
    lines = []

    lines.append(f"Trajectories: {len(all_results)}")
    lines.append("")
    lines.append(f"{'Condition':<35s}  {'rel_l2 mean':>12s}  {'rel_l2 std':>10s}  "
                 f"{'tke mean':>10s}  {'tke std':>10s}")
    lines.append("-" * 85)
    for cond, name in CONDITION_NAMES.items():
        all_rl2 = [e for r in all_results for e in r[f"{cond}_rl2"]]
        all_tke = [e for r in all_results for e in r[f"{cond}_tke"]]
        lines.append(f"{name:<35s}  {np.mean(all_rl2):12.5f}  {np.std(all_rl2):10.5f}  "
                     f"{np.mean(all_tke):10.5f}  {np.std(all_tke):10.5f}")

    lines.append("")
    lines.append("Per-trajectory rel_l2 means:")
    header = f"{'sim_id':25s}  " + "  ".join(f"{c:>10s}" for c in CONDITION_NAMES)
    lines.append(header)
    for r, sid in zip(all_results, sim_ids):
        row = f"{sid:25s}  "
        row += "  ".join(f"{np.mean(r[f'{c}_rl2']):10.5f}" for c in CONDITION_NAMES)
        lines.append(row)

    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nSummary saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"GEPS   : {GEPS_CKPT}")
    print(f"Vanilla: {VANILLA_CKPT}")

    mi, mt, si, st = load_normalizer()
    sim_ids = get_test_sim_ids()
    print(f"\nTest trajectories ({len(sim_ids)}): {sim_ids}\n")

    print("Loading dataset ...")
    dataset = load_from_disk(HF_DATASET_DIR)
    sim_id_to_idx = {dataset[i]["sim_id"]: i for i in range(len(dataset))}

    all_results = []
    for tid, sim_id in enumerate(sim_ids):
        print(f"\n── Trajectory {tid+1}/{len(sim_ids)}: {sim_id} ──")
        traj = load_full_trajectory(dataset, sim_id_to_idx, sim_id)
        T, H, W, C = traj.shape
        print(f"   Shape: {T} x {H}×{W} x {C}")
        result = run_one_trajectory(traj, mi, mt, si, st, sim_id)
        if result is not None:
            all_results.append(result)

    write_summary(all_results, sim_ids, OUTPUT_DIR)
    plot_results(all_results, sim_ids, OUTPUT_DIR)
    print("\nDone.")
