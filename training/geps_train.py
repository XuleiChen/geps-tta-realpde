"""
geps_train.py – Two-stage GEPS training for RealPDE Track 2 (Foil scenario).

Stage 1 – pretrain  : CFD simulation data (train_sim h5 files); all GEPS params trained.
Stage 2 – finetune  : PIV real data (train_real h5 files); continue from pretrain ckpt.

Each stage has its OWN dataset, its OWN sim_id → env_id mapping, and therefore
its OWN codes matrix (n_env rows × code_c cols).  The two stages are run as
separate processes with separate checkpoints.

Dataset layout expected (after unpacking competition tarballs)
-------------------------------------------------------------
    <data_dir>/
        3750_0.h5       # filename: {Re}_{AoA_int}.h5
        5000_2.h5
        ...
    Inside each h5: f['measured_data']['u'], f['measured_data']['v']
    shape: (n_sim_frame, H, W)  e.g. (868, 64, 128) for competition data

Usage (local small-data verification, CPU)
------------------------------------------
    cd RealPDEBench
    python geps_train.py --stage pretrain --data_dir /path/to/train_sim \\
        --device cpu --n_iter 20
    python geps_train.py --stage finetune --data_dir /path/to/train_real \\
        --device cpu --n_iter 20 --checkpoint <pretrain_ckpt.pth>

Usage (server, full data, CUDA)
--------------------------------
    python geps_train.py --stage pretrain --data_dir /path/to/train_sim \\
        --device cuda:0 --n_iter 10000
    python geps_train.py --stage finetune --data_dir /path/to/train_real \\
        --device cuda:0 --n_iter 5000 --checkpoint <pretrain_ckpt.pth>
"""

import os
import sys
import glob
import argparse
import logging
import time
import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from realpdebench.model.unet_geps import Unet3d as GEPSUnet3d
from realpdebench.utils.utils import set_seed, cycle


# ── Bad simulation IDs ─────────────────────────────────────────────────────────
# Known corrupted/degenerate trajectories to exclude from training.
BAD_SIM_IDS = {'7575_0'}


# ── Dataset ───────────────────────────────────────────────────────────────────

class GEPSFoilH5Dataset(Dataset):
    """Reads competition h5 files directly; returns (input, output, env_id).

    Parameters
    ----------
    data_dir     : path to directory containing {Re}_{AoA}.h5 files
    n_sim_frame  : total frames per trajectory (868 for competition data)
    in_step      : number of input frames  (20)
    out_step     : number of output frames (20)
    sub_s        : spatial sub-sampling stride (1 = native resolution)
    interval     : stride between sliding windows (controls number of samples)
    bad_sim_ids  : set of sim_id stems to exclude (e.g. {'7575_0'})

    Notes
    -----
    h5 layout: f['measured_data']['u'] and f['measured_data']['v']
               shape (n_sim_frame, H, W).
    Pressure p is all-zeros in real data; code uses p=zeros_like(u).
    For numerical data, p may exist under f['measured_data']['p'].
    A fallback to top-level f['u'] / f['v'] is included for non-standard files.
    """

    def __init__(
        self,
        data_dir: str,
        n_sim_frame: int = 868,
        in_step: int = 20,
        out_step: int = 20,
        sub_s: int = 1,
        interval: int = 20,
        bad_sim_ids=None,
    ):
        self.data_dir = data_dir
        self.n_sim_frame = n_sim_frame
        self.in_step = in_step
        self.out_step = out_step
        self.horizon = in_step + out_step
        self.sub_s = sub_s
        self.interval = interval
        self.bad_sim_ids = bad_sim_ids if bad_sim_ids is not None else BAD_SIM_IDS

        # Discover h5 files
        h5_paths = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        if not h5_paths:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")

        # Build sim_id list (stem without .h5), excluding bad sims
        sim_ids = []
        for p in h5_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in self.bad_sim_ids:
                logging.info(f"Skipping bad sim: {stem}")
                continue
            sim_ids.append(stem)

        if not sim_ids:
            raise ValueError(f"All h5 files excluded by bad_sim_ids: {self.bad_sim_ids}")

        # Sorted for deterministic env_id assignment
        sim_ids = sorted(sim_ids)
        self._sim_id_to_env_id = {s: i for i, s in enumerate(sim_ids)}
        self._sim_ids = sim_ids
        self.n_env = len(sim_ids)

        # Build flat index list: (sim_id, t_start)
        # Read actual frame count per file to handle variable-length trajectories.
        self._indices = []
        for sim_id in sim_ids:
            h5_path = os.path.join(data_dir, f"{sim_id}.h5")
            with h5py.File(h5_path, 'r') as f:
                try:
                    actual_frames = f['measured_data']['u'].shape[0]
                except KeyError:
                    actual_frames = f['u'].shape[0]
            max_t_start = actual_frames - self.horizon
            if max_t_start < 0:
                logging.warning(f"Skipping {sim_id}: only {actual_frames} frames, need {self.horizon}")
                continue
            for t_start in range(0, max_t_start + 1, interval):
                self._indices.append((sim_id, t_start))

        logging.info(
            f"GEPSFoilH5Dataset: {self.n_env} environments, "
            f"{len(self._indices)} samples  (data_dir={data_dir})"
        )

    # For GaussianNormalizer compatibility
    @property
    def dataset_dir(self):
        return self.data_dir

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sim_id, t_start = self._indices[idx]
        env_id = self._sim_id_to_env_id[sim_id]
        h5_path = os.path.join(self.data_dir, f"{sim_id}.h5")

        with h5py.File(h5_path, 'r') as f:
            u, v, p = self._read_uvp(f, t_start)

        data = np.stack([u, v, p], axis=-1)  # (horizon, H', W', 3)
        inp = torch.tensor(data[:self.in_step], dtype=torch.float32)
        out = torch.tensor(data[self.in_step:], dtype=torch.float32)
        return inp, out, torch.tensor(env_id, dtype=torch.long)

    def _read_uvp(self, f, t_start):
        """Read u, v, p from h5 file with layout fallback.

        h5 arrays have shape (n_sim_frame, H, W).
        We slice time first, then sub-sample spatial dims only.
        """
        t_end = t_start + self.horizon
        ss = self.sub_s  # spatial sub-sampling stride

        # Primary layout: measured_data group (competition training files)
        if 'measured_data' in f:
            g = f['measured_data']
            u = g['u'][t_start:t_end, ::ss, ::ss]
            v = g['v'][t_start:t_end, ::ss, ::ss]
            p = g['p'][t_start:t_end, ::ss, ::ss] if 'p' in g else np.zeros_like(u)
        # Fallback: top-level keys (competition example files)
        elif 'u' in f:
            u = f['u'][t_start:t_end, ::ss, ::ss]
            v = f['v'][t_start:t_end, ::ss, ::ss]
            p = np.zeros_like(u)
        else:
            raise KeyError(
                f"Cannot find 'u'/'v' in h5 file. Available keys: {list(f.keys())}"
            )
        return u, v, p


# ── Normalizer ────────────────────────────────────────────────────────────────

class GEPSNormalizer:
    """Gaussian normalizer for (input, output, env_id) datasets.

    Computes per-channel mean and std over the full dataset and caches them to
    disk.  Respects num_workers (unlike GaussianNormalizer which hardcodes 12).

    stats_path : where to save/load mean_std.pt (default: data_dir/mean_std.pt)
    """

    def __init__(self, dataset: 'GEPSFoilH5Dataset', device,
                 num_workers: int = 0, batch_size: int = 64,
                 stats_path: str = None):
        self.device = device
        if stats_path is None:
            stats_path = os.path.join(dataset.data_dir, "mean_std.pt")

        if os.path.exists(stats_path):
            mean_inp, mean_tgt, std_inp, std_tgt = torch.load(
                stats_path, map_location="cpu", weights_only=False
            )
            logging.info(f"Normalizer stats loaded from {stats_path}")
        else:
            logging.info("Computing normalizer stats …")
            mean_inp, mean_tgt, std_inp, std_tgt = self._compute(
                dataset, num_workers, batch_size
            )
            torch.save((mean_inp, mean_tgt, std_inp, std_tgt), stats_path)
            logging.info(f"Normalizer stats saved to {stats_path}")

        # Replace std=0 with 1 (constant channels, e.g. p=0)
        std_inp  = torch.where(std_inp  == 0, torch.ones_like(std_inp),  std_inp)
        std_tgt  = torch.where(std_tgt  == 0, torch.ones_like(std_tgt),  std_tgt)

        self.mean_inp = mean_inp.to(device)
        self.mean_tgt = mean_tgt.to(device)
        self.std_inp  = std_inp.to(device)
        self.std_tgt  = std_tgt.to(device)

    @staticmethod
    def _compute(dataset, num_workers, batch_size):
        """Welford-style single-pass mean / std over all samples."""
        from tqdm import tqdm

        def _collate(batch):
            # batch: list of (inp, out, env_id); stack only inp/out
            inps = torch.stack([b[0] for b in batch])
            outs = torch.stack([b[1] for b in batch])
            return inps, outs

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_collate)

        n = 0
        mean_inp = mean_tgt = 0.
        m2_inp   = m2_tgt   = 0.

        for inps, outs in tqdm(loader, desc="Normalizer stats"):
            b = inps.size(0)
            c_in, c_out = inps.size(-1), outs.size(-1)
            # flatten spatial+time dims, keep channel
            x = inps.view(b, -1, c_in).double()
            y = outs.view(b, -1, c_out).double()

            # channel-wise mean and variance across (b, spatial*time)
            batch_mean_inp = x.mean(dim=(0, 1))
            batch_mean_tgt = y.mean(dim=(0, 1))
            batch_var_inp  = x.var(dim=(0, 1), unbiased=False)
            batch_var_tgt  = y.var(dim=(0, 1), unbiased=False)

            # parallel Welford update
            n_new = n + b
            delta_inp = batch_mean_inp - mean_inp
            delta_tgt = batch_mean_tgt - mean_tgt
            mean_inp = mean_inp + delta_inp * b / n_new
            mean_tgt = mean_tgt + delta_tgt * b / n_new
            m2_inp   = m2_inp + batch_var_inp * b + delta_inp ** 2 * n * b / n_new
            m2_tgt   = m2_tgt + batch_var_tgt * b + delta_tgt ** 2 * n * b / n_new
            n = n_new

        std_inp = (m2_inp / n).sqrt().float()
        std_tgt = (m2_tgt / n).sqrt().float()
        return mean_inp.float(), mean_tgt.float(), std_inp, std_tgt

    def preprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        x = (x.to(self.device) - self.mean_inp[..., :c1]) / self.std_inp[..., :c1]
        y = (y.to(self.device) - self.mean_tgt[..., :c2]) / self.std_tgt[..., :c2]
        return x, y

    def postprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        x = x.to(self.device) * self.std_inp[..., :c1] + self.mean_inp[..., :c1]
        y = y.to(self.device) * self.std_tgt[..., :c2] + self.mean_tgt[..., :c2]
        return x, y


# ── Model helpers ──────────────────────────────────────────────────────────────

def build_model(args, n_env: int, device) -> GEPSUnet3d:
    model = GEPSUnet3d(
        dim=args.dim,
        n_env=n_env,
        out_channels=3,
        dim_mults=args.dim_mults,
        channels=3,
        in_time=args.in_step,
        out_time=args.out_step,
        code_c=args.code_c,
        factor=args.factor,
    )
    return model.to(device)


def load_pretrain_checkpoint(model, checkpoint_path: str, device):
    """Load pretrain checkpoint; discard codes (n_env differs in finetune)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    state = {k: v for k, v in state.items() if k != "codes"}

    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {"codes"}
    truly_missing = set(missing) - expected_missing
    if truly_missing:
        logging.warning(f"Missing keys (unexpected): {truly_missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint: {unexpected}")
    logging.info(
        f"Loaded pretrain checkpoint: {checkpoint_path} "
        f"(codes discarded; new codes initialised to zeros)"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GEPS 3D U-Net training (Foil)")

    # Stage
    parser.add_argument("--stage", choices=["pretrain", "finetune"],
                        default="pretrain",
                        help="pretrain=sim h5 data; finetune=real h5 data")

    # Data directories (competition tar.gz unpacked)
    parser.add_argument("--data_dir", default=None,
                        help="Directory of h5 files for the current stage. "
                             "Defaults to <dataset_root>/train_sim (pretrain) "
                             "or <dataset_root>/train_real (finetune).")
    parser.add_argument("--dataset_root",
                        default="/scratch2/xulei03/data",
                        help="Root directory; used to derive data_dir if not given")
    parser.add_argument("--sim_data_dir", default=None,
                        help="Override: directory of sim h5 files used to fit "
                             "the normalizer (defaults to train_sim inside dataset_root). "
                             "Always fit normalizer on sim data for train/eval consistency.")
    parser.add_argument("--results_path", default="./results/geps",
                        help="Directory where checkpoints are saved")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to pretrain checkpoint (required for finetune stage)")

    # Dataset
    parser.add_argument("--n_sim_frame", type=int, default=868,
                        help="Total frames per trajectory in competition data")
    parser.add_argument("--in_step",  type=int, default=20,
                        help="Input time steps")
    parser.add_argument("--out_step", type=int, default=20,
                        help="Output time steps")
    parser.add_argument("--sub_s",    type=int, default=1,
                        help="Spatial sub-sampling (1=native 64x128, 2=32x64)")
    parser.add_argument("--interval", type=int, default=20,
                        help="Sliding window stride (frames between samples)")

    # Model
    parser.add_argument("--dim",      type=int, default=64)
    parser.add_argument("--code_c",   type=int, default=8,
                        help="Dimension of per-environment context vector c^e")
    parser.add_argument("--factor",   type=int, default=1,
                        help="Low-rank perturbation scale")
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4])

    # Training
    parser.add_argument("--n_iter",       type=int,   default=10000)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--accum_steps",  type=int,   default=4,
                        help="Gradient accumulation steps. Effective batch = batch_size * accum_steps")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--clip_grad",    type=float, default=0.,
                        help="Gradient clipping norm (0 = disabled)")
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--log_every",    type=int,   default=100)
    parser.add_argument("--save_every",   type=int,   default=5000)
    parser.add_argument("--seed",         type=int,   default=42)

    # Device
    parser.add_argument("--device", default="cpu",
                        help="cpu | cuda:N | mps")

    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="realpde-track2-geps",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None,
                        help="W&B run name (defaults to geps_<stage>_<timestamp>)")

    return parser.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    # Resolve data directories
    if args.data_dir is None:
        sub = "train_sim" if args.stage == "pretrain" else "train_real"
        args.data_dir = os.path.join(args.dataset_root, sub)

    if args.sim_data_dir is None:
        args.sim_data_dir = os.path.join(args.dataset_root, "train_sim")

    logging.info(f"Stage: {args.stage} | data_dir: {args.data_dir} | device: {device}")

    # ── Training dataset ──────────────────────────────────────────────────────
    train_ds = GEPSFoilH5Dataset(
        data_dir=args.data_dir,
        n_sim_frame=args.n_sim_frame,
        in_step=args.in_step,
        out_step=args.out_step,
        sub_s=args.sub_s,
        interval=args.interval,
    )
    n_env = train_ds.n_env
    logging.info(f"Training environments: {n_env}")

    # ── Normalizer: fitted on the actual training data for each stage ─────────
    normalizer = GEPSNormalizer(train_ds, device=device,
                                num_workers=args.num_workers)

    train_loader = cycle(
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, n_env=n_env, device=device)

    if args.stage == "finetune":
        assert args.checkpoint is not None, \
            "--checkpoint is required for the finetune stage"
        load_pretrain_checkpoint(model, args.checkpoint, device)

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")
    logging.info(f"  of which codes: {model.codes.numel():,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_iter
    )

    # ── Output directory ──────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.results_path, f"geps_{args.stage}", ts)
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Results directory: {exp_dir}")

    # ── W&B init ──────────────────────────────────────────────────────────────
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE:
        logging.warning("wandb not installed; run `pip install wandb`. Continuing without W&B.")
    if use_wandb:
        run_name = args.wandb_run_name or f"geps_{args.stage}_{ts}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=exp_dir,
        )
        wandb.config.update({"n_env": n_env, "num_params": num_params,
                              "effective_batch_size": args.batch_size * args.accum_steps},
                             allow_val_change=True)
        logging.info(f"W&B run: {wandb.run.url}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for iteration in range(1, args.n_iter + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(args.accum_steps):
            inp, target, env_id = next(train_loader)
            inp    = inp.to(device)
            target = target.to(device)
            env_id = env_id.to(device)
            inp, target = normalizer.preprocess(inp, target)
            # Divide by accum_steps so the effective gradient equals a single
            # forward pass over (batch_size * accum_steps) samples
            loss = model.train_loss(inp, target, env_id).mean() / args.accum_steps
            loss.backward()
            accum_loss += loss.item()

        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        scheduler.step()

        running_loss += accum_loss

        if iteration % args.log_every == 0:
            avg = running_loss / args.log_every
            elapsed = time.time() - start_time
            codes_norm = model.codes.norm().item()
            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"[iter {iteration:6d}/{args.n_iter}]  "
                f"loss={avg:.6f}  "
                f"codes_norm={codes_norm:.4f}  "
                f"lr={current_lr:.2e}  "
                f"elapsed={elapsed/60:.1f}min"
            )
            if use_wandb:
                wandb.log({
                    "train/loss":       avg,
                    "train/codes_norm": codes_norm,
                    "train/lr":         current_lr,
                    "train/elapsed_min": elapsed / 60,
                }, step=iteration)
            running_loss = 0.0

        if iteration % args.save_every == 0 or iteration == args.n_iter:
            ckpt_path = os.path.join(
                exp_dir, f"geps_{args.stage}_{iteration:05d}.pth"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_env":      n_env,
                "code_c":     args.code_c,
                "factor":     args.factor,
                "dim":        args.dim,
                "dim_mults":  args.dim_mults,
                "in_step":    args.in_step,
                "out_step":   args.out_step,
                "sub_s":      args.sub_s,
                "stage":      args.stage,
            }, ckpt_path)
            logging.info(f"Checkpoint saved: {ckpt_path}")
            if use_wandb:
                wandb.save(ckpt_path)

    logging.info("Training complete.")
    if use_wandb:
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    args = parse_args()
    train(args)
