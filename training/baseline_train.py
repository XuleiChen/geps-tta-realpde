"""
baseline_train.py – Two-stage vanilla Unet3d training for RealPDE Track 2.

Identical training procedure to geps_train.py EXCEPT:
  - Uses vanilla Unet3d (no GEPS, no codes, no env_id)
  - Loss computed on u,v channels only (same as GEPS version)

Usage
-----
    # Pretrain on sim data
    python baseline_train.py --stage pretrain --dataset_root /scratch/xulei03/data \\
        --device cuda:0 --n_iter 20000 --wandb --wandb_run_name baseline_pretrain_20k

    # Finetune on real data
    python baseline_train.py --stage finetune --dataset_root /scratch/xulei03/data \\
        --checkpoint <pretrain_ckpt.pth> \\
        --device cuda:0 --n_iter 10000 --lr 5e-5 --wandb --wandb_run_name baseline_finetune_10k
"""

import os
import sys
import argparse
import logging
import time
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse dataset and normalizer from geps_train (ensures identical data pipeline)
from geps_train import GEPSFoilH5Dataset, GEPSNormalizer

from realpdebench.model.unet import Unet3d as VanillaUnet3d
from realpdebench.utils.metrics import mse_loss
from realpdebench.utils.utils import set_seed, cycle


# ── Model helpers ──────────────────────────────────────────────────────────────

def build_model(args, device) -> VanillaUnet3d:
    model = VanillaUnet3d(
        dim=args.dim,
        out_channels=3,
        dim_mults=args.dim_mults,
        channels=3,
        in_time=args.in_step,
        out_time=args.out_step,
    )
    return model.to(device)


def load_pretrain_checkpoint(model, checkpoint_path: str, device):
    """Load pretrain checkpoint for vanilla UNet (no codes to discard)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")
    logging.info(f"Loaded pretrain checkpoint: {checkpoint_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Baseline Unet3d training for RealPDE Track 2")

    parser.add_argument("--stage", choices=["pretrain", "finetune"],
                        required=True,
                        help="pretrain=sim data; finetune=real data")

    # Data directories
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--dataset_root", default="/scratch2/xulei03/data")
    parser.add_argument("--sim_data_dir", default=None)
    parser.add_argument("--results_path", default="./results/baseline")
    parser.add_argument("--checkpoint", default=None,
                        help="Pretrain checkpoint path (required for finetune)")

    # Dataset
    parser.add_argument("--n_sim_frame", type=int, default=868)
    parser.add_argument("--in_step",     type=int, default=20)
    parser.add_argument("--out_step",    type=int, default=20)
    parser.add_argument("--sub_s",       type=int, default=1)
    parser.add_argument("--interval",    type=int, default=20)

    # Model
    parser.add_argument("--dim",       type=int, default=64)
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4])

    # Training
    parser.add_argument("--n_iter",      type=int,   default=10000)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--accum_steps", type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--clip_grad",   type=float, default=0.)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--log_every",   type=int,   default=100)
    parser.add_argument("--save_every",  type=int,   default=10000)
    parser.add_argument("--seed",        type=int,   default=42)

    # Device
    parser.add_argument("--device", default="cpu")

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="realpde-track2-geps")
    parser.add_argument("--wandb_run_name", default=None)

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

    # Dataset
    train_ds = GEPSFoilH5Dataset(
        data_dir=args.data_dir,
        n_sim_frame=args.n_sim_frame,
        in_step=args.in_step,
        out_step=args.out_step,
        sub_s=args.sub_s,
        interval=args.interval,
    )
    logging.info(f"Training samples: {len(train_ds)}")

    # Normalizer: fitted on the actual training data for each stage
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

    # Model
    model = build_model(args, device)
    if args.stage == "finetune":
        assert args.checkpoint is not None, "--checkpoint is required for finetune"
        load_pretrain_checkpoint(model, args.checkpoint, device)

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_iter
    )

    # Output directory
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.results_path, f"baseline_{args.stage}", ts)
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Results directory: {exp_dir}")

    # W&B
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE:
        logging.warning("wandb not installed. Continuing without W&B.")
    if use_wandb:
        run_name = args.wandb_run_name or f"baseline_{args.stage}_{ts}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=exp_dir,
        )
        wandb.config.update({"num_params": num_params}, allow_val_change=True)
        logging.info(f"W&B run: {wandb.run.url}")

    # Training loop
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for iteration in range(1, args.n_iter + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(args.accum_steps):
            inp, target, _ = next(train_loader)   # env_id ignored for baseline
            inp    = inp.to(device)
            target = target.to(device)
            inp, target = normalizer.preprocess(inp, target)
            # Loss on u,v channels only (same as GEPS version)
            pred = model(inp)
            loss = mse_loss(pred[..., :2], target[..., :2]).mean() / args.accum_steps
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
            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"[iter {iteration:6d}/{args.n_iter}]  "
                f"loss={avg:.6f}  "
                f"lr={current_lr:.2e}  "
                f"elapsed={elapsed/60:.1f}min"
            )
            if use_wandb:
                wandb.log({
                    "train/loss":        avg,
                    "train/lr":          current_lr,
                    "train/elapsed_min": elapsed / 60,
                }, step=iteration)
            running_loss = 0.0

        if iteration % args.save_every == 0 or iteration == args.n_iter:
            ckpt_path = os.path.join(
                exp_dir, f"baseline_{args.stage}_{iteration:05d}.pth"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "dim":       args.dim,
                "dim_mults": args.dim_mults,
                "in_step":   args.in_step,
                "out_step":  args.out_step,
                "stage":     args.stage,
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
        format='%(asctime)s  %(levelname)s  %(message)s',
        handlers=[logging.StreamHandler()],
    )
    args = get_args()
    train(args)
