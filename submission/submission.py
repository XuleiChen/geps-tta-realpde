"""
submission.py — GEPS U-Net + adapt_code TTA for RealPDE Track 2 LTTTA.

Submission layout:
  submission.py               (this file)
  model.pth                   (GEPS finetune checkpoint, ~91 MB)
  geps_layers.py              (GEPSConv3D, vendored)
  unet_geps.py                (GEPSUnet3d, vendored, patched imports)
  einops_exts/                (vendored pure-Python)
  rotary_embedding_torch/     (vendored pure-Python)

TTA mechanism:
  - adapt_code = nn.Parameter, initialized to mean of finetune codes.
  - Only adapt_code is updated (all model weights frozen).
  - Each ttt_step: adapt on (prev_input, prev_target) with N_ADAPT_STEPS Adam steps,
    then predict current input. Causally correct — current target never used.
  - Bilinear wrapper: competition tensors are (1,20,32,64,3); model is 64×128.
    Input is trilinearly upsampled to (1,20,64,128,3) before model, output
    downsampled back to (1,20,32,64,3).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure vendored packages are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from unet_geps import Unet3d as GEPSUnet3d
from ridge_tta import RidgeLastLayerAdapter

# ── Constants ────────────────────────────────────────────────────────────────
_H_MODEL, _W_MODEL = 64, 128    # model's native training resolution
_H_EVAL,  _W_EVAL  = 32, 64     # competition evaluation resolution
N_ADAPT_STEPS  = 1               # gradient steps per adapt_code update
ADAPT_LR       = 1e-3
RIDGE_LAM      = 1e-2           # ridge regularisation strength

# ── SPS interval calibration ─────────────────────────────────────────────────
# Calibrated on 14 held-out foil trajectories (teacher-forcing protocol).
# Physical widths / biases (m/s) from grid search on calibration set.
# channel 0=u, 1=v; buckets: frames 0-6=early, 7-13=mid, 14-19=late
_W_TABLE_PHYS = {
    (0, 'early'): (0.064206, -0.006508),  # full_width_phys, bias_phys
    (0, 'mid'):   (0.074953,  0.000000),
    (0, 'late'):  (0.093429,  0.000000),
    (1, 'early'): (0.014465, -0.001621),
    (1, 'mid'):   (0.024335, -0.003053),
    (1, 'late'):  (0.024940, -0.002779),
}

# std_tgt per channel from the official competition normalizer
# (mean_std_real.pt: std_tgt[0]=0.0968104079, std_tgt[1]=0.0159636438).
# Purpose: convert physical widths/biases → normalised-space so that
#   local_eval.py's postprocess_pred(lower_norm) = lower_norm * std_tgt + mean_tgt
# correctly recovers the intended physical bounds.
# NOTE: this is NOT the same as scoring.py's SIGMA_GLOBAL (= (std_u+std_v)/2 ≈ 0.0564),
# which is a single scalar used only inside the SPS penalty formula nil=w/SIGMA_GLOBAL.
# _STD_TGT is per-channel and is purely a unit-conversion factor, unrelated to scoring.
_STD_TGT = {0: 0.09681041, 1: 0.01596364}

# Pre-convert to normalised space: w_norm = w_phys / std_tgt_ch
W_TABLE = {
    k: (w / _STD_TGT[k[0]], b / _STD_TGT[k[0]])
    for k, (w, b) in _W_TABLE_PHYS.items()
}


def _frame_bucket(frame_idx: int) -> str:
    """Map 0-based frame index (0..19) to lead-time bucket."""
    if frame_idx <= 6:
        return 'early'
    elif frame_idx <= 13:
        return 'mid'
    else:
        return 'late'


def _build_intervals(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build lower/upper in normalised space from calibrated W_TABLE.
    pred: (1, T, H, W, C) normalised — competition shape, C=3 (u, v, p)
    Returns lower, upper of same shape (also normalised).
    After local_eval.py's postprocess_pred, physical bounds = pred_phys ± w_phys/2 + bias_phys.
    p (ch=2) falls back to default ±10% (not scored).
    """
    lower = torch.empty_like(pred)
    upper = torch.empty_like(pred)
    T = pred.shape[1]
    for t in range(T):
        bucket = _frame_bucket(t)
        for ch in range(pred.shape[-1]):
            if ch < 2:
                w_norm, bias_norm = W_TABLE[(ch, bucket)]
                center = pred[:, t, :, :, ch] + bias_norm
                lower[:, t, :, :, ch] = center - w_norm / 2
                upper[:, t, :, :, ch] = center + w_norm / 2
            else:
                lower[:, t, :, :, ch] = pred[:, t, :, :, ch] - 0.1 * pred[:, t, :, :, ch].abs()
                upper[:, t, :, :, ch] = pred[:, t, :, :, ch] + 0.1 * pred[:, t, :, :, ch].abs()
    return lower, upper


# ── Resolution helpers ────────────────────────────────────────────────────────
def _upsample(x: torch.Tensor) -> torch.Tensor:
    """(1, T, 32, 64, C) → (1, T, 64, 128, C)"""
    b, t, h, w, c = x.shape
    z = F.interpolate(
        x.permute(0, 4, 1, 2, 3),                          # (1, C, T, H, W)
        size=(t, _H_MODEL, _W_MODEL),
        mode='trilinear', align_corners=False,
    )
    return z.permute(0, 2, 3, 4, 1)                        # (1, T, H, W, C)


def _downsample(x: torch.Tensor) -> torch.Tensor:
    """(1, T, 64, 128, C) → (1, T, 32, 64, C)"""
    b, t, h, w, c = x.shape
    z = F.interpolate(
        x.permute(0, 4, 1, 2, 3),
        size=(t, _H_EVAL, _W_EVAL),
        mode='trilinear', align_corners=False,
    )
    return z.permute(0, 2, 3, 4, 1)


# ── TTT model ─────────────────────────────────────────────────────────────────
class GEPSTTTModel:
    """GEPS U-Net with per-trajectory adapt_code test-time adaptation."""

    def __init__(
        self,
        model: GEPSUnet3d,
        device: str,
        lr: float = ADAPT_LR,
    ):
        self.model  = model.to(device)
        self.device = device
        self.lr = lr

        # Freeze all model weights — only adapt_code will be updated via Adam;
        # final_conv[1] is updated via ridge regression (in-place .data.copy_())
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Warm start: mean of all finetune environment codes
        self._init_code = self.model.codes.data.mean(dim=0).clone()

        # Ridge adapter — do NOT call freeze_backbone() (would conflict with our setup)
        self.adapter = RidgeLastLayerAdapter(self.model, lam=RIDGE_LAM)

        # Mutable state (reset per trajectory)
        self.adapt_code: Optional[nn.Parameter] = None
        self._optimizer = None
        self._prev_input_hires: Optional[torch.Tensor] = None

    # ── Interface ──────────────────────────────────────────────────────────────
    def reset_ttt_state(self) -> None:
        """Called at the start of every trajectory."""
        self.adapt_code = nn.Parameter(self._init_code.clone().to(self.device))
        self._optimizer = torch.optim.Adam([self.adapt_code], lr=self.lr)
        self._prev_input_hires = None
        self.adapter.reset_to_prior()   # restore final_conv[1] to pretrained weights

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        input_norm       : (1, 20, 32, 64, 3) normalized
        prev_target_norm : (1, 20, 32, 64, 3) GT of previous step, or None
        Returns (pred_norm, info) where pred_norm is (1, 20, 32, 64, 3).
        """
        input_norm = torch.as_tensor(input_norm, dtype=torch.float32).to(self.device)
        input_hires = _upsample(input_norm)    # (1, 20, 64, 128, 3)

        adapt_loss: Optional[float] = None

        if prev_target_norm is not None and self._prev_input_hires is not None:
            prev_target_norm = torch.as_tensor(
                prev_target_norm, dtype=torch.float32
            ).to(self.device)
            prev_target_hires = _upsample(prev_target_norm)  # (1, 20, 64, 128, 3)

            # ── 1. Adapt code: every step, 1 gradient step ────────────────────
            self.model.train()
            for _ in range(N_ADAPT_STEPS):
                pred_prev = self.model(
                    self._prev_input_hires, adapt_code=self.adapt_code
                )
                loss = F.mse_loss(
                    pred_prev[..., :2], prev_target_hires[..., :2]
                )
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            adapt_loss = float(loss.detach().cpu())

            # ── 2. Ridge TTA: update final_conv[1] every step ─────────────────
            self.model.eval()
            self.adapter.enable_capture()
            with torch.no_grad():
                self.model(self._prev_input_hires, adapt_code=self.adapt_code)
            self.adapter.disable_capture()
            self.adapter.ridge_update(prev_target_hires)

        # ── 3. Predict current input (no gradients) ───────────────────────────
        self.model.eval()
        with torch.no_grad():
            pred_hires = self.model(input_hires, adapt_code=self.adapt_code)

        pred_lores = _downsample(pred_hires)   # (1, 20, 32, 64, 3)

        # ── 4. Cache current hires input for next step's adaptation ───────────
        self._prev_input_hires = input_hires.detach()

        # ── 5. Build calibrated SPS intervals ────────────────────────────────
        lower, upper = _build_intervals(pred_lores)

        return pred_lores, {"adapt_loss": adapt_loss, "lower": lower, "upper": upper}


# ── Entry point ───────────────────────────────────────────────────────────────
def get_ttt_model(submission_dir: str, device: str) -> GEPSTTTModel:
    """Called once by the evaluator before the stream starts (not timed)."""
    ckpt_path = os.path.join(submission_dir, "model.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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

    return GEPSTTTModel(model, device=device, lr=ADAPT_LR)
