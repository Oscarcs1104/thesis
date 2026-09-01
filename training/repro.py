"""Reproducibility, LR scheduling, target standardization and metrics helpers
shared by the training entrypoints (FASE 1 of the audit refactor).
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine anneal
# --------------------------------------------------------------------------- #
def build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, int(warmup_epochs))
    max_epochs = max(1, int(max_epochs))

    def lr_lambda(epoch: int) -> float:  # epoch is 0-indexed by LambdaLR
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class WarmupThenPlateau:
    """Linear LR warmup for `warmup_epochs`, then ReduceLROnPlateau driven by val loss.

    Unlike a fixed-horizon cosine schedule, this composes correctly with early
    stopping no matter when it fires: a cosine schedule built for --epochs=100
    assumes training actually runs 100 epochs, so if early stopping cuts it off
    at epoch 61 the LR never reaches its floor and training keeps oscillating at
    a relatively high LR right when val has already plateaued. Plateau-based
    decay instead reacts to val stalling directly, whenever that happens.

    Respects PER-PARAM-GROUP base LRs (reads them from the optimizer at construction
    time) rather than forcing every group to the same value -- this is what makes it
    safe to combine with discriminative learning rates (e.g. linear probing: a head
    param group at --lr and a graph_encoder param group at --lr * --unfreeze-lr-mult).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        factor: float = 0.5,
        patience: int = 5,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.epoch = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=[lr * min_lr_ratio for lr in self.base_lrs],
        )
        if self.warmup_epochs:
            self._set_lr([lr / self.warmup_epochs for lr in self.base_lrs])

    def _set_lr(self, lrs) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def step(self, val_loss: float) -> None:
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            frac = self.epoch / self.warmup_epochs
            self._set_lr([lr * frac for lr in self.base_lrs])
        else:
            self.plateau.step(val_loss)


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr_ratio: float = 0.01,
    plateau_factor: float = 0.5,
    plateau_patience: int = 5,
):
    name = name.lower()
    if name == "cosine":
        return build_warmup_cosine(optimizer, warmup_epochs, max_epochs, min_lr_ratio)
    if name == "plateau":
        return WarmupThenPlateau(optimizer, warmup_epochs, plateau_factor, plateau_patience, min_lr_ratio)
    raise ValueError(f"Unknown --lr-schedule: {name!r} (expected 'cosine' or 'plateau')")


def step_scheduler(scheduler, val_loss: float) -> None:
    """Unified step() call site: WarmupThenPlateau needs the val metric, LambdaLR doesn't."""
    if isinstance(scheduler, WarmupThenPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


# --------------------------------------------------------------------------- #
# Target standardization (fit on TRAIN targets only)
# --------------------------------------------------------------------------- #
class TargetStandardizer:
    """z = (y - mean) / std, fit on the training split only.

    Wraps sklearn's StandardScaler but operates transparently on torch tensors
    and is a no-op when `enabled=False` (e.g. classification tasks).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None

    def fit(self, train_targets: torch.Tensor) -> "TargetStandardizer":
        if not self.enabled:
            return self
        from sklearn.preprocessing import StandardScaler

        y = train_targets.detach().cpu().float().view(train_targets.size(0), -1).numpy()
        scaler = StandardScaler().fit(y)
        self.mean_ = torch.tensor(scaler.mean_, dtype=torch.float32).view(1, -1)
        self.std_ = torch.tensor(scaler.scale_, dtype=torch.float32).view(1, -1)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def _to(self, device: torch.device) -> None:
        if self.mean_ is not None and self.mean_.device != device:
            self.mean_ = self.mean_.to(device)
            self.std_ = self.std_.to(device)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mean_ is None:
            return y
        self._to(y.device)
        return (y - self.mean_) / self.std_  # [1, D] broadcasts over [B, D]

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mean_ is None:
            return z
        self._to(z.device)
        return z * self.std_ + self.mean_

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mean": None if self.mean_ is None else self.mean_.cpu(),
            "std": None if self.std_ is None else self.std_.cpu(),
        }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_range: Optional[tuple] = None,
) -> Dict[str, float]:
    preds = preds.detach().cpu().float().view(-1)
    targets = targets.detach().cpu().float().view(-1)
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = math.sqrt(mse)
    out = {"mse": mse, "rmse": rmse, "mae": mae}
    if target_range is not None and target_range[1] > target_range[0]:
        out["nrmse"] = rmse / (target_range[1] - target_range[0])
    else:
        out["nrmse"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_seed_metrics(per_seed: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    per_seed = list(per_seed)
    keys = set().union(*(m.keys() for m in per_seed)) if per_seed else set()
    agg: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals: List[float] = [float(m[k]) for m in per_seed if k in m and m[k] == m[k]]  # drop NaN
        if not vals:
            continue
        arr = np.asarray(vals, dtype="float64")
        agg[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": len(vals)}
    return agg


def format_seed_table(agg: Dict[str, Dict[str, float]], headline_keys: Iterable[str] = ("rmse", "nrmse", "mae")) -> str:
    lines = ["", "=== Multi-seed test summary (mean +/- std) ==="]
    ordered = [k for k in headline_keys if k in agg] + [k for k in agg if k not in set(headline_keys)]
    for k in ordered:
        s = agg[k]
        lines.append(f"  {k:>12s}: {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n']})")
    return "\n".join(lines)
