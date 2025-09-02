# research/exec/rl_agent/mlp_utils.py
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------ Seeding ------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- Activations ---------------------------------

_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01, inplace=False),
    "identity": nn.Identity,
}

def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name not in _ACTS:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(_ACTS)}")
    return _ACTS[name]()


# ----------------------------- Normalization -------------------------------

def get_norm(name: Optional[str], dim: int) -> nn.Module:
    if not name:
        return nn.Identity()
    n = name.lower()
    if n in ("bn", "batch", "batchnorm", "batch_norm"):
        return nn.BatchNorm1d(dim)
    if n in ("ln", "layer", "layernorm", "layer_norm"):
        return nn.LayerNorm(dim)
    if n in ("id", "identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unknown norm '{name}'")


# ------------------------------ Init ---------------------------------------

def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)


# --------------------------------- MLP -------------------------------------

@dataclass
class MLPConfig:
    in_dim: int
    out_dim: int
    hidden: Sequence[int] = (128, 128)
    act: str = "relu"
    norm: Optional[str] = None       # 'bn' | 'ln' | None
    dropout: float = 0.0
    residual: bool = False           # residual connections between same-sized blocks
    final_act: Optional[str] = None  # e.g., 'tanh' for bounded output
    init: str = "kaiming"            # 'kaiming' or 'none'


class MLP(nn.Module):
    """
    Flexible MLP:
      - optional BatchNorm/LayerNorm per hidden
      - optional dropout
      - optional residuals when shapes match
      - configurable final activation
    """
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        layers: List[nn.Module] = []
        last = cfg.in_dim
        act = get_activation(cfg.act)
        for h in cfg.hidden:
            layers.append(nn.Linear(last, h))
            layers.append(get_norm(cfg.norm, h))
            layers.append(act.__class__())  # fresh module
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            last = h
        layers.append(nn.Linear(last, cfg.out_dim))
        if cfg.final_act:
            layers.append(get_activation(cfg.final_act))
        self.net = nn.Sequential(*layers)

        if cfg.init == "kaiming":
            self.apply(kaiming_init)

        self._residual = bool(cfg.residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._residual:
            return self.net(x)
        # Residual across blocks of equal size (simple skip strategy)
        out = x
        idx = 0
        for mod in self.net:
            prev = out
            out = mod(out)
            # After each activation/dropout, if shape matches previous linear input, add skip
            if isinstance(mod, (nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.Tanh)) and prev.shape == out.shape:
                out = out + prev
            idx += 1
        return out


# ----------------------------- Model utils ---------------------------------

def num_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters() if not trainable_only else (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def to_device(model: nn.Module, prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        dev = torch.device(prefer)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return dev


def save_model(model: nn.Module, path: str, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(model: nn.Module, path: str, map_location: Optional[str] = None) -> dict:
    ckpt = torch.load(path, map_location=map_location or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("extra", {})


# ---------------------------- Optim / Sched --------------------------------

@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    clip_grad_norm: Optional[float] = 5.0


def make_optimizer(model: nn.Module, cfg: OptimConfig) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)


def clip_grads(model: nn.Module, max_norm: Optional[float]):
    if max_norm and max_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def make_cosine_scheduler(optimizer: optim.Optimizer, T_max: int, min_lr: float = 0.0) -> optim.lr_scheduler._LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr) # type: ignore


# ---------------------------- Early Stopping -------------------------------

class EarlyStop:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.count = 0
        self.stop = False

    def update(self, value: float) -> bool:
        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.count = 0
        else:
            self.count += 1
            self.stop = self.count >= self.patience
        return self.stop


# --------------------------- Feature Scaling --------------------------------

class Standardizer(nn.Module):
    """
    Online feature standardizer: y = (x - mean) / (std + eps)
    - Use `fit_batch(x)` on a few batches, then switch to eval() for inference.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.eps = eps
        self._n = 0

    @torch.no_grad()
    def fit_batch(self, x: torch.Tensor):
        # x: [B, D]
        bmean = x.mean(dim=0)
        bvar = x.var(dim=0, unbiased=False)
        if self._n == 0:
            self.mean.copy_(bmean) # type: ignore
            self.var.copy_(bvar) # type: ignore
        else:
            # running average
            m = float(self._n)
            n = float(self._n + x.shape[0])
            w_old = m / n
            w_new = x.shape[0] / n
            self.mean.mul_(w_old).add_(w_new * bmean) # type: ignore
            self.var.mul_(w_old).add_(w_new * bvar) # type: ignore
        self._n += x.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.var + self.eps)) # type: ignore


# ------------------------------ Example ------------------------------------

if __name__ == "__main__":
    seed_everything(123)

    cfg = MLPConfig(in_dim=7, out_dim=25, hidden=(256, 256), act="silu", norm="ln", dropout=0.1, residual=False)
    net = MLP(cfg)
    dev = to_device(net)
    print("Params:", num_parameters(net))

    opt_cfg = OptimConfig(lr=3e-4, weight_decay=0.01, clip_grad_norm=1.0)
    opt = make_optimizer(net, opt_cfg)
    sch = make_cosine_scheduler(opt, T_max=1000, min_lr=1e-5)

    x = torch.randn(32, cfg.in_dim, device=dev)
    y = net(x).sum()
    y.backward()
    clip_grads(net, opt_cfg.clip_grad_norm)
    opt.step(); sch.step()
    opt.zero_grad()
    print("Step OK")