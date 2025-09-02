# research/exec/rl_agent/policy_ddqn.py
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse your evaluator’s ChildDecision object if available
try:
    from research.exec.rl_agent.evaluator import ChildDecision # type: ignore
except Exception:
    # Minimal fallback to keep this file standalone
    @dataclass(frozen=True)
    class ChildDecision:
        qty: float
        limit_px: Optional[float] = None
        reason: str = "ddqn"


# ----------------------------- Model --------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (128, 128), dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------- Replay Buffer ---------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int):
        size = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, size, size=batch_size)
        return (
            torch.from_numpy(self.state[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_state[idx]),
            torch.from_numpy(self.done[idx]),
        )

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr


# -------------------------- Action Discretization -------------------------

@dataclass(frozen=True)
class ActionGrid:
    """
    Discrete grid mapping -> ChildDecision
    - qty_fracs: signed fraction of remaining (e.g., [-0.02, -0.01, 0, +0.01, +0.02])
                 positive means move in the direction of remaining, negative means reduce.
    - px_bps: limit offset in bps from mid (positive/negative). None -> marketable touch.
    """
    qty_fracs: Tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02)
    px_bps: Tuple[Optional[float], ...] = (None, -10.0, -5.0, 0.0, 5.0, 10.0)

    def to_action_list(self) -> List[Tuple[float, Optional[float]]]:
        actions: List[Tuple[float, Optional[float]]] = []
        for q in self.qty_fracs:
            for b in self.px_bps:
                actions.append((q, b))
        return actions


# ------------------------------ Config ------------------------------------

@dataclass
class DDQNConfig:
    state_dim: int
    hidden: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 200_000
    warmup: int = 2_000
    tau: float = 0.005                    # soft target update coefficient
    eps_start: float = 0.10               # exploration ε
    eps_end: float = 0.01
    eps_decay_steps: int = 200_000
    grad_clip: float = 5.0
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    action_grid: ActionGrid = ActionGrid()

    def n_actions(self) -> int:
        return len(self.action_grid.to_action_list())


# ------------------------------ Policy ------------------------------------

class DDQNPolicy:
    """
    Double DQN with:
      - Epsilon-greedy exploration
      - Soft target updates
      - Stable replay buffer
      - Save/Load & eval mode
    Expects state as 1D float array (shape = state_dim).
    """

    def __init__(self, cfg: DDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.online = MLP(cfg.state_dim, cfg.n_actions(), cfg.hidden, cfg.dropout).to(self.device)
        self.target = MLP(cfg.state_dim, cfg.n_actions(), cfg.hidden, cfg.dropout).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = optim.AdamW(self.online.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_size, cfg.state_dim)

        self._steps = 0
        self._eps = cfg.eps_start
        self._actions = cfg.action_grid.to_action_list()

    # ------------- Public API ----------------

    def act(self, state: Dict[str, float]) -> ChildDecision:
        """
        Convert hedge-fund state dict -> vector, choose action, and map to ChildDecision.
        Expected keys in state:
          remaining, bid, ask, mid, bar_v, elapsed, participation_today
        Extra keys are ignored.
        """
        s = self._state_to_vec(state)
        a_idx = self._select_action(s)
        qty_frac, px_bps = self._actions[a_idx]

        remaining = float(state.get("remaining", 0.0))
        side = 1.0 if remaining > 0 else -1.0  # buy if positive remaining
        qty = side * abs(remaining) * float(qty_frac)

        limit_px = None
        if px_bps is not None:
            mid = float(state.get("mid", 0.0))
            limit_px = mid * (1.0 + (px_bps * 1e-4) * (1.0 if side > 0 else -1.0))

        return ChildDecision(qty=qty, limit_px=limit_px, reason="ddqn")

    def remember(self, s: Dict[str, float], a_idx: int, r: float, s2: Dict[str, float], done: bool):
        self.buf.push(self._state_to_vec(s), a_idx, float(r), self._state_to_vec(s2), done)

    def train_step(self) -> Optional[float]:
        if len(self.buf) < max(self.cfg.batch_size, self.cfg.warmup):
            self._steps += 1
            self._anneal_eps()
            return None

        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            # Online net chooses best actions; target net evaluates them (Double DQN)
            q_next_online = self.online(s2)                # [B, A]
            a_next = q_next_online.argmax(dim=1, keepdim=True)  # [B, 1]
            q_next_target = self.target(s2).gather(1, a_next).squeeze(1)  # [B]
            target_q = r + (1.0 - d) * self.cfg.gamma * q_next_target

        q = self.online(s).gather(1, a.view(-1, 1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target_q)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        # Soft update target
        with torch.no_grad():
            tau = self.cfg.tau
            for t, o in zip(self.target.parameters(), self.online.parameters()):
                t.data.mul_(1.0 - tau).add_(tau * o.data)

        self._steps += 1
        self._anneal_eps()
        return float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "steps": self._steps,
                "eps": self._eps,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "DDQNPolicy":
        ckpt = torch.load(path, map_location="cpu")
        cfg_dict = ckpt["cfg"]
        # reconstruct ActionGrid nested dataclass
        cfg_dict["action_grid"] = ActionGrid(**cfg_dict.get("action_grid", {}))
        cfg = DDQNConfig(**cfg_dict)
        pol = DDQNPolicy(cfg)
        pol.online.load_state_dict(ckpt["online"])
        pol.target.load_state_dict(ckpt["target"])
        pol._steps = int(ckpt.get("steps", 0))
        pol._eps = float(ckpt.get("eps", cfg.eps_end))
        return pol

    # ------------- Internals -----------------

    def _state_to_vec(self, state: Dict[str, float]) -> np.ndarray:
        # Order the features explicitly to avoid accidental reordering
        # Missing keys default to 0.0
        keys = ("remaining", "bid", "ask", "mid", "bar_v", "elapsed", "participation_today")
        vec = np.array([float(state.get(k, 0.0)) for k in keys], dtype=np.float32)

        # Normalize some dynamic ranges lightly (helps stability)
        vec[0] = np.tanh(vec[0] / (1e4))                 # remaining shares scale
        vec[4] = np.tanh(vec[4] / (1e6))                 # bar volume
        vec[5] = np.clip(vec[5], 0.0, 1.0)               # elapsed ∈ [0,1]
        vec[6] = np.clip(vec[6], 0.0, 1.0)               # participation ∈ [0,1]
        return vec

    def _select_action(self, state_vec: np.ndarray) -> int:
        self.online.eval()
        if random.random() < self._eps:
            a = random.randrange(self.cfg.n_actions())
        else:
            with torch.no_grad():
                s = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
                q = self.online(s)  # [1, A]
                a = int(q.argmax(dim=1).item())
        self.online.train()
        return a

    def _anneal_eps(self):
        # Linear decay from eps_start -> eps_end over eps_decay_steps
        t = min(self._steps, self.cfg.eps_decay_steps)
        frac = t / max(1, self.cfg.eps_decay_steps)
        self._eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac


# -------------------------- Minimal usage ----------------------------------

if __name__ == "__main__":
    # Quick smoke test on random data
    cfg = DDQNConfig(state_dim=7)
    pol = DDQNPolicy(cfg)
    s = {
        "remaining": 10_000.0,
        "bid": 99.9,
        "ask": 100.1,
        "mid": 100.0,
        "bar_v": 50_000,
        "elapsed": 0.1,
        "participation_today": 0.02,
    }
    a = pol.act(s)
    print("Sample action:", a)