# backend/engine/rl_exec.py
"""
RL Execution Agent
------------------
Wraps a reinforcement learning policy into the execution engine.

Responsibilities
  • Subscribe to state features (market obs, risk metrics, positions)
  • Run RL policy (PyTorch model or custom policy)
  • Map actions → execution intents (buy/sell/hold, size, symbol)
  • Apply risk guard (exposure limits, stop-loss, VaR/vol budgets)
  • Publish normalized order envelopes to STREAM_ORDERS
  • Audit all decisions with hash and optional Merkle ledger

Dependencies
  • torch (for neural nets) -> pip install torch
  • numpy
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

# -------- bus hook --------
try:
    from backend.bus.streams import publish_stream, consume_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload):
        print(f"[stub publish_stream] {stream} <- {json.dumps(payload)[:220]}")
    def consume_stream(stream: str, handler):
        print(f"[stub consume_stream] {stream} (handler={handler.__name__})")

# -------- optional ledger --------
def _ledger_append(payload, ledger_path: Optional[str]):
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "rl_exec", "payload": payload})
    except Exception:
        pass

# -------- config --------

@dataclass
class RLExecConfig:
    stream_orders: str = "STREAM_ORDERS"
    stream_states: str = "STREAM_STATES"
    ledger_path: Optional[str] = None
    max_position: float = 1_000_000.0  # notional cap
    max_trade_size: float = 100_000.0  # per-trade cap
    stop_loss: float = -50_000.0       # absolute PnL floor
    risk_aversion: float = 0.5         # scales down sizes if VaR high
    use_gpu: bool = False

# -------- RL Execution Agent --------

class RLExecAgent:
    def __init__(self, cfg: RLExecConfig, policy: Optional[Any] = None) -> None:
        """
        policy: callable(state_vector: np.ndarray) -> action dict
                or PyTorch nn.Module returning logits/values
        """
        self.cfg = cfg
        self.policy = policy
        self.device = "cuda" if (cfg.use_gpu and torch and torch.cuda.is_available()) else "cpu"

    def start(self):
        """Begin consuming state stream and running policy."""
        consume_stream(self.cfg.stream_states, self._on_state)

    def _on_state(self, state_env: Dict[str, Any]):
        """
        Handler for new state envelope.
        state_env schema:
          {
            "ts": <ms>,
            "symbol": "AAPL",
            "features": [...],
            "position": 123.4,
            "pnl": 500.0,
            "risk": {"var": ..., "vol": ...}
          }
        """
        try:
            sym = state_env.get("symbol")
            feat = np.asarray(state_env.get("features", []), dtype=float)
            position = float(state_env.get("position", 0.0))
            pnl = float(state_env.get("pnl", 0.0))
            risk = state_env.get("risk", {}) or {}

            action = self._decide(feat, position, pnl, risk)

            if action and abs(action.get("size", 0.0)) > 0.0:
                order_env = self._envelope(sym, action) # type: ignore
                publish_stream(self.cfg.stream_orders, order_env)
                _ledger_append(order_env, self.cfg.ledger_path)

        except Exception as e:
            print(f"[rl_exec error] {e}")

    # ----- decision -----

    def _decide(self, feat: np.ndarray, position: float, pnl: float, risk: Dict[str, Any]) -> Dict[str, Any]:
        # Hard stop-loss
        if pnl <= self.cfg.stop_loss:
            return {"side": "flat", "size": -position, "reason": "stop_loss"}

        # Use policy
        if self.policy is None:
            # trivial random policy
            action = np.random.choice(["buy", "sell", "hold"])
            size = np.random.uniform(0.1, 1.0) * self.cfg.max_trade_size
        elif torch and isinstance(self.policy, nn.Module): # type: ignore
            with torch.no_grad():
                x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.policy(x)
                if logits.ndim == 2 and logits.shape[1] >= 3:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    action_idx = np.random.choice(len(probs), p=probs)
                else:
                    action_idx = int(torch.argmax(logits).item())
            action = ["buy", "sell", "hold"][action_idx % 3]
            size = self.cfg.max_trade_size
        elif callable(self.policy):
            out = self.policy(feat)
            action = out.get("action", "hold") # type: ignore
            size = out.get("size", self.cfg.max_trade_size) # type: ignore
        else:
            action, size = "hold", 0.0

        # Risk scaling (simple: shrink size if VaR high)
        var = float(risk.get("var", 0.0) or 0.0)
        scale = 1.0 / (1.0 + self.cfg.risk_aversion * var)
        size = max(0.0, min(size, self.cfg.max_trade_size)) * scale

        # Clamp by position limits
        if action == "buy" and (position + size) > self.cfg.max_position:
            size = max(0.0, self.cfg.max_position - position)
        elif action == "sell" and (position - size) < -self.cfg.max_position:
            size = max(0.0, position + self.cfg.max_position)

        return {"side": action, "size": float(size)}

    # ----- envelope -----

    def _envelope(self, symbol: str, action: Dict[str, Any]) -> Dict[str, Any]:
        env = {
            "ts": int(time.time() * 1000),
            "adapter": "rl_exec",
            "symbol": symbol,
            "action": action,
            "version": 1,
        }
        env["hash"] = hashlib.sha256(
            json.dumps(env, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
        ).hexdigest()
        return env


# -------- script demo --------

if __name__ == "__main__":
    cfg = RLExecConfig()
    agent = RLExecAgent(cfg)

    # Fake state
    state_env = {
        "ts": int(time.time()*1000),
        "symbol": "AAPL",
        "features": [0.1, -0.2, 0.3],
        "position": 0.0,
        "pnl": 100.0,
        "risk": {"var": 0.5, "vol": 0.2}
    }
    agent._on_state(state_env)