from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# -----------------------------------------
# Registry Row = what’s inside CSV/JSONL
# -----------------------------------------
@dataclass
class StrategyRow:
    id: str
    firm: str
    discipline: str
    family: str
    region: str
    horizon: str
    status: str
    name: str
    description: str
    risk_budget: float
    engine: str
    owner: str

    # Optional new fields
    genre: Optional[str] = None           # e.g. Momentum, Carry, Breakeven, TailHedge
    mode: Optional[str] = "auto"          # "auto" or "manual"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# -----------------------------------------
# YAML Config = per-strategy customization
# -----------------------------------------
@dataclass
class StrategyConfig:
    id: str
    firm: str
    name: str
    description: str
    discipline: str
    family: str
    region: str
    horizon: str
    status: str
    engine: str

    # New fields
    genre: Optional[str] = None
    mode: Optional[str] = "auto"

    # Dictionaries allow flexible extension
    params: Dict = field(default_factory=dict)   # model parameters
    risk: Dict = field(default_factory=dict)     # risk settings (budget, vol_target, stop_loss, max_leverage)
    data: Dict = field(default_factory=dict)     # data requirements (lookup_data_sources.csv)

    owner: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# -----------------------------------------
# Bundle = merged registry + config
# (what loader/selector passes to engines)
# -----------------------------------------
@dataclass
class StrategyBundle:
    row: StrategyRow
    config: StrategyConfig
    score: Optional[float] = None          # selector attaches this
    target_weight: Optional[float] = None  # risk_policies attaches this

    def as_dict(self) -> Dict:
        d = {**self.row.__dict__}
        d.update({
            "params": self.config.params,
            "risk": self.config.risk,
            "data": self.config.data,
            "genre": self.config.genre or self.row.genre,
            "mode": self.config.mode or self.row.mode,
            "score": self.score,
            "target_weight": self.target_weight,
        })
        return d