# ai_brain/strategy_generator.py
"""
Strategy Generator (AI CRO stub – catalog + priors + param sampler)

Outputs BacktestSpec-compatible dicts:
    {"region": "US", "name": "labor_union_power", "params": {...}, "id_suffix": "A1"}

Usage:
    from ai_brain.strategy_generator import propose
    ideas = propose(limit=50, seed=42)

Design goals:
- Deterministic catalogs of NEW strategies (25 each US/EU/IN/JP).
- Sane hyper-parameter priors per strategy family.
- Deduplication vs existing YAML configs under backend/config/strategies/.
- Optional priors to bias regions/families or emphasize certain regimes.
- Returns plain dicts to keep this decoupled from AutoResearch types.

No network calls, no model dependencies – plug-and-play today.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------
# Catalog (100 brand-new strategies: 25 per region)
# -----------------------

US: Tuple[str, ...] = (
    "labor_union_power",
    "student_loan_repayment",
    "defense_budget_rotation",
    "ev_subsidy_spread",
    "agri_subsidy_pivot",
    "healthcare_reg_arb",
    "hurricane_insurance_hedge",
    "muni_vs_property_tax",
    "ai_datacenter_power_demand",
    "streaming_subscriber_wars",
    "patent_cliff_pharma",
    "cloud_capex_divergence",
    "railroad_regulation_cycle",
    "endowment_rebalancing",
    "state_pension_alloc_flows",
    "shale_basin_cost_curve",
    "border_trade_policy",
    "debt_buyback_loophole",
    "crypto_etf_gold_substitution",
    "export_control_semis",
    "water_rights_futures",
    "election_year_defense_procurement",
    "trucking_vs_rail_modal_shift",
    "mortgage_origination_cycle",
    "private_equity_exit_pipeline",
)

EU: Tuple[str, ...] = (
    "north_sea_gas_infra",
    "french_nuclear_reliability",
    "german_mittelstand_credit",
    "nordic_hydropower_arb",
    "ee_energy_subsidies",
    "spanish_tourism_cycle",
    "brexit_logistics_friction",
    "eu_fishing_quota",
    "french_cap_subsidies",
    "italian_npl_cycle",
    "wind_feed_in_tariffs",
    "swiss_wealth_flows",
    "dutch_gas_phaseout",
    "greek_shipping_freight",
    "eu_data_sovereignty",
    "polish_coal_transition",
    "german_wage_negotiations",
    "french_pension_protests",
    "ee_fx_remittances",
    "nordic_swf_rotations",
    "uk_food_inflation_hedge",
    "baltic_port_congestion",
    "luxury_vs_cn_consumption",
    "eu_cybersecurity_mandates",
    "banking_union_progress",
)

IN: Tuple[str, ...] = (
    "state_election_infra_spend",
    "festival_season_consumption",
    "textile_export_quotas",
    "coal_imports_power_shortage",
    "rural_electrification",
    "upi_penetration_equities",
    "state_bank_lending_cycle",
    "msp_agribasket_arbitrage",
    "cricket_ad_spike",
    "infra_corridor_ports_roads",
    "budget_defense_psu",
    "pharma_generics_export_cycle",
    "digital_rupee_cbcd",
    "election_year_subsidies",
    "rural_credit_expansion",
    "private_uni_ipo_wave",
    "auto_financing_cycle",
    "reit_vs_developer_spread",
    "airline_fuel_hedge_fail",
    "it_captive_center_growth",
    "hydropower_monsoon_drought",
    "telecom_tariff_hikes",
    "gold_loan_vs_imports",
    "logistics_gst_optimization",
    "space_launch_supply_chain",
)

JP: Tuple[str, ...] = (
    "corporate_cash_hoard_unlock",
    "robotization_capex_cycle",
    "aging_pop_healthcare",
    "anime_media_ip_globalization",
    "quake_rebuild_capex",
    "shinkansen_extensions",
    "sake_export_growth",
    "gender_diversity_reform",
    "cross_shareholding_unwinds",
    "semicap_global_demand",
    "consumption_tax_cycles",
    "inbound_visa_tourism",
    "nuclear_restart_bets",
    "cultural_content_exports",
    "yen_hedged_etf_flows",
    "gpif_rotations",
    "food_security_imports",
    "conglomerate_spinoffs",
    "cherry_blossom_seasonality",
    "tech_standard_adoption",
    "labor_shortage_automation",
    "climate_disaster_insurance",
    "rail_privatization",
    "offshore_wind_developers",
    "digital_yen_experiments",
)

REGIONS: Dict[str, Tuple[str, ...]] = {"US": US, "EU": EU, "IN": IN, "JP": JP}

# -----------------------
# Param families / priors
# -----------------------

@dataclass
class ParamPrior:
    """Sampling bounds for common hyper-params."""
    lookback_days: Tuple[int, int] = (60, 240)
    z_threshold: Tuple[float, float] = (0.5, 3.0)
    vol_target_bps: Tuple[int, int] = (50, 300)
    decay: Tuple[float, float] = (0.90, 0.995)
    rebalance: Tuple[str, ...] = ("daily@15:40", "weekly@Fri-15:40", "intraday@hourly")
    hedge_allow: bool = True

DEFAULT_PRIORS = ParamPrior()


def _sample_params(name: str, priors: ParamPrior, rng: random.Random) -> Dict[str, Any]:
    """Family-aware param sampler (light heuristics by name)."""
    lb = rng.randint(*priors.lookback_days)
    zt = round(rng.uniform(*priors.z_threshold), 2)
    vt = int(rng.randint(*priors.vol_target_bps))
    dc = round(rng.uniform(*priors.decay), 5)
    rb = rng.choice(priors.rebalance)

    p: Dict[str, Any] = {
        "lookback_days": lb,
        "z_thresh": zt,
        "vol_target_bps": vt,
        "decay": dc,
        "rebalance": rb,
        "allow_overlays": priors.hedge_allow,
    }

    # Family tweaks
    if "hedge" in name or "insurance" in name:
        p["overlay_type"] = rng.choice(["index_put_spread", "long_vix", "fx"])
        p["overlay_budget_bps"] = rng.choice([5, 10, 15, 20])
    if "credit" in name or "loan" in name:
        p["carry_window"] = rng.randint(20, 120)
    if "tourism" in name or "festival" in name:
        p["seasonal_win"] = rng.randint(30, 365)
    if "capex" in name or "infra" in name:
        p["event_window"] = rng.randint(10, 90)
    if "subsidi" in name or "subsid" in name:
        p["policy_lag_days"] = rng.randint(5, 45)
    if "export" in name or "imports" in name:
        p["fx_hedge_ratio"] = round(rng.uniform(0.0, 1.0), 2)

    return p


# -----------------------
# Dedup against existing configs
# -----------------------

def _existing_ids(config_root: str = "backend/config/strategies") -> set:
    root = Path(config_root)
    if not root.exists():
        return set()
    out = set()
    for p in root.rglob("*.yml"):
        # Expect filename matches strategy name; region = parent folder
        try:
            region = p.parent.name.upper()
            name = p.stem
            out.add(f"{region}.{name}")
        except Exception:
            continue
    return out


# -----------------------
# Proposer
# -----------------------

def propose(
    *,
    limit: int = 50,
    seed: Optional[int] = None,
    priors: Optional[ParamPrior] = None,
    bias_regions: Optional[List[str]] = None,
    exclude_existing_configs: bool = True,
    config_root: str = "backend/config/strategies",
    mode: str = "random",  # "random" | "grid"
) -> List[Dict[str, Any]]:
    """
    Generate BacktestSpec-compatible dicts.

    Args:
        limit: number of proposals to emit (best-effort).
        seed: RNG seed for reproducibility.
        priors: ParamPrior to override defaults.
        bias_regions: e.g., ["IN","JP"] to overweight those regions.
        exclude_existing_configs: skip <region>.<name> present under config_root.
        config_root: path to YAML configs to dedupe against.
        mode: "random" (default) or "grid" (coarse grid over priors).

    Returns:
        List[dict] with keys: region, name, params, id_suffix
    """
    rng = random.Random(seed or (os.getpid() ^ 0x9E3779B97F4A7C15))
    pri = priors or DEFAULT_PRIORS
    existing = _existing_ids(config_root) if exclude_existing_configs else set()

    # region selection weights
    regions = list(REGIONS.keys())
    if bias_regions:
        w = [2.0 if r in set(bias_regions) else 1.0 for r in regions]
    else:
        w = [1.0] * len(regions)

    ideas: List[Dict[str, Any]] = []

    def emit(region: str, name: str, params: Dict[str, Any], idx: int):
        sid = f"{region}.{name}"
        if exclude_existing_configs and sid in existing:
            return
        suffix = f"{idx:02d}"
        ideas.append({"region": region, "name": name, "params": params, "id_suffix": suffix})

    if mode == "grid":
        # coarse grid: 3x3x3 over lb, z, vol
        lb_grid = [60, 120, 240]
        z_grid = [0.75, 1.5, 2.5]
        vt_grid = [75, 150, 250]
        idx = 0
        for region in regions:
            names = list(REGIONS[region])
            rng.shuffle(names)
            for name in names:
                for lb in lb_grid:
                    for z in z_grid:
                        for vt in vt_grid:
                            p = _sample_params(name, pri, rng)
                            p.update({"lookback_days": lb, "z_thresh": z, "vol_target_bps": vt})
                            emit(region, name, p, idx)
                            idx += 1
                            if len(ideas) >= limit:
                                return ideas
        return ideas

    # random mode
    idx = 0
    while len(ideas) < limit:
        region = rng.choices(regions, weights=w, k=1)[0]
        name = rng.choice(REGIONS[region])
        params = _sample_params(name, pri, rng)
        emit(region, name, params, idx)
        idx += 1
        if idx > limit * 10:  # safety to avoid infinite loop if everything deduped
            break

    return ideas


# -----------------------
# CLI helper (optional)
# -----------------------

def _main():
    import argparse, sys
    ap = argparse.ArgumentParser(description="Generate strategy proposals")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bias", type=str, default="", help="Comma list of regions to overweight, e.g. IN,JP")
    ap.add_argument("--grid", action="store_true", help="Use coarse grid mode")
    ap.add_argument("--exclude-existing", action="store_true", default=True)
    ap.add_argument("--config-root", type=str, default="backend/config/strategies")
    args = ap.parse_args()

    pri = DEFAULT_PRIORS
    bias = [s.strip().upper() for s in args.bias.split(",") if s.strip()] or None
    mode = "grid" if args.grid else "random"

    ideas = propose(
        limit=args.limit,
        seed=args.seed,
        priors=pri,
        bias_regions=bias,
        exclude_existing_configs=args.exclude_existing,
        config_root=args.config_root,
        mode=mode,
    )
    print(json.dumps(ideas, indent=2))


if __name__ == "__main__":
    _main()