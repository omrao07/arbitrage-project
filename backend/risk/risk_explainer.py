# backend/risk/risk_explainer.py
from __future__ import annotations

import json
from typing import Dict, Any, Optional

class RiskExplainer:
    """
    Convert numeric risk metrics into plain-English explanations.
    Use for dashboards, literacy_mode, or alerts.
    """

    @staticmethod
    def explain_var(metrics: Dict[str, Any], alpha: float = 0.99) -> str:
        """
        Explain Value at Risk (VaR).
        metrics should include: {"VaR": float, "ES": float, "mean_pnl": float, "std_pnl": float}
        """
        var = metrics.get("VaR")
        es = metrics.get("ES")
        mean = metrics.get("mean_pnl")
        std = metrics.get("std_pnl")

        if var is None: return "No VaR available."

        return (
            f"At the {int(alpha*100)}% confidence level, the portfolio is expected to "
            f"lose no more than ~{var:,.0f} in a single period. "
            f"The Expected Shortfall (average loss beyond VaR) is ~{es:,.0f}. "
            f"Average PnL is {mean:,.0f} with volatility {std:,.0f}."
        )

    @staticmethod
    def explain_stress(attrib: Dict[str, Any], scenario_name: str = "Crash") -> str:
        """
        Explain a stress test attribution.
        attrib is typically the aggregated PnL from stress_attribution.py
        """
        total = attrib["pnl"].sum()
        drivers = attrib.drop(columns=["pnl"]).sum().to_dict() # type: ignore
        top = sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        msg = f"Under the {scenario_name} scenario, portfolio PnL is ~{total:,.0f}. "
        if top:
            msg += "Key drivers: " + ", ".join([f"{f} {v:,.0f}" for f, v in top])
        return msg

    @staticmethod
    def explain_montecarlo(summary: Dict[str, Any]) -> str:
        """
        Explain Monte Carlo summary results.
        """
        return (
            f"Monte Carlo simulation (using {summary.get('alpha',0.99)*100:.0f}% tail): "
            f"VaR ~{summary.get('VaR',0):,.0f}, ES ~{summary.get('ES',0):,.0f}, "
            f"mean PnL {summary.get('mean_pnl',0):,.0f}, "
            f"vol {summary.get('std_pnl',0):,.0f}, "
            f"max drawdown ~{summary.get('max_drawdown_est',0):,.0f}."
        )

    @staticmethod
    def explain_factors(exposures: Dict[str, float]) -> str:
        """
        exposures: dict of {factor: beta or sensitivity}
        """
        if not exposures: return "No factor exposures found."
        top = sorted(exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return "Main factor exposures: " + ", ".join([f"{f} ({v:+.2f})" for f,v in top])

    @staticmethod
    def explain_alert(kind: str, details: Dict[str, Any]) -> str:
        """
        Generic risk alert explanation.
        """
        if kind == "drawdown":
            return f"Alert: drawdown exceeded {details.get('dd_frac',0):.1%}."
        elif kind == "var_breach":
            return f"Alert: losses exceeded {details.get('alpha',0.99):.0%} VaR estimate."
        else:
            return f"Alert: {kind} triggered with details {json.dumps(details)}"


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    metrics = {"VaR": -250000, "ES": -350000, "mean_pnl": 10000, "std_pnl": 150000}
    print(RiskExplainer.explain_var(metrics))

    mc = {"alpha":0.99,"VaR":-200000,"ES":-300000,"mean_pnl":5000,"std_pnl":120000,"max_drawdown_est":-400000}
    print(RiskExplainer.explain_montecarlo(mc))

    exposures = {"Equity":1.2,"Rates":-0.3,"FX":0.5}
    print(RiskExplainer.explain_factors(exposures))

    alert = {"dd_frac":0.12}
    print(RiskExplainer.explain_alert("drawdown", alert))