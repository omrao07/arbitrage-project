# dashboard_api/routes/signals.py

from flask import Blueprint, jsonify, request
from engine.regions import get_region_strategies, get_supported_regions, get_global_model

signals_bp = Blueprint("signals", __name__)

def run_strategy(strategy_module):
    try:
        if hasattr(strategy_module, 'generate_signal'):
            return strategy_module.generate_signal()
        else:
            return {"signal": None, "error": "No signal method"}
    except Exception as e:
        return {"signal": None, "error": str(e)}

@signals_bp.route("/signals/<region>", methods=["GET"])
def get_signals(region):
    region = region.lower()

    if region not in get_supported_regions() and region != "global":
        return jsonify({"error": "Unsupported region"}), 400

    # Retrieve strategies
    if region == "global":
        region_model = get_global_model()
    else:
        region_model = get_region_strategies(region)

    signals_output = {
        "region": region_model["region"],
        "alpha_signals": {},
        "diversified_signals": {}
    }

    # Alpha strategies
    for strat in region_model["alpha"]:
        strat_name = strat.__name__
        signals_output["alpha_signals"][strat_name] = run_strategy(strat)

    # Diversified strategies
    for strat in region_model["diversified"]:
        strat_name = strat.__name__
        signals_output["diversified_signals"][strat_name] = run_strategy(strat)

    return jsonify(signals_output)