# backend/dashboard_api/routes/weights.py

from flask import Blueprint, jsonify, request
import numpy as np

weights_bp = Blueprint("weights", __name__)

# In-memory store for region-specific weights
current_weights = {}

def compute_strategy_weights(strategy_returns_dict):
    """
    Input:
    {
        'strategy1': [0.01, 0.02, ...],
        'strategy2': [0.005, 0.015, ...],
    }
    Output:
    {
        'strategy1': 0.35,
        'strategy2': 0.65,
    }
    """
    try:
        sharpe_ratios = {}
        for name, returns in strategy_returns_dict.items():
            arr = np.array(returns)
            if arr.std() == 0:
                sharpe = 0
            else:
                sharpe = arr.mean() / arr.std()
            sharpe_ratios[name] = sharpe

        total = sum(max(s, 0) for s in sharpe_ratios.values())
        weights = {
            name: round(max(s, 0) / total, 4) if total > 0 else 0
            for name, s in sharpe_ratios.items()
        }

        return weights

    except Exception as e:
        return {"error": str(e)}

@weights_bp.route("/weights/<region>", methods=["GET"])
def get_weights(region):
    region = region.lower()
    weights = current_weights.get(region)
    if not weights:
        return jsonify({"error": f"No weights found for region: {region}"}), 404
    return jsonify({"region": region, "weights": weights})

@weights_bp.route("/weights/<region>", methods=["POST"])
def update_weights(region):
    region = region.lower()
    data = request.get_json()

    if not data or "strategies" not in data:
        return jsonify({"error": "Invalid input. Expected JSON with 'strategies' key."}), 400

    strategies = data["strategies"]

    weights = compute_strategy_weights(strategies)
    current_weights[region] = weights

    return jsonify({"region": region, "weights": weights})