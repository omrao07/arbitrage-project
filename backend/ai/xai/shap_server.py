from __future__ import annotations

"""
SHAP Server
-----------
A simple API for model explainability:
- Hosts endpoints to compute SHAP values on-demand
- Supports tabular ML models (tree/linear/sklearn/xgboost/lightgbm)
- Returns JSON or numpy arrays for integration into explainability dashboards
- Optional caching
- Can be extended to serve stored SHAP explanations from disk

Run:
  uvicorn backend.analytics.shap_server:app --reload --port 8001

Endpoints:
  POST /explain
    { "model": "xgb1", "data": [[...],[...]] }

  GET /models
    returns list of registered models

  GET /health
    returns {"ok": true}
"""

import os
import json
import joblib # type: ignore
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# optional: SHAP
try:
    import shap # type: ignore
    _has_shap = True
except Exception:
    _has_shap = False

# optional: numpy
try:
    import numpy as np
except Exception:
    np = None

# ---------------------- Registry ----------------------

MODEL_REGISTRY: Dict[str, Any] = {}

def load_models():
    """Load models from disk or memory. Extend as needed."""
    model_dir = os.getenv("SHAP_MODEL_DIR", "models/")
    if not os.path.exists(model_dir):
        return
    for fn in os.listdir(model_dir):
        if fn.endswith(".pkl"):
            name = fn.replace(".pkl","")
            try:
                MODEL_REGISTRY[name] = joblib.load(os.path.join(model_dir, fn))
                print(f"[shap_server] loaded model {name} from {fn}")
            except Exception as e:
                print(f"[shap_server] failed to load {fn}: {e}")

# preload
load_models()

# ---------------------- API Models ----------------------

class ExplainRequest(BaseModel):
    model: str
    data: List[List[float]]  # 2D array

# ---------------------- FastAPI ----------------------

app = FastAPI(title="SHAP Server", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True, "models": list(MODEL_REGISTRY.keys()), "shap": _has_shap}

@app.get("/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.keys())}

@app.post("/explain")
def explain(req: ExplainRequest):
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
    if not _has_shap:
        raise HTTPException(status_code=500, detail="SHAP not installed. pip install shap")

    model = MODEL_REGISTRY[req.model]

    try:
        X = np.array(req.data, dtype=float) # type: ignore
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        out = {
            "shap_values": shap_values.values.tolist(),
            "expected_value": (
                shap_values.base_values.tolist()
                if hasattr(shap_values, "base_values") else None
            ),
            "features": getattr(model, "feature_names_in_", None),
        }
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {e}")

# ---------------------- CLI ----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.analytics.shap_server:app", host="0.0.0.0", port=8001, reload=True)