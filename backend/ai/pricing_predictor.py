# backend/ai/models/pricing_predictor.py
from __future__ import annotations
"""
Pricing Predictor
-----------------
Feature pipeline + multiple model backends (auto-select best available):
 - Baseline EWMA / AR-style features (no deps)
 - scikit-learn (Ridge, RandomForest) if installed
 - XGBoost / LightGBM if installed
 - (optional) PyTorch tiny LSTM (if installed; disabled by default)

APIs:
  PricingPredictor.fit(df)            # train on bars (timestamp, symbol, price[, volume])
  PricingPredictor.predict(rows)      # predict next price/return for rows (dicts or DataFrame)
  PricingPredictor.update(row)        # online update with a new tick/bar
  PricingPredictor.save(path)/load()  # persist model+config
  backtest(df)                        # simple walk-forward evaluation

CLI (CSV: timestamp,symbol,price,volume?):
  Train:
    python -m backend.ai.models.pricing_predictor --train bars.csv --horizon 1 --out model.pkl
  Predict CSV:
    python -m backend.ai.models.pricing_predictor --predict bars_tail.csv --model model.pkl --out preds.csv
"""

import json, os, math, time, pickle, warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# ---------- Optional deps (graceful fallbacks) ----------
try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pandas as _pd
except Exception:
    _pd = None

try:
    from sklearn.linear_model import Ridge  # type: ignore
    from sklearn.ensemble import RandomForestRegressor # type: ignore
    from sklearn.metrics import r2_score, mean_squared_error  # type: ignore
    _has_sklearn = True
except Exception:
    _has_sklearn = False

try:
    import xgboost as _xgb 
    _has_xgb = True
except Exception:
    _has_xgb = False

try:
    import lightgbm as _lgb     
    _has_lgb = True
except Exception:
    _has_lgb = False

try:
    import torch as _torch 
    import torch.nn as _nn
    _has_torch = True
except Exception:
    _has_torch = False

# ---------- Data models ----------
@dataclass
class PredictorConfig:
    target: str = "ret_fwd"         # predict forward return by default
    horizon: int = 1                # steps ahead (bars)
    price_col: str = "price"
    volume_col: str = "volume"
    symbol_col: str = "symbol"
    time_col: str = "timestamp"
    use_logret: bool = True
    lags: Tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21)
    emas: Tuple[int, ...] = (3, 5, 9, 21, 50)
    rsi_windows: Tuple[int, ...] = (7, 14)
    bb_window: int = 20
    bb_k: float = 2.0
    include_volume: bool = True
    model_kind: str = "auto"        # "auto"|"ridge"|"rf"|"xgb"|"lgb"|"ewma"|"lstm"
    task: str = "regression"        # regression predicts next return; you can switch to "price" to predict price deltas
    seed: int = 42

@dataclass
class PredictorState:
    cfg: PredictorConfig
    feature_names: List[str]
    model_info: Dict[str, Any]
    last_symbol_stats: Dict[str, Dict[str, Any]]  # per-symbol running stats for online updates

# ---------- Utility math ----------
def _safe_log(x: float) -> float:
    try:
        return math.log(max(1e-12, x))
    except Exception:
        return 0.0

def _pct(a: float, b: float) -> float:
    if b == 0: return 0.0
    return (a - b) / b

def _ema(prev: float, x: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * prev

# ---------- Feature builder ----------
class FeatureBuilder:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

    def make_frame(self, df):
        """
        df: pandas DataFrame with at least [timestamp,symbol,price,(volume?)]
        Returns DataFrame with engineered features and target.
        """
        if _pd is None:
            raise RuntimeError("pandas is required for bulk training. Use .update() for online.")
        cfg = self.cfg
        df = df.copy()
        # ensure dtypes
        df[cfg.time_col] = _pd.to_datetime(df[cfg.time_col])
        df = df.sort_values([cfg.symbol_col, cfg.time_col])

        # group per symbol
        out = []
        for sym, g in df.groupby(cfg.symbol_col):
            g = g.copy()
            px = g[cfg.price_col].astype(float)
            vol = g[cfg.volume_col].astype(float) if (cfg.include_volume and cfg.volume_col in g.columns) else None

            # returns
            if cfg.use_logret:
                g["ret1"] = _pd.Series(_safe_log(x) for x in px).diff().fillna(0.0) # type: ignore
            else:
                g["ret1"] = px.pct_change().fillna(0.0)

            # lags of ret1
            for L in cfg.lags:
                g[f"lag_{L}"] = g["ret1"].shift(L)

            # EMAs of price + ret
            for W in cfg.emas:
                g[f"ema_{W}"] = px.ewm(span=W, adjust=False).mean()
                g[f"ret_ema_{W}"] = g["ret1"].ewm(span=W, adjust=False).mean()

            # RSI
            for W in cfg.rsi_windows:
                chg = g["ret1"]
                up = _pd.Series(_np.where(chg > 0, chg, 0.0)) if _np is not None else chg.clip(lower=0)
                dn = _pd.Series(_np.where(chg < 0, -chg, 0.0)) if _np is not None else (-chg).clip(lower=0)
                rs = up.rolling(W).mean() / (dn.rolling(W).mean() + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                g[f"rsi_{W}"] = rsi

            # Bollinger (on price)
            W = cfg.bb_window
            m = px.rolling(W).mean()
            s = px.rolling(W).std(ddof=1)
            g["bb_up"] = (px - (m + cfg.bb_k * s)) / (s + 1e-9)
            g["bb_dn"] = (px - (m - cfg.bb_k * s)) / (s + 1e-9)

            # Volume features
            if vol is not None:
                g["zvol_20"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std(ddof=1) + 1e-9)
                g["vol_chg"] = vol.pct_change().replace([_np.inf, -_np.inf], 0.0) if _np is not None else vol.pct_change()

            # forward target (horizon bars ahead)
            if cfg.task == "regression":
                if cfg.use_logret:
                    g[cfg.target] = g["ret1"].shift(-cfg.horizon).fillna(0.0)
                else:
                    g[cfg.target] = px.pct_change(periods=cfg.horizon).shift(-cfg.horizon).fillna(0.0)
            elif cfg.task == "price":
                g[cfg.target] = px.shift(-cfg.horizon).fillna(method="ffill") - px

            out.append(g)
        fd = _pd.concat(out, axis=0)

        # assemble features
        feat_cols = [c for c in fd.columns if c not in (cfg.target, cfg.time_col, cfg.symbol_col, cfg.price_col, cfg.volume_col)]
        # drop rows with NaNs in features/target
        fd = fd.dropna(subset=feat_cols + [cfg.target]).copy()
        return fd, feat_cols

# ---------- Model backends ----------
class _BaseModel:
    kind: str = "base"
    def fit(self, X, y): ...
    def predict(self, X): ...
    def save_bytes(self) -> bytes: return pickle.dumps(self)
    @classmethod
    def load_bytes(cls, b: bytes): return pickle.loads(b)

class _EWMAReg(_BaseModel):
    """No-deps baseline: predict next return via EWMA of past returns."""
    kind = "ewma"
    def __init__(self, span: int = 10):
        self.span = span
        self.alpha = 2.0 / (span + 1)
        self.mu = 0.0
        self.fitted = False
    def fit(self, X, y):
        # X unused; y is returns
        mu = 0.0
        for v in y:
            mu = _ema(mu, float(v), self.alpha)
        self.mu = mu
        self.fitted = True
    def predict(self, X):
        # Return constant EWMA for all rows
        n = len(X) if hasattr(X, "__len__") else 1
        return [self.mu for _ in range(n)]

class _RidgeReg(_BaseModel):
    kind = "ridge"
    def __init__(self, alpha: float = 1.0):
        if not _has_sklearn: raise RuntimeError("sklearn not available")
        self.model = Ridge(alpha=alpha, random_state=42)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class _RFReg(_BaseModel):
    kind = "rf"
    def __init__(self, n_estimators: int = 200, max_depth: int = 8):
        if not _has_sklearn: raise RuntimeError("sklearn not available")
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class _XGBReg(_BaseModel):
    kind = "xgb"
    def __init__(self):
        if not _has_xgb: raise RuntimeError("xgboost not available")
        self.model = _xgb.XGBRegressor(
            max_depth=6, n_estimators=400, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8,
            random_state=42, tree_method="hist", n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y, verbose=False)
    def predict(self, X): return self.model.predict(X)

class _LGBReg(_BaseModel):
    kind = "lgb"
    def __init__(self):
        if not _has_lgb: raise RuntimeError("lightgbm not available")
        self.model = _lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=-1, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

# (Optional) minimal LSTM — disabled by default to avoid torch dependency
class _TinyLSTM(_BaseModel):
    kind = "lstm"
    def __init__(self, input_size: int, hidden: int = 32):
        if not _has_torch: raise RuntimeError("torch not available")
        self.input_size = input_size
        self.model = _nn.LSTM(input_size, hidden, batch_first=True)
        self.head = _nn.Linear(hidden, 1)
        self.opt = _torch.optim.Adam(list(self.model.parameters()) + list(self.head.parameters()), lr=1e-3)
    def fit(self, X, y):
        # X: [N, F] -> reshape into seq length 1
        X = _torch.tensor(X, dtype=_torch.float32).unsqueeze(1)
        y = _torch.tensor(y, dtype=_torch.float32).view(-1, 1)
        for _ in range(10):  # tiny epochs
            self.opt.zero_grad()
            h, _ = self.model(X)
            pred = self.head(h[:, -1, :])
            loss = ((pred - y) ** 2).mean()
            loss.backward(); self.opt.step()
    def predict(self, X):
        with _torch.no_grad():
            X = _torch.tensor(X, dtype=_torch.float32).unsqueeze(1)
            h, _ = self.model(X)
            pred = self.head(h[:, -1, :]).view(-1).cpu().numpy().tolist()
        return pred

# ---------- Predictor wrapper ----------
class PricingPredictor:
    def __init__(self, cfg: PredictorConfig = PredictorConfig()):
        self.cfg = cfg
        self.fb = FeatureBuilder(cfg)
        self.model: _BaseModel = _EWMAReg(span=10)
        self.state = PredictorState(cfg=cfg, feature_names=[], model_info={}, last_symbol_stats={})

    # choose backend
    def _select_model(self, n_features: int) -> _BaseModel:
        kind = self.cfg.model_kind
        if kind == "ewma": return _EWMAReg(10)
        if kind == "ridge": return _RidgeReg()
        if kind == "rf": return _RFReg()
        if kind == "xgb": return _XGBReg()
        if kind == "lgb": return _LGBReg()
        if kind == "lstm": return _TinyLSTM(n_features)

        # auto: prefer tree boosters > RF > Ridge > EWMA
        if _has_lgb: return _LGBReg()
        if _has_xgb: return _XGBReg()
        if _has_sklearn: return _RFReg()
        if _has_sklearn: return _RidgeReg()
        return _EWMAReg(10)

    # bulk fit
    def fit(self, df) -> Dict[str, Any]:
        if _pd is None:
            raise RuntimeError("Training requires pandas. Install pandas/numpy.")
        cfg = self.cfg
        fd, feat_cols = self.fb.make_frame(df)
        X = fd[feat_cols].values if _np is not None else [[float(v) for v in row] for row in fd[feat_cols].to_numpy().tolist()]
        y = fd[cfg.target].values if _np is not None else fd[cfg.target].tolist()

        self.model = self._select_model(len(feat_cols))
        self.model.fit(X, y)
        self.state.feature_names = list(feat_cols)
        self.state.model_info = {"kind": getattr(self.model, "kind", "unknown"), "n_features": len(feat_cols)}
        return {"n_rows": len(fd), "n_features": len(feat_cols), "model": self.state.model_info}

    # predict for DataFrame OR list of dict rows {timestamp,symbol,price,volume?}
    def predict(self, rows: Union[Any, List[Dict[str, Any]]]) -> List[float]:
        if _pd is not None and hasattr(rows, "__class__") and rows.__class__.__name__ in ("DataFrame", "DataFrame"):
            # build features with no target shift; we use latest available feature row per symbol
            df = rows.copy()
            fd, feat_cols = self.fb.make_frame(df)
            # choose last per-symbol
            take_idx = fd.groupby(self.cfg.symbol_col).tail(1).index
            X = fd.loc[take_idx, feat_cols].values
            return list(self.model.predict(X)) # type: ignore
        # else assume list of dicts with running stats per symbol
        feats = []
        for row in rows:
            feats.append(self._features_from_row(row))
        return list(self.model.predict(feats)) # type: ignore

    # online update: feed a new bar and update running stats; optional partial fit for EWMA
    def update(self, row: Dict[str, Any]) -> Optional[float]:
        sym = str(row.get(self.cfg.symbol_col))
        px = float(row.get(self.cfg.price_col, 0.0))
        vol = float(row.get(self.cfg.volume_col, 0.0)) if self.cfg.include_volume else 0.0
        st = self.state.last_symbol_stats.setdefault(sym, {
            "last_px": None, "ema": {}, "ret1": 0.0, "zvol_mu": 0.0, "zvol_sd": 1.0, "vol": 0.0, "n": 0
        })
        # compute ret
        if st["last_px"] is None:
            ret1 = 0.0
        else:
            if self.cfg.use_logret:
                ret1 = _safe_log(px) - _safe_log(st["last_px"])
            else:
                ret1 = _pct(px, st["last_px"])
        st["last_px"] = px
        st["vol"] = vol
        st["n"] += 1

        # update EMAs
        for W in self.cfg.emas:
            a = 2.0 / (W + 1.0)
            st["ema"][W] = _ema(st["ema"].get(W, px), px, a)

        # naive online zvol
        mu = st["zvol_mu"]; sd = st["zvol_sd"]
        mu_new = mu + (vol - mu) / max(1, st["n"])
        sd_new = max(1e-6, sd + (abs(vol - mu_new) - sd) / max(1, st["n"]))
        st["zvol_mu"], st["zvol_sd"], st["ret1"] = mu_new, sd_new, ret1

        # if EWMA baseline, update internal mean
        if isinstance(self.model, _EWMAReg):
            self.model.mu = _ema(self.model.mu, ret1, self.model.alpha)

        # produce features and one-step prediction
        x = self._features_from_row(row, use_state=True, sym=sym)
        try:
            yhat = float(self.model.predict([x])[0]) # type: ignore
        except Exception:
            yhat = 0.0
        return yhat

    # create feature vector from a single row using tracked state
    def _features_from_row(self, row: Dict[str, Any], use_state: bool = True, sym: Optional[str] = None) -> List[float]:
        cfg = self.cfg
        sym = sym or str(row.get(cfg.symbol_col))
        px = float(row.get(cfg.price_col, 0.0))
        vol = float(row.get(cfg.volume_col, 0.0)) if cfg.include_volume else 0.0
        if use_state:
            st = self.state.last_symbol_stats.setdefault(sym, {"last_px": px, "ema": {}, "ret1": 0.0, "zvol_mu": 0.0, "zvol_sd": 1.0, "vol": 0.0, "n": 0})
        else:
            st = {"last_px": px, "ema": {}, "ret1": 0.0, "zvol_mu": 0.0, "zvol_sd": 1.0, "vol": 0.0, "n": 0}

        # basic ret1
        ret1 = st.get("ret1", 0.0)
        feats = [ret1]
        # lags emulated by repeated ret1 entries (for API parity when online)
        for L in cfg.lags:
            feats.append(ret1)  # online mode lacks full lag buffer; acceptable for baseline

        # ema features: ratio to price and ret ema proxy
        for W in cfg.emas:
            ema_px = st["ema"].get(W, px)
            feats.append((px - ema_px) / (ema_px + 1e-9))
            feats.append(ret1)  # placeholder for ret_ema_W in online mode

        # RSI proxies (use ret1)
        for _W in cfg.rsi_windows:
            feats.append(ret1)

        # bollinger proxy
        feats.append(0.0); feats.append(0.0)

        # volume zscore
        if cfg.include_volume:
            mu = st["zvol_mu"]; sd = st["zvol_sd"] or 1.0
            feats.append((vol - mu) / sd)
            feats.append(0.0)  # vol_chg placeholder
        return feats

    # persistence
    def save(self, path: str) -> None:
        blob = {
            "state": asdict(self.state),
            "model_kind": getattr(self.model, "kind", "unknown"),
            "model_blob": self.model.save_bytes(),
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "PricingPredictor":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cfg = PredictorConfig(**blob["state"]["cfg"])
        self = cls(cfg)
        # restore feature names & info
        self.state.feature_names = blob["state"]["feature_names"]
        self.state.model_info = blob["state"]["model_info"]
        self.state.last_symbol_stats = blob["state"]["last_symbol_stats"]
        # restore model
        kind = blob["model_kind"]
        if kind == "ewma": self.model = _EWMAReg(10)
        elif kind == "ridge": self.model = _RidgeReg()
        elif kind == "rf": self.model = _RFReg()
        elif kind == "xgb": self.model = _XGBReg()
        elif kind == "lgb": self.model = _LGBReg()
        elif kind == "lstm": self.model = _TinyLSTM(len(self.state.feature_names) or 8)
        else: self.model = _EWMAReg(10)
        self.model = self.model.load_bytes(blob["model_blob"])
        return self

# ---------- Simple walk-forward backtest ----------
def backtest(df, cfg: PredictorConfig = PredictorConfig()) -> Dict[str, Any]:
    if _pd is None or _np is None:
        raise RuntimeError("Backtest requires pandas & numpy.")
    fb = FeatureBuilder(cfg)
    fd, feats = fb.make_frame(df)
    # chronological split per symbol
    fd = fd.sort_values([cfg.symbol_col, cfg.time_col])
    preds = _pd.Series(index=fd.index, dtype=float)
    model = PricingPredictor(cfg)._select_model(len(feats))
    # rolling-origin: expand window
    split = int(len(fd) * 0.7)
    X_tr, y_tr = fd[feats].iloc[:split].values, fd[cfg.target].iloc[:split].values
    X_te, y_te = fd[feats].iloc[split:].values, fd[cfg.target].iloc[split:].values
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    preds.iloc[split:] = yhat
    # metrics
    mse = float(((y_te - _np.array(yhat)) ** 2).mean())
    rmse = float(mse ** 0.5)
    r2 = float(1.0 - (mse / (_np.var(y_te) + 1e-12))) # type: ignore
    ic = float(_pd.Series(y_te).corr(_pd.Series(yhat), method="spearman"))
    return {"rmse": rmse, "r2": r2, "ic": ic, "n_train": int(split), "n_test": int(len(fd) - split)}

# ---------- CLI ----------
def _read_csv(path: str):
    if _pd is None:
        raise SystemExit("Please install pandas to use the CLI.")
    df = _pd.read_csv(path)
    # best-effort column names
    for c in list(df.columns):
        if c.lower() in ("ts","time","datetime","date"): df = df.rename(columns={c:"timestamp"})
        if c.lower() in ("sym","ticker"): df = df.rename(columns={c:"symbol"})
        if c.lower() in ("close","price_last","last"): df = df.rename(columns={c:"price"})
        if c.lower() in ("vol","volume_qty"): df = df.rename(columns={c:"volume"})
    return df

def _main():
    import argparse, csv
    p = argparse.ArgumentParser(description="Pricing Predictor (train/predict)")
    p.add_argument("--train", type=str, help="CSV for training")
    p.add_argument("--predict", type=str, help="CSV for prediction")
    p.add_argument("--model", type=str, help="Path to load/save model.pkl")
    p.add_argument("--out", type=str, help="Output CSV/JSON")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--kind", type=str, default="auto", choices=["auto","ewma","ridge","rf","xgb","lgb","lstm"])
    args = p.parse_args()

    cfg = PredictorConfig(horizon=args.horizon, model_kind=args.kind)
    pred = PricingPredictor(cfg)

    if args.train:
        df = _read_csv(args.train)
        info = pred.fit(df)
        if args.model:
            pred.save(args.model)
        print(json.dumps({"train_info": info}, indent=2))
        return

    if args.predict:
        if not args.model:
            raise SystemExit("--model is required for prediction")
        pred = PricingPredictor.load(args.model)
        df = _read_csv(args.predict)
        yhat = pred.predict(df)
        if args.out:
            # write alongside input with yhat
            df["_yhat"] = yhat + [None]*(len(df)-len(yhat)) if len(yhat) < len(df) else yhat[:len(df)]
            df.to_csv(args.out, index=False)
        else:
            # print JSON lines
            for v in yhat:
                print(json.dumps({"yhat": float(v)}))
        return

    p.print_help()

if __name__ == "__main__":  # pragma: no cover
    _main()