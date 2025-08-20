# backend/engine/mev_detection.py
from __future__ import annotations

import json, math, os, time, threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis

"""
MEV Detection / Guard (paper)
-----------------------------
Goal:
  • Consume pending tx feed (mempool) + pool state snapshots from Redis.
  • Detect common MEV patterns: Sandwich setup (frontrun+backrun), Backrun clusters, Priority Gas Auctions (PGA),
    JIT LP adds, and “toxic flow” (very large swaps vs pool depth).
  • Provide a fast guard API for your execution engine:
        decision = guard.assess_order(chain="ETH", pool=POOL_ADDR, route=[...], notional_usd=..., max_slip_bps=...)
        -> { action: "ALLOW" | "DELAY" | "WIDEN" | "REROUTE" | "BLOCK", risk_score: float, reasons: [...] }

Redis I/O you publish elsewhere (examples):
  • Mempool stream (push externally from your node/WS):
      XADD mempool:ETH:txs * '{"hash":"0x..","from":"0x..","to":"0xRouter","method":"swapExactTokensForTokens","pool":"0xPool",
                                "amount_in":1234.5,"amount_out_min":..., "token_in":"USDC","token_out":"WETH",
                                "gas_price_gwei": 12.3, "prio_fee_gwei": 0.8, "nonce": 120, "sim_delta_px_bps": -18.4}'
    - Use method names or 4byte selectors; include pool or the first hop pool if available.

  • Pool state (refresh every second or on events):
      HSET amm:ETH:pools "0xPool" '{"dex":"UNI_V3","fee_bps":5, "liquidity_usd": 3_500_000,
                                    "mid_price": 0.000312, "depth_1pct_usd": 450_000,
                                    "vol_5m_usd": 2_100_000, "updated_ms": 1765400000000}'

  • Known bot labels (optional):
      HSET mev:labels "<addr>" "BOT"   # or "FLASHBOTS" / "SEARCHER" / "ARBITRAGE"

  • Kill:
      SET risk:halt 0|1

Outputs the guard publishes:
  • XADD alerts:mev * '{"chain":"ETH","pool":"0xPool","type":"SANDWICH_RISK","score":0.82,"txs":[<hashes>],"window_ms":180}'
  • HSET mev:last_advice "<ctx|pool>" '{"ts_ms":..., "action":"DELAY","score":0.76,"reasons":["sandwich_cluster","pga"]}'
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("MEV_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("MEV_REDIS_PORT", "6379"))
CHAIN      = os.getenv("MEV_CHAIN", "ETH").upper()

# Keys
MEMPOOL_STREAM = os.getenv("MEV_MEMPOOL_STREAM", f"mempool:{CHAIN}:txs")
POOL_HK        = os.getenv("MEV_POOL_HK",        f"amm:{CHAIN}:pools")
LABEL_HK       = os.getenv("MEV_LABEL_HK",       "mev:labels")
HALT_KEY       = os.getenv("MEV_HALT_KEY",       "risk:halt")
ADVICE_HK      = os.getenv("MEV_ADVICE_HK",      "mev:last_advice")
ALERT_STREAM   = os.getenv("MEV_ALERT_STREAM",   "alerts:mev")

# Tunables
WINDOW_MS            = int(os.getenv("MEV_WINDOW_MS", "1200"))    # cluster lookback for patterns
SUBWINDOW_MS         = int(os.getenv("MEV_SUBWINDOW_MS","350"))    # tighter window for sandwich “bread”
PGA_Z_GAS            = float(os.getenv("MEV_PGA_Z_GAS", "2.2"))    # gas gwei z-score to flag PGA
SANDWICH_MIN_TXS     = int(os.getenv("MEV_SANDWICH_MIN_TXS","2"))  # >=2 near-identical-direction swaps
SANDWICH_MIN_EDGE_BPS= float(os.getenv("MEV_SANDWICH_MIN_EDGE_BPS","10.0")) # implied slippage harm to victim
BACKRUN_MIN_NOTIONAL = float(os.getenv("MEV_BACKRUN_MIN_NOTIONAL","50000"))  # USD
TOXIC_SIZE_DEPTH_X   = float(os.getenv("MEV_TOXIC_SIZE_DEPTH_X", "0.35"))    # victim size >= 35% of 1% depth
COOLDOWN_MS          = int(os.getenv("MEV_COOLDOWN_MS","1500"))    # after BLOCK/DELAY, cool down horizon
MAX_CACHE_TX         = int(os.getenv("MEV_MAX_CACHE_TX","4000"))

# Advice thresholds
BLOCK_SCORE      = float(os.getenv("MEV_BLOCK_SCORE", "0.85"))
DELAY_SCORE      = float(os.getenv("MEV_DELAY_SCORE", "0.60"))
WIDEN_SCORE      = float(os.getenv("MEV_WIDEN_SCORE", "0.45"))

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ Data classes ============================
@dataclass
class Tx:
    ts_ms: int
    h: str
    frm: str
    to: str
    method: str
    pool: Optional[str]
    token_in: Optional[str]
    token_out: Optional[str]
    amount_in: float
    gas_gwei: float
    prio_gwei: float
    nonce: int
    sim_delta_px_bps: float  # (optional) simulated price impact vs mid

@dataclass
class Advice:
    action: str      # ALLOW | DELAY | WIDEN | REROUTE | BLOCK
    score: float     # 0..1
    reasons: List[str]
    horizon_ms: int

# ============================ Utils ============================
def _now_ms() -> int: return int(time.time()*1000)

def _parse_tx(js: dict) -> Optional[Tx]:
    try:
        return Tx(
            ts_ms=int(js.get("ts_ms", _now_ms())),
            h=str(js.get("hash")),
            frm=str(js.get("from","")).lower(),
            to=str(js.get("to","")).lower(),
            method=str(js.get("method","")).lower(),
            pool=(str(js.get("pool")).lower() if js.get("pool") else None),
            token_in=(str(js.get("token_in")).upper() if js.get("token_in") else None),
            token_out=(str(js.get("token_out")).upper() if js.get("token_out") else None),
            amount_in=float(js.get("amount_in", 0.0)),
            gas_gwei=float(js.get("gas_price_gwei", js.get("gas_gwei", 0.0))),
            prio_gwei=float(js.get("prio_fee_gwei", 0.0)),
            nonce=int(js.get("nonce", 0)),
            sim_delta_px_bps=float(js.get("sim_delta_px_bps", 0.0))
        )
    except Exception:
        return None

def _pool_info(addr: str) -> dict:
    raw = r.hget(POOL_HK, addr.lower())
    if not raw: return {}
    try: return json.loads(raw)
    except Exception: return {}

def _label(addr: str) -> str:
    return (r.hget(LABEL_HK, addr.lower()) or "").upper()

# ============================ Core Guard ============================
class MEVGuard:
    """
    Maintains per-pool time-ordered tx cache; exposes assess_order() for the OMS/execution engine.
    """
    def __init__(self, chain: str = CHAIN):
        self.chain = chain
        self._lock = threading.Lock()
        # per pool: list[Tx] sorted by ts_ms (we prunes regularly)
        self._by_pool: Dict[str, List[Tx]] = {}
        # gas stats for PGA detection (EWMA mean/var)
        self._gas_mv = {"mean": 20.0, "var": 25.0, "alpha": 0.08}
        self._last_seen_id = "0-0"

    # ---------- ingestion loop (optional) ----------
    def run_stream_loop(self, stop_event: Optional[threading.Event] = None) -> None:
        while True:
            if stop_event and stop_event.is_set(): break
            if (r.get(HALT_KEY) or "0") == "1":
                time.sleep(0.25); continue
            try:
                resp = r.xread({MEMPOOL_STREAM: self._last_seen_id}, count=200, block=500)
                if not resp:
                    continue
                for _, entries in resp:
                    for entry_id, kv in entries:
                        self._last_seen_id = entry_id
                        try:
                            js = json.loads(list(kv.values())[0]) if isinstance(kv, dict) else {}
                        except Exception:
                            js = {}
                        tx = _parse_tx(js)
                        if tx:
                            self._ingest_tx(tx)
            except Exception:
                time.sleep(0.05)

    # ---------- public API ----------
    def assess_order(self,
                     chain: str,
                     pool: str,
                     route: List[str],
                     notional_usd: float,
                     max_slip_bps: float,
                     ctx_key: str = "default") -> Advice:
        """
        Fast decision for the next order.
        """
        if chain.upper() != self.chain:
            return Advice("ALLOW", 0.0, ["different_chain"], horizon_ms=0)

        now = _now_ms()
        txs = self._by_pool.get(pool.lower(), [])
        recent = [t for t in txs if now - t.ts_ms <= WINDOW_MS]
        reasons: List[str] = []
        score = 0.0

        # 1) Sandwich cluster risk
        s_score, s_r = self._sandwich_score(recent, pool, max_slip_bps)
        if s_score > 0:
            score = max(score, s_score); reasons += s_r

        # 2) Backrun / toxic flow risk
        b_score, b_r = self._backrun_score(recent, pool, notional_usd)
        if b_score > 0:
            score = max(score, b_score); reasons += b_r

        # 3) PGA (priority gas auction) surge → poor fill odds
        pga_score, p_r = self._pga_score(recent)
        if pga_score > 0:
            score = max(score, pga_score); reasons += p_r

        # 4) JIT LP (liquidity can move against you)
        jit_score, j_r = self._jit_lp_score(recent, pool)
        if jit_score > 0:
            score = max(score, jit_score); reasons += j_r

        # Decision ladder
        if score >= BLOCK_SCORE:
            act = "BLOCK"; horizon = COOLDOWN_MS
        elif score >= DELAY_SCORE:
            act = "DELAY"; horizon = int(COOLDOWN_MS * 0.6)
        elif score >= WIDEN_SCORE:
            act = "WIDEN"; horizon = int(COOLDOWN_MS * 0.4)
        else:
            act = "ALLOW"; horizon = 0

        # Persist advice (for dashboards)
        key = f"{ctx_key}|{pool.lower()}"
        r.hset(ADVICE_HK, key, json.dumps({"ts_ms": now, "action": act, "score": round(score,3), "reasons": reasons[:6]}))

        # Emit alert when we’re actually constraining
        if act != "ALLOW":
            self._emit_alert(pool, reasons, score, WINDOW_MS)

        return Advice(act, float(round(score, 3)), reasons[:6], horizon_ms=horizon)

    # ---------- internals ----------
    def _ingest_tx(self, tx: Tx) -> None:
        with self._lock:
            if tx.pool:
                arr = self._by_pool.setdefault(tx.pool, [])
                arr.append(tx)
                # prune
                cutoff = _now_ms() - WINDOW_MS*2
                while arr and arr[0].ts_ms < cutoff:
                    arr.pop(0)
                # cap
                if len(arr) > MAX_CACHE_TX:
                    del arr[: len(arr) - MAX_CACHE_TX]
            # update gas EWMA/Var (for PGA)
            self._update_gas(tx.gas_gwei)

    def _update_gas(self, g: float) -> None:
        a = self._gas_mv["alpha"]
        m0 = self._gas_mv["mean"]
        v0 = self._gas_mv["var"]
        m = (1 - a)*m0 + a*g
        v = max(1e-6, (1 - a)*(v0 + (g - m0)*(g - m)))
        self._gas_mv["mean"], self._gas_mv["var"] = m, v

    # --- detectors ---
    def _sandwich_score(self, recent: List[Tx], pool: str, our_slip_bps: float) -> Tuple[float, List[str]]:
        """
        Heuristic: multiple swaps through same pool in SUBWINDOW_MS with:
          • same token direction as us (if known), high gas/priority,
          • simulated negative price delta for middle (victim),
          • labels indicate bots for the bread (optional).
        We don't know our direction here; we use cluster intensity & harm proxy.
        """
        if not recent:
            return 0.0, []

        now = _now_ms()
        window = [t for t in recent if now - t.ts_ms <= SUBWINDOW_MS and (t.method.startswith("swap") or "swap" in t.method)]
        if len(window) < SANDWICH_MIN_TXS:
            return 0.0, []

        # group by token pair dir
        buckets: Dict[Tuple[str,str], List[Tx]] = {}
        for t in window:
            key = (t.token_in or "UNK", t.token_out or "UNK")
            buckets.setdefault(key, []).append(t)

        best = 0.0; reasons: List[str] = []
        for k, arr in buckets.items():
            if len(arr) < SANDWICH_MIN_TXS: 
                continue
            # sort by gas to approximate “bread” priority
            arr_sorted = sorted(arr, key=lambda x: (x.gas_gwei, x.prio_gwei), reverse=True)
            high = arr_sorted[:2] if len(arr_sorted) >= 2 else arr_sorted
            victim_like = sorted(arr, key=lambda x: x.ts_ms)[len(arr)//2]  # middle-ish
            # harm proxy: victim simulated price delta and our slippage tolerance
            harm_bps = abs(victim_like.sim_delta_px_bps)
            # bot labels?
            bot_hits = sum(1 for t in high if _label(t.frm) in ("BOT","SEARCHER","FLASHBOTS"))
            # score components 0..1
            s_harm = min(1.0, max(0.0, (harm_bps - SANDWICH_MIN_EDGE_BPS) / 50.0))
            s_intensity = min(1.0, math.log1p(len(arr))/math.log(8))  # up to ~1 for 7+
            s_bots = min(1.0, bot_hits/2.0)
            s = 0.55*s_harm + 0.30*s_intensity + 0.15*s_bots
            if s > best:
                best = s
        if best <= 0.0:
            return 0.0, []
        return best, ["sandwich_cluster"]

    def _backrun_score(self, recent: List[Tx], pool: str, notional_usd: float) -> Tuple[float, List[str]]:
        """
        Heuristic: presence of very large swaps just behind prior swaps (same dir),
        and your order would be small vs depth → likely poor execution (or being backrun yourself).
        """
        if not recent:
            return 0.0, []

        # get pool stats
        info = _pool_info(pool)
        depth_1pct = float(info.get("depth_1pct_usd", 0.0)) or 1.0
        liq = float(info.get("liquidity_usd", 0.0)) or 1.0

        # cluster big swaps
        big = [t for t in recent if t.amount_in and t.amount_in * (info.get("mid_price") or 1.0) >= BACKRUN_MIN_NOTIONAL]
        if not big:
            return 0.0, []

        # if our size is a big chunk of 1% depth, risk rises
        size_ratio = notional_usd / max(1.0, depth_1pct)
        s_size = min(1.0, max(0.0, (size_ratio - TOXIC_SIZE_DEPTH_X) / (1.0 - TOXIC_SIZE_DEPTH_X))) if size_ratio > TOXIC_SIZE_DEPTH_X else 0.0
        # more big txs in window -> worse
        s_intensity = min(1.0, (len(big) - 1) / 3.0) if len(big) > 1 else 0.0
        score = 0.6*s_size + 0.4*s_intensity
        return score, (["backrun_risk"] if score > 0 else [])

    def _pga_score(self, recent: List[Tx]) -> Tuple[float, List[str]]:
        """
        Priority gas auction: many txs with gas >> EWMA mean → mempool congestion / bots.
        """
        if not recent:
            return 0.0, []
        m = self._gas_mv["mean"]; v = self._gas_mv["var"]; sd = math.sqrt(max(1e-9, v))
        zs = [(t.gas_gwei - m)/sd for t in recent]
        # fraction above PGA_Z_GAS
        frac = sum(1 for z in zs if z >= PGA_Z_GAS) / max(1, len(zs))
        score = min(1.0, frac * 1.5)  # 2/3 high → score ~1
        return (score, ["pga"]) if score > 0 else (0.0, [])

    def _jit_lp_score(self, recent: List[Tx], pool: str) -> Tuple[float, List[str]]:
        """
        JIT LP adds/remo (v3-style) are hard to see without on-chain logs; use proxy:
          • multiple tiny swaps + one large swap within short window → liquidity reshuffle risk.
        """
        if len(recent) < 3:
            return 0.0, []
        amounts = sorted([abs(t.amount_in) for t in recent if t.amount_in > 0.0])
        if not amounts:
            return 0.0, []
        small = amounts[: max(1, len(amounts)//3)]
        large = amounts[-1]
        if len(small) >= 2 and sum(small) > 0 and large / max(1.0, sum(small)) >= 3.0:
            # spiky flow → moderate score
            return 0.45, ["jit_lp_pattern"]
        return 0.0, []

    def _emit_alert(self, pool: str, reasons: List[str], score: float, window_ms: int) -> None:
        try:
            r.xadd(ALERT_STREAM, {"msg": json.dumps({
                "chain": self.chain, "pool": pool, "type": "+".join(sorted(set(reasons))),
                "score": round(score,3), "window_ms": window_ms, "ts_ms": _now_ms()
            })})
        except Exception:
            pass

# ---------------- Convenience singleton ----------------
_guard_singleton: Optional[MEVGuard] = None

def guard() -> MEVGuard:
    global _guard_singleton
    if _guard_singleton is None:
        _guard_singleton = MEVGuard()
    return _guard_singleton

# ---------------- Example CLI loop ----------------
if __name__ == "__main__":
    g = guard()
    stop = threading.Event()
    try:
        print("[mev_guard] starting mempool loop…")
        g.run_stream_loop(stop_event=stop)
    except KeyboardInterrupt:
        stop.set()
        print("\n[mev_guard] stopped.")