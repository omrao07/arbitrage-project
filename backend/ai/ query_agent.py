# backend/ai/query_agent.py
"""
QueryAgent: lightweight NL Q&A over your trading state.

Listens to:
  - ai.query      { "ts_ms":..., "user":"u123", "q":"what's my exposure if usd inr +3%?" }
  - state streams (risk.var, risk.dd, pnl.snap, positions, governor.events)

Emits:
  - ai.answers    { "ts_ms":..., "user":"u123", "q":"...", "answer":"...", "refs":{...} }

Design:
  - Pattern-based interpreter (regex/keywords)
  - Answers from current rolling state (positions, PnL, risk ladders)
  - Extendable with LLM if available (transformers/OpenAI client)
"""

from __future__ import annotations
import os, re, time, json
from typing import Any, Dict, List, Optional

try:
    from backend.bus.streams import consume_stream, publish_stream
except ImportError:
    consume_stream = publish_stream = None  # type: ignore

# Optional: if you want to back LLM
_USE_LLM = False
try:
    import openai  # type: ignore
    _USE_LLM = True and bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _USE_LLM = False


def _utc_ms() -> int:
    return int(time.time() * 1000)


class State:
    """
    Rolling state of risk/positions/pnl, updated from streams.
    """
    def __init__(self):
        self.positions: Dict[str, float] = {}      # sym -> qty
        self.marks: Dict[str, float] = {}          # sym -> price
        self.pnl_snap: Dict[str, float] = {}       # per strategy PnL
        self.var_snap: Dict[str, Any] = {}         # firm_var, per_strategy
        self.dd_snap: Dict[str, Any] = {}          # firm_dd, per_strategy
        self.gov_events: List[Dict[str, Any]] = [] # last few governor actions

    def exposure(self, sym: str, shock: float=0.0) -> float:
        qty = self.positions.get(sym.upper(), 0.0)
        px  = self.marks.get(sym.upper(), 0.0)
        if not px: return 0.0
        base = qty * px
        if shock != 0.0:
            base *= (1.0 + shock)
        return base


class QueryAgent:
    def __init__(self,
        in_stream="ai.query",
        out_stream="ai.answers",
        registry: Optional[Dict[str, Any]]=None
    ):
        self.in_stream = in_stream
        self.out_stream = out_stream
        self.state = State()
        self.registry = registry or {}

        self.patterns = [
            (re.compile(r"exposure.*([A-Z]{2,6}(?:\.NS)?)", re.I), self._answer_exposure),
            (re.compile(r"VaR", re.I), self._answer_var),
            (re.compile(r"drawdown|DD", re.I), self._answer_dd),
            (re.compile(r"PnL|profit|loss", re.I), self._answer_pnl),
            (re.compile(r"governor|trim|reduce", re.I), self._answer_governor),
        ]

    # ------------ Core loop ------------
    def run(self, poll_ms=500):
        assert consume_stream and publish_stream, "bus streams not wired"
        cur = "$"
        while True:
            for _, msg in consume_stream(self.in_stream, start_id=cur, block_ms=poll_ms, count=50):
                cur = "$"
                if isinstance(msg,str):
                    try: msg=json.loads(msg)
                    except: continue
                self._handle_query(msg)

    # ------------ Handle query ------------
    def _handle_query(self, qmsg: Dict[str, Any]):
        q = str(qmsg.get("q") or "").strip()
        if not q: return
        user = qmsg.get("user","anon")

        # Try pattern match
        for pat, fn in self.patterns:
            m = pat.search(q)
            if m:
                ans = fn(q, m)
                self._emit(q, user, ans)
                return

        # fallback: LLM if available
        if _USE_LLM:
            ans = self._answer_llm(q)
            self._emit(q,user,ans)
        else:
            self._emit(q,user,"Sorry, I don’t understand. Try 'exposure RELIANCE.NS' or 'VaR'.")

    # ------------ Answers ------------
    def _answer_exposure(self, q: str, m) -> str:
        sym = m.group(1).upper() if m.group(1) else ""
        shock = 0.0
        m2 = re.search(r"([+-]?\d+)%", q)
        if m2:
            shock = float(m2.group(1))/100.0
        exp = self.state.exposure(sym, shock)
        return f"Exposure {sym} ≈ {exp:,.0f} {'(with shock)' if shock else ''}"

    def _answer_var(self, q, m) -> str:
        firm = self.state.var_snap.get("firm_var_pct_nav", None)
        return f"Firm VaR = {firm*100:.2f}% NAV" if firm else "No VaR snapshot yet."

    def _answer_dd(self, q,m) -> str:
        dd = self.state.dd_snap.get("firm_dd", None)
        return f"Firm Drawdown = {dd*100:.1f}%" if dd else "No drawdown data yet."

    def _answer_pnl(self, q,m) -> str:
        total = sum(self.state.pnl_snap.values())
        return f"Total PnL ≈ {total:,.0f}"

    def _answer_governor(self, q,m) -> str:
        if not self.state.gov_events: return "No governor actions recorded."
        last = self.state.gov_events[-1]
        return f"Governor action: {last.get('action')} reason={last.get('reason')}"

    def _answer_llm(self, q: str) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":"You are a trading risk Q&A assistant."},
                          {"role":"user","content":q}],
                max_tokens=200
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            return f"LLM error: {e}"

    # ------------ Emit ------------
    def _emit(self, q: str, user: str, ans: str):
        out = {
            "ts_ms": _utc_ms(),
            "user": user,
            "q": q,
            "answer": ans,
            "refs": {}
        }
        publish_stream(self.out_stream, out) # type: ignore


# ------------------- CLI --------------------

def main():
    qa = QueryAgent()
    try:
        qa.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()