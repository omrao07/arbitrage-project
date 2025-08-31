// frontend/components/ArbMap.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

/** ---------- Types ---------- */
type PriceTick = { venue: string; price: number; ts: number };
type Opportunity = {
  from: string;
  to: string;
  rawPct: number;     // (to - from)/from
  netPct: number;     // after fees + slippage
};
type SpreadPt = { t: number; spreadPct: number };

interface Props {
  symbol?: string;               // e.g., "BTC/USDT" or "AAPL"
  wsUrl?: string;                // optional WS for live ticks (JSON: {venue, price, ts})
  pricesEndpoint?: string;       // GET -> PriceTick[]
  spreadEndpoint?: string;       // GET -> SpreadPt[]
  defaultFeeBps?: number;        // roundtrip fee assumption if not provided per-venue
}

/** ---------- Component ---------- */
export default function ArbMap({
  symbol = "BTC/USDT",
  wsUrl,
  pricesEndpoint = "/api/arbmap/prices?symbol=",
  spreadEndpoint = "/api/arbmap/history?symbol=",
  defaultFeeBps = 8, // 8 bps roundtrip default
}: Props) {
  const [ticks, setTicks] = useState<PriceTick[]>([]);
  const [spreadHist, setSpreadHist] = useState<SpreadPt[]>([]);
  const [feeBps, setFeeBps] = useState<number>(defaultFeeBps);
  const [slipBps, setSlipBps] = useState<number>(5); // assumed slippage bps
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);

  // ---------- Fetch initial data ----------
  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const [p, s] = await Promise.all([
          fetch(pricesEndpoint + encodeURIComponent(symbol)).then(r => r.json()),
          fetch(spreadEndpoint + encodeURIComponent(symbol)).then(r => r.json()),
        ]);
        setTicks(Array.isArray(p) ? p : []);
        setSpreadHist(Array.isArray(s) ? s : []);
      } catch (e: any) {
        setErr(e?.message || "Failed to load arbitrage data");
      } finally {
        setLoading(false);
      }
    })();
  }, [symbol, pricesEndpoint, spreadEndpoint]);

  // ---------- Optional WS live updates ----------
  useEffect(() => {
    if (!wsUrl) return;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "subscribe", channel: "arbmap", symbol }));
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg?.venue && msg?.price) {
          setTicks(prev => upsertTick(prev, { venue: String(msg.venue), price: Number(msg.price), ts: Number(msg.ts || Date.now()) }));
        }
      } catch {}
    };
    ws.onerror = (e) => console.error("Arb WS error", e);
    ws.onclose = () => (wsRef.current = null);
    return () => ws.close();
  }, [wsUrl, symbol]);

  // ---------- Derived ----------
  const venueList = useMemo(() => unique(ticks.map(t => t.venue)).sort(), [ticks]);

  const priceMap = useMemo(() => {
    const m = new Map<string, PriceTick>();
    for (const t of ticks) {
      if (!m.has(t.venue) || (m.get(t.venue)!.ts < t.ts)) m.set(t.venue, t);
    }
    return m;
  }, [ticks]);

  const matrix = useMemo(() => {
    // M[i][j] = pct edge (%): buy at i, sell at j -> (Pj - Pi)/Pi
    const venues = venueList;
    const M: number[][] = venues.map(() => venues.map(() => NaN));
    for (let i = 0; i < venues.length; i++) {
      const pi = priceMap.get(venues[i])?.price;
      if (!isFinite(pi || NaN)) continue;
      for (let j = 0; j < venues.length; j++) {
        const pj = priceMap.get(venues[j])?.price;
        if (pj === undefined || !isFinite(pj)) continue;
        M[i][j] = (pj - (pi as number)) / (pi as number);
      }
    }
    return M;
  }, [venueList, priceMap]);

  const opps = useMemo<Opportunity[]>(() => {
    const res: Opportunity[] = [];
    const fee = (feeBps + slipBps) / 1e4; // convert bps → fraction
    for (let i = 0; i < venueList.length; i++) {
      for (let j = 0; j < venueList.length; j++) {
        if (i === j) continue;
        const v = matrix?.[i]?.[j];
        if (!Number.isFinite(v)) continue;
        const net = v - fee; // simplistic: subtract roundtrip costs
        res.push({ from: venueList[i], to: venueList[j], rawPct: v, netPct: net });
      }
    }
    // sort by netPct desc
    return res.sort((a, b) => b.netPct - a.netPct).slice(0, 12);
  }, [matrix, venueList, feeBps, slipBps]);

  const bestOpp = opps[0];

  // ---------- Render ----------
  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Arbitrage Map — {symbol}</h2>
          <p className="text-sm opacity-70">Cross-venue price gaps, top routes, and spread history</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs opacity-70">Fees (bps)</label>
          <input
            type="number"
            value={feeBps}
            onChange={(e) => setFeeBps(Number(e.target.value))}
            className="w-20 border rounded-lg px-2 py-1 text-sm dark:bg-gray-800"
          />
          <label className="text-xs opacity-70">Slippage (bps)</label>
          <input
            type="number"
            value={slipBps}
            onChange={(e) => setSlipBps(Number(e.target.value))}
            className="w-20 border rounded-lg px-2 py-1 text-sm dark:bg-gray-800"
          />
        </div>
      </header>

      {loading && <div className="text-sm opacity-70">Loading arbitrage data…</div>}
      {err && !loading && <div className="text-sm text-red-500">Error: {err}</div>}

      {!loading && !err && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Matrix heatmap */}
          <section className="lg:col-span-2">
            <h3 className="text-lg font-medium mb-2">Venue Matrix (buy row → sell column)</h3>
            <Matrix
              venues={venueList}
              matrix={matrix}
            />
          </section>

          {/* Top opportunities */}
          <section className="lg:col-span-1">
            <h3 className="text-lg font-medium mb-2">Top Routes (net after fees/slip)</h3>
            <div className="rounded-xl border dark:border-gray-700 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <Th>From</Th>
                    <Th>To</Th>
                    <Th right>Raw</Th>
                    <Th right>Net</Th>
                  </tr>
                </thead>
                <tbody>
                  {opps.map((o, i) => (
                    <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
                      <Td>{o.from}</Td>
                      <Td>{o.to}</Td>
                      <Td right>{fmtPct(o.rawPct)}</Td>
                      <Td right className={o.netPct > 0 ? "text-green-600" : "text-red-500"}>{fmtPct(o.netPct)}</Td>
                    </tr>
                  ))}
                  {opps.length === 0 && <tr><Td colSpan={4} className="opacity-60">No opportunities</Td></tr>}
                </tbody>
              </table>
            </div>

            {/* Best badge */}
            <div className="mt-3 text-sm">
              {bestOpp ? (
                <div className={`inline-block px-3 py-1 rounded-lg ${bestOpp.netPct > 0 ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-700"}`}>
                  Best: {bestOpp.from} → {bestOpp.to} • {fmtPct(bestOpp.netPct)} net
                </div>
              ) : (
                <span className="opacity-70">—</span>
              )}
            </div>
          </section>

          {/* Spread history */}
          <section className="lg:col-span-3">
            <h3 className="text-lg font-medium mb-2">Best Route Spread (history)</h3>
            <div className="rounded-xl border dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800">
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={spreadHist}>
                  <XAxis dataKey="t" hide />
                  <YAxis tickFormatter={(v) => `${(v * 100).toFixed(2)}%`} />
                  <Tooltip labelFormatter={(t) => new Date(Number(t)).toLocaleTimeString()} formatter={(v: any) => [`${(Number(v) * 100).toFixed(2)}%`, "spread"]} />
                  <Line type="monotone" dataKey="spreadPct" stroke="#0A84FF" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

/** ---------- Matrix (CSS grid heatmap) ---------- */
function Matrix({ venues, matrix }: { venues: string[]; matrix: number[][] }) {
  // color scale: negative → red, 0 → gray, positive → green
  const colorFor = (x: number) => {
    if (!Number.isFinite(x)) return "transparent";
    const v = Math.max(-0.02, Math.min(0.02, x)); // clamp to ±2% for color scaling
    const n = (v + 0.02) / 0.04; // 0..1
    const r = Math.round(200 + (55 - 200) * n);  // 200 → 55 (more green reduces red)
    const g = Math.round(200 + (235 - 200) * n); // 200 → 235
    const b = Math.round(200 + (55 - 200) * n);  // 200 → 55
    return `rgb(${r},${g},${b})`;
  };
  const fmt = (x: number) => (Number.isFinite(x) ? `${(x * 100).toFixed(2)}%` : "—");

  return (
    <div className="overflow-auto">
      <div
        className="grid"
        style={{ gridTemplateColumns: `160px repeat(${venues.length}, 1fr)` }}
      >
        {/* top header */}
        <div className="sticky left-0 bg-white dark:bg-gray-900 z-10 p-2 font-medium border-b dark:border-gray-700">Buy ⟶ / Sell ⤵</div>
        {venues.map((v) => (
          <div key={`col-${v}`} className="p-2 text-xs font-medium border-b dark:border-gray-700 text-center">{v}</div>
        ))}

        {/* rows */}
        {venues.map((rowV, i) => (
          <React.Fragment key={`row-${rowV}`}>
            <div className="sticky left-0 bg-white dark:bg-gray-900 z-10 p-2 text-sm border-b dark:border-gray-800">{rowV}</div>
            {venues.map((colV, j) => {
              const val = matrix?.[i]?.[j];
              const bg = colorFor(val);
              const text =
                i === j ? "—" :
                Number.isFinite(val) ? fmt(val) : "—";
              return (
                <div
                  key={`${rowV}-${colV}`}
                  className={`h-9 border-b border-r dark:border-gray-800 flex items-center justify-center text-xs ${i===j ? "opacity-60" : ""}`}
                  style={{ backgroundColor: i === j ? "transparent" : bg }}
                  title={i!==j ? `${rowV} → ${colV}: ${text}` : ""}
                >
                  <span className="px-1 font-medium">{text}</span>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <div className="mt-2 text-xs opacity-70">
        Cell = % edge if you **buy row** venue and **sell column** venue at current quotes.
      </div>
    </div>
  );
}

/** ---------- Small table helpers ---------- */
function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={`p-2 ${right ? "text-right" : "text-left"}`}>{children}</th>;
}
function Td({ children, right, colSpan, className }: { children?: React.ReactNode; right?: boolean; colSpan?: number; className?: string }) {
  return <td colSpan={colSpan} className={`p-2 ${right ? "text-right" : "text-left"} ${className || ""}`}>{children}</td>;
}

/** ---------- Utils ---------- */
function unique<T>(arr: T[]) { return Array.from(new Set(arr)); }

function upsertTick(prev: PriceTick[], t: PriceTick) {
  const i = prev.findIndex(x => x.venue === t.venue);
  if (i >= 0) {
    const next = prev.slice();
    next[i] = t.ts >= next[i].ts ? t : next[i];
    return next;
  }
  return [t, ...prev];
}

function fmtPct(x: number) {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(2)}%`;
}