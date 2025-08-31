// frontend/components/OrderbookPanel.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

/* ------------------------------- Types ------------------------------- */

export type Level = { price: number; size: number };
export type Side = "bids" | "asks";

export interface L2Snapshot {
  symbol: string;
  ts: number;              // epoch ms
  bids: Level[];           // sorted desc by price preferred
  asks: Level[];           // sorted asc by price preferred
  venue?: string;
}

interface Props {
  symbol?: string;
  /** If provided, will connect and expect JSON snapshots or deltas (L2Snapshot or {bids:[p,s], asks:[p,s]}). */
  wsUrl?: string;
  /** If no wsUrl, will poll this endpoint (GET returns L2Snapshot). */
  endpoint?: string;         // default: /api/orderbook?symbol=XXX
  pollMs?: number;           // default: 1500
  rowLimit?: number;         // default: 15 rows per side
  groupTick?: number;        // e.g. 0.01 to group by 1 cent
  title?: string;
}

/* ------------------------------ Component ---------------------------- */

export default function OrderbookPanel({
  symbol = "BTC-USD",
  wsUrl,
  endpoint = "/api/orderbook",
  pollMs = 1500,
  rowLimit = 15,
  groupTick = 0,
  title = "Order Book",
}: Props) {
  const [snap, setSnap] = useState<L2Snapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastUpdate = snap?.ts ? new Date(snap.ts).toLocaleTimeString() : "—";

  /* ----------------------------- Data feed ---------------------------- */

  useEffect(() => {
    let mounted = true;

    if (wsUrl) {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        ws.onopen = () => setErr(null);
        ws.onerror = () => setErr("WebSocket error");
        ws.onclose = () => { if (mounted) setErr("WebSocket closed"); };
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            const next = normalizeInbound(symbol, snap, msg);
            if (mounted && next) setSnap(limitAndGroup(next, rowLimit, groupTick));
          } catch (e: any) {
            // ignore parse errors
          }
        };
      } catch (e: any) {
        setErr(e?.message || "Failed to open WebSocket");
      }
      return () => { mounted = false; wsRef.current?.close(); wsRef.current = null; };
    }

    // Fallback: HTTP polling
    async function poll() {
      try {
        const q = new URLSearchParams({ symbol }).toString();
        const res = await fetch(`${endpoint}?${q}`);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as L2Snapshot;
        if (mounted) { setSnap(limitAndGroup(json, rowLimit, groupTick)); setErr(null); }
      } catch (e: any) {
        if (mounted) setErr(e?.message || "Failed to load order book");
      }
    }
    poll();
    const t = setInterval(poll, pollMs);
    return () => { mounted = false; clearInterval(t); };
  }, [wsUrl, endpoint, pollMs, symbol, rowLimit, groupTick]);

  /* ------------------------------ Derived ----------------------------- */

  const bestBid = snap?.bids?.[0]?.price ?? NaN;
  const bestAsk = snap?.asks?.[0]?.price ?? NaN;
  const spread = Number.isFinite(bestBid) && Number.isFinite(bestAsk) ? bestAsk - bestBid : NaN;
  const mid = Number.isFinite(bestBid) && Number.isFinite(bestAsk) ? (bestBid + bestAsk) / 2 : NaN;

  const withCum = useMemo(() => {
    if (!snap) return null;
    // ensure bids desc, asks asc
    const bids = [...snap.bids].sort((a, b) => b.price - a.price).slice(0, rowLimit);
    const asks = [...snap.asks].sort((a, b) => a.price - b.price).slice(0, rowLimit);
    // cumulative sizes
    let acc = 0;
    const bidsCum = bids.map((l) => ({ ...l, cum: (acc += l.size) }));
    acc = 0;
    const asksCum = asks.map((l) => ({ ...l, cum: (acc += l.size) }));
    return { ...snap, bids: bidsCum, asks: asksCum };
  }, [snap, rowLimit]);

  // depth curve for chart (relative to mid)
  const depthSeries = useMemo(() => {
    if (!withCum || !Number.isFinite(mid)) return [];
    const left = withCum.bids.map((l) => ({ px: l.price, depth: l.cum }));
    const right = withCum.asks.map((l) => ({ px: l.price, depth: l.cum }));
    // merge and sort by price
    return [...left, ...right].sort((a, b) => a.px - b.px);
  }, [withCum, mid]);

  /* ------------------------------- UI -------------------------------- */

  if (!withCum) {
    return (
      <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
        <div className="text-sm">Loading order book…</div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{title} — {withCum.symbol}{withCum.venue ? ` @ ${withCum.venue}` : ""}</h2>
          <div className="text-xs opacity-70">
            Best Bid: <b className="text-green-600">{fmt(bestBid)}</b> • Best Ask: <b className="text-red-600">{fmt(bestAsk)}</b> •
            {" "}Spread: <b>{fmt(spread)}</b> • Mid: <b>{fmt(mid)}</b> • {lastUpdate}
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => exportCSV(withCum)}>Export CSV</button>
          {err && <span className="text-red-600">{err}</span>}
        </div>
      </header>

      {/* Book + Depth */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* L2 tables */}
        <div className="md:col-span-2 grid grid-cols-2 gap-3">
          {/* Bids */}
          <div className="rounded-xl border dark:border-gray-800 overflow-hidden">
            <div className="px-3 py-2 text-sm font-medium bg-green-50 dark:bg-green-900/20">Bids</div>
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white dark:bg-gray-900">
                  <tr className="text-xs opacity-70">
                    <th className="p-2 text-left">Price</th>
                    <th className="p-2 text-right">Size</th>
                    <th className="p-2 text-right">Cum</th>
                  </tr>
                </thead>
                <tbody>
                  {withCum.bids.map((l, i) => (
                    <tr key={`b-${i}`} className="border-t dark:border-gray-800">
                      <td className="p-2 text-left text-green-600 font-medium">{fmt(l.price)}</td>
                      <td className="p-2 text-right">{fmt(l.size)}</td>
                      <td className="p-2 text-right">
                        <div className="relative">
                          <div className="absolute right-0 top-1/2 -translate-y-1/2 h-3 bg-green-200 dark:bg-green-900/40 rounded"
                               style={{ width: `${barWidth(l.cum, withCum.bids[withCum.bids.length - 1].cum)}%` }} />
                          <span className="relative">{fmt(l.cum as number)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {withCum.bids.length === 0 && (
                    <tr><td className="p-3 text-center opacity-60" colSpan={3}>No bids</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Asks */}
          <div className="rounded-xl border dark:border-gray-800 overflow-hidden">
            <div className="px-3 py-2 text-sm font-medium bg-red-50 dark:bg-red-900/20">Asks</div>
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white dark:bg-gray-900">
                  <tr className="text-xs opacity-70">
                    <th className="p-2 text-left">Price</th>
                    <th className="p-2 text-right">Size</th>
                    <th className="p-2 text-right">Cum</th>
                  </tr>
                </thead>
                <tbody>
                  {withCum.asks.map((l, i) => (
                    <tr key={`a-${i}`} className="border-t dark:border-gray-800">
                      <td className="p-2 text-left text-red-600 font-medium">{fmt(l.price)}</td>
                      <td className="p-2 text-right">{fmt(l.size)}</td>
                      <td className="p-2 text-right">
                        <div className="relative">
                          <div className="absolute left-0 top-1/2 -translate-y-1/2 h-3 bg-red-200 dark:bg-red-900/40 rounded"
                               style={{ width: `${barWidth(l.cum, withCum.asks[withCum.asks.length - 1].cum)}%` }} />
                          <span className="relative">{fmt(l.cum as number)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {withCum.asks.length === 0 && (
                    <tr><td className="p-3 text-center opacity-60" colSpan={3}>No asks</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Mini depth chart */}
        <div className="rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">Depth Around Mid</div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={depthSeries}>
                <XAxis dataKey="px" tickFormatter={(v) => shortNum(v as number)} />
                <YAxis />
                <Tooltip
                  formatter={(v, n) => [fmt(v as number), n as string]}
                  labelFormatter={(v) => `Price: ${fmt(v as number)}`}
                />
                <Legend />
                <Area type="monotone" dataKey="depth" stroke="#475569" fill="#cbd5e1" fillOpacity={0.4} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 text-xs opacity-70">
            Spread: <b>{fmt(spread)}</b> • Mid: <b>{fmt(mid)}</b>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------- Helpers ------------------------------ */

function fmt(x?: number) {
  if (x == null || !Number.isFinite(x)) return "—";
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 8 }); } catch { return String(x); }
}
function shortNum(x: number) {
  if (Math.abs(x) >= 1e9) return (x / 1e9).toFixed(2) + "B";
  if (Math.abs(x) >= 1e6) return (x / 1e6).toFixed(2) + "M";
  if (Math.abs(x) >= 1e3) return (x / 1e3).toFixed(2) + "K";
  return fmt(x);
}
function barWidth(cum: number, maxCum: number) {
  if (!maxCum) return 0;
  return Math.max(4, Math.min(100, (cum / maxCum) * 100));
}

/** Limit rows per side and group by tick if requested */
function limitAndGroup(s: L2Snapshot, rowLimit: number, tick: number): L2Snapshot {
  const g = (levels: Level[], isBid: boolean) => {
    const xs = groupByTick(levels, tick, isBid);
    return isBid
      ? xs.sort((a, b) => b.price - a.price).slice(0, rowLimit)
      : xs.sort((a, b) => a.price - b.price).slice(0, rowLimit);
  };
  return { ...s, bids: g(s.bids || [], true), asks: g(s.asks || [], false) };
}

/** Groups price levels into buckets of size `tick` (e.g., 0.01) and sums sizes. */
function groupByTick(levels: Level[], tick: number, isBid: boolean): Level[] {
  if (!tick || tick <= 0) return levels.slice();
  const m = new Map<number, number>();
  for (const { price, size } of levels) {
    // snap to bucket: bids floor, asks ceil → visually conservative around top of book
    const b = isBid ? Math.floor(price / tick) * tick : Math.ceil(price / tick) * tick;
    m.set(b, (m.get(b) ?? 0) + size);
  }
  return Array.from(m.entries()).map(([price, size]) => ({ price: roundTick(price, tick), size }));
}
function roundTick(px: number, tick: number) {
  const d = Math.max(0, (tick.toString().split(".")[1] || "").length);
  return Number(px.toFixed(d));
}

/** Accepts full snapshot or common delta formats and returns a normalized snapshot. */
function normalizeInbound(symbol: string, prev: L2Snapshot | null, msg: any): L2Snapshot | null {
  // Full snapshot
  if (msg && Array.isArray(msg.bids) && Array.isArray(msg.asks)) {
    return {
      symbol: msg.symbol ?? symbol,
      ts: msg.ts ?? Date.now(),
      venue: msg.venue,
      bids: coerceLevels(msg.bids, true),
      asks: coerceLevels(msg.asks, false),
    };
  }
  // Deltas { bids:[[p,s], ...], asks:[[p,s], ...] }
  if (msg && (msg.bids || msg.asks) && prev) {
    const bids = applyDeltas(prev.bids, msg.bids ?? [], true);
    const asks = applyDeltas(prev.asks, msg.asks ?? [], false);
    return { ...prev, bids, asks, ts: msg.ts ?? Date.now() };
  }
  return prev;
}
function coerceLevels(xs: any[], isBid: boolean): Level[] {
  const to = (v: any): Level | null => {
    if (Array.isArray(v)) return { price: Number(v[0]), size: Number(v[1]) };
    if (typeof v === "object" && v) return { price: Number(v.price), size: Number(v.size) };
    return null;
  };
  const out = xs.map(to).filter(Boolean) as Level[];
  return isBid ? out.sort((a, b) => b.price - a.price) : out.sort((a, b) => a.price - b.price);
}
function applyDeltas(book: Level[], deltas: any[], isBid: boolean): Level[] {
  // mutate a copy using a Map for performance
  const m = new Map<number, number>();
  for (const { price, size } of book) m.set(price, size);
  for (const d of deltas) {
    const p = Array.isArray(d) ? Number(d[0]) : Number(d.price);
    const s = Array.isArray(d) ? Number(d[1]) : Number(d.size);
    if (!Number.isFinite(p)) continue;
    if (!Number.isFinite(s) || s <= 0) m.delete(p); else m.set(p, s);
  }
  const out = Array.from(m.entries()).map(([price, size]) => ({ price, size }));
  return isBid ? out.sort((a, b) => b.price - a.price) : out.sort((a, b) => a.price - b.price);
}

/** CSV export of current book (both sides). */
function exportCSV(s: L2Snapshot) {
  const headers = ["side", "price", "size", "cum"];
  const bids = s.bids.map((l) => ["bid", l.price, l.size, (l as any).cum ?? ""].join(","));
  const asks = s.asks.map((l) => ["ask", l.price, l.size, (l as any).cum ?? ""].join(","));
  const csv = [headers.join(","), ...bids, ...asks].join("\n");
  download(`orderbook_${s.symbol}_${s.ts}.csv`, csv, "text/csv;charset=utf-8;");
}
function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}