// frontend/components/livetape.js
// A production‑ready, live market tape widget.
// - Connects via WebSocket (preferred) or Server‑Sent Events (SSE)
// - Shows rolling trade prints + best bid/ask per symbol
// - Color‑codes buys/sells, highlights stale feeds, and shows latency
// - Filter by symbol, pause/resume stream, and adjust buffer size
// - Drop‑in for Bolt/Next.js/React apps. No external state libs.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---------- Expected message shapes (examples) ----------
// Trade:
// { "type":"trade", "sym":"AAPL", "px":230.11, "sz":200, "side":"buy", "ts": 1765400000123 }
// Quote (best book):
// { "type":"quote", "sym":"AAPL", "bid":230.10, "bsz":1800, "ask":230.12, "asz":1500, "ts": 1765400000100 }
// Heartbeat (optional):
// { "type":"hb", "ts": 1765400000200 }

const DEFAULT_WS = "wss://localhost:8081/ws/tape"; // change to your gateway
const DEFAULT_SSE = "/api/tape"; // SSE fallback (EventSource)

// Simple money and time formatters
const fmtPx = (n) =>
  Number.isFinite(n) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 8 }).format(n) : "—";
const fmtSz = (n) =>
  Number.isFinite(n) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(n) : "—";
const fmtTime = (ms) => {
  try {
    const d = new Date(ms);
    return d.toLocaleTimeString(undefined, { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
  } catch {
    return "—";
  }
};

// Derive microprice for a quote row
function microprice(bid, bsz, ask, asz) {
  if (!Number.isFinite(bid) || !Number.isFinite(ask) || !Number.isFinite(bsz) || !Number.isFinite(asz)) return null;
  const den = Math.max(1e-9, bsz + asz);
  return (ask * bsz + bid * asz) / den;
}

/**
 * LiveTape props:
 * @param {object} props
 * @param {"ws"|"sse"|"auto"} [props.mode="auto"] connection mode
 * @param {string} [props.wsUrl] websocket URL
 * @param {string} [props.sseUrl] SSE URL (GET that streams events)
 * @param {string[]} [props.subscribe] symbols to request (backend should handle)
 * @param {number} [props.maxRows=200] ring buffer length for trades table
 * @param {boolean} [props.compact=false] tighter row height
 */
export default function LiveTape({
  mode = "auto",
  wsUrl = DEFAULT_WS,
  sseUrl = DEFAULT_SSE,
  subscribe = [],
  maxRows = 200,
  compact = false,
}) {
  const [rows, setRows] = useState([]); // rolling trades
  const [quotes, setQuotes] = useState({}); // sym -> {bid,bsz,ask,asz,ts}
  const [status, setStatus] = useState("disconnected"); // disconnected | connecting | connected
  const [latency, setLatency] = useState(null);
  const [filter, setFilter] = useState("");
  const [paused, setPaused] = useState(false);
  const [using, setUsing] = useState("auto"); // ws | sse
  const [bufSize, setBufSize] = useState(maxRows);

  const wsRef = useRef(null);
  const esRef = useRef(null);
  const lastTsRef = useRef(0);
  const rafRef = useRef(0);
  const queueRef = useRef([]); // aggregate UI updates per frame

  // Derived filtered rows
  const filtered = useMemo(() => {
    const f = filter.trim().toUpperCase();
    const list = f ? rows.filter((r) => r.sym?.toUpperCase()?.includes(f)) : rows;
    return list.slice(-bufSize).reverse();
  }, [rows, filter, bufSize]);

  // Clean up connection
  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onopen = wsRef.current.onmessage = wsRef.current.onerror = wsRef.current.onclose = null;
      try { wsRef.current.close(1000); } catch {}
      wsRef.current = null;
    }
    if (esRef.current) {
      try { esRef.current.close(); } catch {}
      esRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    queueRef.current = [];
    setStatus("disconnected");
  }, []);

  // Flush queued UI updates (once per animation frame)
  const flush = useCallback(() => {
    rafRef.current = 0;
    if (paused) {
      queueRef.current = [];
      return;
    }
    if (queueRef.current.length) {
      // Partition into trades and quotes
      let trades = [];
      let qmap = {};
      for (const msg of queueRef.current) {
        if (msg.type === "trade") trades.push(msg);
        else if (msg.type === "quote") qmap[msg.sym] = msg;
        else if (msg.type === "hb") setLatency(Math.max(0, Date.now() - Number(msg.ts)));
      }
      queueRef.current = [];

      if (trades.length) {
        setRows((prev) => {
          const merged = prev.concat(trades);
          // Trim ring buffer (keep last N * 1.2 to reduce GC)
          const cap = Math.max(bufSize, maxRows) * 1.2;
          return merged.length > cap ? merged.slice(merged.length - Math.ceil(cap)) : merged;
        });
      }
      if (Object.keys(qmap).length) {
        setQuotes((prev) => ({ ...prev, ...qmap }));
      }
    }
  }, [paused, bufSize, maxRows]);

  const enqueue = useCallback(
    (msg) => {
      // Basic sanity + latency
      if (msg && typeof msg === "object" && Number.isFinite(Number(msg.ts))) {
        lastTsRef.current = Number(msg.ts);
      }
      queueRef.current.push(msg);
      if (!rafRef.current) rafRef.current = requestAnimationFrame(flush);
    },
    [flush]
  );

  // Connection bootstrap
  useEffect(() => {
    cleanup();
    setStatus("connecting");

    // Decide mode
    const wantWs = mode === "ws" || (mode === "auto" && typeof WebSocket !== "undefined");
    let connected = false;

    if (wantWs) {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        setUsing("ws");

        ws.onopen = () => {
          connected = true;
          setStatus("connected");
          // optional: send subscription list
          if (Array.isArray(subscribe) && subscribe.length) {
            ws.send(JSON.stringify({ op: "subscribe", symbols: subscribe }));
          }
        };

        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            Array.isArray(msg) ? msg.forEach(enqueue) : enqueue(msg);
          } catch {
            // ignore malformed
          }
        };

        ws.onerror = () => {
          if (!connected) {
            // fallback to SSE
            trySSE();
          }
        };

        ws.onclose = () => {
          setStatus("disconnected");
        };
      } catch {
        trySSE();
      }
    } else {
      trySSE();
    }

    function trySSE() {
      setUsing("sse");
      try {
        const es = new EventSource(sseUrl + makeQuery(subscribe));
        esRef.current = es;

        es.onopen = () => setStatus("connected");
        es.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            Array.isArray(msg) ? msg.forEach(enqueue) : enqueue(msg);
          } catch {
            // ignore malformed
          }
        };
        es.onerror = () => {
          setStatus("disconnected");
        };
      } catch {
        setStatus("disconnected");
      }
    }

    function makeQuery(list) {
      if (!Array.isArray(list) || !list.length) return "";
      const p = new URLSearchParams();
      p.set("symbols", list.join(","));
      return `?${p.toString()}`;
    }

    return () => cleanup();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, wsUrl, sseUrl, JSON.stringify(subscribe)]);

  // Staleness indicator
  const staleMs = useMemo(() => (lastTsRef.current ? Date.now() - lastTsRef.current : null), [rows, quotes]);

  // UI bits
  const statusBadge = useMemo(() => {
    const base =
      "inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium select-none border";
    if (status === "connected")
      return <span className={`${base} border-emerald-300/40 text-emerald-600 dark:text-emerald-300 bg-emerald-50/60 dark:bg-emerald-950/20`}>● live ({using})</span>;
    if (status === "connecting")
      return <span className={`${base} border-amber-300/40 text-amber-600 dark:text-amber-300 bg-amber-50/60 dark:bg-amber-950/20`}>● connecting…</span>;
    return <span className={`${base} border-rose-300/40 text-rose-600 dark:text-rose-300 bg-rose-50/60 dark:bg-rose-950/20`}>● offline</span>;
  }, [status, using]);

  return (
    <div className="w-full rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4 shadow-sm">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
        <div className="flex items-center gap-3">
          <h3 className="text-base md:text-lg font-semibold">Live Tape</h3>
          {statusBadge}
          {Number.isFinite(latency) && (
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              latency: {Math.max(0, latency)} ms
            </span>
          )}
          {Number.isFinite(staleMs) && staleMs > 2500 && (
            <span className="text-xs text-rose-500">stale {Math.round(staleMs)} ms</span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            aria-label="Filter symbols"
            className="px-3 py-1.5 rounded-xl border border-black/10 dark:border-white/10 bg-white/60 dark:bg-zinc-900/60 text-sm outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Filter e.g. AAPL,BTC…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label className="text-xs opacity-70">buffer</label>
          <input
            type="number"
            min={50}
            max={1000}
            value={bufSize}
            onChange={(e) => setBufSize(Math.max(50, Math.min(1000, Number(e.target.value) || 200)))}
            className="w-20 px-2 py-1.5 rounded-xl border border-black/10 dark:border-white/10 bg-white/60 dark:bg-zinc-900/60 text-sm"
          />
          <button
            onClick={() => setPaused((p) => !p)}
            className={`px-3 py-1.5 rounded-xl text-sm border ${paused ? "border-emerald-300/40 text-emerald-600 dark:text-emerald-300 bg-emerald-50/60 dark:bg-emerald-950/20" : "border-zinc-300/40 text-zinc-700 dark:text-zinc-200 bg-zinc-50/60 dark:bg-zinc-900/40"}`}
            title="Pause/resume live updates (space)"
          >
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>

      {/* Quotes board */}
      <div className="overflow-x-auto rounded-xl border border-black/5 dark:border-white/10 mb-3">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-50/70 dark:bg-zinc-900/40 text-zinc-600 dark:text-zinc-300">
            <tr>
              <Th>Symbol</Th>
              <Th align="right">Bid</Th>
              <Th align="right">Bid Sz</Th>
              <Th align="right">Ask</Th>
              <Th align="right">Ask Sz</Th>
              <Th align="right">Micro</Th>
              <Th align="right">Spread</Th>
              <Th align="right">Updated</Th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(quotes)
              .filter(([sym]) => (filter ? sym.toUpperCase().includes(filter.toUpperCase()) : true))
              .sort((a, b) => a[0].localeCompare(b[0]))
              .map(([sym, q]) => {
                const spr = Number.isFinite(q.ask) && Number.isFinite(q.bid) ? q.ask - q.bid : null;
                const mi = microprice(q.bid, q.bsz, q.ask, q.asz);
                const ageMs = Date.now() - (q.ts || 0);
                const stale = ageMs > 2500;
                return (
                  <tr key={sym} className={stale ? "opacity-60" : ""}>
                    <Td>{sym}</Td>
                    <Td align="right">{fmtPx(q.bid)}</Td>
                    <Td align="right">{fmtSz(q.bsz)}</Td>
                    <Td align="right">{fmtPx(q.ask)}</Td>
                    <Td align="right">{fmtSz(q.asz)}</Td>
                    <Td align="right">{fmtPx(mi)}</Td>
                    <Td align="right">{spr != null ? fmtPx(spr) : "—"}</Td>
                    <Td align="right">{q.ts ? fmtTime(q.ts) : "—"}</Td>
                  </tr>
                );
              })}
            {!Object.keys(quotes).length && (
              <tr>
                <Td colSpan={8} className="text-center text-zinc-500 py-4">
                  No quotes yet…
                </Td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Trades tape */}
      <div className="overflow-x-auto rounded-xl border border-black/5 dark:border-white/10">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-50/70 dark:bg-zinc-900/40 text-zinc-600 dark:text-zinc-300">
            <tr>
              <Th>Time</Th>
              <Th>Symbol</Th>
              <Th align="right">Price</Th>
              <Th align="right">Size</Th>
              <Th>Side</Th>
            </tr>
          </thead>
        </table>
        <div className={`max-h-[40vh] overflow-auto ${compact ? "leading-5" : "leading-6"}`}>
          <table className="min-w-full text-sm">
            <tbody>
              {filtered.map((r, idx) => {
                const sideClass =
                  r.side === "buy"
                    ? "text-emerald-600 dark:text-emerald-400"
                    : r.side === "sell"
                    ? "text-rose-600 dark:text-rose-400"
                    : "text-zinc-500";
                return (
                  <tr key={`${r.ts}-${idx}`} className="odd:bg-zinc-50/30 dark:odd:bg-zinc-900/20">
                    <Td>{fmtTime(r.ts)}</Td>
                    <Td>{r.sym ?? "—"}</Td>
                    <Td align="right" className={sideClass}>
                      {fmtPx(r.px)}
                    </Td>
                    <Td align="right">{fmtSz(r.sz)}</Td>
                    <Td className={sideClass}>{(r.side || "").toUpperCase() || "—"}</Td>
                  </tr>
                );
              })}
              {!filtered.length && (
                <tr>
                  <Td colSpan={5} className="text-center text-zinc-500 py-4">
                    Waiting for prints…
                  </Td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer help */}
      <div className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
        Tip: Press <kbd className="px-1.5 py-0.5 rounded bg-zinc-200/70 dark:bg-zinc-800/80">Space</kbd> to pause/resume.
        &nbsp; Mode: <span className="font-mono">{using.toUpperCase()}</span>
      </div>
    </div>
  );
}

// ---------- Small presentational helpers ----------
function Th({ children, align = "left" }) {
  return (
    <th
      className={`px-3 py-2 font-medium ${align === "right" ? "text-right" : "text-left"} whitespace-nowrap`}
    >
      {children}
    </th>
  );
}
function Td({ children, align = "left", colSpan, className = "" }) {
  return (
    <td
      colSpan={colSpan}
      className={`px-3 py-2 ${align === "right" ? "text-right" : "text-left"} whitespace-nowrap ${className}`}
    >
      {children}
    </td>
  );
}

// ---------- Optional: keyboard shortcuts ----------
if (typeof window !== "undefined") {
  window.__livetape_keybinds__ ||= (() => {
    const handlers = new Set();
    const onKey = (e) => {
      if (e.code === "Space") {
        handlers.forEach((h) => h());
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return {
      register(h) {
        handlers.add(h);
        return () => handlers.delete(h);
      },
    };
  })();
}

// Hook up Pause/Resume global space bar for this instance
export function useSpaceToToggle(paused, setPaused) {
  useEffect(() => {
    const reg = window.__livetape_keybinds__?.register?.(() => setPaused((p) => !p));
    return () => reg && reg();
  }, [setPaused, paused]);
}