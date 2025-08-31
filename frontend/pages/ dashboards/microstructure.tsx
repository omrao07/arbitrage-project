// frontend/components/Microstructure.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  LineChart,
  Line,
} from "recharts";

type Side = "bid" | "ask";

export interface BookLevel {
  price: number;
  size: number;
  side: Side;
}

export interface TradePrint {
  ts: number;       // epoch ms
  price: number;
  size: number;
  side: Side;       // taker side (buy=hit ask, sell=hit bid)
}

interface Props {
  wsUrl?: string;             // optional WebSocket for live data
  symbol?: string;
  // Fallback/static data (used if no ws)
  initialBook?: BookLevel[];
  initialTrades?: TradePrint[];
  depthLevels?: number;       // how many price levels per side to display
  spreadHistoryPoints?: number;
}

export default function Microstructure({
  wsUrl,
  symbol = "AAPL",
  initialBook = [],
  initialTrades = [],
  depthLevels = 8,
  spreadHistoryPoints = 60,
}: Props) {
  const [book, setBook] = useState<BookLevel[]>(initialBook);
  const [tape, setTape] = useState<TradePrint[]>(initialTrades);
  const [spreadHistory, setSpreadHistory] = useState<{ t: number; spread: number }[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // --- WebSocket wiring (optional) ---
  useEffect(() => {
    if (!wsUrl) return;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "subscribe", channel: "microstructure", symbol }));
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "book") {
          // expect msg.levels: [{price, size, side}]
          setBook(msg.levels);
        } else if (msg.type === "trade") {
          // expect msg.print: {ts, price, size, side}
          setTape((prev) => {
            const next = [msg.print, ...prev].slice(0, 100);
            return next;
          });
        }
      } catch (e) {
        console.error("WS parse error", e);
      }
    };
    ws.onerror = (e) => console.error("WS error", e);
    ws.onclose = () => (wsRef.current = null);
    return () => ws.close();
  }, [wsUrl, symbol]);

  // --- Derived metrics ---
  const bestBid = useMemo(
    () => book.filter(b => b.side === "bid").reduce((m, x) => Math.max(m, x.price), -Infinity),
    [book]
  );
  const bestAsk = useMemo(
    () => book.filter(b => b.side === "ask").reduce((m, x) => Math.min(m, x.price), Infinity),
    [book]
  );
  const spread = useMemo(() => (isFinite(bestBid) && isFinite(bestAsk) ? bestAsk - bestBid : NaN), [bestBid, bestAsk]);

  const bidDepth = useMemo(() => sumDepth(book, "bid", depthLevels), [book, depthLevels]);
  const askDepth = useMemo(() => sumDepth(book, "ask", depthLevels), [book, depthLevels]);

  const imbalance = useMemo(() => {
    const total = bidDepth + askDepth;
    return total > 0 ? (bidDepth - askDepth) / total : 0;
  }, [bidDepth, askDepth]);

  // Spread history ring buffer
  useEffect(() => {
    if (!Number.isFinite(spread)) return;
    setSpreadHistory((prev) => {
      const next = [...prev, { t: Date.now(), spread }];
      if (next.length > spreadHistoryPoints) next.shift();
      return next;
    });
  }, [spread, spreadHistoryPoints]);

  // --- Charts data prep ---
  const depthChartData = useMemo(() => {
    // take top N per side by proximity to mid
    const [bids, asks] = splitTopLevels(book, depthLevels);
    // Merge into aligned rows by price (string key to keep order stable)
    const priceSet = new Set<string>([...bids.map(b => fmtPx(b.price)), ...asks.map(a => fmtPx(a.price))]);
    const rows = Array.from(priceSet)
      .map(px => {
        const b = bids.find(l => fmtPx(l.price) === px)?.size ?? 0;
        const a = asks.find(l => fmtPx(l.price) === px)?.size ?? 0;
        return { price: px, bid: b, ask: a };
      })
      // sort by numeric price ascending
      .sort((r1, r2) => parseFloat(r1.price) - parseFloat(r2.price));
    return rows;
  }, [book, depthLevels]);

  // --- Render ---
  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-semibold">Microstructure — {symbol}</h2>
        <div className="text-sm opacity-80">
          <span className="mr-3">Best Bid: <b>{isFinite(bestBid) ? bestBid.toFixed(2) : "-"}</b></span>
          <span className="mr-3">Best Ask: <b>{isFinite(bestAsk) ? bestAsk.toFixed(2) : "-"}</b></span>
          <span>Spread: <b>{Number.isFinite(spread) ? spread.toFixed(2) : "-"}</b></span>
        </div>
      </header>

      {/* Spread mini chart */}
      <div className="mb-4">
        <div className="text-sm mb-1">Spread (last {spreadHistory.length}s)</div>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={spreadHistory}>
            <XAxis dataKey="t" hide />
            <YAxis />
            <Tooltip labelFormatter={(t) => new Date(Number(t)).toLocaleTimeString()} />
            <Line type="monotone" dataKey="spread" stroke="#0A84FF" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Depth by price level */}
      <div className="mb-4">
        <div className="text-sm mb-1">Order Book Depth (top {depthLevels} levels/side)</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={depthChartData} stackOffset="sign">
            <XAxis dataKey="price" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="bid" stackId="d" fill="#34C759" />
            <Bar dataKey="ask" stackId="d" fill="#FF3B30" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-2 text-sm">
          Imbalance: <b className={imbalance > 0 ? "text-green-500" : imbalance < 0 ? "text-red-500" : ""}>
            {(imbalance * 100).toFixed(1)}%
          </b>
        </div>
      </div>

      {/* Trade tape */}
      <div>
        <div className="text-sm mb-1">Recent Trades</div>
        <div className="h-40 overflow-auto rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="text-left p-2">Time</th>
                <th className="text-right p-2">Price</th>
                <th className="text-right p-2">Size</th>
                <th className="text-right p-2">Side</th>
              </tr>
            </thead>
            <tbody>
              {tape.map((t, i) => (
                <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
                  <td className="p-2">{new Date(t.ts).toLocaleTimeString()}</td>
                  <td className="p-2 text-right">{t.price.toFixed(2)}</td>
                  <td className="p-2 text-right">{t.size}</td>
                  <td className={`p-2 text-right ${t.side === "ask" ? "text-green-500" : "text-red-500"}`}>
                    {t.side.toUpperCase()}
                  </td>
                </tr>
              ))}
              {tape.length === 0 && (
                <tr><td className="p-2 opacity-60" colSpan={4}>No prints yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------- helpers ----------
function fmtPx(x: number) { return x.toFixed(2); }

function splitTopLevels(book: BookLevel[], levels: number): [BookLevel[], BookLevel[]] {
  const bids = book.filter(l => l.side === "bid").sort((a,b) => b.price - a.price).slice(0, levels);
  const asks = book.filter(l => l.side === "ask").sort((a,b) => a.price - b.price).slice(0, levels);
  return [bids, asks];
}

function sumDepth(book: BookLevel[], side: Side, levels: number): number {
  const lvls = book
    .filter(l => l.side === side)
    .sort((a,b) => side === "bid" ? b.price - a.price : a.price - b.price)
    .slice(0, levels);
  return lvls.reduce((s, l) => s + l.size, 0);
}