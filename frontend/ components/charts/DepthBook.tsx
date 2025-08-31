// frontend/components/DepthBook.tsx
import React, { useEffect, useState, useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

type Level = { price: number; size: number };
type OrderBookPayload = { bids: Level[]; asks: Level[] };

interface Props {
  endpoint?: string;   // REST endpoint: GET -> { bids: Level[], asks: Level[] }
  symbol?: string;     // trading pair / ticker
  depth?: number;      // how many levels to render
}

export default function DepthBook({
  endpoint = "/api/orderbook",
  symbol = "BTC/USDT",
  depth = 20,
}: Props) {
  const [bids, setBids] = useState<Level[]>([]);
  const [asks, setAsks] = useState<Level[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${endpoint}?symbol=${encodeURIComponent(symbol)}&depth=${depth}`);
        const json: OrderBookPayload = await res.json();
        setBids(json.bids || []);
        setAsks(json.asks || []);
      } catch (e: any) {
        setErr(e?.message || "Failed to fetch depth book");
      }
    })();
  }, [endpoint, symbol, depth]);

  // cumulative depth
  const chartData = useMemo(() => {
    const b: any[] = [];
    const a: any[] = [];

    let cum = 0;
    for (const lv of bids) {
      cum += lv.size;
      b.push({ price: lv.price, bid: cum });
    }
    cum = 0;
    for (const lv of asks) {
      cum += lv.size;
      a.push({ price: lv.price, ask: cum });
    }
    return { bids: b.sort((x, y) => y.price - x.price), asks: a.sort((x, y) => x.price - y.price) };
  }, [bids, asks]);

  const bestBid = bids.length ? bids[0].price : null;
  const bestAsk = asks.length ? asks[0].price : null;
  const mid = bestBid && bestAsk ? ((bestBid + bestAsk) / 2).toFixed(2) : "—";

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Depth Book — {symbol}</h2>
        <div className="text-sm opacity-70">Mid: {mid}</div>
      </header>

      {err && <div className="text-sm text-red-500">{err}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bids */}
        <section>
          <h3 className="text-sm font-medium mb-1">Bids</h3>
          <table className="w-full text-xs border dark:border-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr><Th>Price</Th><Th right>Size</Th></tr>
            </thead>
            <tbody>
              {bids.slice(0, depth).map((lv, i) => (
                <tr key={i} className="border-t dark:border-gray-800">
                  <Td>{fmtNum(lv.price)}</Td>
                  <Td right>{fmtNum(lv.size, 4)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {/* Asks */}
        <section>
          <h3 className="text-sm font-medium mb-1">Asks</h3>
          <table className="w-full text-xs border dark:border-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr><Th>Price</Th><Th right>Size</Th></tr>
            </thead>
            <tbody>
              {asks.slice(0, depth).map((lv, i) => (
                <tr key={i} className="border-t dark:border-gray-800">
                  <Td>{fmtNum(lv.price)}</Td>
                  <Td right>{fmtNum(lv.size, 4)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      {/* Depth curve */}
      <div className="mt-6">
        <h3 className="text-sm font-medium mb-2">Cumulative Depth</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={[...chartData.bids, ...chartData.asks]}>
              <XAxis type="number" dataKey="price" domain={["auto", "auto"]} />
              <YAxis />
              <Tooltip />
              <Area
                type="stepAfter"
                dataKey="bid"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.3}
              />
              <Area
                type="stepAfter"
                dataKey="ask"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ---------------- helpers ---------------- */
function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={`p-2 ${right ? "text-right" : "text-left"}`}>{children}</th>;
}
function Td({ children, right }: { children?: React.ReactNode; right?: boolean }) {
  return <td className={`p-2 ${right ? "text-right" : "text-left"}`}>{children}</td>;
}
function fmtNum(x: number, fixed = 2) {
  if (x >= 1000) return x.toFixed(0);
  if (x >= 100) return x.toFixed(1);
  return x.toFixed(fixed);
}