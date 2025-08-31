// frontend/components/Terminal.tsx
import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
} from "recharts";

interface Candle {
  t: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface OrderBookLevel {
  price: number;
  size: number;
  side: "bid" | "ask";
}

interface TradeLogRow {
  ts: number;
  price: number;
  qty: number;
  side: "buy" | "sell";
}

interface Alert {
  ts: number;
  msg: string;
  severity: "info" | "warn" | "critical";
}

export default function Terminal() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [book, setBook] = useState<OrderBookLevel[]>([]);
  const [trades, setTrades] = useState<TradeLogRow[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const [candleRes, bookRes, tradeRes, alertRes] = await Promise.all([
          fetch("/api/terminal/candles"),
          fetch("/api/terminal/book"),
          fetch("/api/terminal/trades"),
          fetch("/api/terminal/alerts"),
        ]);
        setCandles(await candleRes.json());
        setBook(await bookRes.json());
        setTrades(await tradeRes.json());
        setAlerts(await alertRes.json());
      } catch (err) {
        console.error("Terminal fetch error:", err);
      }
    }
    fetchData();
  }, []);

  const bidData = book.filter((b) => b.side === "bid").slice(0, 10);
  const askData = book.filter((b) => b.side === "ask").slice(0, 10);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4 bg-white dark:bg-gray-900 rounded-2xl shadow-md">
      {/* Candlestick (simplified as close line here) */}
      <div className="col-span-1">
        <h3 className="text-lg font-semibold mb-2">Candlestick Chart</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={candles}>
            <XAxis dataKey="t" hide />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="close" stroke="#0A84FF" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Order Book Depth */}
      <div className="col-span-1">
        <h3 className="text-lg font-semibold mb-2">Order Book</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={[...bidData, ...askData]}>
            <XAxis dataKey="price" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="size" fill="#34C759" name="Bids" />
            <Bar dataKey="size" fill="#FF3B30" name="Asks" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Trade Log */}
      <div className="col-span-1">
        <h3 className="text-lg font-semibold mb-2">Trade Log</h3>
        <div className="h-60 overflow-auto border rounded-lg dark:border-gray-700">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="p-2 text-left">Time</th>
                <th className="p-2 text-right">Price</th>
                <th className="p-2 text-right">Qty</th>
                <th className="p-2 text-right">Side</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i} className="border-t dark:border-gray-700">
                  <td className="p-2">{new Date(t.ts).toLocaleTimeString()}</td>
                  <td className="p-2 text-right">{t.price.toFixed(2)}</td>
                  <td className="p-2 text-right">{t.qty}</td>
                  <td className={`p-2 text-right ${t.side === "buy" ? "text-green-500" : "text-red-500"}`}>
                    {t.side.toUpperCase()}
                  </td>
                </tr>
              ))}
              {trades.length === 0 && <tr><td colSpan={4} className="p-2 opacity-60">No trades</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      {/* Alerts */}
      <div className="col-span-1">
        <h3 className="text-lg font-semibold mb-2">Alerts</h3>
        <div className="h-60 overflow-auto border rounded-lg dark:border-gray-700">
          {alerts.map((a, i) => (
            <div
              key={i}
              className={`p-2 text-sm border-b ${
                a.severity === "critical"
                  ? "bg-red-100 text-red-700"
                  : a.severity === "warn"
                  ? "bg-yellow-100 text-yellow-700"
                  : "bg-blue-100 text-blue-700"
              }`}
            >
              <span className="font-medium">{new Date(a.ts).toLocaleTimeString()} — </span>
              {a.msg}
            </div>
          ))}
          {alerts.length === 0 && <div className="p-2 opacity-60">No alerts</div>}
        </div>
      </div>
    </div>
  );
}