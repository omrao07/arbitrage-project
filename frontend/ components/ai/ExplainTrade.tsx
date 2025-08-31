// frontend/components/ExplainTrade.tsx
import React, { useEffect, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

interface Trade {
  id: string;
  symbol: string;
  side: "buy" | "sell";
  qty: number;
  price: number;
  ts: number;
  pnl?: number;
}

interface Explanation {
  summary: string;
  drivers: Record<string, number>; // e.g., factor → contribution
  risk_flags: string[];
}

export default function ExplainTrade({ tradeId }: { tradeId: string }) {
  const [trade, setTrade] = useState<Trade | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const [t, e] = await Promise.all([
          fetch(`/api/trades/${tradeId}`).then(r => r.json()),
          fetch(`/api/trades/${tradeId}/explain`).then(r => r.json()),
        ]);
        setTrade(t);
        setExplanation(e);
      } catch (e: any) {
        setErr(e?.message || "Error loading trade explanation");
      } finally {
        setLoading(false);
      }
    })();
  }, [tradeId]);

  if (loading) return <div className="p-4">Loading trade explanation...</div>;
  if (err) return <div className="p-4 text-red-500">Error: {err}</div>;
  if (!trade || !explanation) return <div className="p-4">No trade explanation available</div>;

  const driverData = Object.entries(explanation.drivers).map(([k, v]) => ({
    factor: k,
    impact: v,
  }));

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4">
        <h2 className="text-xl font-semibold">Trade Explanation</h2>
        <p className="text-sm opacity-70">
          Why this trade was executed and what drove its performance
        </p>
      </header>

      {/* Trade summary */}
      <div className="mb-4 text-sm">
        <div><b>Symbol:</b> {trade.symbol}</div>
        <div><b>Side:</b> {trade.side.toUpperCase()}</div>
        <div><b>Quantity:</b> {trade.qty}</div>
        <div><b>Price:</b> ${trade.price.toFixed(2)}</div>
        <div>
          <b>Timestamp:</b> {new Date(trade.ts).toLocaleString()}
        </div>
        <div>
          <b>PnL:</b>{" "}
          <span className={trade.pnl && trade.pnl < 0 ? "text-red-500" : "text-green-600"}>
            {trade.pnl !== undefined ? `$${trade.pnl.toFixed(2)}` : "—"}
          </span>
        </div>
      </div>

      {/* Narrative summary */}
      <div className="mb-4">
        <h3 className="text-lg font-medium mb-1">Summary</h3>
        <p className="text-sm">{explanation.summary}</p>
      </div>

      {/* Drivers chart */}
      <div className="mb-4">
        <h3 className="text-lg font-medium mb-1">Risk/Alpha Drivers</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={driverData}>
            <XAxis dataKey="factor" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="impact" fill="#0A84FF" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Risk flags */}
      {explanation.risk_flags && explanation.risk_flags.length > 0 && (
        <div>
          <h3 className="text-lg font-medium mb-1">Risk Flags</h3>
          <ul className="list-disc pl-5 text-sm">
            {explanation.risk_flags.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}