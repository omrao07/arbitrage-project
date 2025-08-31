// frontend/components/PositionStrip.tsx
import React, { useMemo } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

/* ------------------------------- Types ------------------------------- */

export type PositionSide = "long" | "short";

export type Position = {
  id: string;
  symbol: string;
  side: PositionSide;
  qty: number;            // signed/absolute accepted; we infer if needed
  avgPrice: number;
  lastPrice: number;
  currency?: string;      // e.g., "USD"
  leverage?: number | null;
  venue?: string | null;

  // optional extras
  todayPnl?: number | null;
  pnl?: number | null;        // total PnL in ccy
  pnlPct?: number | null;     // total return %
  sector?: string | null;

  // sparkline (optional): recent prices or PnL
  spark?: { t: number; v: number }[];
};

interface Props {
  positions: Position[];
  title?: string;
  dense?: boolean;              // tighter vertical padding
  showSpark?: boolean;
  sparkField?: "price" | "pnl"; // how you computed v in spark
  onClose?: (pos: Position) => void;
  onReduce?: (pos: Position, fraction: number) => void; // e.g. 0.5 = reduce half
  onHedge?: (pos: Position) => void;
}

/* ------------------------------ Component ---------------------------- */

export default function PositionStrip({
  positions,
  title = "Open Positions",
  dense = false,
  showSpark = true,
  sparkField = "price",
  onClose,
  onReduce,
  onHedge,
}: Props) {
  const rows = useMemo(() => normalize(positions), [positions]);

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <div className="text-xs opacity-70">{rows.length} positions</div>
        </div>
      </header>

      {/* Horizontal strip */}
      <div className={`flex gap-3 overflow-x-auto pb-1 ${dense ? "-mt-1" : ""}`}>
        {rows.map((p) => (
          <div
            key={p.id}
            className={`min-w-[280px] rounded-xl border dark:border-gray-800 ${dense ? "p-2" : "p-3"} bg-white dark:bg-gray-900`}
          >
            {/* Top row: symbol / venue / side badge */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="text-base font-semibold">{p.symbol}</div>
                {p.venue && <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800">{p.venue}</span>}
                {p.sector && <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800">{p.sector}</span>}
              </div>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${p.side === "long" ? "bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300" : "bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300"}`}
              >
                {p.side.toUpperCase()}
              </span>
            </div>

            {/* Middle: qty / prices / leverage */}
            <div className={`mt-1 grid ${dense ? "grid-cols-2" : "grid-cols-3"} gap-2 text-xs`}>
              <Labeled k="Qty" v={fmtNum(p.qty)} />
              <Labeled k="Avg" v={fmtNum(p.avgPrice)} />
              <Labeled k="Last" v={fmtNum(p.lastPrice)} />
              <Labeled k="PnL" v={fmtMoney(p.pnl, p.currency)} strong color={p.pnl ?? 0} />
              <Labeled k="Return" v={p.pnlPct != null ? `${p.pnlPct.toFixed(2)}%` : "—"} color={p.pnlPct ?? 0} />
              <Labeled k="Today" v={fmtMoney(p.todayPnl, p.currency)} color={p.todayPnl ?? 0} />
              {p.leverage != null && <Labeled k="Lev" v={`${p.leverage.toFixed(1)}×`} />}
            </div>

            {/* Sparkline */}
            {showSpark && p.spark && p.spark.length > 1 && (
              <div className="h-14 mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={p.spark}>
                    <XAxis dataKey="t" hide />
                    <YAxis hide domain={["auto", "auto"]} />
                    <Tooltip
                      contentStyle={{ fontSize: 11 }}
                      labelFormatter={(t) => new Date(t as number).toLocaleString()}
                      formatter={(v) => [fmtNum(v as number), sparkField]}
                    />
                    <Line
                      type="monotone"
                      dataKey="v"
                      dot={false}
                      stroke={sparkColor(p)}
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Actions */}
            {(onClose || onReduce || onHedge) && (
              <div className="mt-2 flex gap-2">
                {onClose && (
                  <button className="px-2 py-1 rounded-md border text-xs dark:border-gray-800" onClick={() => onClose(p)}>
                    Close
                  </button>
                )}
                {onReduce && (
                  <>
                    <button className="px-2 py-1 rounded-md border text-xs dark:border-gray-800" onClick={() => onReduce(p, 0.5)}>
                      −50%
                    </button>
                    <button className="px-2 py-1 rounded-md border text-xs dark:border-gray-800" onClick={() => onReduce(p, 0.25)}>
                      −25%
                    </button>
                  </>
                )}
                {onHedge && (
                  <button className="px-2 py-1 rounded-md border text-xs dark:border-gray-800" onClick={() => onHedge(p)}>
                    Hedge
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
        {rows.length === 0 && (
          <div className="text-sm opacity-60">No open positions</div>
        )}
      </div>
    </div>
  );
}

/* ----------------------------- Subcomponents ----------------------------- */

function Labeled({ k, v, strong = false, color }: { k: string; v: string; strong?: boolean; color?: number }) {
  const cls =
    color == null
      ? ""
      : color > 0
      ? "text-green-600"
      : color < 0
      ? "text-red-600"
      : "opacity-80";
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="opacity-70">{k}</span>
      <span className={`${strong ? "font-semibold" : ""} ${cls}`}>{v}</span>
    </div>
  );
}

/* --------------------------------- Utils --------------------------------- */

function normalize(xs: Position[]): Position[] {
  return xs.map((p) => {
    const side = p.side ?? (p.qty < 0 ? "short" : "long");
    const qtyAbs = Math.abs(p.qty);
    // If pnl not provided, compute rough mark-to-market (not including fees)
    const pnl = p.pnl ?? (qtyAbs && p.avgPrice ? (p.lastPrice - p.avgPrice) * (side === "long" ? 1 : -1) * qtyAbs : null);
    const ret = p.pnlPct ?? (p.avgPrice ? (((p.lastPrice - p.avgPrice) / p.avgPrice) * (side === "long" ? 100 : -100)) : null);
    return { ...p, side, pnl, pnlPct: ret };
  });
}

function fmtNum(x: number | null | undefined) {
  if (x == null || !Number.isFinite(x)) return "—";
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 6 }); } catch { return String(x); }
}
function fmtMoney(x: number | null | undefined, ccy = "USD") {
  if (x == null || !Number.isFinite(x)) return "—";
  try { return x.toLocaleString(undefined, { style: "currency", currency: ccy }); } catch { return `${ccy} ${x.toFixed(2)}`; }
}
function sparkColor(p: Position) {
  const s = (p.pnl ?? 0) >= 0 ? "#16a34a" : "#ef4444";
  return s;
}