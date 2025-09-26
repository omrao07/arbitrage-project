"use client";

/**
 * fundingratespane.tsx
 * Zero-import, dependency-free pane for perpetual swap funding rates.
 * - Tailwind + inline SVG only (no Recharts, no icon libs)
 * - Filters: exchange, symbol
 * - Summary tiles (Latest, 24h avg, Next funding ETA)
 * - Bars chart (positive/negative funding prints)
 * - Recent table
 */

type FundingPoint = {
  ts: string;         // ISO timestamp
  exchange: string;   // e.g., "Binance", "Bybit", "OKX", "Deribit"
  symbol: string;     // e.g., "BTCUSDT", "ETHUSDT", "BTC-PERP"
  rate: number;       // funding rate as a DECIMAL per period (e.g., 0.0001 = 0.01%)
  period?: "8H" | "1H" | "12H"; // optional, defaults to 8H
};

export type FundingRatesPaneProps = {
  title?: string;
  data: FundingPoint[];
  height?: number;            // chart height
  defaultExchange?: string;
  defaultSymbol?: string;
  compact?: boolean;
  className?: string;
};

export default function FundingRatesPane({
  title = "Perp Funding Rates",
  data,
  height = 220,
  defaultExchange,
  defaultSymbol,
  compact = false,
  className = "",
}: FundingRatesPaneProps) {
  // state
  const [ex, setEx] = useState(defaultExchange || "");
  const [sym, setSym] = useState(defaultSymbol || "");

  // derive lists
  const exchanges = useMemo(() => Array.from(new Set(data.map(d => d.exchange))).sort(), [data]);
  const symbols = useMemo(
    () => Array.from(new Set(data.filter(d => !ex || d.exchange === ex).map(d => d.symbol))).sort(),
    [data, ex]
  );

  // auto-select if not set
  useEffect(() => {
    if (!ex && exchanges.length) setEx(exchanges[0]);
  }, [exchanges]);
  useEffect(() => {
    if (!sym && symbols.length) setSym(symbols[0]);
  }, [symbols]);

  // filter rows for selection
  const rows = useMemo(() => {
    const r = data
      .filter(d => (!ex || d.exchange === ex) && (!sym || d.symbol === sym))
      .slice()
      .sort((a, b) => +new Date(a.ts) - +new Date(b.ts));
    return r;
  }, [data, ex, sym]);

  // stats
  const latest = rows.at(-1) || null;
  const last24h = useMemo(() => {
    if (!latest) return [] as FundingPoint[];
    const cutoff = +new Date(latest.ts) - 24 * 3600 * 1000;
    return rows.filter(r => +new Date(r.ts) >= cutoff);
  }, [rows, latest]);

  const avg24h = useMemo(() => {
    if (!last24h.length) return 0;
    return last24h.reduce((s, r) => s + r.rate, 0) / last24h.length;
  }, [last24h]);

  const nextEta = useMemo(() => {
    if (!latest) return "—";
    const period = (latest.period ?? "8H") === "1H" ? 3600e3 : (latest.period ?? "8H") === "12H" ? 12 * 3600e3 : 8 * 3600e3;
    const next = +new Date(latest.ts) + period;
    const dt = next - Date.now();
    if (dt <= 0) return "imminent";
    const h = Math.floor(dt / 3600e3);
    const m = Math.floor((dt % 3600e3) / 60e3);
    return `${h}h ${m}m`;
  }, [latest]);

  // chart data
  const labels = rows.map(r => shortLabel(r.ts));
  const series = rows.map(r => r.rate * 100); // to percent units

  const dens = compact
    ? { padX: "px-3", padY: "py-2", text: "text-xs", gap: "gap-2" }
    : { padX: "px-4", padY: "py-3", text: "text-sm", gap: "gap-3" };

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className={`flex flex-wrap items-center justify-between border-b border-neutral-800 ${dens.padX} ${dens.padY}`}>
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {ex || "—"} · {sym || "—"} · {rows.length} pts
          </p>
        </div>
        <div className={`flex items-center ${dens.gap}`}>
          <select
            value={ex}
            onChange={(e) => { setEx(e.target.value); setSym(""); }}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 text-xs"
          >
            {exchanges.map(x => <option key={x} value={x}>{x}</option>)}
          </select>
          <select
            value={sym}
            onChange={(e) => setSym(e.target.value)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 text-xs"
          >
            {symbols.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {/* Summary tiles */}
      <div className={`grid grid-cols-1 md:grid-cols-3 ${dens.gap} ${dens.padX} ${dens.padY} ${dens.text}`}>
        <Tile label="Latest"
              value={latest ? fmtPct(latest.rate) : "—"}
              accent={latest ? (latest.rate >= 0 ? "pos" : "neg") : "mut"} />
        <Tile label="24h Avg"
              value={fmtPct(avg24h)}
              accent={avg24h >= 0 ? "pos" : "neg"} />
        <Tile label="Next Funding"
              value={nextEta}
              hint={latest?.period ?? "8H"} />
      </div>

      {/* Chart */}
      <div className="px-2 pb-2">
        <BarsSVG
          width={800}
          height={height}
          data={series}
          labels={labels}
          positive="#10b981"
          negative="#ef4444"
        />
      </div>

      {/* Table */}
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Time</th>
              <th className="px-3 py-2 text-left font-medium">Exchange</th>
              <th className="px-3 py-2 text-left font-medium">Symbol</th>
              <th className="px-3 py-2 text-right font-medium">Funding</th>
              <th className="px-3 py-2 text-right font-medium">Period</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-20).reverse().map((r, i) => (
              <tr key={`${r.ts}-${i}`} className="border-t border-neutral-800">
                <td className="px-3 py-2">{longLabel(r.ts)}</td>
                <td className="px-3 py-2">{r.exchange}</td>
                <td className="px-3 py-2">{r.symbol}</td>
                <td className={`px-3 py-2 text-right ${r.rate >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {fmtPct(r.rate)}
                </td>
                <td className="px-3 py-2 text-right">{r.period ?? "8H"}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-6 text-center text-neutral-400">No data</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* --------------------------------- Tiles --------------------------------- */

function Tile({
  label,
  value,
  accent = "mut",
  hint,
}: {
  label: string;
  value: string;
  accent?: "pos" | "neg" | "mut";
  hint?: string;
}) {
  const color =
    accent === "pos" ? "text-emerald-400" :
    accent === "neg" ? "text-rose-400" : "text-neutral-100";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
      <div className="text-neutral-400 text-xs">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-neutral-500">{hint}</div>}
    </div>
  );
}

/* ------------------------------ Charts (SVG) ------------------------------ */

function BarsSVG({
  width = 640,
  height = 200,
  data,
  labels,
  positive = "#10b981",
  negative = "#ef4444",
}: {
  width?: number;
  height?: number;
  data: number[];    // values in PERCENT (e.g., 0.01% => 0.01)
  labels?: string[];
  positive?: string;
  negative?: string;
}) {
  const w = width;
  const h = height;
  const pad = 24;
  // symmetric domain around zero for better visual balance
  const maxAbs = Math.max(0.01, Math.max(...data.map(v => Math.abs(v))));
  const dom = Math.max(maxAbs, 0.05); // at least ±0.05%
  const zeroY = h / 2;
  const bw = (w - pad * 2) / Math.max(1, data.length);
  const scale = (v: number) => ((v / (dom || 1)) * (h / 2 - pad));

  const yTickVals = [-dom, -dom/2, 0, dom/2, dom];

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
      {/* grid */}
      {yTickVals.map((v, i) => {
        const yy = zeroY - scale(v);
        return (
          <g key={i}>
            <line x1={pad} y1={yy} x2={w - pad} y2={yy} stroke="#27272a" strokeDasharray="3 3" />
            <text x={4} y={yy + 4} fill="#9ca3af" fontSize="10">
              {fmtPctDisplay(v / 100)}
            </text>
          </g>
        );
      })}
      {/* zero line */}
      <line x1={pad} y1={zeroY} x2={w - pad} y2={zeroY} stroke="#6b7280" strokeDasharray="4 4" />

      {/* bars */}
      {data.map((v, i) => {
        const barH = Math.abs(scale(v));
        const x = pad + i * bw + bw * 0.12;
        const y = v >= 0 ? zeroY - barH : zeroY;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={bw * 0.76}
            height={barH}
            fill={v >= 0 ? positive : negative}
            rx="3"
          />
        );
      })}

      {/* x labels (sparse) */}
      {labels && labels.map((lab, i) => (
        (i === 0 || i === labels.length - 1 || i % Math.ceil(labels.length / 6 || 1) === 0) ? (
          <text key={i} x={pad + i * bw + bw / 2} y={h - 4} textAnchor="middle" fill="#9ca3af" fontSize="10">
            {lab}
          </text>
        ) : null
      ))}
    </svg>
  );
}

/* --------------------------------- utils --------------------------------- */

function fmtPct(x: number) {
  // x is decimal (0.0001 = 0.01%)
  return `${(x * 100).toFixed(4)}%`;
}
function fmtPctDisplay(x: number) {
  // x is decimal again
  const v = x * 100;
  if (Math.abs(v) >= 1) return `${v.toFixed(2)}%`;
  if (Math.abs(v) >= 0.1) return `${v.toFixed(3)}%`;
  return `${v.toFixed(4)}%`;
}
function shortLabel(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
function longLabel(iso: string) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

/* ------------------- Ambient React (to keep zero imports) ------------------- */
// Remove these if you prefer actual imports.
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useMemo<T>(cb: () => T, deps: any[]): T;

/* ----------------------------- Example mock data -----------------------------
const now = Date.now();
const mock: FundingPoint[] = Array.from({ length: 36 }).flatMap((_, i) => {
  const ts = new Date(now - (35 - i) * 2 * 3600e3).toISOString(); // every 2h
  return [
    { ts, exchange: "Binance", symbol: "BTCUSDT", rate: (Math.sin(i/6) * 0.00005), period: "8H" },
    { ts, exchange: "Binance", symbol: "ETHUSDT", rate: (Math.cos(i/5) * 0.00007), period: "8H" },
    { ts, exchange: "Bybit",   symbol: "BTCUSDT", rate: (Math.sin(i/7) * 0.00006), period: "8H" },
  ];
});

<FundingRatesPane data={mock} defaultExchange="Binance" defaultSymbol="BTCUSDT" />
------------------------------------------------------------------------------- */