"use client";

/**
 * openinterestpane.tsx
 * Zero-import, dependency-free pane to visualize Open Interest (OI).
 * - Tailwind + inline SVG only (no Recharts, no icon libs)
 * - Filters: exchange, symbol
 * - Summary tiles: Latest OI, Δ 24h, OI/Vol (24h)
 * - Area/line chart of OI over time (+ optional volume bars)
 * - Recent table
 */

type OIRow = {
  ts: string;         // ISO timestamp
  exchange: string;   // e.g., "Binance", "OKX", "Bybit"
  symbol: string;     // e.g., "BTCUSDT", "ETHUSDT"
  oi: number;         // open interest (contracts or notional)
  vol?: number;       // traded volume over bucket (optional)
  currency?: string;  // optional "USD", "INR" (for formatting)
};

export type OpenInterestPaneProps = {
  title?: string;
  data: OIRow[];
  height?: number;            // chart height
  defaultExchange?: string;
  defaultSymbol?: string;
  showVolume?: boolean;
  compact?: boolean;
  className?: string;
  /** Format multiplier for OI and volume (e.g., 1e-6 to display in "M") */
  scale?: number;
  /** Unit label after scaling, e.g., "USD", "contracts", "M USD" */
  unit?: string;
};

export default function OpenInterestPane({
  title = "Open Interest",
  data,
  height = 260,
  defaultExchange,
  defaultSymbol,
  showVolume = true,
  compact = false,
  className = "",
  scale = 1,
  unit = "",
}: OpenInterestPaneProps) {
  // state
  const [ex, setEx] = useState(defaultExchange || "");
  const [sym, setSym] = useState(defaultSymbol || "");

  // derived lists
  const exchanges = useMemo(() => Array.from(new Set(data.map(d => d.exchange))).sort(), [data]);
  const symbols = useMemo(
    () => Array.from(new Set(data.filter(d => !ex || d.exchange === ex).map(d => d.symbol))).sort(),
    [data, ex]
  );

  // auto-select defaults
  useEffect(() => {
    if (!ex && exchanges.length) setEx(exchanges[0]);
  }, [exchanges]);
  useEffect(() => {
    if (!sym && symbols.length) setSym(symbols[0]);
  }, [symbols]);

  // rows for selection
  const rows = useMemo(() => {
    const r = data
      .filter(d => (!ex || d.exchange === ex) && (!sym || d.symbol === sym))
      .slice()
      .sort((a, b) => +new Date(a.ts) - +new Date(b.ts));
    return r;
  }, [data, ex, sym]);

  // stats
  const latest = rows.at(-1) || null;
  const prev = rows.at(-2) || null;

  const delta24h = useMemo(() => {
    if (!latest) return 0;
    const cutoff = +new Date(latest.ts) - 24 * 3600e3;
    const base = [...rows].reverse().find(r => +new Date(r.ts) <= cutoff) ?? rows[0];
    return latest.oi - (base?.oi ?? latest.oi);
  }, [rows, latest]);

  const vol24h = useMemo(() => {
    if (!latest) return 0;
    const cutoff = +new Date(latest.ts) - 24 * 3600e3;
    return rows
      .filter(r => +new Date(r.ts) >= cutoff)
      .reduce((s, r) => s + (r.vol ?? 0), 0);
  }, [rows, latest]);

  const oiVolRatio = vol24h ? (latest?.oi ?? 0) / vol24h : 0;

  // chart series
  const labels = rows.map(r => shortLabel(r.ts));
  const oiSeries = rows.map(r => (r.oi ?? 0) / (scale || 1));
  const volSeries = rows.map(r => (r.vol ?? 0) / (scale || 1));

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
        <Tile
          label="Latest OI"
          value={latest ? fmtNum((latest.oi ?? 0) / (scale || 1)) : "—"}
          suffix={unit}
        />
        <Tile
          label="Δ 24h"
          value={`${delta24h >= 0 ? "+" : ""}${fmtNum(delta24h / (scale || 1))}`}
          suffix={unit}
          accent={delta24h >= 0 ? "pos" : "neg"}
        />
        <Tile
          label="OI / Vol (24h)"
          value={vol24h ? fmtNum(oiVolRatio, 2) : "—"}
          suffix=""
        />
      </div>

      {/* Chart */}
      <div className="px-2 pb-2">
        <OIChart
          width={900}
          height={height}
          oi={oiSeries}
          vol={showVolume ? volSeries : undefined}
          labels={labels}
          unit={unit}
        />
      </div>

      {/* Table */}
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Time</th>
              <th className="px-3 py-2 text-right font-medium">Open Interest {unit && `(${unit})`}</th>
              <th className="px-3 py-2 text-right font-medium">Volume {unit && `(${unit})`}</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-24).reverse().map((r, i) => (
              <tr key={`${r.ts}-${i}`} className="border-t border-neutral-800">
                <td className="px-3 py-2">{longLabel(r.ts)}</td>
                <td className="px-3 py-2 text-right">{fmtNum((r.oi ?? 0) / (scale || 1))}</td>
                <td className="px-3 py-2 text-right">{r.vol != null ? fmtNum((r.vol ?? 0) / (scale || 1)) : "—"}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={3} className="px-3 py-6 text-center text-neutral-400">No data</td>
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
  suffix,
  accent = "mut",
}: {
  label: string;
  value: string;
  suffix?: string;
  accent?: "pos" | "neg" | "mut";
}) {
  const color =
    accent === "pos" ? "text-emerald-400" :
    accent === "neg" ? "text-rose-400" : "text-neutral-100";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
      <div className="text-neutral-400 text-xs">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color}`}>
        {value} {suffix ? <span className="text-xs font-normal text-neutral-400">{suffix}</span> : null}
      </div>
    </div>
  );
}

/* ------------------------------ Charts (SVG) ------------------------------ */

function OIChart({
  width = 800,
  height = 240,
  oi,
  vol,
  labels,
  unit,
}: {
  width?: number;
  height?: number;
  oi: number[];
  vol?: number[];
  labels?: string[];
  unit?: string;
}) {
  const w = width;
  const h = height;
  const pad = { l: 40, r: 12, t: 12, b: vol ? 60 : 28 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);
  const volH = vol ? Math.floor(innerH * 0.28) : 0;
  const oiH = innerH - volH - (vol ? 8 : 0);

  // scales
  const n = Math.max(1, oi.length);
  const minOI = Math.min(...oi, 0);
  const maxOI = Math.max(...oi, 1);
  const padOI = (maxOI - minOI) * 0.06 || maxOI * 0.06;
  const yMin = Math.max(0, minOI - padOI);
  const yMax = maxOI + padOI;

  const maxVol = vol ? Math.max(...vol, 1) : 1;

  const x = (i: number) => pad.l + (i * innerW) / Math.max(1, n - 1);
  const yOI = (v: number) => pad.t + (oiH - ((v - yMin) / (yMax - yMin || 1)) * oiH);
  const yVOL = (v: number) => pad.t + oiH + 8 + (volH - (v / (maxVol || 1)) * volH);

  const pathD = oi.map((v, i) => `${i ? "L" : "M"} ${x(i).toFixed(2)} ${yOI(v).toFixed(2)}`).join(" ");
  const areaD =
    `M ${x(0)} ${yOI(oi[0] ?? yMin)} ` +
    oi.map((v, i) => `L ${x(i)} ${yOI(v)}`).join(" ") +
    ` L ${x(oi.length - 1)} ${pad.t + oiH} L ${x(0)} ${pad.t + oiH} Z`;

  // ticks
  const yTicks = 4;
  const yVals = Array.from({ length: yTicks + 1 }, (_, i) => yMin + ((yMax - yMin) * i) / yTicks);

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
      {/* grid + y labels */}
      {yVals.map((val, i) => {
        const yy = yOI(val);
        return (
          <g key={i}>
            <line x1={pad.l} y1={yy} x2={w - pad.r} y2={yy} stroke="#27272a" strokeDasharray="3 3" />
            <text x={4} y={yy + 4} fill="#9ca3af" fontSize="10">
              {fmtNum(val)} {unit || ""}
            </text>
          </g>
        );
      })}

      {/* OI area + line */}
      <path d={areaD} fill="#10b98122" />
      <path d={pathD} fill="none" stroke="#10b981" strokeWidth="2" />

      {/* sparsified OI dots */}
      {oi.map((v, i) =>
        (n <= 80 || i % Math.ceil(n / 80) === 0) ? (
          <circle key={i} cx={x(i)} cy={yOI(v)} r="1.7" fill="#10b981" />
        ) : null
      )}

      {/* Volume bars */}
      {vol && volH > 0 && vol.map((v, i) => {
        const bw = Math.max(1, innerW / Math.max(1, n - 1)) * 0.64;
        const vx = x(i) - bw / 2;
        const vy = yVOL(v);
        const vh = pad.t + oiH + 8 + volH - vy;
        return (
          <rect
            key={`v-${i}`}
            x={vx}
            y={vy}
            width={bw}
            height={vh}
            fill="#334155"
            opacity={0.9}
            rx="2"
          />
        );
      })}

      {/* x labels (sparse) */}
      {labels && labels.map((lab, i) =>
        (i === 0 || i === labels.length - 1 || i % Math.ceil(labels.length / 6 || 1) === 0) ? (
          <text key={i} x={x(i)} y={h - 6} textAnchor="middle" fill="#9ca3af" fontSize="10">
            {lab}
          </text>
        ) : null
      )}
    </svg>
  );
}

/* --------------------------------- utils --------------------------------- */

function fmtNum(n: number, d = 2) {
  return n.toLocaleString("en-US", { maximumFractionDigits: d });
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
const mock: OIRow[] = Array.from({ length: 48 }).map((_, i) => {
  const ts = new Date(now - (47 - i) * 3600e3).toISOString(); // hourly
  const base = 8_000_000 + Math.sin(i / 7) * 300_000 + Math.random() * 120_000;
  return {
    ts,
    exchange: "Binance",
    symbol: "BTCUSDT",
    oi: base,
    vol: 300_000 + Math.max(0, Math.sin(i / 5)) * 400_000 + Math.random() * 100_000,
    currency: "USD",
  };
});

<OpenInterestPane
  data={mock}
  defaultExchange="Binance"
  defaultSymbol="BTCUSDT"
  unit="USD"
  scale={1e6} // show in millions
/>
------------------------------------------------------------------------------- */