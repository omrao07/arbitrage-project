"use client";

/**
 * optionssurfacepane.tsx
 * Zero-import, dependency-free pane to visualize an option surface.
 *
 * - Tailwind + inline SVG only (no Recharts / no icon libs)
 * - Shows a heatmap (expiries × strikes) of a chosen metric (IV / Price / Delta / Vega / Theta)
 * - Per-expiry or global normalization, signed diverging or mono scales
 * - Tooltips on hover, keyboard shortcuts:
 *     • ←/→ cycle metric   • ↑/↓ toggle CALL/PUT/BOTH
 *
 * Wire it with the surface from fetchoptionsurface.server.ts
 */

type OptionType = "CALL" | "PUT";
type GreekKey = "iv" | "price" | "delta" | "vega" | "theta";

type SurfacePoint = {
  symbol: string;
  expiry: string; // YYYY-MM-DD
  t: number;      // years
  type: OptionType;
  strike: number;
  moneyness: number;
  iv: number;     // decimal (0.22 = 22%)
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;  // per day
};
type ExpirySlice = { expiry: string; t: number; baseVol: number; points: SurfacePoint[]; };
export type OptionSurface = { symbol: string; spot: number; rate: number; div: number; ts: string; slices: ExpirySlice[]; };

type NormMode = "signed" | "global" | "per-expiry";

export default function OptionSurfacePane({
  surface,
  defaultMetric = "iv",
  defaultType = "CALL" as OptionType | "PUT" | "BOTH",
  normalize = "signed" as NormMode,
  className = "",
}: {
  surface: OptionSurface;
  defaultMetric?: GreekKey;
  defaultType?: OptionType | "PUT" | "BOTH";
  normalize?: NormMode;
  className?: string;
}) {
  const [metric, setMetric] = useState<GreekKey>(defaultMetric);
  const [otype, setOtype] = useState<OptionType | "PUT" | "BOTH">(defaultType);
  const [norm, setNorm] = useState<NormMode>(normalize);
  const [hover, setHover] = useState<null | { e: string; k: number; v: number }>(null);

  // flatten points by expiry/strike, avg when BOTH
  const { expiries, strikes, grid } = useMemo(() => toGrid(surface, metric, otype), [surface, metric, otype]);

  const bounds = useMemo(() => calcBounds(grid, metric, norm), [grid, metric, norm]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName ?? "")) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const order: GreekKey[] = ["iv", "price", "delta", "vega", "theta"];
        const i = order.indexOf(metric);
        setMetric(order[(i + (e.key === "ArrowRight" ? 1 : 4)) % order.length]);
      }
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const order: Array<OptionType | "PUT" | "BOTH"> = ["CALL", "PUT", "BOTH"];
        const i = order.indexOf(otype);
        setOtype(order[(i + (e.key === "ArrowUp" ? 1 : 2)) % 3]);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [metric, otype]);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">
            {surface.symbol} Option Surface
          </h3>
          <p className="text-xs text-neutral-400">
            Spot {fmt(surface.spot)} · {expiries.length} expiries · {strikes.length} strikes
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Select
            label="Metric"
            value={metric}
            onChange={(v) => setMetric(v as GreekKey)}
            options={[
              { value: "iv", label: "IV %" },
              { value: "price", label: "Price" },
              { value: "delta", label: "Delta" },
              { value: "vega", label: "Vega" },
              { value: "theta", label: "Theta/day" },
            ]}
          />
          <Select
            label="Type"
            value={otype}
            onChange={(v) => setOtype(v as OptionType | "PUT" | "BOTH")}
            options={[
              { value: "CALL", label: "CALL" },
              { value: "PUT", label: "PUT" },
              { value: "BOTH", label: "BOTH (avg)" },
            ]}
          />
          <Select
            label="Scale"
            value={norm}
            onChange={(v) => setNorm(v as NormMode)}
            options={[
              { value: "signed", label: "Signed ±" },
              { value: "global", label: "Global 0..max" },
              { value: "per-expiry", label: "Per-expiry 0..max" },
            ]}
          />
        </div>
      </div>

      {/* legend */}
      <div className="flex items-center gap-3 px-4 pt-3 text-[11px] text-neutral-400">
        <span>Legend:</span>
        <Legend gradient={norm === "global" ? "mono" : "diverge"} />
        <span className="ml-auto">{legend(bounds, metric, norm)}</span>
      </div>

      {/* heatmap */}
      <div className="overflow-x-auto px-3 py-3">
        <table className="min-w-full border-collapse text-sm">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-neutral-900 px-2 py-1 text-left text-xs font-medium text-neutral-400">
                Expiry ↓ / Strike →
              </th>
              {strikes.map((k) => (
                <th key={k} className="px-2 py-1 text-right text-xs font-medium text-neutral-400">
                  {fmt(k, 0)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {expiries.map((e, ei) => (
              <tr key={e} className="border-t border-neutral-800">
                <td className="sticky left-0 z-10 bg-neutral-900 px-2 py-1 text-xs text-neutral-300">
                  {e}
                </td>
                {strikes.map((k, ki) => {
                  const v = grid[ei][ki];
                  const color = isFinite(v) ? colorFor(v, grid, bounds, ei, norm) : "transparent";
                  const txt = isFinite(v) ? formatMetric(metric, v) : "—";
                  const txtColor = contrastText(color);
                  return (
                    <td
                      key={`${e}-${k}`}
                      className="px-1 py-1 text-right align-middle"
                      style={{ background: color, color: txtColor, cursor: "default" }}
                      onMouseEnter={() => setHover({ e, k, v })}
                      onMouseLeave={() => setHover(null)}
                      title={`${e} @ ${fmt(k,0)} → ${metricLabel(metric)}: ${txt}`}
                    >
                      <span className="inline-block min-w-[60px]">{txt}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
            {expiries.length === 0 && (
              <tr>
                <td colSpan={strikes.length + 1} className="px-3 py-6 text-center text-neutral-500">
                  No data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* footer tooltip / status */}
      <div className="flex items-center justify-between border-t border-neutral-800 px-4 py-2 text-xs text-neutral-400">
        <div>
          {hover
            ? (
              <>
                <span className="text-neutral-300">{hover.e}</span>{" · "}
                K {fmt(hover.k, 0)} — <span className="text-neutral-300">{metricLabel(metric)}:</span>{" "}
                {formatMetric(metric, hover.v)}
              </>
            )
            : "Hover any cell for details"}
        </div>
        <div>←/→ metric · ↑/↓ type</div>
      </div>
    </div>
  );
}

/* -------------------------------- helpers -------------------------------- */

function toGrid(surface: OptionSurface, metric: GreekKey, otype: OptionType | "PUT" | "BOTH") {
  // collect unique strikes (2dp) and expiries
  const all: SurfacePoint[] = [];
  for (const sl of surface.slices) for (const p of sl.points) {
    if (otype === "BOTH" || p.type === otype) all.push(p);
  }
  const strikes = Array.from(new Set(all.map((p) => round(p.strike, 2)))).sort((a, b) => a - b);
  const expiries = Array.from(new Set(all.map((p) => p.expiry))).sort();

  // map expiry->strike->list
  const map = new Map<string, Map<number, SurfacePoint[]>>();
  for (const e of expiries) map.set(e, new Map());
  for (const p of all) {
    const e = map.get(p.expiry)!;
    const k = round(p.strike, 2);
    const list = e.get(k) ?? [];
    list.push(p);
    e.set(k, list);
  }

  const grid = expiries.map(() => strikes.map(() => NaN));
  for (let ei = 0; ei < expiries.length; ei++) {
    const e = expiries[ei];
    const row = map.get(e)!;
    for (let ki = 0; ki < strikes.length; ki++) {
      const k = strikes[ki];
      const ps = row.get(k) ?? [];
      if (!ps.length) continue;
      // average if BOTH, else single
      let v = 0;
      for (const p of ps) v += pickMetric(p, metric);
      v /= ps.length;
      grid[ei][ki] = v;
    }
  }

  return { expiries, strikes, grid };
}

function pickMetric(p: SurfacePoint, m: GreekKey): number {
  switch (m) {
    case "iv": return p.iv * 100; // show in %
    case "price": return p.price;
    case "delta": return p.delta;
    case "vega": return p.vega;
    case "theta": return p.theta;
  }
}

function formatMetric(m: GreekKey, v: number): string {
  if (!isFinite(v)) return "—";
  if (m === "iv") return `${v.toFixed(2)}%`;
  if (m === "price") return fmt(v);
  if (m === "delta") return v.toFixed(2);
  if (m === "vega") return shortNum(v);
  if (m === "theta") return v.toFixed(3);
  return v.toFixed(3);
}
function metricLabel(m: GreekKey) {
  return m === "iv" ? "IV" : m[0].toUpperCase() + m.slice(1);
}

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-US", { maximumFractionDigits: d });
}
function shortNum(n: number) {
  const a = Math.abs(n), s = n < 0 ? "-" : "";
  if (a >= 1e9) return s + (a / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return s + (a / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return s + (a / 1e3).toFixed(2) + "K";
  if (a >= 1) return s + a.toFixed(2);
  if (a >= 0.01) return s + a.toFixed(4);
  return s + a.toExponential(2);
}
function round(n: number, d = 2) {
  const p = 10 ** d;
  return Math.round(n * p) / p;
}

/* ------------------------------ bounds & color ----------------------------- */

type Bounds = { min: number; max: number; maxAbs: number; perRow?: number[] };

function calcBounds(grid: number[][], metric: GreekKey, norm: NormMode): Bounds {
  // signed normalization is symmetric on absolute magnitude
  let min = Infinity, max = -Infinity, maxAbs = 0;
  for (const row of grid) for (const v of row) {
    if (!isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  }
  if (!isFinite(min)) min = 0;
  if (!isFinite(max)) max = 0;
  if (norm === "per-expiry") {
    const per = grid.map((row) => row.reduce((m, v) => (isFinite(v) ? Math.max(m, Math.abs(v)) : m), 0));
    return { min, max, maxAbs: Math.max(maxAbs, 1e-12), perRow: per.map((x) => (x <= 0 ? 1 : x)) };
  }
  return { min, max, maxAbs: Math.max(maxAbs, 1e-12) };
}

function colorFor(v: number, grid: number[][], b: Bounds, ei: number, norm: NormMode): string {
  if (!isFinite(v)) return "transparent";
  if (norm === "global") {
    // 0..max mono-green
    const M = Math.max(b.max, 1e-12);
    const t = clamp01(v / M);
    return monoGreen(t);
  }
  const M = norm === "per-expiry" ? (b.perRow?.[ei] ?? b.maxAbs) : b.maxAbs;
  const t = clamp01((v + M) / (2 * M)); // map [-M..+M] → [0..1]
  return diverge(t);
}

function legend(b: Bounds, metric: GreekKey, norm: NormMode) {
  if (norm === "global") {
    return `${metricLabel(metric)}: 0 → ${formatMetric(metric, metric === "iv" ? b.max : b.max)}`;
  }
  return `${metricLabel(metric)}: −max ↔ +max (|max| ≈ ${formatMetric(metric, b.maxAbs)})`;
}

/* -------------------------------- UI bits -------------------------------- */

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="text-neutral-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 text-xs"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Legend({ gradient }: { gradient: "diverge" | "mono" }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-neutral-500">min</span>
      <div
        className="h-2 w-36 rounded"
        style={{
          background:
            gradient === "diverge"
              ? "linear-gradient(90deg, #ef4444, #f59e0b, #10b981)" // red→amber→green
              : "linear-gradient(90deg, #0b132b, #10b981)",          // slate→green
        }}
      />
      <span className="text-[10px] text-neutral-500">max</span>
    </div>
  );
}

/* -------------------------------- colors -------------------------------- */

function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }
function mix(a: number, b: number, t: number) { return a + (b - a) * clamp01(t); }
// Diverging Red→Amber→Green
function diverge(t: number) {
  const r = mix(239, 16, t), g = mix(68, 185, t), b = mix(68, 129, t);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
// Monochrome green
function monoGreen(t: number) {
  const r = mix(11, 16, t), g = mix(19, 185, t), b = mix(43, 129, t);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
function contrastText(rgb: string) {
  // naive luminance for rgb(r,g,b) strings
  const m = rgb.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/i);
  if (!m) return "#e5e7eb";
  const r = +m[1], g = +m[2], b = +m[3];
  const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return L > 140 ? "#111827" : "#e5e7eb"; // dark text on light cells, else light text
}

/* ------------------- Ambient React (to keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;

/* ----------------------------- Example usage -----------------------------
import { fetchOptionSurface } from "@/server/fetchoptionsurface.server";

const surf = await fetchOptionSurface({
  symbol: "NIFTY",
  spot: 22500,
  expiriesDays: [2,7,14,30,60],
  kMin: 0.9, kMax: 1.1, kSteps: 11,
});

<OptionSurfacePane surface={surf} defaultMetric="iv" defaultType="BOTH" />
---------------------------------------------------------------------------- */