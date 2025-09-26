"use client";

/**
 * panes/greeks-grid.tsx
 * Zero-import, dependency-free heatmap grid for option greeks.
 *
 * Input: an option surface (like the one from fetchoptionsurface.server.ts)
 * Renders strikes × expiries grid with a selectable greek (Δ, Γ, Vega, Θ).
 * - Tailwind only, inline SVG for legend, no external libs
 * - Filters: Option type (CALL/PUT/BOTH), Greek, normalize mode
 * - Keyboard: ←/→ cycles greek, ↑/↓ cycles type
 */

type OptionType = "CALL" | "PUT";

type SurfacePoint = {
  symbol: string;
  expiry: string; // YYYY-MM-DD
  t: number;
  type: OptionType;
  strike: number;
  moneyness: number;
  iv: number;
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
};

type ExpirySlice = {
  expiry: string;
  t: number;
  baseVol: number;
  points: SurfacePoint[];
};

export type OptionSurface = {
  symbol: string;
  spot: number;
  rate: number;
  div: number;
  ts: string;
  slices: ExpirySlice[];
};

type GreekKey = "delta" | "gamma" | "vega" | "theta";
type Norm = "global" | "per-expiry" | "signed"; // signed = symmetric diverging

export default function GreeksGridPane({
  surface,
  defaultGreek = "delta",
  defaultType = "CALL" as OptionType | "BOTH",
  normalize = "signed" as Norm,
  className = "",
}: {
  surface: OptionSurface;
  defaultGreek?: GreekKey;
  defaultType?: OptionType | "BOTH";
  normalize?: Norm;
  className?: string;
}) {
  const [greek, setGreek] = useState<GreekKey>(defaultGreek);
  const [otype, setOtype] = useState<OptionType | "BOTH">(defaultType);
  const [norm, setNorm] = useState<Norm>(normalize);

  // flatten surface -> rows of points
  const rows = useMemo(() => flattenPoints(surface, otype), [surface, otype]);

  const strikes = useMemo(() => {
    const s = Array.from(new Set(rows.map((r) => tidy(r.strike)))).sort((a, b) => a - b);
    return s;
  }, [rows]);

  const expiries = useMemo(() => {
    const e = Array.from(new Set(rows.map((r) => r.expiry))).sort();
    return e;
  }, [rows]);

  const grid = useMemo(() => {
    // matrix[expiryIdx][strikeIdx] = value
    const map = new Map<string, Map<number, SurfacePoint[]>>();
    expiries.forEach((e) => map.set(e, new Map()));
    for (const r of rows) {
      const e = map.get(r.expiry)!;
      const k = tidy(r.strike);
      const list = e.get(k) ?? [];
      list.push(r);
      e.set(k, list);
    }
    // choose value per cell (avg if BOTH)
    const vals: number[][] = expiries.map(() => strikes.map(() => NaN));
    for (let ei = 0; ei < expiries.length; ei++) {
      const e = expiries[ei];
      const m = map.get(e)!;
      for (let ki = 0; ki < strikes.length; ki++) {
        const k = strikes[ki];
        const ps = m.get(k) ?? [];
        if (ps.length === 0) continue;
        const v = avgGreek(ps, greek);
        vals[ei][ki] = v;
      }
    }
    return vals;
  }, [rows, greek, strikes, expiries]);

  // color scale bounds
  const bounds = useMemo(() => computeBounds(grid, greek, norm), [grid, greek, norm]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName ?? "")) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const order: GreekKey[] = ["delta", "gamma", "vega", "theta"];
        const idx = order.indexOf(greek);
        const next = order[(idx + (e.key === "ArrowRight" ? 1 : 3)) % 4];
        setGreek(next);
      }
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const order: Array<OptionType | "BOTH"> = ["CALL", "PUT", "BOTH"];
        const idx = order.indexOf(otype);
        const next = order[(idx + (e.key === "ArrowUp" ? 1 : 2)) % 3];
        setOtype(next);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [greek, otype]);

    function contrastText(color: string) {
        throw new Error("Function not implemented.");
    }

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header / controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">
            {surface.symbol} Greeks Grid
          </h3>
          <p className="text-xs text-neutral-400">
            Spot: {fmt(surface.spot)} · Strikes: {strikes.length} · Expiries: {expiries.length}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            label="Greek"
            value={greek}
            onChange={(v) => setGreek(v as GreekKey)}
            options={[
              { value: "delta", label: "Δ Delta" },
              { value: "gamma", label: "Γ Gamma" },
              { value: "vega", label: "V Vega" },
              { value: "theta", label: "Θ Theta" },
            ]}
          />
          <Select
            label="Type"
            value={otype}
            onChange={(v) => setOtype(v as OptionType | "BOTH")}
            options={[
              { value: "CALL", label: "CALL" },
              { value: "PUT", label: "PUT" },
              { value: "BOTH", label: "BOTH (avg)" },
            ]}
          />
          <Select
            label="Scale"
            value={norm}
            onChange={(v) => setNorm(v as Norm)}
            options={[
              { value: "signed", label: "Signed (±)" },
              { value: "global", label: "Global [0..max]" },
              { value: "per-expiry", label: "Per-expiry [0..max]" },
            ]}
          />
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-3 px-4 pt-3 text-[11px] text-neutral-400">
        <span>Legend:</span>
        <Legend gradient={norm === "signed" ? "diverge" : "mono"} />
        <span className="ml-auto">
          {legendText(bounds, greek, norm)}
        </span>
      </div>

      {/* Grid */}
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
                  const val = grid[ei][ki];
                  const color = isNaN(val) ? "transparent" : colorFor(val, grid, bounds, ei, greek, norm);
                  const text =
                    isNaN(val) ? "—" :
                    greek === "delta" ? val.toFixed(2) :
                    greek === "gamma" ? sci(val) :
                    greek === "vega" ? shortNum(val) :
                    greek === "theta" ? val.toFixed(3) :
                    val.toFixed(3);
                  const txtColor = contrastText(color);
                  return (
                    <td
  key={`${e}-${k}`}
  className="px-1 py-1 text-right align-middle"
  style={{ backgroundColor: color as string, color: txtColor as unknown as string }}
  title={`${e} @ ${fmt(k, 0)} → ${(greek as string).toUpperCase()}: ${text}`}
>
  <span className="inline-block min-w-[56px]">{text}</span>
</td>
                  );
                })}
              </tr>
            ))}
            {expiries.length === 0 && (
              <tr>
                <td className="px-3 py-6 text-center text-neutral-400" colSpan={strikes.length + 1}>
                  No data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* -------------------------------- helpers -------------------------------- */

function flattenPoints(surface: OptionSurface, otype: OptionType | "BOTH"): SurfacePoint[] {
  const arr: SurfacePoint[] = [];
  for (const sl of surface.slices) {
    for (const p of sl.points) {
      if (otype === "BOTH" || p.type === otype) arr.push(p);
    }
  }
  return arr;
}

function avgGreek(ps: SurfacePoint[], g: GreekKey): number {
  if (!ps.length) return NaN;
  const s = ps.reduce((acc, p) => acc + (p[g] as number), 0);
  return s / ps.length;
}

function sci(n: number): string {
  // compact gamma display
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a >= 0.01 && a < 1000) return n.toFixed(4);
  const e = Math.floor(Math.log10(a));
  const m = n / 10 ** e;
  return `${m.toFixed(2)}e${e}`;
}

function shortNum(n: number): string {
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (a >= 1e9) return `${sign}${(a / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${sign}${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${sign}${(a / 1e3).toFixed(2)}K`;
  if (a >= 1) return `${sign}${a.toFixed(2)}`;
  if (a >= 0.01) return `${sign}${a.toFixed(4)}`;
  return `${sign}${a.toExponential(2)}`;
}

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-US", { maximumFractionDigits: d });
}

function tidy(n: number) {
  return Math.round(n * 100) / 100; // 2dp strikes
}

/* ------------------------------ bounds & colors ----------------------------- */

type Bounds = { min: number; max: number; maxAbs: number; perExpiryMax?: number[] };

function computeBounds(grid: number[][], greek: GreekKey, norm: Norm): Bounds {
  let min = Infinity, max = -Infinity, maxAbs = 0;
  for (const row of grid) {
    for (const v of row) {
      if (!isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  }
  if (!isFinite(min)) min = 0;
  if (!isFinite(max)) max = 0;
  if (norm === "per-expiry") {
    const per = grid.map((row) =>
      row.reduce((m, v) => (isFinite(v) ? Math.max(m, Math.abs(v)) : m), 0)
    );
    return { min, max, maxAbs, perExpiryMax: per.map((x) => (x <= 0 ? 1 : x)) };
  }
  return { min, max, maxAbs: Math.max(maxAbs, 1e-12) };
}

function colorFor(
  v: number,
  grid: number[][],
  b: Bounds,
  ei: number,
  greek: GreekKey,
  norm: Norm
): string {
  if (!isFinite(v)) return "transparent";
  if (norm === "global") {
    // 0..max → green scale
    const m = Math.max(b.max, 1e-12);
    const t = clamp01(v / m);
    return monoGreen(t);
  }
  if (norm === "per-expiry") {
    const m = b.perExpiryMax?.[ei] ?? b.maxAbs;
    const t = clamp01((v + m) / (2 * m));
    return diverge(t);
  }
  // signed, global symmetric
  const m = Math.max(b.maxAbs, 1e-12);
  const t = clamp01((v + m) / (2 * m));
  return diverge(t);
}

function legendText(b: Bounds, greek: GreekKey, norm: Norm) {
  if (norm === "global") {
    return `0 → max (${formatGreekUnit(greek, b.max)})`;
  }
  return `−max ↔ +max (${formatGreekUnit(greek, b.maxAbs)})`;
}

function formatGreekUnit(g: GreekKey, sample: number) {
  switch (g) {
    case "delta": return "unitless";
    case "gamma": return "per 1% move (approx)";
    case "vega":  return "per 1.00 vol";
    case "theta": return "per day";
  }
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
    <label className="flex items-center gap-2 text-xs text-neutral-400">
      <span>{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
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
              ? "linear-gradient(90deg, #ef4444, #f59e0b, #10b981)"
              : "linear-gradient(90deg, #0b132b, #10b981)",
        }}
      />
      <span className="text-[10px] text-neutral-500">max</span>
    </div>
  );
}

/* -------------------------------- colors -------------------------------- */

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// Diverging Red→Amber→Green
function diverge(t: number) {
  // t in [0,1]; 0=red, .5=amber, 1=green
  const r = mix(239, 16, t);   // 0.0->239 (red), 1.0->16 (green)
  const g = mix(68, 185, t);
  const b = mix(68, 129, t);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
// Monochrome green  (0=deep slate, 1=emerald)
function monoGreen(t: number) {
  const r = mix(11, 16, t);
  const g = mix(19, 185, t);
  const b = mix(43, 129, t);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
function mix(a: number, b: number, t: number) {
  return a + (b - a) * clamp01(t);
}

/* ------------------- Ambient React (keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;

/* ---------------------------- Example wiring ----------------------------
import { fetchOptionSurface } from "@/server/fetchoptionsurface.server";

const surface = await fetchOptionSurface({ symbol: "NIFTY", spot: 22500 });
<GreeksGridPane surface={surface} defaultGreek="gamma" defaultType="BOTH" />
--------------------------------------------------------------------------- */