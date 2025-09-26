"use client";

/**
 * volsmilepane.tsx (fixed + self-contained)
 * - No external imports; includes local <Select> and <ExpMulti> helpers
 * - Toggle CALL / PUT / BOTH
 * - X-axis: Strike or Moneyness (K/F)
 * - Choose which expiries to show
 * - Tailwind + inline SVG only
 */

type OptionType = "CALL" | "PUT";
type AxisMode = "moneyness" | "strike";

type SurfacePoint = {
  symbol: string;
  expiry: string; // YYYY-MM-DD
  t: number;
  type: OptionType;
  strike: number;
  moneyness: number; // K/F
  iv: number;        // decimal 0.22 = 22%
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
};
type ExpirySlice = { expiry: string; t: number; baseVol: number; points: SurfacePoint[] };
export type OptionSurface = { symbol: string; spot: number; rate: number; div: number; ts: string; slices: ExpirySlice[] };

export default function VolSmilePane({
  surface,
  defaultType = "CALL" as OptionType | "BOTH",
  defaultAxis = "moneyness" as AxisMode,
  className = "",
}: {
  surface: OptionSurface;
  defaultType?: OptionType | "BOTH";
  defaultAxis?: AxisMode;
  className?: string;
}) {
  const expiriesAll = surface.slices.map((s) => s.expiry).sort();

  const [otype, setOtype] = useState<OptionType | "BOTH">(defaultType);
  const [axis, setAxis] = useState<AxisMode>(defaultAxis);
  const [activeExps, setActiveExps] = useState<string[]>(expiriesAll.slice(0, Math.max(1, Math.min(4, expiriesAll.length))));
  const [hover, setHover] = useState<{ x: number; y: number; exp: string } | null>(null);

  // width observer (kept super-safe for TS)
  const hostRef = useRef<any>(null);
  const [w, setW] = useState(900);
  useEffect(() => {
    const el = hostRef.current as HTMLElement | null;
    if (!el) return;
    const set = () => setW(Math.max(360, Math.floor(el.clientWidth)));
    set();
    // @ts-ignore – allow missing lib DOM types
    const ro: any = typeof ResizeObserver !== "undefined" ? new ResizeObserver(set) : null;
    ro?.observe(el);
    return () => ro?.disconnect?.();
  }, []);

  // Build series per expiry
  const series = useMemo(() => buildSeries(surface, otype, axis, activeExps), [surface, otype, axis, activeExps]);

  // Bounds
  const xMin = Math.min(...series.flatMap((s) => s.xs));
  const xMax = Math.max(...series.flatMap((s) => s.xs));
  const yMin = Math.min(...series.flatMap((s) => s.ivs));
  const yMax = Math.max(...series.flatMap((s) => s.ivs));
  const padY = (yMax - yMin) * 0.12 || 0.05;
  const [Ymin, Ymax] = [Math.max(0, yMin - padY), yMax + padY];

  // Layout
  const h = 320;
  const pad = { l: 52, r: 14, t: 16, b: 26 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);
  const X = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const Y = (v: number) => pad.t + (1 - (v - Ymin) / (Ymax - Ymin || 1)) * innerH;

  const palette = ["#10b981", "#60a5fa", "#f59e0b", "#f97316", "#e879f9", "#22c55e", "#a78bfa", "#ef4444"];

  return (
    <div ref={hostRef} className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{surface.symbol} Vol Smile</h3>
          <p className="text-xs text-neutral-400">
            Spot {fmt(surface.spot, 0)} · {series.length} expiry{series.length !== 1 ? "ies" : ""} · Type {otype}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
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
            label="X-Axis"
            value={axis}
            onChange={(v) => setAxis(v as AxisMode)}
            options={[
              { value: "moneyness", label: "Moneyness (K/F)" },
              { value: "strike", label: "Strike (K)" },
            ]}
          />
          <ExpMulti
            all={expiriesAll}
            selected={activeExps}
            onToggle={(e) =>
              setActiveExps((prev) => (prev.includes(e) ? prev.filter((x) => x !== e) : [...prev, e]))
            }
          />
        </div>
      </div>

      {/* Chart */}
      <div className="px-2 py-2">
        <svg
          width="100%"
          viewBox={`0 0 ${w} ${h}`}
          className="block"
          onMouseMove={(e) => {
            const rect = (e.currentTarget as any).getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            // pick nearest point across all series
            let best: { x: number; y: number; exp: string } | null = null;
            let bestDist = Infinity;
            for (const s of series) {
              for (let i = 0; i < s.xs.length; i++) {
                const cx = X(s.xs[i]);
                const cy = Y(s.ivs[i]);
                const d = (cx - px) ** 2 + (cy - py) ** 2;
                if (d < bestDist) {
                  bestDist = d;
                  best = { x: cx, y: cy, exp: s.expiry };
                }
              }
            }
            setHover(best);
          }}
          onMouseLeave={() => setHover(null)}
        >
          {/* grid + axes */}
          {yTicks(Ymin, Ymax).map((v, i) => {
            const yy = Y(v);
            return (
              <g key={`y-${i}`}>
                <line x1={pad.l} y1={yy} x2={w - pad.r} y2={yy} stroke="#27272a" strokeDasharray="3 3" />
                <text x={4} y={yy + 4} fontSize="10" fill="#9ca3af">
                  {(v * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
          {xTicks(xMin, xMax, axis).map((xv, i) => {
            const xx = X(xv);
            return (
              <text key={`x-${i}`} x={xx} y={h - 6} textAnchor="middle" fontSize="10" fill="#9ca3af">
                {axis === "moneyness" ? xv.toFixed(2) : fmt(xv, 0)}
              </text>
            );
          })}

          {/* lines + dots */}
          {series.map((s, i) => (
            <g key={s.expiry}>
              <path d={toPath(s.xs, s.ivs, X, Y)} fill="none" stroke={palette[i % palette.length]} strokeWidth="2" />
              {s.xs.map((xv, j) =>
                j % Math.ceil(s.xs.length / 60 || 1) === 0 ? (
                  <circle key={j} cx={X(xv)} cy={Y(s.ivs[j])} r="1.7" fill={palette[i % palette.length]} />
                ) : null
              )}
            </g>
          ))}

          {/* hover */}
          {hover && (
            <>
              <line x1={hover.x} y1={pad.t} x2={hover.x} y2={h - pad.b} stroke="#6b7280" strokeDasharray="4 4" />
              <circle cx={hover.x} cy={hover.y} r="3" fill="#10b981" stroke="#0a0a0a" strokeWidth="1" />
            </>
          )}
        </svg>
      </div>

      {/* Legend + readout */}
      <div className="flex flex-wrap items-center justify-between border-t border-neutral-800 px-4 py-2 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          {series.map((s, i) => (
            <span key={s.expiry} className="inline-flex items-center gap-2">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ backgroundColor: palette[i % palette.length] }}
              />
              <span className={activeExps.includes(s.expiry) ? "text-neutral-200" : "text-neutral-500"}>
                {s.expiry} ({(s.t * 365).toFixed(0)}d)
              </span>
            </span>
          ))}
        </div>
        <div className="text-neutral-400">{hover ? "Hover point selected" : "Hover to inspect"}</div>
      </div>
    </div>
  );
}

/* -------------------------- data → series builder -------------------------- */

function buildSeries(
  surface: OptionSurface,
  otype: OptionType | "BOTH",
  axis: AxisMode,
  onlyExpiries: string[]
): Array<{ expiry: string; t: number; xs: number[]; ivs: number[] }> {
  const out: Array<{ expiry: string; t: number; xs: number[]; ivs: number[] }> = [];
  for (const sl of surface.slices) {
    if (!onlyExpiries.includes(sl.expiry)) continue;
    const pts = sl.points.filter((p) => otype === "BOTH" || p.type === otype);
    const map = new Map<number, number[]>();
    for (const p of pts) {
      const key = axis === "moneyness" ? round(p.moneyness, 3) : round(p.strike, 2);
      const arr = map.get(key) ?? [];
      arr.push(p.iv * 100); // store in %
      map.set(key, arr);
    }
    const xs = Array.from(map.keys()).sort((a, b) => a - b);
    const ivs = xs.map((x) => avg(map.get(x)!));
    const cleanXs: number[] = [], cleanIvs: number[] = [];
    for (let i = 0; i < xs.length; i++) if (isFinite(ivs[i])) { cleanXs.push(xs[i]); cleanIvs.push(ivs[i]); }
    out.push({ expiry: sl.expiry, t: sl.t, xs: cleanXs, ivs: cleanIvs });
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

/* --------------------------------- helpers -------------------------------- */

function yTicks(a: number, b: number) { const n = 4, r: number[] = []; for (let i=0;i<=n;i++) r.push(a+((b-a)*i)/n); return r; }
function xTicks(a: number, b: number, axis: AxisMode) {
  const n = 6, r: number[] = []; for (let i=0;i<=n;i++) r.push(a+((b-a)*i)/n);
  return axis === "moneyness" ? r.map((x) => round(x, 2)) : r.map((x) => round(x, 0));
}
function toPath(xs: number[], ys: number[], X: (x: number) => number, Y: (v: number) => number) {
  if (!xs.length) return ""; let d = `M ${X(xs[0])} ${Y(ys[0])}`; for (let i=1;i<xs.length;i++) d += ` L ${X(xs[i])} ${Y(ys[i])}`; return d;
}
function avg(a: number[]) { return a.reduce((s, v) => s + v, 0) / (a.length || 1); }
function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function round(n: number, d = 2) { const p = 10 ** d; return Math.round(n * p) / p; }

/* -------------------------- Local UI helpers (fixed) -------------------------- */

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
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-200"
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

function ExpMulti({
  all,
  selected,
  onToggle,
}: {
  all: string[];
  selected: string[];
  onToggle: (exp: string) => void;
}) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-neutral-400">Expiries</span>
      <div className="flex flex-wrap items-center gap-1">
        {all.map((e) => {
          const on = selected.includes(e);
          return (
            <button
              key={e}
              onClick={() => onToggle(e)}
              className={`rounded border px-2 py-1 text-xs ${
                on
                  ? "border-emerald-600 bg-emerald-600/20 text-emerald-300"
                  : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:bg-neutral-800/60"
              }`}
              title={e}
            >
              {e.slice(5)}{/* MM-DD */}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------- Ambient React (keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useRef<T>(v: T | null): { current: T | null };