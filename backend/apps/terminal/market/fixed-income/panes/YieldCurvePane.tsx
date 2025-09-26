"use client";

/**
 * yieldcurvepane.tsx
 * Zero-import, self-contained yield-curve pane with comparison, spreads, tooltips, CSV.
 * Tailwind + inline SVG only. No links/imports.
 *
 * Props:
 *  - curves: { label: string; currency?: string; points: { tenor: string; years: number; yield: number }[] }[]
 *      • Provide 1–3 curves (e.g., "US", "DE", "GB"). Points can be unsorted; years in decimal.
 *  - title?: string = "Yield Curves"
 *  - height?: number = 320
 *  - showTable?: boolean = true
 *  - baseIndex?: number = 0   // which curve is the base for spreads
 */

export type CurvePoint = { tenor: string; years: number; yield: number }; // yield in decimal (0.0425 = 4.25%)
export type CurveSeries = { label: string; currency?: string; points: CurvePoint[] };

export default function YieldCurvePane({
  curves = [],
  title = "Yield Curves",
  className = "",
  height = 320,
  showTable = true,
  baseIndex = 0,
}: {
  curves: CurveSeries[];
  title?: string;
  className?: string;
  height?: number;
  showTable?: boolean;
  baseIndex?: number;
}) {
  /* ------------------------------- Normalize ------------------------------- */

  const series = useMemo(() => {
    return (curves || []).map((c) => {
      const pts = (c.points || [])
        .map((p) => ({ years: +p.years, yield: +p.yield, tenor: p.tenor || fmtTenor(+p.years) }))
        .filter((p) => Number.isFinite(p.years) && Number.isFinite(p.yield))
        .sort((a, b) => a.years - b.years);
      return { label: c.label, currency: c.currency || "", color: color(c.label), pts };
    });
  }, [curves]);

  // Union tenor grid across curves (for spread table/CSV)
  const grid = useMemo(() => {
    const u = new Set<number>();
    for (const s of series) for (const p of s.pts) u.add(round(p.years, 6));
    return Array.from(u).sort((a, b) => a - b);
  }, [series]);

  // Interpolators for each curve
  const interpolators = useMemo(() => {
    return series.map((s) => {
      const xs = s.pts.map((p) => p.years);
      const ys = s.pts.map((p) => p.yield);
      return (x: number) => interp(xs, ys, x);
    });
  }, [series]);

  // Sampled values on grid for each series
  const sampled = useMemo(() => {
    return series.map((s, i) => ({
      ...s,
      tenors: grid,
      yields: grid.map((x) => interpolators[i](x)),
    }));
  }, [series, grid, interpolators]);

  // Spreads vs base curve
  const base = sampled[baseIndex];
  const spreads = useMemo(() => {
    if (!base) return [];
    return sampled.map((s, i) => {
      if (i === baseIndex) return { label: s.label, values: s.yields.map(() => 0) };
      return { label: s.label, values: s.yields.map((y, k) => y - base.yields[k]) };
    });
  }, [sampled, base, baseIndex]);

  /* --------------------------------- Layout -------------------------------- */

  const h = Math.max(240, height);
  const w = 980;
  const pad = { l: 56, r: 16, t: 16, b: 44 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);

  const xMin = grid[0] ?? 0;
  const xMax = grid.at(-1) ?? 30;

  // Y bounds from all series yields
  const allY = sampled.flatMap((s) => s.yields).filter((v) => Number.isFinite(v));
  const yMin = Math.min(...(allY.length ? allY : [0]));
  const yMax = Math.max(...(allY.length ? allY : [0.05]));
  const padY = (yMax - yMin) * 0.12 || 0.005;
  const y0 = yMin - padY;
  const y1 = yMax + padY;

  const X = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const Y = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0 || 1)) * innerH;

  const yTicks = niceTicks(y0, y1, 5);
  const xTicks = niceTenorTicks(xMin, xMax);

  // Build path for each series
  const paths = sampled.map((s) => ({
    label: s.label,
    color: s.color,
    d: toPath(s.tenors, s.yields, X, Y),
  }));

  /* -------------------------------- Tooltip -------------------------------- */

  const [tip, setTip] = useState<{ x: number; y: number; years: number } | null>(null);

  const tipHTML = useMemo(() => {
    if (!tip) return "";
    const t = clamp(tip.years, xMin, xMax);
    const lines: string[] = [];
    lines.push(`<div class="text-neutral-200 font-medium">${fmtTenor(t)}</div>`);
    for (let i = 0; i < sampled.length; i++) {
      const s = sampled[i];
      const y = interpolators[i](t);
      if (!Number.isFinite(y)) continue;
      const diff = base ? y - interp(sampled[baseIndex].tenors, sampled[baseIndex].yields, t) : NaN;
      const diffTxt = base && i !== baseIndex && Number.isFinite(diff) ? ` <span class="text-neutral-500">(${fmtBps(diff)})</span>` : "";
      lines.push(
        `<div class="flex items-center gap-2"><span class="inline-block h-2 w-2 rounded-sm" style="background:${s.color}"></span><span class="text-neutral-300">${esc(s.label)}</span><span class="ml-auto font-mono text-neutral-200">${fmtPct(y)}${diffTxt}</span></div>`
      );
    }
    return lines.join("");
  }, [tip, sampled, interpolators, base, baseIndex, xMin, xMax]);

  /* --------------------------------- Render -------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {series.length} curve{series.length!==1?"s":""} · Base: {series[baseIndex]?.label ?? "—"}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => copyCSV(sampled, grid)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 hover:bg-neutral-800"
          >
            Copy CSV
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="relative p-3">
        <svg
          width="100%"
          viewBox={`0 0 ${w} ${h}`}
          className="block"
          onMouseMove={(e) => {
            const svg = (e.target as SVGElement).closest("svg")!;
            const r = svg.getBoundingClientRect();
            const mx = e.clientX - r.left;
            const tx = xMin + ((mx - pad.l) / (innerW || 1)) * (xMax - xMin);
            setTip({ x: mx, y: e.clientY - r.top, years: tx });
          }}
          onMouseLeave={() => setTip(null)}
        >
          {/* Y grid + ticks */}
          {yTicks.map((v, i) => (
            <g key={`gy-${i}`}>
              <line x1={pad.l} y1={Y(v)} x2={w - pad.r} y2={Y(v)} stroke="#27272a" strokeDasharray="3 3" />
              <text x={4} y={Y(v) + 4} fill="#9ca3af" fontSize="10">{fmtPct(v)}</text>
            </g>
          ))}

          {/* Curves */}
          {paths.map((p, i) => (
            <path key={i} d={p.d} fill="none" stroke={p.color} strokeWidth="2" />
          ))}

          {/* Dots on actual points */}
          {series.map((s) =>
            s.pts.map((p, i) => (
              <circle key={`${s.label}-${i}`} cx={X(p.years)} cy={Y(p.yield)} r="2.5" fill={color(s.label)} />
            ))
          )}

          {/* X ticks */}
          {xTicks.map((t, i) => (
            <text key={`tx-${i}`} x={X(t)} y={h - 10} textAnchor="middle" fontSize="10" fill="#9ca3af">
              {fmtTenorShort(t)}
            </text>
          ))}

          {/* Axis line */}
          <line x1={pad.l} y1={pad.t + innerH} x2={w - pad.r} y2={pad.t + innerH} stroke="#3f3f46" />

          {/* Crosshair */}
          {tip && (
            <line x1={X(clamp(tip.years, xMin, xMax))} y1={pad.t} x2={X(clamp(tip.years, xMin, xMax))} y2={pad.t + innerH} stroke="#52525b" strokeDasharray="4 4" />
          )}
        </svg>

        {/* Legend */}
        <div className="mt-2 flex flex-wrap items-center gap-3 px-1 text-xs">
          {series.map((s, i) => (
            <div key={s.label} className="inline-flex items-center gap-2">
              <span className="h-2 w-2 rounded-sm" style={{ background: color(s.label) }} />
              <span className={`text-neutral-300 ${i === baseIndex ? "font-semibold" : ""}`}>
                {s.label}{s.currency ? ` · ${s.currency}` : ""}{i===baseIndex ? " (base)" : ""}
              </span>
            </div>
          ))}
        </div>

        {/* Tooltip */}
        {tip && (
          <div
            className="pointer-events-none absolute z-50 max-w-xs rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 shadow"
            style={{ left: Math.min(w - 180, Math.max(8, tip.x + 12)), top: Math.max(8, tip.y + 12) }}
            dangerouslySetInnerHTML={{ __html: tipHTML }}
          />
        )}
      </div>

      {/* Table (optional) */}
      {showTable && (
        <div className="overflow-x-auto border-t border-neutral-800">
          <table className="w-full text-sm">
            <thead className="bg-neutral-950 text-[11px] uppercase text-neutral-500">
              <tr className="border-b border-neutral-800">
                <th className="px-3 py-2 text-left">Tenor</th>
                {sampled.map((s) => (
                  <th key={s.label} className="px-3 py-2 text-right">{s.label}</th>
                ))}
                {sampled.length > 1 && <th className="px-3 py-2 text-right">Spread vs {series[baseIndex]?.label}</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              {grid.map((x, r) => (
                <tr key={x} className="hover:bg-neutral-800/40">
                  <td className="px-3 py-2">{fmtTenor(x)}</td>
                  {sampled.map((s, i) => (
                    <td key={i} className="px-3 py-2 text-right font-mono tabular-nums">{fmtPct(s.yields[r])}</td>
                  ))}
                  {sampled.length > 1 && (
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {spreads[baseIndex] ? "—" : null}
                      {spreads[baseIndex] ? null : null}
                      {/* show first non-base spread as example aggregate: if 2+ comps, show “…” */}
                      {sampled.length >= 2 ? fmtBps(sampled[1].yields[r] - sampled[baseIndex].yields[r]) : "—"}
                      {sampled.length > 2 ? " …" : ""}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------- Utils ---------------------------------- */

function fmtTenor(y: number) { return y < 1 ? `${Math.round(y * 12)}M` : `${round(y)}Y`; }
function fmtTenorShort(y: number) { return y < 1 ? `${Math.round(y*12)}M` : `${round(y)}Y`; }

function toPath(xs: number[], ys: number[], X: (x: number) => number, Y: (y: number) => number) {
  let d = "";
  for (let i = 0; i < xs.length; i++) {
    const cmd = i === 0 ? "M" : "L";
    d += `${cmd} ${X(xs[i])} ${Y(ys[i])} `;
  }
  return d.trim();
}

function interp(xs: number[], ys: number[], x: number) {
  if (!xs.length) return NaN;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (i < xs.length && xs[i] < x) i++;
  const x0 = xs[i - 1], x1 = xs[i], y0 = ys[i - 1], y1 = ys[i];
  const t = (x - x0) / (x1 - x0 || 1);
  return y0 + (y1 - y0) * t;
}

function niceTicks(min: number, max: number, n = 5) {
  if (!(max > min)) return [min];
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / Math.max(1, n))));
  const err = (span / n) / step0;
  const mult = err >= 7.5 ? 10 : err >= 3 ? 5 : err >= 1.5 ? 2 : 1;
  const step = mult * step0;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

function niceTenorTicks(x0: number, x1: number) {
  const buckets = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30];
  return buckets.filter((t) => t >= x0 - 1e-9 && t <= x1 + 1e-9);
}

function copyCSV(sampled: { label: string; tenors: number[]; yields: number[] }[], grid: number[]) {
  const head = ["tenor", ...sampled.map((s) => s.label)].join(",");
  const lines = [head];
  for (let i = 0; i < grid.length; i++) {
    const row = [tenorCSV(grid[i]), ...sampled.map((s) => String(s.yields[i]))];
    lines.push(row.map(csv).join(","));
  }
  const csvStr = lines.join("\n");
  try { (navigator as any).clipboard?.writeText(csvStr); } catch {}
}
function tenorCSV(y: number) { return y < 1 ? `${Math.round(y*12)}M` : `${Math.round(y)}Y`; }

function fmtPct(x: number) {
  return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%";
}
function fmtBps(x: number) {
  return (x * 10000).toLocaleString("en-US", { maximumFractionDigits: 0 }) + " bps";
}
function round(n: number, d = 0) { const p = 10 ** d; return Math.round(n * p) / p; }
function clamp(x: number, a: number, b: number) { return Math.max(a, Math.min(b, x)); }
function csv(x: any) { if (x == null) return ""; const s = String(x); return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s; }
function color(label: string) { const h = hash(label) % 360; return `hsl(${h}, 70%, 55%)`; }
function hash(s: string) { let h=2166136261>>>0; for (let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0; }
function esc(s: string) { return s.replace(/[&<>"']/g,(m)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]!)); }

/* ------------------------ Ambient React (no imports) ------------------------ */
declare const React: any;
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];

/* ---------------------------------- Example ---------------------------------
import { fetchCurves } from "@/server/fetchcurves.server";
const snap = await fetchCurves({ countries: ["US","DE"], tenors: [0.25,0.5,1,2,3,5,7,10,20,30] });
<YieldCurvePane
  curves={snap.series.map(s => ({ label: s.country, currency: s.currency, points: s.points }))}
  baseIndex={0}
/>
-------------------------------------------------------------------------------- */