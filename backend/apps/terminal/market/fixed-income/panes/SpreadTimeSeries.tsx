"use client";

/**
 * spreadtimeseries.tsx
 * Zero-import, self-contained multi-series spread time-series chart.
 *
 * Features
 * - Multiple line series (e.g., "BBB – Gov", "HY – Gov")
 * - Date range filter, optional moving-average smoothing
 * - Hover crosshair + tooltip (values & diffs vs baseline)
 * - Autoscaled Y with nice ticks, compact legend, CSV export
 * - Tailwind + inline SVG only (no imports, no links)
 */

export type SpreadPoint = { t: string; value: number }; // t = ISO "YYYY-MM-DD" or full ISO
export type SpreadSeries = { label: string; points: SpreadPoint[] };

export default function SpreadTimeSeries({
  series = [],
  title = "Credit Spreads",
  className = "",
  height = 320,
  smoothDefault = 0, // moving average window in days (0=off)
  baselineIndex = 0, // series index to compute diffs against in tooltip
}: {
  series: SpreadSeries[];
  title?: string;
  className?: string;
  height?: number;
  smoothDefault?: number;
  baselineIndex?: number;
}) {
  /* ------------------------------ State/Derived ------------------------------ */

  // Normalize & sort points by time for each series
  const clean = useMemo(() => {
    return series.map((s) => ({
      label: s.label,
      color: color(s.label),
      points: (s.points || [])
        .map((p) => ({ x: +new Date(p.t), y: Number(p.value) }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
        .sort((a, b) => a.x - b.x),
    }));
  }, [series]);

  const allTimes = useMemo(
    () => dedupe(clean.flatMap((s) => s.points.map((p) => p.x))).sort((a, b) => a - b),
    [clean]
  );

  // Initial date range: full span
  const [from, setFrom] = useState<string>(() => (allTimes[0] ? iso(allTimes[0]) : ""));
  const [to, setTo] = useState<string>(() => (allTimes.at(-1) ? iso(allTimes.at(-1)!) : ""));
  const [win, setWin] = useState<number>(smoothDefault);

  // Clamp range
  const tMin = useMemo(() => (from ? +new Date(from) : (allTimes[0] ?? 0)), [from, allTimes]);
  const tMax = useMemo(() => (to ? +new Date(to) : (allTimes.at(-1) ?? 1)), [to, allTimes]);

  // Re-sample each series to the union time grid in range (forward-fill)
  const grid = useMemo(() => allTimes.filter((t) => t >= tMin && t <= tMax), [allTimes, tMin, tMax]);

  const sampled = useMemo(() => {
    return clean.map((s) => {
      const res: { x: number; y: number }[] = [];
      let i = 0;
      for (const gx of grid) {
        while (i + 1 < s.points.length && s.points[i + 1].x <= gx) i++;
        const y = s.points[i]?.x <= gx ? s.points[i].y : NaN;
        res.push({ x: gx, y });
      }
      return { label: s.label, color: s.color, points: res };
    });
  }, [clean, grid]);

  // Optional moving average smoothing (simple)
  const smoothed = useMemo(() => {
    if (!win || win < 2) return sampled;
    return sampled.map((s) => ({
      ...s,
      points: movAvg(s.points, win),
    }));
  }, [sampled, win]);

  // Y domain
  const yVals = smoothed.flatMap((s) => s.points.map((p) => p.y)).filter((v) => Number.isFinite(v));
  const yMin = Math.min(...(yVals.length ? yVals : [0]));
  const yMax = Math.max(...(yVals.length ? yVals : [1]));
  const pad = (yMax - yMin) * 0.08 || 0.5;
  const y0 = yMin - pad;
  const y1 = yMax + pad;

  // X domain
  const x0 = grid[0] ?? 0;
  const x1 = grid.at(-1) ?? 1;

  // Ticks
  const yTicks = niceTicks(y0, y1, 5);
  const xTicks = timeTicks(x0, x1, 6);

  /* --------------------------------- Layout --------------------------------- */

  const w = 980;
  const h = Math.max(220, height);
  const padL = 56, padR = 16, padT = 16, padB = 40;
  const innerW = Math.max(1, w - padL - padR);
  const innerH = Math.max(1, h - padT - padB);

  const X = (t: number) => padL + ((t - x0) / (x1 - x0 || 1)) * innerW;
  const Y = (v: number) => padT + (1 - (v - y0) / (y1 - y0 || 1)) * innerH;

  // Paths
  const paths = smoothed.map((s) => ({
    label: s.label,
    color: s.color,
    d: toPath(s.points.map((p) => p.x), s.points.map((p) => p.y), X, Y, true),
  }));

  /* -------------------------------- Tooltip -------------------------------- */

  const [tip, setTip] = useState<{ x: number; y: number; t: number } | null>(null);

  // Closest grid index to mouse x
  const nearestIdx = (tx: number) => {
    if (!grid.length) return -1;
    let lo = 0, hi = grid.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (grid[mid] < tx) lo = mid; else hi = mid;
    }
    return Math.abs(grid[lo] - tx) < Math.abs(grid[hi] - tx) ? lo : hi;
  };

  const baseline = smoothed[baselineIndex]?.points ?? [];

  const tipHTML = useMemo(() => {
    if (!tip) return "";
    const idx = nearestIdx(tip.t);
    const dt = grid[idx];
    if (idx < 0 || dt == null) return "";
    const lines: string[] = [];
    lines.push(`<div class="text-neutral-200 font-medium">${fmtDate(dt)}</div>`);
    for (const s of smoothed) {
      const y = s.points[idx]?.y;
      if (!Number.isFinite(y)) continue;
      const diff = Number.isFinite(baseline[idx]?.y) ? (y - baseline[idx].y) : NaN;
      const diffTxt = Number.isFinite(diff) ? ` <span class="text-neutral-500">(${fmtBps(diff)})</span>` : "";
      lines.push(
        `<div class="flex items-center gap-2"><span class="inline-block h-2 w-2 rounded-sm" style="background:${s.color}"></span><span class="text-neutral-300">${esc(s.label)}</span><span class="ml-auto text-neutral-200 font-mono">${fmtBps(y)}${diffTxt}</span></div>`
      );
    }
    return lines.join("");
  }, [tip, smoothed, grid, baseline]);

  /* --------------------------------- Render --------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {series.length} series · {grid.length} pts · {win ? `MA(${win})` : "raw"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="flex items-center gap-1">
            <span className="text-neutral-400">From</span>
            <input
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              className="rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
            />
          </label>
          <label className="flex items-center gap-1">
            <span className="text-neutral-400">To</span>
            <input
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              className="rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-neutral-400">Smooth</span>
            <input
              type="number"
              min={0}
              step={1}
              value={win}
              onChange={(e) => setWin(Math.max(0, Math.round(+e.target.value || 0)))}
              className="w-16 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
              title="Moving average window (days)"
            />
          </label>
          <button
            onClick={() => copyCSV(smoothed, grid)}
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
            const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const tx = x0 + ((mx - padL) / (innerW || 1)) * (x1 - x0);
            setTip({ x: mx, y: e.clientY - rect.top, t: tx });
          }}
          onMouseLeave={() => setTip(null)}
        >
          {/* Y grid & ticks */}
          {yTicks.map((v, i) => (
            <g key={`gy-${i}`}>
              <line x1={padL} y1={Y(v)} x2={w - padR} y2={Y(v)} stroke="#27272a" strokeDasharray="3 3" />
              <text x={4} y={Y(v) + 4} fill="#9ca3af" fontSize="10">{fmtBps(v)}</text>
            </g>
          ))}

          {/* Series paths */}
          {paths.map((p, i) => (
            <path key={i} d={p.d} fill="none" stroke={p.color} strokeWidth="2" />
          ))}

          {/* X ticks */}
          {xTicks.map((t, i) => (
            <text key={`tx-${i}`} x={X(t)} y={h - 10} textAnchor="middle" fontSize="10" fill="#9ca3af">
              {fmtTickDate(t, x0, x1)}
            </text>
          ))}

          {/* Crosshair */}
          {tip && (
            <>
              <line x1={X(clamp(tip.t, x0, x1))} y1={padT} x2={X(clamp(tip.t, x0, x1))} y2={padT + innerH} stroke="#52525b" strokeDasharray="4 4" />
            </>
          )}

          {/* X axis line */}
          <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} stroke="#3f3f46" />
        </svg>

        {/* Legend */}
        <div className="mt-2 flex flex-wrap items-center gap-3 px-1 text-xs">
          {smoothed.map((s, i) => (
            <div key={i} className="inline-flex items-center gap-2">
              <span className="h-2 w-2 rounded-sm" style={{ background: s.color }} />
              <span className={`text-neutral-300 ${i === baselineIndex ? "font-semibold" : ""}`}>{s.label}{i===baselineIndex?" (base)":""}</span>
            </div>
          ))}
        </div>

        {/* Tooltip */}
        {tip && (
          <div
            className="pointer-events-none absolute z-50 max-w-xs rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 shadow"
            style={{ left: Math.min(w - 160, Math.max(8, tip.x + 12)), top: Math.max(8, tip.y + 12) }}
            dangerouslySetInnerHTML={{ __html: tipHTML }}
          />
        )}
      </div>
    </div>
  );
}

/* ---------------------------------- Utils ---------------------------------- */

function movAvg(pts: { x: number; y: number }[], k: number) {
  const n = Math.max(1, Math.floor(k));
  if (n <= 1) return pts;
  const out: { x: number; y: number }[] = [];
  let sum = 0, cnt = 0;
  const q: number[] = [];
  for (let i = 0; i < pts.length; i++) {
    const y = pts[i].y;
    if (Number.isFinite(y)) { q.push(y); sum += y; cnt++; }
    else { q.push(NaN); }
    if (q.length > n) {
      const old = q.shift()!;
      if (Number.isFinite(old)) { sum -= old; cnt--; }
    }
    out.push({ x: pts[i].x, y: cnt > 0 ? sum / cnt : NaN });
  }
  return out;
}

function toPath(xs: number[], ys: number[], X: (x: number) => number, Y: (y: number) => number, skipNaN = false) {
  let d = "", started = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (!Number.isFinite(y)) { if (skipNaN) { started = false; continue; } else continue; }
    const cmd = started ? "L" : "M";
    d += `${cmd} ${X(xs[i])} ${Y(y)} `;
    started = true;
  }
  return d.trim();
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

function timeTicks(x0: number, x1: number, n = 6) {
  if (!(x1 > x0)) return [x0];
  const span = x1 - x0;
  const step = Math.round(span / n);
  const out: number[] = [];
  for (let t = x0; t <= x1 + 1; t += step) out.push(t);
  return out;
}

function fmtDate(t: number) {
  const d = new Date(t);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}
function fmtTickDate(t: number, x0: number, x1: number) {
  const d = new Date(t);
  const sameYear = new Date(x0).getFullYear() === new Date(x1).getFullYear();
  const y = d.getFullYear();
  const m = d.toLocaleString("en-US", { month: "short" });
  const day = String(d.getDate()).padStart(2, "0");
  return sameYear ? `${m} ${day}` : `${m} ${y}`;
}

function fmtBps(x: number) { return (x * 10000).toLocaleString("en-US", { maximumFractionDigits: 0 }) + " bps"; }
function iso(t: number) { return new Date(t).toISOString().slice(0, 10); }
function esc(s: string) { return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]!)); }
function clamp(x: number, a: number, b: number) { return Math.max(a, Math.min(b, x)); }

function copyCSV(series: { label: string; color: string; points: { x: number; y: number }[] }[], grid: number[]) {
  const head = ["date", ...series.map((s) => s.label)].join(",");
  const lines = [head];
  for (let i = 0; i < grid.length; i++) {
    const row = [iso(grid[i]), ...series.map((s) => (Number.isFinite(s.points[i]?.y) ? String(s.points[i].y) : ""))];
    lines.push(row.map(csv).join(","));
  }
  const csvStr = lines.join("\n");
  try { (navigator as any).clipboard?.writeText(csvStr); } catch {}
}

function csv(x: any) {
  if (x == null) return "";
  if (typeof x === "number") return String(x);
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function dedupe<T>(a: T[]) { return Array.from(new Set(a)); }
function color(label: string) {
  const h = hash(label) % 360;
  return `hsl(${h}, 70%, 55%)`;
}
function hash(s: string) { let h = 2166136261 >>> 0; for (let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0; }

/* ------------------------ Ambient React (no imports) ------------------------ */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;

/* ---------------------------------- Example ---------------------------------
const s1 = { label: "BBB – Gov", points: [...Array(120)].map((_,i)=>({ t: new Date(Date.now()- (120-i)*86400000).toISOString().slice(0,10), value: 0.015 + Math.sin(i/15)/100 }))};
const s2 = { label: "HY – Gov",  points: [...Array(120)].map((_,i)=>({ t: new Date(Date.now()- (120-i)*86400000).toISOString().slice(0,10), value: 0.045 + Math.cos(i/18)/80 }))};
<SpreadTimeSeries series={[s1,s2]} smoothDefault={5} />
-------------------------------------------------------------------------------- */