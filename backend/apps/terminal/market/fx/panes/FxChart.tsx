"use client";

/**
 * fxchart.tsx
 * Zero-import, self-contained multi-pair FX price chart.
 *
 * Features
 * - Multiple pairs (EURUSD, USDJPY, …) as line series
 * - Rebase options: Raw, Rebased to 100 (at range start), % Change
 * - Date range filter (From / To)
 * - Optional Moving Average (MA-N) per series
 * - Hover crosshair + tooltip with per-pair values & deltas
 * - CSV export (current view)
 *
 * Tailwind + inline SVG only. No imports/links.
 */

export type FxPoint = { t: string; mid: number }; // ISO date/time + mid
export type FxSeries = { pair: string; points: FxPoint[]; label?: string };

type Mode = "raw" | "rebase" | "pct";

export default function FXChart({
  series = [],
  title = "FX Chart",
  className = "",
  height = 320,
  defaultMode = "raw",
  smoothDefault = 0, // MA window (0 = off)
}: {
  series: FxSeries[];
  title?: string;
  className?: string;
  height?: number;
  defaultMode?: Mode;
  smoothDefault?: number;
}) {
  /* --------------------------- Normalize & options -------------------------- */

  // Canonicalize input -> sorted numeric timestamps
  const clean = useMemo(() => {
    return series.map((s) => ({
      label: s.label || prettyPair(normPair(s.pair) || s.pair),
      pair: normPair(s.pair) || s.pair,
      color: color(s.label || s.pair),
      pts: (s.points || [])
        .map((p) => ({ x: +new Date(p.t), y: Number(p.mid) }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
        .sort((a, b) => a.x - b.x),
    }));
  }, [series]);

  // Union grid of timestamps (so all series align in tooltips/CSV)
  const gridAll = useMemo(
    () => dedupe(clean.flatMap((s) => s.pts.map((p) => p.x))).sort((a, b) => a - b),
    [clean]
  );

  // Controls: range / smoothing / mode
  const [from, setFrom] = useState<string>(() => (gridAll[0] ? iso(gridAll[0]) : ""));
  const [to, setTo] = useState<string>(() => (gridAll.at(-1) ? iso(gridAll.at(-1)!) : ""));
  const [win, setWin] = useState<number>(smoothDefault);
  const [mode, setMode] = useState<Mode>(defaultMode);

  const tMin = useMemo(() => (from ? +new Date(from) : (gridAll[0] ?? 0)), [from, gridAll]);
  const tMax = useMemo(() => (to ? +new Date(to) : (gridAll.at(-1) ?? 1)), [to, gridAll]);
  const grid = useMemo(() => gridAll.filter((t) => t >= tMin && t <= tMax), [gridAll, tMin, tMax]);

  // Forward-fill sample each series on the union grid
  const sampled = useMemo(() => {
    return clean.map((s) => {
      const out: { x: number; y: number }[] = [];
      let i = 0;
      for (const gx of grid) {
        while (i + 1 < s.pts.length && s.pts[i + 1].x <= gx) i++;
        const y = s.pts[i]?.x <= gx ? s.pts[i].y : NaN;
        out.push({ x: gx, y });
      }
      return { label: s.label, pair: s.pair, color: s.color, points: out };
    });
  }, [clean, grid]);

  // Apply MA smoothing (simple moving average ignoring NaNs)
  const smoothed = useMemo(() => {
    if (!win || win < 2) return sampled;
    return sampled.map((s) => ({ ...s, points: movAvg(s.points, win) }));
  }, [sampled, win]);

  // Rebase modes
  const transformed = useMemo(() => {
    if (mode === "raw") return smoothed;
    return smoothed.map((s) => {
      // find first valid y in range
      let base = NaN;
      for (let i = 0; i < s.points.length; i++) {
        if (Number.isFinite(s.points[i].y)) { base = s.points[i].y; break; }
      }
      if (!Number.isFinite(base)) return { ...s };
      if (mode === "rebase") {
        return { ...s, points: s.points.map((p) => ({ x: p.x, y: Number.isFinite(p.y) ? (p.y / base) * 100 : NaN })) };
      } else {
        // % change
        return { ...s, points: s.points.map((p) => ({ x: p.x, y: Number.isFinite(p.y) ? (p.y / base - 1) * 100 : NaN })) };
      }
    });
  }, [smoothed, mode]);

  // Domains
  const yVals = transformed.flatMap((s) => s.points.map((p) => p.y)).filter((v) => Number.isFinite(v));
  const yMin = Math.min(...(yVals.length ? yVals : [0]));
  const yMax = Math.max(...(yVals.length ? yVals : [1]));
  const yPad = (yMax - yMin) * 0.08 || (mode === "raw" ? 0.5 : 1);
  const y0 = yMin - yPad;
  const y1 = yMax + yPad;

  const x0 = grid[0] ?? 0;
  const x1 = grid.at(-1) ?? 1;

  const yTicks = niceTicks(y0, y1, 5);
  const xTicks = timeTicks(x0, x1, 6);

  /* --------------------------------- Layout -------------------------------- */

  const w = 980;
  const h = Math.max(240, height);
  const padL = 56, padR = 16, padT = 16, padB = 40;
  const innerW = Math.max(1, w - padL - padR);
  const innerH = Math.max(1, h - padT - padB);

  const X = (t: number) => padL + ((t - x0) / (x1 - x0 || 1)) * innerW;
  const Y = (v: number) => padT + (1 - (v - y0) / (y1 - y0 || 1)) * innerH;

  // Paths
  const paths = transformed.map((s) => ({
    label: s.label,
    color: s.color,
    pair: s.pair,
    d: toPath(s.points.map((p) => p.x), s.points.map((p) => p.y), X, Y, true),
  }));

  /* -------------------------------- Tooltip -------------------------------- */

  const [tip, setTip] = useState<{ x: number; y: number; t: number } | null>(null);

  const nearestIdx = (tx: number) => {
    if (!grid.length) return -1;
    let lo = 0, hi = grid.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (grid[mid] < tx) lo = mid; else hi = mid;
    }
    return Math.abs(grid[lo] - tx) < Math.abs(grid[hi] - tx) ? lo : hi;
  };

  const tipHTML = useMemo(() => {
    if (!tip) return "";
    const idx = nearestIdx(tip.t);
    if (idx < 0) return "";
    const dt = grid[idx];
    const lines: string[] = [];
    lines.push(`<div class="text-neutral-200 font-medium">${fmtDate(dt)}</div>`);
    for (const s of transformed) {
      const y = s.points[idx]?.y;
      if (!Number.isFinite(y)) continue;
      let val = "";
      if (mode === "raw") val = fmtNum(y, s.pair);
      else if (mode === "rebase") val = `${fmt(y, 2)} (index)`;
      else val = `${fmt(y, 2)}%`;
      // Change vs previous day (in mode space)
      const prev = idx > 0 ? s.points[idx - 1]?.y : NaN;
      const d = Number.isFinite(prev) ? y - (prev as number) : NaN;
      const delta = Number.isFinite(d) ? (mode === "raw" ? fmtSigned(d, s.pair) : (mode === "rebase" ? fmtSigned(d, 2) : fmtSigned(d, 2) + "%")) : "";
      lines.push(
        `<div class="flex items-center gap-2">
          <span class="inline-block h-2 w-2 rounded-sm" style="background:${color(s.label)}"></span>
          <span class="text-neutral-300">${esc(s.label)}</span>
          <span class="ml-auto font-mono text-neutral-200">${val}${delta ? ` <span class="text-neutral-500">(${delta})</span>` : ""}</span>
        </div>`
      );
    }
    return lines.join("");
  }, [tip, transformed, grid, mode]);

  /* --------------------------------- Render -------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header / Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {series.length} pair{series.length!==1?"s":""} · {grid.length} points · {win ? `MA(${win})` : "raw"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="flex items-center gap-1">
            <span className="text-neutral-400">From</span>
            <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} className="rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200" />
          </label>
          <label className="flex items-center gap-1">
            <span className="text-neutral-400">To</span>
            <input type="date" value={to} onChange={(e) => setTo(e.target.value)} className="rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200" />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-neutral-400">MA</span>
            <input
              type="number" min={0} step={1} value={win}
              onChange={(e) => setWin(Math.max(0, Math.round(+e.target.value || 0)))}
              className="w-16 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
              title="Moving average window (points)"
            />
          </label>
          <Select
            label="Mode"
            value={mode}
            onChange={(v) => setMode(v as Mode)}
            options={[
              { value: "raw", label: "Raw" },
              { value: "rebase", label: "Rebased (100)" },
              { value: "pct", label: "% Change" },
            ]}
          />
          <button
            onClick={() => copyCSV(transformed, grid, mode)}
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
            const tx = x0 + ((mx - padL) / (innerW || 1)) * (x1 - x0);
            setTip({ x: mx, y: e.clientY - r.top, t: tx });
          }}
          onMouseLeave={() => setTip(null)}
        >
          {/* Y grid & ticks */}
          {yTicks.map((v, i) => (
            <g key={`gy-${i}`}>
              <line x1={padL} y1={Y(v)} x2={w - padR} y2={Y(v)} stroke="#27272a" strokeDasharray="3 3" />
              <text x={4} y={Y(v) + 4} fill="#9ca3af" fontSize="10">
                {mode === "raw" ? fmtAxis(v) : (mode === "rebase" ? fmt(v, 0) : fmt(v, 1) + "%")}
              </text>
            </g>
          ))}

          {/* Series */}
          {paths.map((p, i) => (
            <path key={i} d={p.d} fill="none" stroke={p.color} strokeWidth="2" />
          ))}

          {/* X ticks */}
          {xTicks.map((t, i) => (
            <text key={`tx-${i}`} x={X(t)} y={h - 10} textAnchor="middle" fontSize="10" fill="#9ca3af">
              {fmtTickDate(t, x0, x1)}
            </text>
          ))}

          {/* Axis baseline */}
          <line x1={padL} y1={padT + innerH} x2={w - padR} y2={padT + innerH} stroke="#3f3f46" />

          {/* Crosshair */}
          {tip && (
            <line x1={X(clamp(tip.t, x0, x1))} y1={padT} x2={X(clamp(tip.t, x0, x1))} y2={padT + innerH} stroke="#52525b" strokeDasharray="4 4" />
          )}
        </svg>

        {/* Legend */}
        <div className="mt-2 flex flex-wrap items-center gap-3 px-1 text-xs">
          {transformed.map((s, i) => (
            <div key={i} className="inline-flex items-center gap-2">
              <span className="h-2 w-2 rounded-sm" style={{ background: color(s.label) }} />
              <span className="text-neutral-300">{s.label}</span>
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
    </div>
  );
}

/* --------------------------------- Select --------------------------------- */

function Select({
  label, value, onChange, options,
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
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
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
    if (Number.isFinite(y)) { q.push(y); sum += y; cnt++; } else { q.push(NaN); }
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

function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtNum(x: number, pair: string) { return x.toLocaleString("en-US", { maximumFractionDigits: dp(pair) }); }
function fmtSigned(x: number, pOrD: string | number) {
  const sign = x >= 0 ? "+" : "";
  if (typeof pOrD === "string") return sign + fmtNum(x, pOrD);
  return sign + fmt(x, pOrD);
}
function fmtAxis(v: number) {
  // compact formatter for raw mode axis (mix pairs -> generic)
  if (Math.abs(v) >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (Math.abs(v) >= 10) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return v.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

function copyCSV(
  ser: { label: string; pair: string; color: string; points: { x: number; y: number }[] }[],
  grid: number[],
  mode: Mode
) {
  const head = ["date", ...ser.map((s) => s.label)].join(",");
  const lines = [head];
  for (let i = 0; i < grid.length; i++) {
    const row = [iso(grid[i]), ...ser.map((s) => (Number.isFinite(s.points[i]?.y) ? String(s.points[i].y) : ""))];
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

function iso(t: number) { return new Date(t).toISOString().slice(0, 10); }
function dedupe<T>(a: T[]) { return Array.from(new Set(a)); }
function color(label: string) { const h = hash(label) % 360; return `hsl(${h}, 70%, 55%)`; }
function hash(s: string) { let h=2166136261>>>0; for (let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0; }
function clamp(x: number, a: number, b: number) { return Math.max(a, Math.min(b, x)); }

function normPair(p: string) {
  const s = (p || "").toUpperCase().replace(/[^A-Z]/g, "");
  return s.length === 6 ? s : "";
}
function prettyPair(p: string) { const n = normPair(p) || p; return `${n.slice(0,3)}/${n.slice(3)}`; }
function dp(pair: string) { return pair.slice(3).toUpperCase() === "JPY" ? 3 : 5; }

/* ------------------------ Ambient React (no imports) ------------------------ */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;


function esc(label: string) {
    throw new Error("Function not implemented.");
}
/* ---------------------------------- Example ---------------------------------
const a = { pair: "EURUSD", points: [...Array(120)].map((_,i)=>({ t: new Date(Date.now()-(120-i)*86400000).toISOString().slice(0,10), mid: 1.08 + Math.sin(i/15)/100 }))};
const b = { pair: "USDJPY", points: [...Array(120)].map((_,i)=>({ t: new Date(Date.now()-(120-i)*86400000).toISOString().slice(0,10), mid: 155 + Math.cos(i/18) }))};
<FXChart series={[a,b]} defaultMode="raw" smoothDefault={5} />
-------------------------------------------------------------------------------- */