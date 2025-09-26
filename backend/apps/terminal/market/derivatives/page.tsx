"use client";

/**
 * app/derivatives/page.tsx
 * Zero-import, self-contained Derivatives dashboard (no links/imports).
 *
 * What you get:
 * - Lightweight mock option surface generator (deterministic per day)
 * - Vol Smile chart (moneyness vs IV%)
 * - Greeks Grid heatmap (expiries × strikes for Δ/Γ/V/Θ)
 * - Simple controls (spot / r / q / greek)
 *
 * Tailwind + inline SVG only. Drop in as-is.
 */

type OptionType = "CALL" | "PUT";
type SurfacePoint = {
  symbol: string;
  expiry: string; // YYYY-MM-DD
  t: number;      // years
  type: OptionType;
  strike: number;
  moneyness: number; // K/F
  iv: number;        // decimal (0.24 = 24%)
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;     // per day
};
type ExpirySlice = { expiry: string; t: number; baseVol: number; points: SurfacePoint[] };
type OptionSurface = { symbol: string; spot: number; rate: number; div: number; ts: string; slices: ExpirySlice[] };

type GreekKey = "delta" | "gamma" | "vega" | "theta";

export default function DerivativesPage() {
  // market params
  const [symbol] = useState("NIFTY");
  const [spot, setSpot] = useState(22500);
  const [r, setR] = useState(0.05);
  const [q, setQ] = useState(0);
  const [greek, setGreek] = useState<GreekKey>("delta");

  // build a compact mock surface
  const surface = useMemo(
    () => makeSurface({ symbol, spot, r, q }),
    [symbol, spot, r, q]
  );

  const tiles = useMemo(() => summarize(surface), [surface]);

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      {/* Header */}
      <div className="sticky top-0 z-20 border-b border-neutral-800 bg-neutral-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
          <h1 className="text-lg font-semibold tracking-tight">Derivatives</h1>
          <div className="flex items-center gap-2 text-xs">
            <Num label="Spot" value={spot} step={50} onChange={(v) => setSpot(safe(v))} />
            <Num label="r" value={r} step={0.005} onChange={(v) => setR(clamp(v, -0.1, 0.5))} />
            <Num label="q" value={q} step={0.005} onChange={(v) => setQ(clamp(v, -0.1, 0.5))} />
          </div>
        </div>
      </div>

      {/* Tiles */}
      <div className="mx-auto grid max-w-7xl grid-cols-2 gap-4 px-4 py-6 md:grid-cols-4">
        <Tile label="Symbol" value={symbol} />
        <Tile label="Spot" value={fmt(surface.spot, 0)} />
        <Tile label="Front IV (ATM)" value={(tiles.frontAtmIv * 100).toFixed(2) + "%"} />
        <Tile label="Skew (25Δ C−P)" value={(tiles.frontSkew25 * 100).toFixed(2) + "%"} accent={tiles.frontSkew25 >= 0 ? "pos" : "neg"} />
      </div>

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 pb-10 lg:grid-cols-2">
        {/* Vol smile panel */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900">
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
            <div>
              <div className="text-sm font-semibold">IV Smile (K/F)</div>
              <div className="text-xs text-neutral-400">
                {surface.slices.length} expiries · Spot {fmt(surface.spot, 0)}
              </div>
            </div>
            <span className="text-xs text-neutral-400">CALL/PUT avg</span>
          </div>
          <VolSmile surface={surface} />
        </div>

        {/* Greeks heatmap */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900">
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
            <div>
              <div className="text-sm font-semibold">Greeks Grid</div>
              <div className="text-xs text-neutral-400">
                Expiries × Strikes
              </div>
            </div>
            <label className="flex items-center gap-2 text-xs text-neutral-400">
              <span>Greek</span>
              <select
                value={greek}
                onChange={(e) => setGreek(e.target.value as GreekKey)}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
              >
                <option value="delta">Δ Delta</option>
                <option value="gamma">Γ Gamma</option>
                <option value="vega">V Vega</option>
                <option value="theta">Θ Theta</option>
              </select>
            </label>
          </div>
          <GreeksHeatmap surface={surface} greek={greek} />
        </div>
      </div>
    </div>
  );
}

/* ============================== Vol Smile ============================== */

function VolSmile({ surface }: { surface: OptionSurface }) {
  // average CALL+PUT by strike moneyness per expiry
  const series = useMemo(() => {
    const out: Array<{ expiry: string; t: number; xs: number[]; ivs: number[] }> = [];
    for (const sl of surface.slices) {
      const map = new Map<number, number[]>();
      for (const p of sl.points) {
        const key = round(p.moneyness, 3);
        const arr = map.get(key) ?? [];
        arr.push(p.iv * 100);
        map.set(key, arr);
      }
      const xs = Array.from(map.keys()).sort((a, b) => a - b);
      const ivs = xs.map((x) => avg(map.get(x)!));
      out.push({ expiry: sl.expiry, t: sl.t, xs, ivs });
    }
    out.sort((a, b) => a.t - b.t);
    return out;
  }, [surface]);

  const xMin = Math.min(...series.flatMap((s) => s.xs));
  const xMax = Math.max(...series.flatMap((s) => s.xs));
  const yMin = Math.min(...series.flatMap((s) => s.ivs));
  const yMax = Math.max(...series.flatMap((s) => s.ivs));
  const padY = (yMax - yMin) * 0.12 || 0.05;
  const [Ymin, Ymax] = [Math.max(0, yMin - padY), yMax + padY];

  const w = 900, h = 300;
  const pad = { l: 52, r: 14, t: 16, b: 28 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);
  const X = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const Y = (v: number) => pad.t + (1 - (v - Ymin) / (Ymax - Ymin || 1)) * innerH;
  const palette = ["#10b981", "#60a5fa", "#f59e0b", "#f97316", "#a78bfa", "#ef4444"];

  return (
    <div className="px-2 py-2">
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
        {/* grid + axes */}
        {yTicks(Ymin, Ymax).map((v, i) => {
          const yy = Y(v);
          return (
            <g key={`y-${i}`}>
              <line x1={pad.l} y1={yy} x2={w - pad.r} y2={yy} stroke="#27272a" strokeDasharray="3 3" />
              <text x={4} y={yy + 4} fontSize="10" fill="#9ca3af">{v.toFixed(2)}%</text>
            </g>
          );
        })}
        {xTicks(xMin, xMax).map((xv, i) => (
          <text key={`x-${i}`} x={X(xv)} y={h - 6} textAnchor="middle" fontSize="10" fill="#9ca3af">
            {xv.toFixed(2)}
          </text>
        ))}

        {/* lines */}
        {series.map((s, i) => (
          <g key={s.expiry}>
            <path d={toPath(s.xs, s.ivs, X, Y)} fill="none" stroke={palette[i % palette.length]} strokeWidth="2" />
            {s.xs.map((xv, j) =>
              j % Math.ceil(s.xs.length / 60 || 1) === 0 ? (
                <circle key={j} cx={X(xv)} cy={Y(s.ivs[j])} r="1.6" fill={palette[i % palette.length]} />
              ) : null
            )}
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap items-center gap-3 px-3 pb-2 text-xs">
        {series.map((s, i) => (
          <span key={s.expiry} className="inline-flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: palette[i % palette.length] }} />
            <span className="text-neutral-300">{s.expiry} ({(s.t * 365).toFixed(0)}d)</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ============================= Greeks Heatmap ============================= */

function GreeksHeatmap({ surface, greek }: { surface: OptionSurface; greek: GreekKey }) {
  // collect unique strikes (2dp)
  const strikes = Array.from(
    new Set(surface.slices.flatMap((sl) => sl.points.map((p) => round(p.strike, 2))))
  ).sort((a, b) => a - b);

  // expiries
  const expiries = surface.slices.map((s) => s.expiry);

  // grid[ei][ki] = avg CALL/PUT greek
  const grid = expiries.map((e) => {
    const row = strikes.map((k) => {
      const pts = surface.slices
        .find((sl) => sl.expiry === e)!
        .points.filter((p) => round(p.strike, 2) === k);
      if (!pts.length) return NaN;
      const v = avg(pts.map((p) => pickGreek(p, greek)));
      return v;
    });
    return row;
  });

  // bounds for diverging color
  let maxAbs = 0;
  for (const r of grid) for (const v of r) if (isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
  maxAbs = Math.max(maxAbs, 1e-12);

  const w = 900, h = 300;
  const pad = { l: 60, r: 12, t: 12, b: 28 };
  const cellW = (w - pad.l - pad.r) / Math.max(1, strikes.length);
  const cellH = (h - pad.t - pad.b) / Math.max(1, expiries.length);

    function greekLabel(greek: string): import("react").ReactNode {
        throw new Error("Function not implemented.");
    }

  return (
    <div className="px-2 py-2">
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
        {/* headers */}
        {strikes.map((k, i) => (
          <text key={`k-${i}`} x={pad.l + i * cellW + cellW / 2} y={h - 6} textAnchor="middle" fontSize="10" fill="#9ca3af">
            {fmt(k, 0)}
          </text>
        ))}
        {expiries.map((e, r) => (
          <text key={`e-${r}`} x={4} y={pad.t + r * cellH + cellH * 0.65} fontSize="10" fill="#9ca3af">
            {e}
          </text>
        ))}

        {/* cells */}
        {grid.map((row, r) =>
          row.map((v, c) => {
            const t = (v + maxAbs) / (2 * maxAbs); // [-M..M] → [0..1]
            const fill = diverge(t);
            const x = pad.l + c * cellW;
            const y = pad.t + r * cellH;
              function formatGreek(greek: string, v: number): import("react").ReactNode {
                  throw new Error("Function not implemented.");
              }

            return (
              <g key={`${r}-${c}`}>
                <rect x={x} y={y} width={Math.max(0, cellW - 1)} height={Math.max(0, cellH - 1)} fill={fill} />
                <text
                  x={x + cellW - 4}
                  y={y + cellH - 6}
                  fontSize="10"
                  textAnchor="end"
                  fill={contrastText(fill)}
                >
                  {formatGreek(greek, v)}
                </text>
              </g>
            );
          })
        )}

        {/* legend */}
        <text x={4} y={12} fontSize="10" fill="#9ca3af">
          {greekLabel(greek)} (signed scale)
        </text>
        <rect x={140} y={6} width={120} height={8} fill="url(#grad)" />
        <defs>
          <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}

/* ============================== Surface Builder ============================== */

function makeSurface({
  symbol,
  spot,
  r,
  q,
}: {
  symbol: string;
  spot: number;
  r: number;
  q: number;
}): OptionSurface {
  // expiries in days
  const ds = [7, 14, 30, 60];
  const ks = linspace(0.9, 1.1, 13); // K/F grid
  const dayKey = new Date().toISOString().slice(0, 10);
  const rng = mulberry32(hash(`${symbol}|${dayKey}|${Math.round(spot)}|${Math.round(r*10000)}|${Math.round(q*10000)}`));

  const slices: ExpirySlice[] = ds.map((d) => {
    const t = Math.max(1 / 365, d / 365);
    const fwd = forward(spot, r, q, t);
    const base = clamp(0.22 * Math.pow(t, -0.18), 0.06, 2.0);
    const pts: SurfacePoint[] = [];

    for (const x of ks) {
      const K = fwd * x;
      const klog = Math.log(Math.max(1e-12, x)); // log-moneyness
      let iv = base * (1 + (-0.10) * klog + 0.30 * klog * klog) + (rng() - 0.5) * 0.01;
      iv = clamp(iv, 0.05, 3.0);

      // CALL + PUT
      const call = bs("CALL", spot, K, r, q, iv, t);
      const put = bs("PUT", spot, K, r, q, iv, t);

      pts.push(
        mkPt(symbol, addDays(d), t, "CALL", K, fwd, x, iv, call),
        mkPt(symbol, addDays(d), t, "PUT", K, fwd, x, iv, put)
      );
    }

    return { expiry: addDays(d), t, baseVol: base, points: pts };
  });

  return { symbol, spot, rate: r, div: q, ts: new Date().toISOString(), slices };
}

function mkPt(
  symbol: string,
  expiry: string,
  t: number,
  type: OptionType,
  K: number,
  F: number,
  x: number,
  iv: number,
  g: Greeks
): SurfacePoint {
  return {
    symbol, expiry, t, type,
    strike: round(K, 2),
    moneyness: x,
    iv: round(iv, 6),
    price: round(g.price, 6),
    delta: round(g.delta, 6),
    gamma: round(g.gamma, 8),
    vega: round(g.vega, 6),
    theta: round(g.theta, 6),
  };
}

/* ============================== Pricing & Math ============================== */

type Greeks = { price: number; delta: number; gamma: number; vega: number; theta: number };

function bs(type: OptionType, S: number, K: number, r: number, q: number, vol: number, t: number): Greeks {
  S = safe(S); K = safe(K); vol = clamp(vol, 1e-6, 5); t = clamp(t, 1e-6, 100);
  const sqrtT = Math.sqrt(t), sigT = vol * sqrtT;
  const d1 = (Math.log(S / K) + (r - q + 0.5 * vol * vol) * t) / sigT;
  const d2 = d1 - sigT;
  const dfR = Math.exp(-r * t), dfQ = Math.exp(-q * t);
  const Nd1 = cnd(d1), Nd2 = cnd(d2), pdfd1 = pdf(d1);

  const call = dfQ * S * Nd1 - dfR * K * Nd2;
  const put  = dfR * K * cnd(-d2) - dfQ * S * cnd(-d1);
  const price = type === "CALL" ? call : put;
  const delta = type === "CALL" ? dfQ * Nd1 : dfQ * (Nd1 - 1);
  const gamma = (dfQ * pdfd1) / (S * sigT);
  const vega  = dfQ * S * pdfd1 * sqrtT;
  const thetaAnnual =
    -(dfQ * S * pdfd1 * vol) / (2 * sqrtT)
    - (type === "CALL"
        ? -r * dfR * K * Nd2 + q * dfQ * S * Nd1
        : -r * dfR * K * cnd(-d2) + q * dfQ * S * cnd(-d1));
  const theta = thetaAnnual / 365;
  return { price, delta, gamma, vega, theta };
}

function forward(S: number, r: number, q: number, t: number) { return clamp(S * Math.exp((r - q) * t), 1e-12, 1e12); }
function pdf(x: number) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
function cnd(x: number) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function erf(z: number) {
  const sign = z < 0 ? -1 : 1; const x = Math.abs(z);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return sign * y;
}

/* ============================== Summaries ============================== */

function summarize(surface: OptionSurface) {
  // Front expiry ATM metrics
  const front = surface.slices[0];
  // ATM ~ closest to moneyness 1.0
  const atm = front.points.reduce((best, p) => Math.abs(p.moneyness - 1) < Math.abs(best.moneyness - 1) ? p : best, front.points[0]);
  const iv = atm.iv;
  // crude 25Δ skew proxy: compare moneyness 0.9 vs 1.1 average (CALL - PUT avg IV)
  const kLo = nearestBy(front.points, (p) => Math.abs(p.moneyness - 0.9));
  const kHi = nearestBy(front.points, (p) => Math.abs(p.moneyness - 1.1));
  const ivCall = avg(front.points.filter((p) => p.type === "CALL" && (p === kLo || p === kHi)).map((p) => p.iv));
  const ivPut  = avg(front.points.filter((p) => p.type === "PUT"  && (p === kLo || p === kHi)).map((p) => p.iv));
  return { frontAtmIv: iv, frontSkew25: ivCall - ivPut };
}

/* ============================== UI atoms ============================== */

function Tile({ label, value, accent = "mut" }: { label: string; value: string; accent?: "pos" | "neg" | "mut" }) {
  const color = accent === "pos" ? "text-emerald-400" : accent === "neg" ? "text-rose-400" : "text-neutral-100";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color}`}>{value}</div>
    </div>
  );
}
function Num({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label?: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="flex items-center gap-1">
      {label && <span className="text-neutral-400">{label}</span>}
      <input
        type="number"
        value={String(value)}
        step={step}
        min={min as any}
        max={max as any}
        onChange={(e) => onChange(safe(parseFloat(e.target.value)))}
        className="w-24 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-xs text-neutral-100"
      />
    </label>
  );
}

/* ============================== Utils & Colors ============================== */

function avg(a: number[]) { return a.reduce((s, v) => s + v, 0) / (a.length || 1); }
function nearestBy<T>(arr: T[], dist: (x: T) => number) { return arr.reduce((b, x) => (dist(x) < dist(b) ? x : b), arr[0]); }
function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function safe(n: number) { return Number.isFinite(n) ? n : 0; }
function round(n: number, d = 2) { const p = 10 ** d; return Math.round(n * p) / p; }
function addDays(d: number) { const t = new Date(); t.setDate(t.getDate() + d); return t.toISOString().slice(0, 10); }
function linspace(a: number, b: number, n: number) { const out: number[] = []; if (n <= 1) return [a]; const step = (b - a) / (n - 1); for (let i=0;i<n;i++) out.push(a + i*step); return out; }

function yTicks(a: number, b: number) { const n = 4, r: number[] = []; for (let i=0;i<=n;i++) r.push(a + ((b - a) * i) / n); return r; }
function xTicks(a: number, b: number) { const n = 6, r: number[] = []; for (let i=0;i<=n;i++) r.push(a + ((b - a) * i) / n); return r.map((x) => round(x, 2)); }
function toPath(xs: number[], ys: number[], X: (x: number) => number, Y: (v: number) => number) {
  if (!xs.length) return ""; let d = `M ${X(xs[0])} ${Y(ys[0])}`; for (let i=1;i<xs.length;i++) d += ` L ${X(xs[i])} ${Y(ys[i])}`; return d;
}

function diverge(t: number) {
  const mix = (a: number, b: number, u: number) => a + (b - a) * Math.max(0, Math.min(1, u));
  const r = mix(239, 16, t), g = mix(68, 185, t), b = mix(68, 129, t);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
function contrastText(rgb: string) {
  const m = rgb.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/i);
  if (!m) return "#e5e7eb";
  const r = +m[1], g = +m[2], b = +m[3];
  const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return L > 140 ? "#111827" : "#e5e7eb";
}

// deterministic RNG for mocks
function mulberry32(seed: number) { let t = seed >>> 0; return () => { t += 0x6D2B79F5; let x = Math.imul(t ^ (t >>> 15), 1 | t); x ^= x + Math.imul(x ^ (x >>> 7), 61 | x); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; }; }
function hash(s: string) { let h = 2166136261 >>> 0; for (let i=0;i<s.length;i++){ h ^= s.charCodeAt(i); h = Math.imul(h, 16777619);} return h >>> 0; }
 
/* ---------------- Ambient React (keep zero imports) ---------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;

function pickGreek(p: SurfacePoint, greek: string): any {
    throw new Error("Function not implemented.");
}
