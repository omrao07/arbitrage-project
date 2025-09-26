"use client";

/**
 * strategy-builderpane.tsx
 * Zero-import options Strategy Builder pane.
 *
 * Features
 * - Add/Edit legs (CALL/PUT, Buy/Sell, Qty, Strike, Expiry (days), IV)
 * - Live pricing with Black–Scholes (r, q) and simple total Greeks
 * - Payoff chart (expiry payoff solid) + Mark-to-Market P&L now (dashed)
 * - Quick presets (Long Call, Bull Call Spread, Short Strangle)
 *
 * Tailwind only. No 3rd-party libs. Ambient React at bottom to keep imports zero.
 */

type LegType = "CALL" | "PUT";
type Side = "BUY" | "SELL";

type Leg = {
  id: string;
  side: Side;
  type: LegType;
  qty: number;       // positive contract count
  strike: number;
  days: number;      // days to expiry
  iv: number;        // decimal (e.g., 0.22)
};

export default function StrategyBuilderPane({
  spotDefault = 22500,
  rateDefault = 0.05, // r
  divDefault = 0,     // q
  lotSize = 1,        // multiply P&L if your contracts have size
  title = "Options Strategy Builder",
  className = "",
}: {
  spotDefault?: number;
  rateDefault?: number;
  divDefault?: number;
  lotSize?: number;
  title?: string;
  className?: string;
}) {
  // book/market params
  const [spot, setSpot] = useState(spotDefault);
  const [r, setR] = useState(rateDefault);
  const [q, setQ] = useState(divDefault);

  // legs
  const [legs, setLegs] = useState<Leg[]>([
    mkLeg("BUY", "CALL", 1, round(spotDefault * 1.02, 0), 30, 0.22),
  ]);

  // derived totals (now)
  const nowGreeks = useMemo(() => sumGreeks(legs, spot, r, q), [legs, spot, r, q]);
  const nowPnl = useMemo(() => legsValue(legs, spot, r, q) * lotSize, [legs, spot, r, q, lotSize]);

  // payoff arrays for chart
  const chart = useMemo(() => {
    const Smin = Math.max(1, spot * 0.6);
    const Smax = spot * 1.4;
    const n = 160;
    const xs = Array.from({ length: n }, (_, i) => Smin + (i * (Smax - Smin)) / (n - 1));
    const payoff = xs.map((S) => expiryPayoff(legs, S) * lotSize);
    const mtm = xs.map((S) => legsValue(legs, S, r, q) * lotSize);
    const yMin = Math.min(0, ...payoff, ...mtm);
    const yMax = Math.max(0, ...payoff, ...mtm);
    return { xs, payoff, mtm, yMin, yMax };
  }, [legs, spot, r, q, lotSize]);

  // helpers
  const addLeg = (l: Partial<Leg> = {}) =>
    setLegs((p) => [...p, mkLeg(l.side ?? "BUY", l.type ?? "CALL", l.qty ?? 1, l.strike ?? round(spot, 0), l.days ?? 30, l.iv ?? 0.22)]);
  const updateLeg = (id: string, patch: Partial<Leg>) =>
    setLegs((p) => p.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  const removeLeg = (id: string) => setLegs((p) => p.filter((x) => x.id !== id));
  const clear = () => setLegs([]);

  // presets
  const presetLongCall = () => setLegs([mkLeg("BUY", "CALL", 1, round(spot * 1.02, 0), 30, 0.22)]);
  const presetBullCall = () =>
    setLegs([
      mkLeg("BUY", "CALL", 1, round(spot * 0.98, 0), 30, 0.22),
      mkLeg("SELL", "CALL", 1, round(spot * 1.02, 0), 30, 0.21),
    ]);
  const presetShortStrangle = () =>
    setLegs([
      mkLeg("SELL", "PUT", 1, round(spot * 0.95, 0), 21, 0.23),
      mkLeg("SELL", "CALL", 1, round(spot * 1.05, 0), 21, 0.23),
    ]);

  const dens = { pad: "px-4 py-3" };

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* header */}
      <div className={`flex flex-wrap items-center justify-between border-b border-neutral-800 ${dens.pad}`}>
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            Spot {fmt(spot, 2)} · {legs.length} leg{legs.length !== 1 ? "s" : ""} · Lot {lotSize}x
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Num label="Spot" value={spot} onChange={(v) => setSpot(safe(v))} />
          <Num label="r" value={r} step={0.005} onChange={(v) => setR(clamp(v, -0.1, 0.5))} />
          <Num label="q" value={q} step={0.005} onChange={(v) => setQ(clamp(v, -0.1, 0.5))} />
          <div className="ml-2 flex items-center gap-1">
            <Btn onClick={presetLongCall}>Long Call</Btn>
            <Btn onClick={presetBullCall}>Bull Call</Btn>
            <Btn onClick={presetShortStrangle}>Short Strangle</Btn>
          </div>
        </div>
      </div>

      {/* summary tiles */}
      <div className="grid grid-cols-1 gap-3 border-b border-neutral-800 px-4 py-3 sm:grid-cols-3">
        <Tile label="MTM (now)" value={money(nowPnl)} accent={nowPnl >= 0 ? "pos" : "neg"} />
        <Tile label="Delta" value={nowGreeks.delta.toFixed(2)} />
        <Tile label="Vega / Theta" value={`${shortNum(nowGreeks.vega)} / ${nowGreeks.theta.toFixed(3)}`} />
      </div>

      {/* legs table */}
      <div className="px-2 py-2">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <Th>Side</Th>
              <Th>Type</Th>
              <Th className="text-right">Qty</Th>
              <Th className="text-right">Strike</Th>
              <Th className="text-right">Days</Th>
              <Th className="text-right">IV %</Th>
              <Th className="text-right">Price</Th>
              <Th className="text-right">Δ</Th>
              <Th className="text-right">Γ</Th>
              <Th className="text-right">V</Th>
              <Th className="text-right">Θ/day</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {legs.map((l) => {
              const t = Math.max(1 / 365, l.days / 365);
              const g = bs(l.type, spot, l.strike, r, q, l.iv, t);
              const sign = l.side === "BUY" ? 1 : -1;
              return (
                <tr key={l.id} className="border-t border-neutral-800">
                  <Td>
                    <Select
                      value={l.side}
                      options={["BUY", "SELL"]}
                      onChange={(v) => updateLeg(l.id, { side: v as Side })}
                    />
                  </Td>
                  <Td>
                    <Select
                      value={l.type}
                      options={["CALL", "PUT"]}
                      onChange={(v) => updateLeg(l.id, { type: v as LegType })}
                    />
                  </Td>
                  <TdRight>
                    <Num value={l.qty} min={0} step={1} onChange={(v) => updateLeg(l.id, { qty: Math.max(0, Math.round(v)) })} />
                  </TdRight>
                  <TdRight>
                    <Num value={l.strike} onChange={(v) => updateLeg(l.id, { strike: safe(v) })} />
                  </TdRight>
                  <TdRight>
                    <Num value={l.days} step={1} min={0} onChange={(v) => updateLeg(l.id, { days: Math.max(0, Math.round(v)) })} />
                  </TdRight>
                  <TdRight>
                    <Num value={l.iv * 100} step={0.5} onChange={(v) => updateLeg(l.id, { iv: Math.max(0.01, v / 100) })} />
                  </TdRight>
                  <TdRight className={sign * g.price >= 0 ? "text-emerald-400" : "text-rose-400"}>
                    {money(sign * g.price * l.qty * lotSize)}
                  </TdRight>
                  <TdRight>{(sign * g.delta * l.qty).toFixed(2)}</TdRight>
                  <TdRight>{sci(sign * g.gamma * l.qty)}</TdRight>
                  <TdRight>{shortNum(sign * g.vega * l.qty)}</TdRight>
                  <TdRight>{(sign * g.theta * l.qty).toFixed(3)}</TdRight>
                  <Td>
                    <button onClick={() => removeLeg(l.id)} className="rounded border border-neutral-700 px-2 py-1 text-xs text-neutral-300 hover:bg-neutral-800">
                      ✕
                    </button>
                  </Td>
                </tr>
              );
            })}
            {legs.length === 0 && (
              <tr>
                <td colSpan={12} className="px-3 py-6 text-center text-neutral-500">
                  No legs. Add one below.
                </td>
              </tr>
            )}
          </tbody>
        </table>

        {/* add row */}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Btn onClick={() => addLeg({ side: "BUY", type: "CALL" })}>+ Buy Call</Btn>
          <Btn onClick={() => addLeg({ side: "BUY", type: "PUT" })}>+ Buy Put</Btn>
          <Btn onClick={() => addLeg({ side: "SELL", type: "CALL" })}>+ Sell Call</Btn>
          <Btn onClick={() => addLeg({ side: "SELL", type: "PUT" })}>+ Sell Put</Btn>
          <button onClick={clear} className="ml-auto rounded border border-neutral-800 px-3 py-1.5 text-xs text-neutral-400 hover:bg-neutral-800">
            Clear
          </button>
        </div>
      </div>

      {/* chart */}
      <div className="border-t border-neutral-800 px-2 py-3">
        <PayoffChart
          width={900}
          height={280}
          xs={chart.xs}
          payoff={chart.payoff}
          mtm={chart.mtm}
          yMin={chart.yMin}
          yMax={chart.yMax}
          spot={spot}
        />
      </div>
    </div>
  );
}

/* ------------------------------- UI atoms ------------------------------- */

function Tile({ label, value, accent = "mut" }: { label: string; value: string; accent?: "pos" | "neg" | "mut" }) {
  const color = accent === "pos" ? "text-emerald-400" : accent === "neg" ? "text-rose-400" : "text-neutral-100";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color}`}>{value}</div>
    </div>
  );
}

function Btn({ children, onClick }: { children: any; onClick: () => void }) {
  return (
    <button onClick={onClick} className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-xs text-neutral-200 hover:bg-neutral-800">
      {children}
    </button>
  );
}

function Th({ children, className = "" }: any) {
  return <th className={`px-2 py-2 text-left font-medium ${className}`}>{children}</th>;
}
function Td({ children, className = "" }: any) {
  return <td className={`px-2 py-2 ${className}`}>{children}</td>;
}
function TdRight({ children, className = "" }: any) {
  return <td className={`px-2 py-2 text-right ${className}`}>{children}</td>;
}

function Select({
  value,
  options,
  onChange,
}: {
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-200"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function Num({
  label,
  value,
  onChange,
  step = 0.5,
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

/* ------------------------------- Chart SVG ------------------------------- */

function PayoffChart({
  width = 900,
  height = 280,
  xs,
  payoff,
  mtm,
  yMin,
  yMax,
  spot,
}: {
  width?: number;
  height?: number;
  xs: number[];
  payoff: number[];
  mtm: number[];
  yMin: number;
  yMax: number;
  spot: number;
}) {
  const pad = { l: 60, r: 12, t: 12, b: 28 };
  const w = width;
  const h = height;
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);
  const X = (i: number) => pad.l + (i * innerW) / Math.max(1, xs.length - 1);
  const Y = (v: number) => pad.t + (yMax - v) / (yMax - yMin || 1) * innerH;

  const pathPayoff = arrToPath(xs, payoff, X, Y);
  const pathMtm = arrToPath(xs, mtm, X, Y);
  const ticks = 4;
  const yVals = Array.from({ length: ticks + 1 }, (_, i) => yMin + (i * (yMax - yMin)) / ticks);

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
      {/* grid & axes */}
      {yVals.map((v, i) => {
        const yy = Y(v);
        return (
          <g key={i}>
            <line x1={pad.l} y1={yy} x2={w - pad.r} y2={yy} stroke="#27272a" strokeDasharray="3 3" />
            <text x={4} y={yy + 4} fill="#9ca3af" fontSize="10">
              {money(v)}
            </text>
          </g>
        );
      })}
      {/* zero */}
      <line x1={pad.l} y1={Y(0)} x2={w - pad.r} y2={Y(0)} stroke="#6b7280" strokeDasharray="4 4" />

      {/* spot marker */}
      <line x1={X(nearestIdx(xs, spot))} y1={pad.t} x2={X(nearestIdx(xs, spot))} y2={pad.t + innerH} stroke="#93c5fd" strokeDasharray="2 3" />
      <text x={X(nearestIdx(xs, spot)) + 4} y={pad.t + 12} fontSize="10" fill="#93c5fd">
        Spot {fmt(spot, 0)}
      </text>

      {/* paths */}
      <path d={pathPayoff} fill="none" stroke="#10b981" strokeWidth="2" />
      <path d={pathMtm} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="6 4" />

      {/* end labels */}
      <text x={w - pad.r - 4} y={pad.t + 12} textAnchor="end" fontSize="10" fill="#10b981">Payoff @ Expiry</text>
      <text x={w - pad.r - 4} y={pad.t + 24} textAnchor="end" fontSize="10" fill="#f59e0b">Mark-to-Market (now)</text>
    </svg>
  );
}

function arrToPath(xs: number[], ys: number[], X: (i: number) => number, Y: (v: number) => number) {
  if (!xs.length) return "";
  let d = `M ${X(0)} ${Y(ys[0])}`;
  for (let i = 1; i < xs.length; i++) d += ` L ${X(i)} ${Y(ys[i])}`;
  return d;
}
function nearestIdx(xs: number[], s: number) {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - s) < Math.abs(xs[best] - s)) best = i;
  return best;
}

/* ------------------------------ Pricing core ------------------------------ */

type Greeks = { price: number; delta: number; gamma: number; vega: number; theta: number };

function legsValue(legs: Leg[], S: number, r: number, q: number) {
  let v = 0;
  for (const l of legs) {
    const t = Math.max(1 / 365, l.days / 365);
    const g = bs(l.type, S, l.strike, r, q, l.iv, t);
    const sign = l.side === "BUY" ? 1 : -1;
    v += sign * g.price * l.qty;
  }
  return v;
}
function expiryPayoff(legs: Leg[], S: number) {
  let v = 0;
  for (const l of legs) {
    const intrinsic = l.type === "CALL" ? Math.max(0, S - l.strike) : Math.max(0, l.strike - S);
    const sign = l.side === "BUY" ? 1 : -1;
    v += sign * intrinsic * l.qty;
  }
  return v;
}
function sumGreeks(legs: Leg[], S: number, r: number, q: number): Greeks {
  let d = 0, g = 0, v = 0, th = 0, p = 0;
  for (const l of legs) {
    const t = Math.max(1 / 365, l.days / 365);
    const gg = bs(l.type, S, l.strike, r, q, l.iv, t);
    const sgn = l.side === "BUY" ? 1 : -1;
    p += sgn * gg.price * l.qty;
    d += sgn * gg.delta * l.qty;
    g += sgn * gg.gamma * l.qty;
    v += sgn * gg.vega * l.qty;
    th += sgn * gg.theta * l.qty;
  }
  return { price: p, delta: d, gamma: g, vega: v, theta: th };
}

// Black–Scholes (per-day theta)
function bs(type: LegType, S: number, K: number, r: number, q: number, vol: number, t: number): Greeks {
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

function pdf(x: number) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
function cnd(x: number) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
// Abramowitz–Stegun erf approx
function erf(z: number) {
  const sign = z < 0 ? -1 : 1; const x = Math.abs(z);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return sign * y;
}

/* --------------------------------- utils --------------------------------- */

function mkLeg(side: Side, type: LegType, qty: number, strike: number, days: number, iv: number): Leg {
  return { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, side, type, qty, strike, days, iv };
}
function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function money(n: number) { const s = n < 0 ? "-" : ""; const a = Math.abs(n);
  if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(2)}K`;
  return `${s}$${a.toFixed(2)}`; }
function shortNum(n: number) { const a = Math.abs(n), s = n < 0 ? "-" : "";
  if (a >= 1e6) return `${s}${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${s}${(a / 1e3).toFixed(2)}K`;
  if (a >= 1) return `${s}${a.toFixed(2)}`;
  if (a >= 0.01) return `${s}${a.toFixed(4)}`;
  return `${s}${a.toExponential(2)}`; }
function sci(n: number) { const a = Math.abs(n); if (a === 0) return "0"; if (a >= 0.01 && a < 1000) return n.toFixed(4);
  const e = Math.floor(Math.log10(a)); const m = n / 10 ** e; return `${m.toFixed(2)}e${e}`; }
function safe(n: number) { return Number.isFinite(n) ? n : 0; }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function round(n: number, d = 2) { const p = 10 ** d; return Math.round(n * p) / p; }

/* ------------------- Ambient React (to keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;