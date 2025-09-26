"use client";

/**
 * perpbasispane.tsx
 * Zero-import, dependency-free pane to visualize PERP vs SPOT basis.
 * - Tailwind + inline SVG only (no Recharts, no icon libs)
 * - Filters: exchange, symbol
 * - Summary tiles: Latest Basis %, 24h Avg %, Premium/Discount
 * - Line chart of basis % over time (+ optional secondary series: spot/perp price)
 * - Recent table
 *
 * Data options:
 *  A) Provide `basisPct` directly as DECIMAL (e.g., 0.0123 = 1.23%)
 *  B) Or provide `perp` and `spot` and we compute basis: (perp/spot - 1)
 */

type PerpBasisRow = {
  ts: string;          // ISO timestamp
  exchange: string;    // "Binance", "Bybit", "OKX", ...
  symbol: string;      // "BTCUSDT", "ETHUSDT"
  // Either give basisPct OR perp+spot (we’ll compute basis if missing)
  basisPct?: number;   // DECIMAL (0.01 = 1%)
  perp?: number;       // perp price
  spot?: number;       // spot price (or index)
};

export type PerpBasisPaneProps = {
  title?: string;
  data: PerpBasisRow[];
  height?: number;              // chart height
  defaultExchange?: string;
  defaultSymbol?: string;
  showPrices?: boolean;         // overlay spot/perp on a compact axis (right)
  compact?: boolean;
  className?: string;
};

export default function PerpBasisPane({
  title = "Perp Basis (Perp vs Spot)",
  data,
  height = 240,
  defaultExchange,
  defaultSymbol,
  showPrices = false,
  compact = false,
  className = "",
}: PerpBasisPaneProps) {
  // state (filters)
  const [ex, setEx] = useState(defaultExchange || "");
  const [sym, setSym] = useState(defaultSymbol || "");

  // lists
  const exchanges = useMemo(() => Array.from(new Set(data.map(d => d.exchange))).sort(), [data]);
  const symbols = useMemo(
    () => Array.from(new Set(data.filter(d => !ex || d.exchange === ex).map(d => d.symbol))).sort(),
    [data, ex]
  );

  // auto defaults
  useEffect(() => {
    if (!ex && exchanges.length) setEx(exchanges[0]);
  }, [exchanges]);
  useEffect(() => {
    if (!sym && symbols.length) setSym(symbols[0]);
  }, [symbols]);

  // rows for selection (compute basis if needed)
  const rows = useMemo(() => {
    return data
      .filter(d => (!ex || d.exchange === ex) && (!sym || d.symbol === sym))
      .map((r) => {
        const basis = r.basisPct != null
          ? r.basisPct
          : (r.perp != null && r.spot != null && r.spot !== 0)
            ? (r.perp / r.spot) - 1
            : 0;
        return { ...r, basisPct: basis };
      })
      .sort((a, b) => +new Date(a.ts) - +new Date(b.ts));
  }, [data, ex, sym]);

  // stats
  const latest = rows.at(-1) || null;

  const last24 = useMemo(() => {
    if (!latest) return [] as typeof rows;
    const cutoff = +new Date(latest.ts) - 24 * 3600e3;
    return rows.filter(r => +new Date(r.ts) >= cutoff);
  }, [rows, latest]);

  const avg24 = useMemo(() => {
    if (!last24.length) return 0;
    return last24.reduce((s, r) => s + (r.basisPct ?? 0), 0) / last24.length;
  }, [last24]);

  const premiumState = latest
    ? latest.basisPct! >= 0 ? "PERP > SPOT (Premium)" : "PERP < SPOT (Discount)"
    : "—";

  // chart series
  const labels = rows.map(r => shortLabel(r.ts));
  const basisPctSeries = rows.map(r => (r.basisPct ?? 0) * 100); // in %
  const spotSeries = showPrices ? rows.map(r => r.spot ?? 0) : undefined;
  const perpSeries = showPrices ? rows.map(r => r.perp ?? 0) : undefined;

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
          label="Latest Basis"
          value={latest ? fmtPct(latest.basisPct ?? 0, 4) : "—"}
          accent={latest ? ((latest.basisPct ?? 0) >= 0 ? "pos" : "neg") : "mut"}
        />
        <Tile
          label="24h Avg"
          value={fmtPct(avg24, 4)}
          accent={avg24 >= 0 ? "pos" : "neg"}
        />
        <Tile
          label="State"
          value={premiumState}
          accent={latest ? ((latest.basisPct ?? 0) >= 0 ? "pos" : "neg") : "mut"}
        />
      </div>

      {/* Chart */}
      <div className="px-2 pb-2">
        <BasisChart
          width={900}
          height={height}
          basisPct={basisPctSeries}
          labels={labels}
          spot={spotSeries}
          perp={perpSeries}
          showPrices={showPrices}
        />
      </div>

      {/* Table */}
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Time</th>
              <th className="px-3 py-2 text-right font-medium">Basis %</th>
              <th className="px-3 py-2 text-right font-medium">Perp</th>
              <th className="px-3 py-2 text-right font-medium">Spot</th>
              <th className="px-3 py-2 text-left font-medium">Exchange</th>
              <th className="px-3 py-2 text-left font-medium">Symbol</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-24).reverse().map((r, i) => (
              <tr key={`${r.ts}-${i}`} className="border-t border-neutral-800">
                <td className="px-3 py-2">{longLabel(r.ts)}</td>
                <td className={`px-3 py-2 text-right ${(r.basisPct ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {fmtPct(r.basisPct ?? 0, 4)}
                </td>
                <td className="px-3 py-2 text-right">{r.perp != null ? fmtNum(r.perp) : "—"}</td>
                <td className="px-3 py-2 text-right">{r.spot != null ? fmtNum(r.spot) : "—"}</td>
                <td className="px-3 py-2">{r.exchange}</td>
                <td className="px-3 py-2">{r.symbol}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-3 py-6 text-center text-neutral-400">No data</td>
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
}: {
  label: string;
  value: string;
  accent?: "pos" | "neg" | "mut";
}) {
  const color =
    accent === "pos" ? "text-emerald-400" :
    accent === "neg" ? "text-rose-400" : "text-neutral-100";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
      <div className="text-neutral-400 text-xs">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color}`}>{value}</div>
    </div>
  );
}

/* ------------------------------ Chart (SVG) ------------------------------ */

function BasisChart({
  width = 900,
  height = 240,
  basisPct,           // in PERCENT units (e.g., 1.2 means 1.2%)
  labels,
  spot,
  perp,
  showPrices = false,
}: {
  width?: number;
  height?: number;
  basisPct: number[];
  labels?: string[];
  spot?: number[];
  perp?: number[];
  showPrices?: boolean;
}) {
  const w = width;
  const h = height;
  const pad = { l: 44, r: showPrices ? 44 : 12, t: 12, b: 28 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);

  // BASIS scale (symmetric around 0)
  const maxAbs = basisPct.length ? Math.max(...basisPct.map(v => Math.abs(v))) : 1;
  const dom = Math.max(maxAbs, 0.1); // at least ±0.1%
  const zeroY = pad.t + innerH / 2;
  const scaleB = (v: number) => ((v / (dom || 1)) * (innerH / 2 - 2));

  // PRICE scale (optional, right axis, normalized into same vertical area)
  const hasPrice = !!(showPrices && spot && spot.length && perp && perp.length);
  const pMin = hasPrice ? Math.min(...(spot as number[]), ...(perp as number[])) : 0;
  const pMax = hasPrice ? Math.max(...(spot as number[]), ...(perp as number[])) : 1;
  const yPrice = (p: number) => pad.t + (innerH - ((p - pMin) / (pMax - pMin || 1)) * innerH);

  const n = Math.max(1, basisPct.length);
  const x = (i: number) => pad.l + (i * innerW) / Math.max(1, n - 1);

  const pathBasis = basisPct.map((v, i) => `${i ? "L" : "M"} ${x(i).toFixed(2)} ${(zeroY - scaleB(v)).toFixed(2)}`).join(" ");

  const sparseEvery = Math.ceil(n / 80) || 1;

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
      {/* basis grid and ticks */}
      {[-dom, -dom/2, 0, dom/2, dom].map((v, i) => {
        const yy = zeroY - scaleB(v);
        return (
          <g key={i}>
            <line x1={pad.l} y1={yy} x2={w - pad.r} y2={yy} stroke="#27272a" strokeDasharray={v === 0 ? "4 4" : "3 3"} />
            <text x={4} y={yy + 4} fill="#9ca3af" fontSize="10">
              {fmtPct(v / 100)}
            </text>
          </g>
        );
      })}

      {/* basis line */}
      <path d={pathBasis} fill="none" stroke="#10b981" strokeWidth="2" />
      {/* basis dots (sparse) */}
      {basisPct.map((v, i) =>
        (i % sparseEvery === 0) ? (
          <circle key={i} cx={x(i)} cy={zeroY - scaleB(v)} r="1.7" fill="#10b981" />
        ) : null
      )}

      {/* optional price overlays */}
      {hasPrice && (
        <>
          {/* spot */}
          <path
            d={spot!.map((p, i) => `${i ? "L" : "M"} ${x(i).toFixed(2)} ${yPrice(p).toFixed(2)}`).join(" ")}
            fill="none"
            stroke="#93c5fd"
            strokeWidth="1.5"
            opacity="0.9"
          />
          {/* perp */}
          <path
            d={perp!.map((p, i) => `${i ? "L" : "M"} ${x(i).toFixed(2)} ${yPrice(p).toFixed(2)}`).join(" ")}
            fill="none"
            stroke="#fca5a5"
            strokeWidth="1.5"
            opacity="0.9"
          />
          {/* right axis ticks for price */}
          {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
            const val = pMin + (pMax - pMin) * t;
            const yy = yPrice(val);
            return (
              <text key={`pr-${i}`} x={w - pad.r + 2} y={yy + 4} fill="#9ca3af" fontSize="10">
                {fmtNum(val)}
              </text>
            );
          })}
        </>
      )}

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
function fmtPct(x: number, d = 2) {
  return `${(x * 100).toFixed(d)}%`;
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
const mock: PerpBasisRow[] = Array.from({ length: 50 }).map((_, i) => {
  const ts = new Date(now - (49 - i) * 60 * 60e3).toISOString(); // hourly
  const spot = 60000 + Math.sin(i/6) * 800 + Math.random() * 200;
  const perp = spot * (1 + (Math.sin(i/7) * 0.004 - 0.0015)); // small premium/discount
  return { ts, exchange: "Binance", symbol: "BTCUSDT", spot, perp };
});

<PerpBasisPane
  data={mock}
  defaultExchange="Binance"
  defaultSymbol="BTCUSDT"
  showPrices
/>
------------------------------------------------------------------------------- */