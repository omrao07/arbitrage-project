"use client";

/**
 * equityorderbook.tsx
 * Zero-import, self-contained Level-2 order book pane for an equity.
 *
 * - Inputs: bids/asks [{price,size}], last, symbol
 * - Shows top-of-book, spread, depth bars, cumulative sizes
 * - Grouping (tick size), depth limit, side alignment toggle
 * - Hover highlight + keyboard: g/G adjust group, +/- depth, a flip align
 * - Tailwind + inline SVG only
 */

export type L2Level = { price: number; size: number };

export default function EquityOrderBook({
  symbol,
  last,
  bids = [],
  asks = [],
  className = "",
  depthDefault = 10,
  groupDefault = 0.01,
  align = "inside" as "inside" | "center",
}: {
  symbol: string;
  last: number;
  bids: L2Level[];
  asks: L2Level[];
  className?: string;
  depthDefault?: number;
  groupDefault?: number;
  align?: "inside" | "center";
}) {
  const [depth, setDepth] = useState(depthDefault);
  const [group, setGroup] = useState(groupDefault);
  const [layout, setLayout] = useState<"inside" | "center">(align);
  const [hover, setHover] = useState<{ side: "bid" | "ask"; i: number } | null>(null);

  const gBids = useMemo(() => groupBook(bids, group, "bid"), [bids, group]);
  const gAsks = useMemo(() => groupBook(asks, group, "ask"), [asks, group]);
  const bookBids = useMemo(() => withCum(gBids.slice(0, depth)), [gBids, depth]);
  const bookAsks = useMemo(() => withCum(gAsks.slice(0, depth)), [gAsks, depth]);

  const bestBid = bookBids[0]?.price ?? NaN;
  const bestAsk = bookAsks[0]?.price ?? NaN;
  const spread = isFinite(bestBid) && isFinite(bestAsk) ? bestAsk - bestBid : NaN;

  const maxSz = Math.max(
    1,
    ...bookBids.map((r) => r.size),
    ...bookAsks.map((r) => r.size),
    ...bookBids.map((r) => r.cum),
    ...bookAsks.map((r) => r.cum)
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "+" || e.key === "=") setDepth((d) => Math.min(50, d + 1));
      if (e.key === "-" || e.key === "_") setDepth((d) => Math.max(1, d - 1));
      if (e.key === "g") setGroup((t) => round(stepTick(t, -1), 4));
      if (e.key === "G") setGroup((t) => round(stepTick(t, +1), 4));
      if (e.key.toLowerCase() === "a") setLayout((s) => (s === "inside" ? "center" : "inside"));
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{symbol} Order Book</h3>
          <p className="text-xs text-neutral-400">
            Last ${fmt(last, 2)}
            {isFinite(spread) && isFinite(bestBid) && isFinite(bestAsk) && (
              <>
                {" · "}Bid {fmt(bestBid, 2)} / Ask {fmt(bestAsk, 2)}{" "}
                <span className="text-neutral-500">({fmt(spread, 3)} spread)</span>
              </>
            )}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="flex items-center gap-2">
            <span className="text-neutral-400">Depth</span>
            <input
              type="number"
              value={String(depth)}
              min={1}
              max={50}
              onChange={(e) => setDepth(clamp(Math.round(+e.target.value || 1), 1, 50))}
              className="w-16 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-neutral-400">Group</span>
            <input
              type="number"
              step="0.01"
              value={String(group)}
              onChange={(e) => setGroup(Math.max(0.0001, +e.target.value || group))}
              className="w-24 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
            />
          </label>
          <button
            onClick={() => setLayout((s) => (s === "inside" ? "center" : "inside"))}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 hover:bg-neutral-800"
            title="Toggle alignment"
          >
            Align: {layout}
          </button>
        </div>
      </div>

      {/* Book grid */}
      <div className="grid grid-cols-1 gap-0 md:grid-cols-2">
        <SideTable side="bid" rows={bookBids} maxSz={maxSz} layout={layout} onHover={setHover} hover={hover} />
        <SideTable side="ask" rows={bookAsks} maxSz={maxSz} layout={layout} onHover={setHover} hover={hover} />
      </div>
    </div>
  );
}

/* ------------------------------ Side table ------------------------------ */

function SideTable({
  side,
  rows,
  maxSz,
  layout,
  onHover,
  hover,
}: {
  side: "bid" | "ask";
  rows: Array<{ price: number; size: number; cum: number }>;
  maxSz: number;
  layout: "inside" | "center";
  onHover: (h: { side: "bid" | "ask"; i: number } | null) => void;
  hover: { side: "bid" | "ask"; i: number } | null;
}) {
  const isBid = side === "bid";
  const headClr = isBid ? "text-emerald-300" : "text-rose-300";
  const barClr = isBid ? "#064e3b" : "#7f1d1d";
  const barClrLite = isBid ? "#065f46" : "#991b1b";
  const textRight = isBid ? "text-right" : "text-left";
  const px = isBid ? "pr-3" : "pl-3";

  const alignStyles =
    layout === "inside"
      ? isBid
        ? "justify-end"
        : "justify-start"
      : "justify-center";

  return (
    <div className="border-t border-neutral-800 md:border-t-0 md:border-l">
      <div className={`flex items-center ${alignStyles} gap-2 border-b border-neutral-800 px-3 py-2 text-xs`}>
        <span className={`${headClr} font-medium`}>{isBid ? "Bids" : "Asks"}</span>
        <span className="text-neutral-500">({rows.length})</span>
      </div>

      <div className="divide-y divide-neutral-800">
        <div className={`grid grid-cols-12 gap-2 px-3 py-2 text-[11px] text-neutral-400 ${textRight}`}>
          <div className={`col-span-4 ${isBid ? "col-start-1" : "order-3"}`}>{isBid ? "Price" : "Size"}</div>
          <div className="col-span-4">{isBid ? "Size" : "Price"}</div>
          <div className={`col-span-4 ${isBid ? "order-3" : "col-start-1"}`}>Cum</div>
        </div>
        {rows.map((r, i) => {
          const pct = clamp(r.size / maxSz, 0, 1);
          const cumPct = clamp(r.cum / maxSz, 0, 1);
          const on = hover?.side === side && hover?.i === i;
          return (
            <div
              key={i}
              className={`relative grid grid-cols-12 items-center gap-2 px-3 py-1.5 text-sm ${textRight}`}
              onMouseEnter={() => onHover({ side, i })}
              onMouseLeave={() => onHover(null)}
              title={`${isBid ? "Bid" : "Ask"} ${fmt(r.price, 2)} · Size ${fmtInt(r.size)} · Cum ${fmtInt(r.cum)}`}
            >
              {/* depth bars */}
              <div
                className="absolute inset-0"
                style={{
                  background: `linear-gradient(${isBid ? "to left" : "to right"}, ${barClrLite} ${pct * 100}%, transparent ${pct * 100}%)`,
                }}
              />
              <div
                className="absolute inset-y-0"
                style={{
                  [isBid ? "right" : "left"]: 0,
                  width: `${cumPct * 100}%`,
                  background: `${barClr}`,
                  opacity: 0.25,
                } as any}
              />
              {/* cells */}
              <div className={`relative col-span-4 ${isBid ? "col-start-1" : "order-3"} ${px} font-mono tabular-nums`}>
                {isBid ? `$${fmt(r.price, 2)}` : fmtInt(r.size)}
              </div>
              <div className="relative col-span-4 font-mono tabular-nums">
                {isBid ? fmtInt(r.size) : `$${fmt(r.price, 2)}`}
              </div>
              <div className={`relative col-span-4 ${isBid ? "order-3" : "col-start-1"} font-mono tabular-nums`}>
                {fmtInt(r.cum)}
              </div>

              {/* row highlight */}
              {on && <div className="pointer-events-none absolute inset-0 border border-neutral-600/60" />}
            </div>
          );
        })}
        {rows.length === 0 && (
          <div className="px-3 py-6 text-center text-xs text-neutral-500">No liquidity.</div>
        )}
      </div>
    </div>
  );
}

/* ----------------------------- helpers/core ----------------------------- */

function groupBook(levels: L2Level[], tick: number, side: "bid" | "ask") {
  if (!Array.isArray(levels) || levels.length === 0) return [];
  const sorted = [...levels].sort((a, b) => (side === "bid" ? b.price - a.price : a.price - b.price));
  const map = new Map<number, number>();
  const factor = 1 / Math.max(0.0001, tick);
  for (const { price, size } of sorted) {
    const bucket = side === "bid" ? Math.floor(price * factor) / factor : Math.ceil(price * factor) / factor;
    map.set(bucket, (map.get(bucket) ?? 0) + Math.max(0, size | 0));
  }
  return Array.from(map.entries())
    .map(([price, size]) => ({ price, size }))
    .sort((a, b) => (side === "bid" ? b.price - a.price : a.price - b.price));
}

function withCum(rows: Array<{ price: number; size: number }>) {
  const out: Array<{ price: number; size: number; cum: number }> = [];
  let s = 0;
  for (const r of rows) { s += r.size; out.push({ ...r, cum: s }); }
  return out;
}

function stepTick(t: number, dir: -1 | 1) {
  const ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1];
  let i = ticks.findIndex((x) => Math.abs(x - t) < 1e-9);
  if (i === -1) { i = ticks.findIndex((x) => x >= t); if (i < 0) i = ticks.length - 1; }
  i = clamp(i + dir, 0, ticks.length - 1);
  return ticks[i];
}

/* -------------------------------- utils -------------------------------- */

function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtInt(n: number) { return n.toLocaleString("en-US"); }
function round(n: number, d = 4) { const p = 10 ** d; return Math.round(n * p) / p; }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }

/* --------------- Ambient React (keep zero imports; no imports) --------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;

/* ------------------------------- Example --------------------------------
<EquityOrderBook
  symbol="AAPL"
  last={189.42}
  bids={[{price:189.40,size:300},{price:189.39,size:500},{price:189.38,size:200}]}
  asks={[{price:189.44,size:250},{price:189.45,size:400},{price:189.46,size:350}]}
/>
--------------------------------------------------------------------------- */