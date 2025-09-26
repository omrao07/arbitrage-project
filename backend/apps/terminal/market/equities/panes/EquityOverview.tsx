"use client";

/**
 * equity-overview.tsx
 * Zero-import, self-contained overview pane for an equity.
 *
 * What it shows:
 * - Header with symbol, name (optional), last price & day change
 * - Metric tiles (MCAP, Volume, 52w, P/E, EPS, Beta, Div Yield)
 * - Intraday mini sparkline with VWAP + prevClose guides
 * - Compact peers list (price & %change)
 * - Recent headlines list (title + time ago)
 *
 * No imports, no external links. Tailwind + inline SVG only.
 */

export type Bar = { t: string; o: number; h: number; l: number; c: number; v: number };
export type Quote = {
  price: number; prevClose: number; change: number; changePct: number;
  open: number; dayHigh: number; dayLow: number; bid: number; bidSize: number; ask: number; askSize: number;
  volume: number; vwap: number;
};
export type Fundamentals = {
  marketCap: number; sharesOut: number; pe: number; eps: number; beta: number;
  divYield: number; week52High: number; week52Low: number; currency: string; exchange: string; sector: string; industry: string;
};
export type EquitySnap = {
  symbol: string;
  quote: Quote;
  fundamentals: Fundamentals;
  intraday: Bar[];
  lastUpdated: string;
};

export type EquityPeer = {
  symbol: string; name: string; sector: string; industry: string;
  marketCap: number; price: number; changePct: number; beta: number; pe: number;
};

export type EquityNewsItem = {
  id: string | number;
  title: string;
  source: string;
  publishedAt: string; // ISO
};

export default function EquityOverviewPane({
  snapshot,
  name,
  peers = [],
  headlines = [],
  className = "",
}: {
  snapshot: EquitySnap;
  name?: string;
  peers?: EquityPeer[];
  headlines?: EquityNewsItem[];
  className?: string;
}) {
  const { symbol, quote: q, fundamentals: f, intraday: bars } = snapshot;

  // sparkline data
  const times = bars.map((b) => +new Date(b.t));
  const closes = bars.map((b) => b.c);
  const vols = bars.map((b) => b.v);
  const cumPV = cumSum(bars.map((b) => b.c * b.v));
  const cumV = cumSum(vols);
  const vwap = bars.map((_, i) => (cumV[i] ? cumPV[i] / cumV[i] : NaN));

  // bounds
  const xMin = Math.min(...times, Date.now());
  const xMax = Math.max(...times, Date.now());
  const yMin0 = Math.min(...closes, q.prevClose);
  const yMax0 = Math.max(...closes, q.prevClose);
  const pad = (yMax0 - yMin0) * 0.1 || 0.5;
  const yMin = Math.max(0, yMin0 - pad);
  const yMax = yMax0 + pad;

  // sparkline layout
  const w = 640, h = 160;
  const padL = 48, padR = 10, padT = 10, padB = 20;
  const innerW = Math.max(1, w - padL - padR);
  const innerH = Math.max(1, h - padT - padB);
  const X = (t: number) => padL + ((t - xMin) / (xMax - xMin || 1)) * innerW;
  const Y = (p: number) => padT + (1 - (p - yMin) / (yMax - yMin || 1)) * innerH;
  const pricePath = toPath(times, closes, X, Y);
  const vwapPath = toPath(times, vwap, X, Y, true);

  const pos = q.change >= 0;
  const chColor = pos ? "text-emerald-400" : "text-rose-400";
  const chBg = pos ? "bg-emerald-600/15" : "bg-rose-600/15";

  // peers (limit to 6, sorted by mcap desc)
  const peersView = [...peers].sort((a, b) => b.marketCap - a.marketCap).slice(0, 6);

  // headlines (limit 6)
  const newsView = headlines
    .slice()
    .sort((a, b) => +new Date(b.publishedAt) - +new Date(a.publishedAt))
    .slice(0, 6);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h2 className="truncate text-base font-semibold text-neutral-100">
              {symbol}{name ? ` · ${name}` : ""}
            </h2>
            <span className={`rounded px-2 py-0.5 text-xs ${chBg} ${chColor}`}>
              {fmtPct(q.changePct)} ({q.change >= 0 ? "+" : ""}{fmt(q.change, 2)})
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-400">
            <span className="text-neutral-200 text-sm font-semibold">${fmt(q.price, 2)} {f.currency ? f.currency.toUpperCase() : ""}</span>
            <span>•</span>
            <span>Open {fmt(q.open, 2)}</span>
            <span>•</span>
            <span>Day {fmt(q.dayLow, 2)}–{fmt(q.dayHigh, 2)}</span>
            <span>•</span>
            <span>{f.exchange} · {f.sector}</span>
            <span className="text-neutral-500">Updated {timeAgo(snapshot.lastUpdated)}</span>
          </div>
        </div>
        <div className="text-right">
          <TopOfBook bid={q.bid} bidSize={q.bidSize} ask={q.ask} askSize={q.askSize} />
        </div>
      </div>

      {/* Tiles + Sparkline */}
      <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-3">
        {/* Sparkline */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-sm font-medium text-neutral-200">Intraday</div>
            <div className="text-xs text-neutral-400">VWAP & Prev Close</div>
          </div>
          <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
            {/* y grid */}
            {niceTicks(yMin, yMax, 3).map((p, i) => (
              <g key={`yg-${i}`}>
                <line x1={padL} y1={Y(p)} x2={w - padR} y2={Y(p)} stroke="#27272a" strokeDasharray="3 3" />
                <text x={4} y={Y(p) + 4} fill="#9ca3af" fontSize="10">${fmt(p, 2)}</text>
              </g>
            ))}
            {/* prev close */}
            <line x1={padL} y1={Y(q.prevClose)} x2={w - padR} y2={Y(q.prevClose)} stroke="#f59e0b" strokeDasharray="5 4" />
            {/* price */}
            <path d={pricePath} fill="none" stroke="#60a5fa" strokeWidth="2" />
            {/* vwap */}
            <path d={vwapPath} fill="none" stroke="#10b981" strokeWidth="1.8" strokeDasharray="6 4" />
            {/* last marker */}
            {closes.length > 0 && (
              <>
                <circle cx={X(times.at(-1)!)} cy={Y(closes.at(-1)!)} r="3" fill="#60a5fa" />
                <text x={X(times.at(-1)!)+6} y={Y(closes.at(-1)!)-6} fontSize="10" fill="#cbd5e1">
                  ${fmt(closes.at(-1)!, 2)}
                </text>
              </>
            )}
            {/* x ticks */}
            {timeTicks(xMin, xMax, 4).map((t, i) => (
              <text key={`tx-${i}`} x={X(t)} y={h - 6} textAnchor="middle" fontSize="10" fill="#9ca3af">
                {fmtTime(t)}
              </text>
            ))}
          </svg>
        </div>

        {/* Tiles */}
        <div className="grid grid-cols-2 gap-3">
          <Tile label="Market Cap" value={`$${fmtInt(f.marketCap)}`} />
          <Tile label="Volume" value={fmtInt(q.volume)} />
          <Tile label="52w Range" value={`${fmt(f.week52Low,2)} – ${fmt(f.week52High,2)}`} />
          <Tile label="P/E" value={fmt(f.pe, 2)} />
          <Tile label="EPS (ttm)" value={fmt(f.eps, 2)} />
          <Tile label="Beta" value={fmt(f.beta, 2)} />
          <Tile label="Dividend Yield" value={fmtPct(f.divYield)} />
          <Tile label="Shares Out" value={fmtInt(f.sharesOut)} />
        </div>
      </div>

      {/* Peers + Headlines */}
      <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-2">
        <div className="rounded-xl border border-neutral-800 bg-neutral-950">
          <div className="flex items-center justify-between border-b border-neutral-800 px-3 py-2">
            <div className="text-sm font-medium text-neutral-200">Peers</div>
            <div className="text-xs text-neutral-500">{peersView.length} shown</div>
          </div>
          <div className="divide-y divide-neutral-800">
            <PeerHeader />
            {peersView.length === 0 && (
              <div className="px-3 py-6 text-center text-xs text-neutral-500">No peers provided.</div>
            )}
            {peersView.map((p) => (
              <PeerRow key={p.symbol} row={p} />
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-neutral-800 bg-neutral-950">
          <div className="flex items-center justify-between border-b border-neutral-800 px-3 py-2">
            <div className="text-sm font-medium text-neutral-200">Recent Headlines</div>
            <div className="text-xs text-neutral-500">{newsView.length} shown</div>
          </div>
          <div className="divide-y divide-neutral-800">
            {newsView.length === 0 && (
              <div className="px-3 py-6 text-center text-xs text-neutral-500">No headlines provided.</div>
            )}
            {newsView.map((n) => (
              <NewsRow key={n.id} item={n} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* --------------------------------- UI Bits --------------------------------- */

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className="mt-1 text-base font-semibold text-neutral-100">{value}</div>
    </div>
  );
}

function TopOfBook({ bid, bidSize, ask, askSize }: { bid: number; bidSize: number; ask: number; askSize: number }) {
  const spread = ask - bid;
  return (
    <div className="rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-xs">
      <div className="grid grid-cols-3 items-center gap-3 font-mono tabular-nums">
        <div className="text-right text-emerald-400">${fmt(bid, 2)}</div>
        <div className="text-center text-neutral-400">Spread {fmt(spread, 3)}</div>
        <div className="text-left text-rose-400">${fmt(ask, 2)}</div>
      </div>
      <div className="mt-1 grid grid-cols-3 items-center gap-3 text-[11px] text-neutral-500">
        <div className="text-right">{fmtInt(bidSize)}</div>
        <div className="text-center">×</div>
        <div className="text-left">{fmtInt(askSize)}</div>
      </div>
    </div>
  );
}

function PeerHeader() {
  return (
    <div className="grid grid-cols-12 gap-2 px-3 py-2 text-[11px] text-neutral-500">
      <div className="col-span-4">Symbol</div>
      <div className="col-span-3 text-right">Price</div>
      <div className="col-span-3 text-right">% Chg</div>
      <div className="col-span-2 text-right">P/E</div>
    </div>
  );
}
function PeerRow({ row }: { row: EquityPeer }) {
  const pos = row.changePct >= 0;
  return (
    <div className="grid grid-cols-12 items-center gap-2 px-3 py-2 text-sm">
      <div className="col-span-4 min-w-0">
        <div className="truncate text-neutral-100">{row.symbol}</div>
        <div className="truncate text-[11px] text-neutral-500">{row.name}</div>
      </div>
      <div className="col-span-3 text-right font-mono tabular-nums">${fmt(row.price, 2)}</div>
      <div className={`col-span-3 text-right font-mono tabular-nums ${pos ? "text-emerald-400" : "text-rose-400"}`}>
        {fmtPct(row.changePct)}
      </div>
      <div className="col-span-2 text-right font-mono tabular-nums">{fmt(row.pe, 1)}</div>
    </div>
  );
}

function NewsRow({ item }: { item: EquityNewsItem }) {
  return (
    <div className="flex items-start gap-3 px-3 py-3">
      <div className="mt-0.5 h-2 w-2 flex-none rounded-full bg-neutral-700" />
      <div className="min-w-0">
        <div className="truncate text-[13px] font-medium text-neutral-100">{item.title}</div>
        <div className="text-[11px] text-neutral-500">{item.source} • {timeAgo(item.publishedAt)}</div>
      </div>
    </div>
  );
}

/* --------------------------------- Utils --------------------------------- */

function toPath(xs: number[], ys: number[], X: (x: number) => number, Y: (y: number) => number, skipNaN = false) {
  let d = ""; let started = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (!isFinite(y)) { if (skipNaN) { started = false; continue; } else continue; }
    const cmd = started ? "L" : "M";
    d += `${cmd} ${X(xs[i])} ${Y(y)} `;
    started = true;
  }
  return d.trim();
}
function cumSum(a: number[]) { const out = new Array(a.length); let s = 0; for (let i=0;i<a.length;i++){ s += a[i]; out[i] = s; } return out; }
function niceTicks(min: number, max: number, n = 3) {
  if (!(max > min)) return [min];
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step;
  const mult = err >= 7.5 ? 10 : err >= 3 ? 5 : err >= 1.5 ? 2 : 1;
  const incr = mult * step;
  const start = Math.ceil(min / incr) * incr;
  const ticks: number[] = [];
  for (let v = start; v <= max + 1e-9; v += incr) ticks.push(v);
  return ticks;
}
function timeTicks(xMin: number, xMax: number, n = 4) {
  if (!(xMax > xMin)) return [xMin];
  const span = xMax - xMin;
  const step = Math.round(span / n);
  const out: number[] = [];
  for (let k = 0; k <= n; k++) out.push(xMin + k * step);
  return out;
}
function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtInt(n: number) { return n.toLocaleString("en-US"); }
function fmtPct(x: number) { return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%"; }
function timeAgo(iso: string) {
  const ms = Date.now() - +new Date(iso);
  const s = Math.max(1, Math.floor(ms / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  const dt = new Date(iso);
  return `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,"0")}-${String(dt.getDate()).padStart(2,"0")}`;
}
function fmtTime(t: number | string) {
  const d = new Date(typeof t === "number" ? t : +new Date(t));
  const hh = d.getHours(), mm = d.getMinutes();
  const pad = (x: number) => (x < 10 ? "0" + x : String(x));
  return `${pad(hh)}:${pad(mm)}`;
}

/* ------------------- Ambient React (keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void]; // not used but OK to keep