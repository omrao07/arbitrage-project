"use client";

/**
 * app/equities/page.tsx
 * Zero-import, self-contained Equities dashboard page.
 *
 * What you get (no imports, no links):
 * - Symbol switcher with mock serverless data generators
 * - Overview tiles (price, change, mcap, 52w, PE, EPS, beta, dividend)
 * - Intraday price + VWAP mini chart
 * - Level-2 order book (grouping + depth)
 * - Peers table (sortable) and recent headlines list
 *
 * Tailwind + inline SVG only. Drop it in as a single file.
 */

/* ================================ Types ================================ */

type Bar = { t: string; o: number; h: number; l: number; c: number; v: number };
type Quote = {
  price: number; prevClose: number; change: number; changePct: number;
  open: number; dayHigh: number; dayLow: number; bid: number; bidSize: number; ask: number; askSize: number;
  volume: number; vwap: number;
};
type Fundamentals = {
  marketCap: number; sharesOut: number; pe: number; eps: number; beta: number;
  divYield: number; week52High: number; week52Low: number; currency: string; exchange: string; sector: string; industry: string;
};
type EquitySnap = { symbol: string; quote: Quote; fundamentals: Fundamentals; intraday: Bar[]; lastUpdated: string };

type L2Level = { price: number; size: number };

type EquityPeer = {
  symbol: string; name: string; sector: string; industry: string;
  marketCap: number; price: number; changePct: number; beta: number; pe: number;
};

type EquityNewsItem = {
  id: string | number;
  title: string;
  source: string;
  publishedAt: string; // ISO
  summary?: string;
};

/* ============================= Page Component ============================= */

export default function EquitiesPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [snap, setSnap] = useState<EquitySnap>(() => makeSnapshot("AAPL"));
  const [peers, setPeers] = useState<EquityPeer[]>(() => makePeers("AAPL"));
  const [news, setNews] = useState<EquityNewsItem[]>(() => makeNews("AAPL"));
  const [book, setBook] = useState<{ bids: L2Level[]; asks: L2Level[] }>(() => makeBook(snap.quote.price));

  // regen when symbol changes
  useEffect(() => {
    const s = makeSnapshot(symbol);
    setSnap(s);
    setPeers(makePeers(symbol));
    setNews(makeNews(symbol));
    setBook(makeBook(s.quote.price));
  }, [symbol]);

  const pos = snap.quote.change >= 0;

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      {/* Top bar */}
      <div className="sticky top-0 z-10 border-b border-neutral-800 bg-neutral-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold">Equities</h1>
            <span className={`rounded px-2 py-0.5 text-xs ${pos ? "bg-emerald-600/15 text-emerald-400" : "bg-rose-600/15 text-rose-400"}`}>
              {snap.symbol} {fmtPct(snap.quote.changePct)} ({pos ? "+" : ""}{fmt(snap.quote.change, 2)})
            </span>
          </div>
          <SymbolPicker value={symbol} onChange={setSymbol} />
        </div>
      </div>

      {/* Tiles */}
      <div className="mx-auto grid max-w-7xl grid-cols-2 gap-4 px-4 py-6 md:grid-cols-4">
        <Tile label="Last Price" value={`$${fmt(snap.quote.price, 2)}`} />
        <Tile label="Market Cap" value={`$${fmtInt(snap.fundamentals.marketCap)}`} />
        <Tile label="52w Range" value={`${fmt(snap.fundamentals.week52Low,2)} – ${fmt(snap.fundamentals.week52High,2)}`} />
        <Tile label="P/E · EPS" value={`${fmt(snap.fundamentals.pe,2)} · ${fmt(snap.fundamentals.eps,2)}`} />
      </div>

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 pb-10 lg:grid-cols-3">
        {/* Chart card */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-3 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-sm font-medium">Intraday · VWAP & Prev Close</div>
            <div className="text-xs text-neutral-400">
              Open {fmt(snap.quote.open, 2)} · Day {fmt(snap.quote.dayLow,2)}–{fmt(snap.quote.dayHigh,2)}
            </div>
          </div>
          <IntradayChart bars={snap.intraday} prevClose={snap.quote.prevClose} />
        </div>

        {/* Order book */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900">
          <OrderBookPane symbol={snap.symbol} last={snap.quote.price} bids={book.bids} asks={book.asks} />
        </div>
      </div>

      {/* Peers & Headlines */}
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 pb-16 lg:grid-cols-2">
        <PeersPane baseSymbol={snap.symbol} peers={peers} />
        <NewsPane symbol={snap.symbol} items={news} />
      </div>
    </div>
  );
}

/* ================================ Widgets ================================ */

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className="mt-1 text-lg font-semibold text-neutral-100">{value}</div>
    </div>
  );
}

function SymbolPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (s: string) => void;
}) {
  const popular = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "XOM"];
  const [text, setText] = useState(value);
  return (
    <div className="flex items-center gap-2 text-xs">
      <input
        value={text}
        onChange={(e) => setText(e.target.value.toUpperCase())}
        onKeyDown={(e) => { if (e.key === "Enter") onChange(text.trim() || value); }}
        className="w-28 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
        placeholder="Symbol"
      />
      <button
        onClick={() => onChange(text.trim() || value)}
        className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1 text-neutral-200 hover:bg-neutral-800"
      >
        Load
      </button>
      <div className="hidden items-center gap-1 md:flex">
        {popular.map((s) => (
          <button
            key={s}
            onClick={() => { setText(s); onChange(s); }}
            className={`rounded border px-2 py-1 ${value===s ? "border-emerald-600 bg-emerald-600/20 text-emerald-300" : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:bg-neutral-800/60"}`}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------ Intraday Chart ------------------------------ */

function IntradayChart({
  bars,
  prevClose,
  height = 220,
}: {
  bars: Bar[];
  prevClose?: number;
  height?: number;
}) {
  const [w, setW] = useState(900);
  const hostRef = useRef<any>(null);
  useEffect(() => {
    const el = hostRef.current as HTMLElement | null;
    if (!el) return;
    const set = () => setW(Math.max(360, Math.floor(el.clientWidth)));
    set();
    // @ts-ignore
    const ro: any = typeof ResizeObserver !== "undefined" ? new ResizeObserver(set) : null;
    ro?.observe(el);
    return () => ro?.disconnect?.();
  }, []);

  const data = (bars || []).filter(Boolean);
  const times = data.map((b) => +new Date(b.t));
  const closes = data.map((b) => b.c);
  const vols = data.map((b) => b.v);

  const cumPV = cumSum(data.map((b) => b.c * b.v));
  const cumV = cumSum(vols);
  const vwap = data.map((_, i) => (cumV[i] ? cumPV[i] / cumV[i] : NaN));

  const minP = Math.min(...closes, ...(prevClose ? [prevClose] : []));
  const maxP = Math.max(...closes, ...(prevClose ? [prevClose] : []));
  const pad = (maxP - minP) * 0.08 || 0.5;
  const yMin = Math.max(0, minP - pad);
  const yMax = maxP + pad;

  const vMax = Math.max(1, Math.max(...vols));
  const xMin = Math.min(...times);
  const xMax = Math.max(...times);

  const padL = 56, padR = 12, padT = 10, padB = 36;
  const priceH = Math.max(80, Math.floor((height - padT - padB) * 0.72));
  const volH = Math.max(40, (height - padT - padB) - priceH - 8);
  const priceTop = padT, volTop = padT + priceH + 8;

  const X  = (t: number) => padL + ((t - xMin) / (xMax - xMin || 1)) * Math.max(1, w - padL - padR);
  const Yp = (p: number) => priceTop + (1 - (p - yMin) / (yMax - yMin || 1)) * priceH;
  const Yv = (v: number) => volTop + (1 - v / (vMax || 1)) * volH;

  const pricePath = toPath(times, closes, X, Yp);
  const vwapPath  = toPath(times, vwap,   X, Yp, true);

  const yTicks = niceTicks(yMin, yMax, 4);
  const tTicks = timeTicks(xMin, xMax, 5);

  return (
    <div ref={hostRef} className="w-full">
      <svg width="100%" viewBox={`0 0 ${w} ${height}`} className="block">
        {yTicks.map((p, i) => (
          <g key={`gy-${i}`}>
            <line x1={padL} y1={Yp(p)} x2={w - padR} y2={Yp(p)} stroke="#27272a" strokeDasharray="3 3" />
            <text x={4} y={Yp(p) + 4} fill="#9ca3af" fontSize="10">${fmt(p, 2)}</text>
          </g>
        ))}

        {prevClose ? (
          <>
            <line x1={padL} y1={Yp(prevClose)} x2={w - padR} y2={Yp(prevClose)} stroke="#f59e0b" strokeDasharray="5 4" />
            <text x={w - padR - 4} y={Yp(prevClose) - 4} textAnchor="end" fontSize="10" fill="#f59e0b">
              Prev {fmt(prevClose, 2)}
            </text>
          </>
        ) : null}

        <path d={pricePath} fill="none" stroke="#60a5fa" strokeWidth="2" />
        <path d={vwapPath} fill="none" stroke="#10b981" strokeWidth="1.8" strokeDasharray="6 4" />

        {/* Volume */}
        <text x={4} y={volTop - 4} fontSize="10" fill="#9ca3af">Volume</text>
        {data.map((b, i) => {
          const x = X(times[i]);
          const x0 = i === 0 ? X(times[0]) : X(times[i - 1]);
          const bw = Math.max(1, Math.min(10, (x - x0) * 0.8));
          const color = b.c >= (data[i - 1]?.c ?? b.o) ? "#10b981" : "#ef4444";
          return <rect key={i} x={x - bw / 2} y={Yv(b.v)} width={bw} height={volTop + volH - Yv(b.v)} fill={color} opacity="0.85" />;
        })}

        {tTicks.map((t, i) => (
          <text key={`tx-${i}`} x={X(t)} y={height - 8} textAnchor="middle" fontSize="10" fill="#9ca3af">
            {fmtTime(t)}
          </text>
        ))}
      </svg>
    </div>
  );
}

/* ------------------------------ Order Book ------------------------------ */

function OrderBookPane({
  symbol, last, bids, asks,
}: {
  symbol: string; last: number; bids: L2Level[]; asks: L2Level[];
}) {
  const [depth, setDepth] = useState(10);
  const [group, setGroup] = useState(0.01);
  const gBids = useMemo(() => groupBook(bids, group, "bid").slice(0, depth), [bids, group, depth]);
  const gAsks = useMemo(() => groupBook(asks, group, "ask").slice(0, depth), [asks, group, depth]);
  const bid0 = gBids[0]?.price, ask0 = gAsks[0]?.price;
  const spread = isFinite(bid0 as number) && isFinite(ask0 as number) ? (ask0 as number) - (bid0 as number) : NaN;
  const maxSz = Math.max(1, ...gBids.map((r) => r.size), ...gAsks.map((r) => r.size));

  return (
    <div className="w-full">
      <div className="flex items-center justify-between border-b border-neutral-800 px-3 py-2">
        <div className="space-y-0.5">
          <div className="text-sm font-medium">{symbol} Order Book</div>
          <div className="text-xs text-neutral-400">
            Last ${fmt(last, 2)} {isFinite(spread) ? <span className="text-neutral-500">· Spread {fmt(spread, 3)}</span> : null}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <label className="flex items-center gap-1">
            <span className="text-neutral-400">Depth</span>
            <input
              type="number"
              min={1}
              max={50}
              value={depth}
              onChange={(e) => setDepth(clamp(Math.round(+e.target.value || 1), 1, 50))}
              className="w-16 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
            />
          </label>
          <label className="flex items-center gap-1">
            <span className="text-neutral-400">Group</span>
            <input
              type="number"
              step="0.01"
              value={group}
              onChange={(e) => setGroup(Math.max(0.0001, +e.target.value || group))}
              className="w-20 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
            />
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2">
        <BookSide side="bid" rows={withCum(gBids)} maxSz={maxSz} />
        <BookSide side="ask" rows={withCum(gAsks)} maxSz={maxSz} />
      </div>
    </div>
  );
}

function BookSide({
  side, rows, maxSz,
}: {
  side: "bid" | "ask"; rows: Array<{ price: number; size: number; cum: number }>; maxSz: number;
}) {
  const isBid = side === "bid";
  const HeadClr = isBid ? "text-emerald-300" : "text-rose-300";
  const bar = isBid ? "#065f46" : "#991b1b";
  return (
    <div className="border-t border-neutral-800 md:border-t-0 md:border-l">
      <div className="flex items-center gap-2 border-b border-neutral-800 px-3 py-2 text-xs">
        <span className={`${HeadClr} font-medium`}>{isBid ? "Bids" : "Asks"}</span>
        <span className="text-neutral-500">({rows.length})</span>
      </div>
      <div className="divide-y divide-neutral-800">
        <div className="grid grid-cols-12 gap-2 px-3 py-2 text-[11px] text-neutral-500">
          <div className={`col-span-4 ${isBid ? "" : "order-3"}`}>{isBid ? "Price" : "Size"}</div>
          <div className="col-span-4">{isBid ? "Size" : "Price"}</div>
          <div className={`col-span-4 ${isBid ? "order-3" : "col-start-1"}`}>Cum</div>
        </div>
        {rows.map((r, i) => {
          const pct = clamp(r.size / maxSz, 0, 1);
          const cumPct = clamp(r.cum / maxSz, 0, 1);
          return (
            <div key={i} className="relative grid grid-cols-12 items-center gap-2 px-3 py-1.5 text-sm">
              <div className="absolute inset-0" style={{ background: `linear-gradient(${isBid ? "to left" : "to right"}, ${bar} ${pct*100}%, transparent ${pct*100}%)`, opacity: 0.25 }} />
              <div className="absolute inset-y-0" style={{ [isBid ? "right" : "left"]: 0, width: `${cumPct*100}%`, background: bar, opacity: 0.15 } as any} />
              <div className={`relative col-span-4 ${isBid ? "" : "order-3"} font-mono tabular-nums`}>{isBid ? `$${fmt(r.price,2)}` : fmtInt(r.size)}</div>
              <div className="relative col-span-4 font-mono tabular-nums">{isBid ? fmtInt(r.size) : `$${fmt(r.price,2)}`}</div>
              <div className={`relative col-span-4 ${isBid ? "order-3" : "col-start-1"} font-mono tabular-nums`}>{fmtInt(r.cum)}</div>
            </div>
          );
        })}
        {rows.length === 0 && <div className="px-3 py-6 text-center text-xs text-neutral-500">No liquidity.</div>}
      </div>
    </div>
  );
}

/* --------------------------------- Peers --------------------------------- */

function PeersPane({ baseSymbol, peers }: { baseSymbol: string; peers: EquityPeer[] }) {
  const [sort, setSort] = useState<{ key: keyof EquityPeer; dir: "asc" | "desc" }>({ key: "marketCap", dir: "desc" });
  const rows = peers.slice().sort((a, b) => {
    const k = sort.key as keyof EquityPeer;
    const dir = sort.dir === "asc" ? 1 : -1;
    const av = a[k] as any, bv = b[k] as any;
    if (typeof av === "string" && typeof bv === "string") return av.localeCompare(bv) * dir;
    return ((av as number) - (bv as number)) * dir;
  });

  const Th = ({ label, k, right = false }: { label: string; k: keyof EquityPeer; right?: boolean }) => {
    const active = sort.key === k;
    const dir = active ? sort.dir : undefined;
    return (
      <th
        className={`cursor-pointer select-none px-3 py-2 ${right ? "text-right" : "text-left"} text-[11px] uppercase text-neutral-500`}
        onClick={() => setSort({ key: k, dir: active && dir === "desc" ? "asc" : "desc" })}
      >
        <span className="inline-flex items-center gap-1">
          {label}
          <svg width="10" height="10" viewBox="0 0 24 24" className={`opacity-70 ${active ? "" : "invisible"}`}>
            {dir === "asc"
              ? <path d="M7 14l5-5 5 5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />
              : <path d="M7 10l5 5 5-5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />}
          </svg>
        </span>
      </th>
    );
  };

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="flex items-center justify-between border-b border-neutral-800 px-3 py-2">
        <div className="text-sm font-medium">{baseSymbol} Peers</div>
        <div className="text-xs text-neutral-500">{rows.length} shown</div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-neutral-950">
            <tr className="border-b border-neutral-800">
              <Th label="Symbol" k="symbol" />
              <Th label="Name" k="name" />
              <Th label="Price" k="price" right />
              <Th label="% Chg" k="changePct" right />
              <Th label="Mkt Cap" k="marketCap" right />
              <Th label="P/E" k="pe" right />
              <Th label="Beta" k="beta" right />
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {rows.map((p) => {
              const pos = p.changePct >= 0;
              return (
                <tr key={p.symbol} className="hover:bg-neutral-800/40">
                  <td className="px-3 py-2 font-medium text-neutral-100">{p.symbol}</td>
                  <td className="px-3 py-2 text-neutral-300">{p.name}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">${fmt(p.price, 2)}</td>
                  <td className={`px-3 py-2 text-right font-mono tabular-nums ${pos ? "text-emerald-400" : "text-rose-400"}`}>{fmtPct(p.changePct)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmtInt(p.marketCap)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(p.pe, 1)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(p.beta, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* --------------------------------- News --------------------------------- */

function NewsPane({ symbol, items }: { symbol: string; items: EquityNewsItem[] }) {
  const [q, setQ] = useState("");
  const rows = items
    .filter((n) => (q ? (n.title + " " + n.source).toLowerCase().includes(q.toLowerCase()) : true))
    .slice(0, 12);

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="flex items-center justify-between gap-2 border-b border-neutral-800 px-3 py-2">
        <div className="text-sm font-medium">{symbol} Headlines</div>
        <label className="relative">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search…"
            className="w-48 rounded-md border border-neutral-700 bg-neutral-950 pl-7 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
          />
          <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
            <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
            <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </label>
      </div>
      <div className="divide-y divide-neutral-800">
        {rows.length === 0 && <div className="px-3 py-6 text-center text-xs text-neutral-500">No headlines.</div>}
        {rows.map((n) => (
          <div key={n.id} className="flex items-start gap-3 px-3 py-3">
            <div className="mt-0.5 h-2 w-2 flex-none rounded-full bg-neutral-700" />
            <div className="min-w-0">
              <div className="truncate text-[13px] font-medium text-neutral-100">{n.title}</div>
              <div className="text-[11px] text-neutral-500">{n.source} • {timeAgo(n.publishedAt)}</div>
              {n.summary && <p className="mt-1 line-clamp-2 text-[12px] leading-5 text-neutral-300">{n.summary}</p>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ============================ Mock Data Makers ============================ */

function makeSnapshot(symbol: string): EquitySnap {
  const dayKey = new Date().toISOString().slice(0, 10);
  const rng = mulberry32(hash(`${symbol}|${dayKey}`));
  const base = 50 + (hash(symbol) % 500);
  const prevClose = round(base * (0.98 + rng() * 0.04), 2);
  const last = round(prevClose * (0.985 + rng() * 0.03), 2);
  const open = round(prevClose * (0.99 + rng() * 0.02), 2);

  const now = new Date();
  const start = withTime(now, 9, 30, 0, -240);
  const bars = makeBars({ start, count: 150, prevClose, last, rng });

  const dayHigh = round(Math.max(...bars.map((b) => b.h), last, open), 2);
  const dayLow = round(Math.min(...bars.map((b) => b.l), last, open), 2);
  const volume = Math.max(1, Math.round(bars.reduce((s, b) => s + b.v, 0)));
  const vwap = round(bars.reduce((s, b) => s + b.c * b.v, 0) / volume, 4);

  const sharesOut = 1e9 * (0.5 + rng());
  const eps = round((1 + rng() * 9) / 12, 2);
  const pe = round(Math.max(5, Math.min(80, (last / Math.max(eps, 0.01)) * (0.6 + rng() * 0.8))), 2);
  const mcap = Math.round(sharesOut * last);

  return {
    symbol: symbol.toUpperCase(),
    quote: {
      price: last, prevClose, change: round(last - prevClose, 2), changePct: round((last - prevClose) / (prevClose || 1), 4),
      open, dayHigh, dayLow, bid: round(last - 0.02, 2), bidSize: 100 + Math.floor(rng() * 900),
      ask: round(last + 0.02, 2), askSize: 100 + Math.floor(rng() * 900), volume, vwap,
    },
    fundamentals: {
      marketCap: mcap, sharesOut: Math.round(sharesOut), pe, eps, beta: round(0.6 + rng() * 1.2, 2),
      divYield: round(rng() * 0.03, 4),
      week52High: round(last * (1.1 + rng() * 0.15), 2),
      week52Low: round(last * (0.8 - rng() * 0.1), 2),
      currency: "USD", exchange: guessExchange(symbol), sector: guessSector(symbol), industry: guessIndustry(symbol),
    },
    intraday: bars,
    lastUpdated: new Date().toISOString(),
  };
}

function makeBars({
  start, count, prevClose, last, rng,
}: { start: Date; count: number; prevClose: number; last: number; rng: () => number }): Bar[] {
  const n = Math.max(60, count);
  const bars: Bar[] = [];
  let px = prevClose;
  let t0 = +start;
  let volBase = Math.max(5000, Math.round(30000 * (0.8 + rng() * 0.6)));
  for (let i = 0; i < n; i++) {
    const t = new Date(t0 + i * 60_000);
    const blend = (i + 1) / n;
    const target = prevClose + (last - prevClose) * blend;
    const noise = (rng() - 0.5) * prevClose * 0.0025;
    const next = clamp(target + noise, 0.01, 1e9);
    const o = px, c = round(next, 4);
    const h = round(Math.max(o, c) * (1 + rng() * 0.0015), 4);
    const l = round(Math.min(o, c) * (1 - rng() * 0.0015), 4);
    const v = Math.max(1, Math.round(volBase * (0.6 + 0.8 * rng())));
    bars.push({ t: t.toISOString(), o: round(o, 4), h, l, c, v });
    px = c;
  }
  return bars;
}

function makeBook(last: number) {
  const rng = mulberry32(hash(String(Math.round(last * 100))));
  const mkSide = (dir: -1 | 1) => {
    const out: L2Level[] = [];
    for (let i = 0; i < 40; i++) {
      const px = round(last + dir * (0.01 * i + (rng() - 0.5) * 0.003), 2);
      const sz = 50 + Math.floor(rng() * 2000);
      out.push({ price: px, size: sz });
    }
    return out;
  };
  return { bids: mkSide(-1), asks: mkSide(+1) };
}

function makePeers(symbol: string): EquityPeer[] {
  const key = symbol.toUpperCase();
  const rng = mulberry32(hash(key));
  const sector = guessSector(key), industry = guessIndustry(key);
  const basePrice = 50 + (hash(key) % 500);
  const n = 6 + Math.floor(rng() * 4);
  const peers: EquityPeer[] = [];
  for (let i = 0; i < n; i++) {
    const ticker = fakeTicker(key, i);
    const price = round(basePrice * (0.5 + rng()), 2);
    const mc = Math.round(price * (1e8 + rng() * 9e8));
    peers.push({
      symbol: ticker, name: fakeName(ticker), sector, industry, marketCap: mc,
      price, changePct: round((rng() - 0.5) * 0.04, 4), beta: round(0.5 + rng() * 1.5, 2), pe: round(5 + rng() * 40, 1),
    });
  }
  return peers;
}

function makeNews(symbol: string): EquityNewsItem[] {
  const rng = mulberry32(hash(symbol + "|news"));
  const now = Date.now();
  const items: EquityNewsItem[] = [];
  const sources = ["Bloomberg", "Reuters", "FT", "WSJ", "CNBC", "MarketWatch"];
  for (let i = 0; i < 14; i++) {
    const agoMin = Math.floor(rng() * 60 * 48); // last ~2 days
    items.push({
      id: i + 1,
      title: `${symbol.toUpperCase()} ${headlineVerb(rng())} as ${buzzword(rng())}`,
      source: sources[Math.floor(rng() * sources.length)],
      publishedAt: new Date(now - agoMin * 60_000).toISOString(),
      summary: Math.random() < 0.6 ? lorem(16 + Math.floor(rng() * 24)) : undefined,
    });
  }
  return items.sort((a, b) => +new Date(b.publishedAt) - +new Date(a.publishedAt));
}

/* ================================ Helpers ================================ */

function groupBook(levels: L2Level[], tick: number, side: "bid" | "ask") {
  if (!Array.isArray(levels) || levels.length === 0) return [];
  const sorted = [...levels].sort((a, b) => (side === "bid" ? b.price - a.price : a.price - b.price));
  const map = new Map<number, number>();
  const factor = 1 / Math.max(0.0001, tick);
  for (const { price, size } of sorted) {
    const bucket = side === "bid" ? Math.floor(price * factor) / factor : Math.ceil(price * factor) / factor;
    map.set(bucket, (map.get(bucket) ?? 0) + Math.max(0, size | 0));
  }
  return Array.from(map.entries()).map(([price, size]) => ({ price, size })).sort((a, b) => (side === "bid" ? b.price - a.price : a.price - b.price));
}
function withCum(rows: Array<{ price: number; size: number }>) { const out: Array<{ price: number; size: number; cum: number }> = []; let s = 0; for (const r of rows) { s += r.size; out.push({ ...r, cum: s }); } return out; }

function toPath(xs: number[], ys: number[], X: (x: number) => number, Y: (v: number) => number, skipNaN = false) {
  let d = "", started = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (!isFinite(y)) { if (skipNaN) { started = false; continue; } else continue; }
    const cmd = started ? "L" : "M";
    d += `${cmd} ${X(xs[i])} ${Y(y)} `;
    started = true;
  }
  return d.trim();
}
function niceTicks(min: number, max: number, n = 4) {
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
function timeTicks(xMin: number, xMax: number, n = 5) {
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
  const ms = Date.now() - +new Date(iso); const s = Math.max(1, Math.floor(ms / 1000));
  if (s < 60) return `${s}s ago`; const m = Math.floor(s / 60); if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60); if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24); if (d < 7) return `${d}d ago`;
  const dt = new Date(iso);
  return `${dt.getFullYear()}-${String(dt.getMonth()+1).padStart(2,"0")}-${String(dt.getDate()).padStart(2,"0")}`;
}
function fmtTime(t: number | string) { const d = new Date(typeof t === "number" ? t : +new Date(t)); const hh = d.getHours(), mm = d.getMinutes(); const pad = (x: number) => (x < 10 ? "0" + x : String(x)); return `${pad(hh)}:${pad(mm)}`; }
function round(n: number, d = 2) { const p = 10 ** d; return Math.round(n * p) / p; }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function withTime(now: Date, hh: number, mm: number, ss: number, tzOffsetMin: number) { const d = new Date(now); d.setHours(hh, mm, ss, 0); d.setMinutes(d.getMinutes() + tzOffsetMin); return d; }

function mulberry32(seed: number) { let t = seed >>> 0; return () => { t += 0x6D2B79F5; let x = Math.imul(t ^ (t >>> 15), 1 | t); x ^= x + Math.imul(x ^ (x >>> 7), 61 | x); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; }; }
function hash(s: string) { let h = 2166136261 >>> 0; for (let i=0;i<s.length;i++){ h ^= s.charCodeAt(i); h = Math.imul(h, 16777619);} return h >>> 0; }

function guessExchange(symbol: string) { if (/^\^?GSPC|^SPY$/.test(symbol)) return "NYSE Arca"; if (/AAPL|MSFT|GOOGL|AMZN|META|NVDA|TSLA|NFLX/.test(symbol)) return "NASDAQ"; return "NYSE"; }
function guessSector(symbol: string) { if (/AAPL|MSFT|NVDA/.test(symbol)) return "Information Technology"; if (/AMZN|WMT|TGT/.test(symbol)) return "Consumer Discretionary"; if (/XOM|CVX/.test(symbol)) return "Energy"; if (/JPM|GS|BAC/.test(symbol)) return "Financials"; return "Industrials"; }
function guessIndustry(symbol: string) { if (/AAPL|MSFT/.test(symbol)) return "Consumer Electronics"; if (/NVDA/.test(symbol)) return "Semiconductors"; if (/AMZN/.test(symbol)) return "Internet & Direct Marketing"; return "Diversified"; }

function fakeTicker(base: string, i: number) { const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; const code = letters[(hash(base + i) % 26)] + letters[(hash(base + i + 7) % 26)]; return (base.slice(0, 2) + code + (i + 1)).slice(0, 4).toUpperCase(); }
function fakeName(ticker: string) { const suffix = ["Corp", "Inc", "Ltd", "Group", "Holdings"]; return ticker + " " + suffix[hash(ticker) % suffix.length]; }

function headlineVerb(r: number) { return r < 0.33 ? "rallies" : r < 0.66 ? "slips" : "steadies"; }
function buzzword(r: number) { const a = ["AI optimism", "rate jitters", "earnings beat", "macro caution", "guidance update", "supply news"]; return a[Math.floor(r * a.length)] || a[0]; }
function lorem(n = 20) { const w = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".split(" "); const out: string[] = []; for (let i=0;i<n;i++) out.push(w[i % w.length]); return out.join(" ") + "."; }

/* ------------------- Ambient React (keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useRef<T>(v: T | null): { current: T | null };

function cumSum(arg0: number[]) {
    throw new Error("Function not implemented.");
}
