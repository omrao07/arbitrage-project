'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { JSX } from 'react';

/** ======================== Types ======================== */
type Level = { px: number; sz: number; venue?: string };
type Book = { ts: number; bids: Level[]; asks: Level[]; symbol?: string };

type Patch =
  | { book: Book }
  | { bids: Level[] }
  | { asks: Level[] }
  | { note: string };

type Props = {
  symbol?: string;
  title?: string;
  depth?: number;                // max levels per side to show
  snapshotEndpoint?: string;     // POST -> {book}
  sseEndpoint?: string;          // POST -> text/event-stream of {data: JSON}
  className?: string;
};

/** ====================== Small utils ===================== */
const fmt2 = (n?: number) => (n != null && Number.isFinite(n) ? n.toFixed(2) : '');
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const byPxDesc = (a: Level, b: Level) => b.px - a.px;
const byPxAsc = (a: Level, b: Level) => a.px - b.px;

function cum(arr: number[]) {
  const out: number[] = [];
  let c = 0;
  for (const v of arr) {
    c += v;
    out.push(c);
  }
  return out;
}

function download(name: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

/** =============== Isometric projection (SVG) =============== */
/** We use an isometric-like projection for a "3D" wall illusion.
 * Axes: X = cumulative size; Z = price steps from mid; Y = bar height.
 * Screen coords from (x,y,z):
 */
function iso(x: number, y: number, z: number, originX: number, originY: number) {
  const cos = Math.cos(Math.PI / 6); // ≈0.866
  const sin = Math.sin(Math.PI / 6); // 0.5
  const sx = originX + (x - z) * cos;
  const sy = originY + (x + z) * sin - y;
  return [sx, sy] as const;
}

/** Build an extruded block polygon set (top + left + right faces)
 * given a base rectangle [x0..x1]×[z0..z1] with height h on Y.
 */
function blockPolys(
  x0: number, x1: number, z0: number, z1: number, h: number,
  originX: number, originY: number
) {
  const p = (x: number, y: number, z: number) => iso(x, y, z, originX, originY);

  const A = p(x0, 0, z0), B = p(x1, 0, z0), C = p(x1, 0, z1), D = p(x0, 0, z1); // base
  const A2 = p(x0, h, z0), B2 = p(x1, h, z0), C2 = p(x1, h, z1), D2 = p(x0, h, z1); // top

  const polyTop = [A2, B2, C2, D2];
  const polyLeft = [D, D2, A2, A];
  const polyRight = [B, B2, C2, C];

  return { polyTop, polyLeft, polyRight };
}

function pathFrom(poly: readonly (readonly [number, number])[]) {
  return poly.map(([x, y], i) => (i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`)).join(' ') + ' Z';
}

/** ===================== Component ====================== */
export default function Orderbook3D({
  symbol = 'AAPL',
  title = 'Order Book (3D)',
  depth = 16,
  snapshotEndpoint = '/api/book/snapshot',
  sseEndpoint = '/api/book/stream',
  className = '',
}: Props) {
  const [book, setBook] = useState<Book>({ ts: Date.now(), bids: [], asks: [], symbol });
  const [connected, setConnected] = useState(false);
  const [note, setNote] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Controls
  const [h, setH] = useState(18);          // bar height
  const [zScale, setZScale] = useState(14); // spacing per level (price step depth)
  const [xScale, setXScale] = useState(180); // size scale
  const [levels, setLevels] = useState(depth);
  const [show2D, setShow2D] = useState(true);

  // SSE
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    seed();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  async function seed() {
    stop();
    setError(null);
    try {
      const res = await fetch(snapshotEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, depth: levels }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Patch = await res.json();
      if ('book' in data && data.book) setBook(data.book);
    } catch (e: any) {
      setError(e.message);
    }
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setConnected(false);
  }

  async function stream() {
    stop();
    setError(null);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await fetch(sseEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ symbol }),
        signal: ac.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (!res.body) throw new Error('No stream body');
      setConnected(true);

      const reader = res.body.getReader();
      const dec = new TextDecoder('utf-8');
      let acc = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        acc += dec.decode(value, { stream: true });
        const lines = acc.split('\n');
        acc = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            const patch = JSON.parse(payload) as Patch;
            applyPatch(patch);
          } catch {
            setNote(n => (n ? n + '\n' : '') + payload);
          }
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e.message);
    } finally {
      setConnected(false);
    }
  }

  function applyPatch(p: Patch) {
    setBook(prev => {
      let next = { ...prev };
      if ('book' in p && p.book) next = p.book;
      if ('bids' in p && p.bids) next = { ...next, bids: p.bids };
      if ('asks' in p && p.asks) next = { ...next, asks: p.asks };
      if ('note' in p && p.note) setNote(n => (n ? n + '\n' : '') + p.note);
      next.ts = Date.now();
      return next;
    });
  }

  /** =============== Derived depth data =============== */
  const norm = useMemo(() => normalizeBook(book, levels), [book, levels]);
  const mid = norm.mid;
  const spread = norm.spread;

  /** =============== SVG geometry =============== */
  const W = 840;                 // view width
  const H = 420;                 // view height
  const originX = W * 0.50;      // center
  const originY = H * 0.72;      // floor baseline

  // Build polygons for each level (bids to the left, asks to the right)
  const bidPolys = useMemo(() => {
    const polys: JSX.Element[] = [];
    const zs = norm.bids.map((_, i) => i * zScale);
    const xs = norm.bidsCum.map(v => v * xScale);
    const n = norm.bids.length;

    for (let i = 0; i < n; i++) {
      const x0 = i === 0 ? 0 : xs[i - 1];
      const x1 = xs[i];
      const z0 = zs[i];
      const z1 = i === n - 1 ? zs[i] + zScale : zs[i + 1];

      const { polyTop, polyLeft, polyRight } = blockPolys(-x1, -x0, z0, z1, h, originX, originY);
      polys.push(
        <g key={'b' + i}>
          <path d={pathFrom(polyLeft)} fill="#064e3b" opacity={0.9} />
          <path d={pathFrom(polyRight)} fill="#065f46" opacity={0.85} />
          <path d={pathFrom(polyTop)} fill="#059669" opacity={0.9}>
            <title>{`Bid ${i + 1}\nPrice: ${fmt2(norm.bids[i].px)}\nSize: ${fmt2(norm.bids[i].sz)}\nCum: ${fmt2(norm.bidsCum[i])}`}</title>
          </path>
        </g>
      );
    }
    return polys;
  }, [norm.bids, norm.bidsCum, h, originX, originY, xScale, zScale]);

  const askPolys = useMemo(() => {
    const polys: JSX.Element[] = [];
    const zs = norm.asks.map((_, i) => i * zScale);
    const xs = norm.asksCum.map(v => v * xScale);
    const n = norm.asks.length;

    for (let i = 0; i < n; i++) {
      const x0 = i === 0 ? 0 : xs[i - 1];
      const x1 = xs[i];
      const z0 = zs[i];
      const z1 = i === n - 1 ? zs[i] + zScale : zs[i + 1];

      const { polyTop, polyLeft, polyRight } = blockPolys(x0, x1, z0, z1, h, originX, originY);
      polys.push(
        <g key={'a' + i}>
          <path d={pathFrom(polyRight)} fill="#7f1d1d" opacity={0.9} />
          <path d={pathFrom(polyLeft)} fill="#991b1b" opacity={0.85} />
          <path d={pathFrom(polyTop)} fill="#ef4444" opacity={0.9}>
            <title>{`Ask ${i + 1}\nPrice: ${fmt2(norm.asks[i].px)}\nSize: ${fmt2(norm.asks[i].sz)}\nCum: ${fmt2(norm.asksCum[i])}`}</title>
          </path>
        </g>
      );
    }
    return polys;
  }, [norm.asks, norm.asksCum, h, originX, originY, xScale, zScale]);

  /** =============== 2D fallback data =============== */
  const twoD = useMemo(() => {
    const maxCum = Math.max(
      norm.bidsCum[norm.bidsCum.length - 1] || 1,
      norm.asksCum[norm.asksCum.length - 1] || 1
    );
    return {
      bids: norm.bids.map((lv, i) => ({
        px: lv.px, w: (norm.bidsCum[i] / maxCum) * 100,
      })),
      asks: norm.asks.map((lv, i) => ({
        px: lv.px, w: (norm.asksCum[i] / maxCum) * 100,
      })),
    };
  }, [norm.bids, norm.asks, norm.bidsCum, norm.asksCum]);

  /** ==================== UI ==================== */
  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <input
            value={symbol}
            onChange={(e) => setBook(b => ({ ...b, symbol: e.target.value.toUpperCase() }))}
            className="rounded border px-2 py-1 text-sm w-28"
          />
          <button onClick={seed} className="px-2 py-1 text-xs rounded border">Seed</button>
          {connected ? (
            <button onClick={stop} className="px-3 py-1 text-xs rounded border border-rose-500 text-rose-600">Stop</button>
          ) : (
            <button onClick={stream} className="px-3 py-1 text-xs rounded border border-indigo-600 bg-indigo-600 text-white">Stream</button>
          )}
          <button
            onClick={() => download(`${book.symbol || symbol}_book.json`, JSON.stringify(book, null, 2), 'application/json')}
            className="px-2 py-1 text-xs rounded border"
          >
            JSON
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 grid grid-cols-2 md:grid-cols-6 gap-3 border-b border-neutral-200 dark:border-neutral-800 text-sm">
        <Field label="Levels">
          <input type="number" min={4} max={50} value={levels} onChange={e => setLevels(Math.max(4, Number(e.target.value) || levels))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="Bar Height">
          <input type="number" min={8} max={48} value={h} onChange={e => setH(clamp(Number(e.target.value) || h, 8, 48))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="X Scale (size)">
          <input type="number" min={60} max={400} value={xScale} onChange={e => setXScale(clamp(Number(e.target.value) || xScale, 60, 400))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="Z Scale (depth)">
          <input type="number" min={8} max={40} value={zScale} onChange={e => setZScale(clamp(Number(e.target.value) || zScale, 8, 40))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="2D Fallback">
          <div className="flex items-center gap-2">
            <input id="twod" type="checkbox" checked={show2D} onChange={e => setShow2D(e.target.checked)} />
            <label htmlFor="twod" className="text-xs">Show</label>
          </div>
        </Field>
        <div className="flex items-end">
          <div className={`text-[11px] px-2 py-1 rounded ${connected ? 'bg-emerald-100 text-emerald-700' : 'bg-neutral-100 text-neutral-700'}`}>
            {connected ? 'LIVE' : 'IDLE'}
          </div>
        </div>
      </div>

      {/* 3D SVG */}
      <div className="p-3">
        <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="3D depth">
          {/* floor grid */}
          <g opacity={0.25}>
            {Array.from({ length: 8 }).map((_, i) => {
              const z = i * zScale * 2;
              const [x1, y1] = iso(-xScale * 1.1, 0, z, originX, originY);
              const [x2, y2] = iso(xScale * 1.1, 0, z, originX, originY);
              return <line key={'gz' + i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1} />;
            })}
            {Array.from({ length: 6 }).map((_, i) => {
              const x = (i - 2.5) * (xScale * 0.5);
              const [x1, y1] = iso(x, 0, 0, originX, originY);
              const [x2, y2] = iso(x, 0, zScale * 12, originX, originY);
              return <line key={'gx' + i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1} />;
            })}
          </g>

          {/* bid wall (left), ask wall (right) */}
          {bidPolys}
          {askPolys}

          {/* mid marker */}
          <g>
            {(() => {
              const [mx1, my1] = iso(0, 0, 0, originX, originY);
              const [mx2, my2] = iso(0, h * 1.5, 0, originX, originY);
              return <line x1={mx1} y1={my1} x2={mx2} y2={my2} stroke="#f59e0b" strokeWidth={2} />;
            })()}
          </g>
        </svg>

        {/* Legend */}
        <div className="mt-2 flex items-center gap-4 text-xs text-neutral-600 dark:text-neutral-300">
          <span><span className="inline-block w-3 h-3 align-middle rounded-sm" style={{ background: '#059669' }} /> Bids</span>
          <span><span className="inline-block w-3 h-3 align-middle rounded-sm" style={{ background: '#ef4444' }} /> Asks</span>
          <span>Mid: <b>{fmt2(mid)}</b></span>
          <span>Spread: <b>{fmt2(spread)}</b></span>
          <span>Updated: {new Date(book.ts).toLocaleTimeString()}</span>
        </div>
      </div>

      {/* 2D fallback histogram */}
      {show2D && (
        <div className="px-3 pb-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Card title="Bids (2D)">
              <ul className="space-y-1">
                {twoD.bids.map((r, i) => (
                  <li key={'b2d' + i} className="flex items-center gap-2">
                    <div className="text-[11px] w-16 text-right">{fmt2(r.px)}</div>
                    <div className="h-3 bg-emerald-600/80 rounded" style={{ width: r.w + '%' }} />
                  </li>
                ))}
              </ul>
            </Card>
            <Card title="Asks (2D)">
              <ul className="space-y-1">
                {twoD.asks.map((r, i) => (
                  <li key={'a2d' + i} className="flex items-center gap-2">
                    <div className="text-[11px] w-16 text-right">{fmt2(r.px)}</div>
                    <div className="h-3 bg-rose-600/80 rounded" style={{ width: r.w + '%' }} />
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        </div>
      )}

      {/* Notes / Errors */}
      {error && <div className="px-3 pb-3 text-xs text-rose-600">{error}</div>}
      {note && <pre className="px-3 pb-3 text-[11px] whitespace-pre-wrap text-neutral-600 dark:text-neutral-300">{note}</pre>}
    </div>
  );
}

/** ================== Subcomponents ================== */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="block text-[11px] text-neutral-500 mb-1">{label}</label>{children}</div>;
}
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}

/** =================== Data shaping =================== */
function normalizeBook(book: Book, levels: number) {
  const bids = (book.bids || []).slice().sort(byPxDesc).slice(0, levels);
  const asks = (book.asks || []).slice().sort(byPxAsc).slice(0, levels);

  const bestBid = bids[0]?.px ?? NaN;
  const bestAsk = asks[0]?.px ?? NaN;
  const mid = Number.isFinite(bestBid) && Number.isFinite(bestAsk) ? (bestBid + bestAsk) / 2 : NaN;
  const spread = Number.isFinite(bestBid) && Number.isFinite(bestAsk) ? (bestAsk - bestBid) : NaN;

  const bidsSz = bids.map(x => Math.max(0, x.sz));
  const asksSz = asks.map(x => Math.max(0, x.sz));
  const bidsCum = cum(bidsSz);
  const asksCum = cum(asksSz);

  // Normalize sizes so 3D scaling is stable even if one side is much larger
  const maxCum = Math.max(bidsCum[bidsCum.length - 1] || 1, asksCum[asksCum.length - 1] || 1);
  const bidsCumN = bidsCum.map(v => v / maxCum);
  const asksCumN = asksCum.map(v => v / maxCum);

  return {
    bids, asks,
    bidsCum: bidsCumN,
    asksCum: asksCumN,
    mid, spread,
  };
}