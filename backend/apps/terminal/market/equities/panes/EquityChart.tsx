"use client";

/**
 * panes/equitychart.tsx
 * Zero-import, self-contained equity intraday chart pane.
 *
 * - Inputs: symbol, bars (1-min or coarser), prevClose (for ref line)
 * - Price line + last marker, VWAP line, prevClose line
 * - Volume columns on a secondary track
 * - Hover crosshair + tooltip (time, price, volume, vwap)
 * - Tailwind + inline SVG only (no 3rd-party libs)
 */

export type Bar = {
  t: string; // ISO timestamp
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
};

export default function EquityChartPane({
  symbol,
  bars,
  prevClose,
  className = "",
  height = 320,
}: {
  symbol: string;
  bars: Bar[];
  prevClose?: number;
  className?: string;
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

  // Guard
  const data = Array.isArray(bars) ? bars.filter(Boolean) : [];
  const vols = data.map((b) => b.v);
  const closes = data.map((b) => b.c);
  const times = data.map((b) => +new Date(b.t));

  // VWAP
  const cumPV = cumSum(data.map((b) => b.c * b.v));
  const cumV = cumSum(vols);
  const vwap = data.map((_, i) => (cumV[i] ? cumPV[i] / cumV[i] : NaN));

  // Bounds
  const minP = Math.min(...closes, ...(prevClose ? [prevClose] : []));
  const maxP = Math.max(...closes, ...(prevClose ? [prevClose] : []));
  const pad = (maxP - minP) * 0.08 || 0.5;
  const yMin = Math.max(0, minP - pad);
  const yMax = maxP + pad;

  const vMax = Math.max(1, Math.max(...vols));
  const xMin = Math.min(...times);
  const xMax = Math.max(...times);

  // Layout
  const wView = w;
  const hView = height;
  const padL = 56, padR = 12, padT = 12, padB = 40;
  const innerW = Math.max(1, wView - padL - padR);
  const priceH = Math.max(80, Math.floor((hView - padT - padB) * 0.74));
  const volH = Math.max(40, (hView - padT - padB) - priceH - 8);
  const priceTop = padT;
  const volTop = padT + priceH + 8;

  const X = (t: number) => padL + ((t - xMin) / (xMax - xMin || 1)) * innerW;
  const Yp = (p: number) => priceTop + (1 - (p - yMin) / (yMax - yMin || 1)) * priceH;
  const Yv = (v: number) => volTop + (1 - v / (vMax || 1)) * volH;

  // Paths
  const pricePath = toPath(times, closes, X, Yp);
  const vwapPath = toPath(times, vwap, X, Yp, true);

  // Hover
  const [hover, setHover] = useState<null | { i: number; x: number }>(null);
  const nearestIdx = (px: number) => {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < times.length; i++) {
      const dx = Math.abs(X(times[i]) - px);
      if (dx < bestD) { bestD = dx; best = i; }
    }
    return best;
  };

  // Axis ticks
  const yTicks = niceTicks(yMin, yMax, 4);
  const tTicks = timeTicks(xMin, xMax, 5);

  return (
    <div ref={hostRef} className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{symbol} — Intraday</h3>
          <p className="text-xs text-neutral-400">
            {data.length} bars · VWAP & prev close reference
          </p>
        </div>
        {closes.length > 0 && (
          <div className="text-right">
            <div className="text-sm font-semibold text-neutral-100">${fmt(closes.at(-1)!, 2)}</div>
            <div className="text-[11px] text-neutral-400">{fmtTime(times.at(-1)!)} (last)</div>
          </div>
        )}
      </div>

      <div className="px-2 py-2">
        <svg
          width="100%"
          viewBox={`0 0 ${wView} ${hView}`}
          className="block"
          onMouseMove={(e) => {
            const rect = (e.currentTarget as any).getBoundingClientRect();
            const i = nearestIdx(e.clientX - rect.left);
            setHover({ i, x: X(times[i]) });
          }}
          onMouseLeave={() => setHover(null)}
        >
          {/* --- PRICE PANEL --- */}
          {/* grid Y */}
          {yTicks.map((p, i) => {
            const y = Yp(p);
            return (
              <g key={`gy-${i}`}>
                <line x1={padL} y1={y} x2={wView - padR} y2={y} stroke="#27272a" strokeDasharray="3 3" />
                <text x={4} y={y + 4} fill="#9ca3af" fontSize="10">${fmt(p, 2)}</text>
              </g>
            );
          })}

          {/* prev close */}
          {prevClose ? (
            <>
              <line x1={padL} y1={Yp(prevClose)} x2={wView - padR} y2={Yp(prevClose)} stroke="#f59e0b" strokeDasharray="5 4" />
              <text x={wView - padR - 4} y={Yp(prevClose) - 4} textAnchor="end" fontSize="10" fill="#f59e0b">
                Prev {fmt(prevClose, 2)}
              </text>
            </>
          ) : null}

          {/* price & VWAP */}
          <path d={pricePath} fill="none" stroke="#60a5fa" strokeWidth="2" />
          <path d={vwapPath} fill="none" stroke="#10b981" strokeWidth="1.8" strokeDasharray="6 4" />

          {/* last marker */}
          {closes.length > 0 && (
            <>
              <circle cx={X(times.at(-1)!)} cy={Yp(closes.at(-1)!)} r="3" fill="#60a5fa" />
              <text x={X(times.at(-1)!)+6} y={Yp(closes.at(-1)!)-6} fontSize="10" fill="#cbd5e1">
                ${fmt(closes.at(-1)!, 2)}
              </text>
            </>
          )}

          {/* --- VOLUME PANEL --- */}
          {/* axis label */}
          <text x={4} y={volTop - 4} fontSize="10" fill="#9ca3af">Volume</text>
          {/* columns */}
          {data.map((b, i) => {
            const x = X(times[i]);
            const x0 = i === 0 ? X(times[0]) : X(times[i - 1]);
            const wCol = Math.max(1, Math.min(10, (x - x0) * 0.8));
            const color = b.c >= (data[i-1]?.c ?? b.o) ? "#10b981" : "#ef4444";
            return (
              <rect key={`v-${i}`} x={x - wCol / 2} y={Yv(b.v)} width={wCol} height={volTop + volH - Yv(b.v)} fill={color} opacity="0.8" />
            );
          })}

          {/* X ticks */}
          {tTicks.map((t, i) => {
            const xx = X(t);
            return (
              <text key={`tx-${i}`} x={xx} y={hView - 8} textAnchor="middle" fontSize="10" fill="#9ca3af">
                {fmtTime(t)}
              </text>
            );
          })}

          {/* HOVER */}
          {hover && (
            <HoverGroup
              x={hover.x}
              y={Yp(closes[hover.i])}
              top={priceTop}
              bottom={volTop + volH}
              padL={padL}
              padR={padR}
              w={wView}
              readout={{
                time: fmtTime(times[hover.i]),
                price: closes[hover.i],
                vwap: vwap[hover.i],
                vol: vols[hover.i],
              }}
            />
          )}
        </svg>
      </div>
    </div>
  );
}

/* ------------------------------- Hover UI ------------------------------- */

function HoverGroup({
  x, y, top, bottom, padL, padR, w,
  readout,
}: {
  x: number; y: number; top: number; bottom: number; padL: number; padR: number; w: number;
  readout: { time: string; price: number; vwap: number; vol: number };
}) {
  const boxW = 170, boxH = 54;
  const leftSide = x < w / 2;
  const bx = leftSide ? x + 8 : x - boxW - 8;
  const by = Math.max(top + 4, Math.min(bottom - boxH - 4, y - boxH / 2));
  return (
    <>
      {/* vertical guide */}
      <line x1={x} y1={top} x2={x} y2={bottom} stroke="#6b7280" strokeDasharray="4 4" />
      {/* marker */}
      <circle cx={x} cy={y} r="3" fill="#60a5fa" stroke="#0a0a0a" strokeWidth="1" />
      {/* tooltip */}
      <g>
        <rect x={bx} y={by} width={boxW} height={boxH} rx="8" fill="#0b0f19" stroke="#1f2937" />
        <text x={bx + 8} y={by + 14} fontSize="11" fill="#e5e7eb">{readout.time}</text>
        <text x={bx + 8} y={by + 30} fontSize="11" fill="#93c5fd">Price: ${fmt(readout.price, 2)}</text>
        <text x={bx + 8} y={by + 44} fontSize="11" fill="#10b981">VWAP: ${fmt(readout.vwap, 2)} · Vol {fmtInt(readout.vol)}</text>
      </g>
    </>
  );
}

/* ------------------------------- Helpers ------------------------------- */

function toPath(
  xs: number[],
  ys: number[],
  X: (x: number) => number,
  Y: (v: number) => number,
  skipNaN = false
) {
  let d = "";
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (!isFinite(y)) { started = false; continue; }
    const cmd = started ? "L" : "M";
    d += `${cmd} ${X(xs[i])} ${Y(y)} `;
    started = true;
  }
  return d.trim();
}

function cumSum(arr: number[]) {
  const out = new Array(arr.length);
  let s = 0;
  for (let i = 0; i < arr.length; i++) { s += arr[i]; out[i] = s; }
  return out;
}

function niceTicks(min: number, max: number, n = 4) {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
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
function fmtTime(t: number | string) {
  const d = new Date(typeof t === "number" ? t : +new Date(t));
  const hh = d.getHours(), mm = d.getMinutes();
  const pad = (x: number) => (x < 10 ? "0" + x : String(x));
  return `${pad(hh)}:${pad(mm)}`;
}

/* ------------------- Ambient React (keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useRef<T>(v: T | null): { current: T | null };