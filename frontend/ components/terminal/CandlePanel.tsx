// frontend/components/CandlePanel.tsx
import React, { JSX, useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Brush,
  Bar,
  Line,
  ReferenceLine,
  Customized,
} from "recharts";

/** ------------------------------- Types ------------------------------- */
export type Candle = {
  t: number;         // epoch ms
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

interface Props {
  symbol?: string;
  candles?: Candle[];          // if provided, no fetch
  endpoint?: string;           // GET -> Candle[]
  height?: number;
  showSMA?: boolean;
  showEMA?: boolean;
  showVWAP?: boolean;
  sma1?: number;               // SMA window 1
  sma2?: number;               // SMA window 2
  ema1?: number;               // EMA window 1
  ema2?: number;               // EMA window 2
  colorUp?: string;
  colorDown?: string;
}

/** ------------------------------ Component ---------------------------- */
export default function CandlePanel({
  symbol = "AAPL",
  candles,
  endpoint = "/api/candles",
  height = 420,
  showSMA = true,
  showEMA = false,
  showVWAP = false,
  sma1 = 20,
  sma2 = 50,
  ema1 = 12,
  ema2 = 26,
  colorUp = "#16a34a",
  colorDown = "#ef4444",
}: Props) {
  const [rows, setRows] = useState<Candle[] | null>(candles ?? null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (candles) { setRows(candles); return; }
    (async () => {
      try {
        const q = symbol ? `?symbol=${encodeURIComponent(symbol)}` : "";
        const res = await fetch(`${endpoint}${q}`);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as Candle[];
        setRows(Array.isArray(json) ? json : []);
      } catch (e: any) {
        setErr(e?.message || "Failed to load candles");
        setRows([]);
      }
    })();
  }, [candles, endpoint, symbol]);

  const data = useMemo(() => {
    const xs = rows ?? [];
    if (!xs.length) return [];
    // compute indicators
    const closes = xs.map((d) => d.close);
    const highs = xs.map((d) => d.high);
    const lows  = xs.map((d) => d.low);
    const vols  = xs.map((d) => d.volume ?? 0);
    const s1 = showSMA ? SMA(closes, sma1) : [];
    const s2 = showSMA ? SMA(closes, sma2) : [];
    const e1 = showEMA ? EMA(closes, ema1) : [];
    const e2 = showEMA ? EMA(closes, ema2) : [];
    const vwap = showVWAP ? VWAP(highs, lows, closes, vols) : [];

    return xs.map((d, i) => ({
      ...d,
      sma1: s1[i] ?? null,
      sma2: s2[i] ?? null,
      ema1: e1[i] ?? null,
      ema2: e2[i] ?? null,
      vwap: vwap[i] ?? null,
      dir: d.close >= d.open ? "up" : "down",
    }));
  }, [rows, showSMA, showEMA, showVWAP, sma1, sma2, ema1, ema2]);

  if (err) return <div className="rounded-xl border p-3 text-sm text-red-600">Error: {err}</div>;
  if (!data.length) return <div className="rounded-xl border p-3 text-sm opacity-70">Loading candles…</div>;

  const latest = data[data.length - 1];

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Candles — {symbol}</h2>
          <div className="text-xs opacity-70">
            Last: <b className={latest.close >= latest.open ? "text-green-600" : "text-red-600"}>
              {fmt(latest.close)}
            </b>{" "}
            • O:{fmt(latest.open)} H:{fmt(latest.high)} L:{fmt(latest.low)} C:{fmt(latest.close)} Vol:{fmt(latest.volume ?? 0)}
          </div>
        </div>
        <div className="text-xs opacity-70">
          {new Date(data[0].t).toLocaleDateString()} → {new Date(latest.t).toLocaleDateString()}
        </div>
      </header>

      {/* Main chart */}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <XAxis
              dataKey="t"
              type="number"
              domain={["auto", "auto"]}
              tickFormatter={(t) => miniDate(t as number)}
            />
            <YAxis yAxisId="price" width={60} />
            <YAxis yAxisId="vol" orientation="right" width={48} hide />
            <Tooltip content={<OHLCtooltip currency="" />} />
            <Legend />

            {/* Volume (background) */}
            <Bar
              yAxisId="vol"
              dataKey="volume"
              barSize={2}
              fill="#9ca3af55"
              stroke="#9ca3af"
              name="Volume"
            />

            {/* Candles (customized) */}
            <Customized
              component={
                <CandleSeries
                  up={colorUp}
                  down={colorDown}
                  yAxisId="price"
                  xKey="t"
                />
              }
            />

            {/* Indicators */}
            {showSMA && (
              <>
                <Line yAxisId="price" type="monotone" dataKey="sma1" name={`SMA ${sma1}`} dot={false} stroke="#2563eb" isAnimationActive={false} />
                <Line yAxisId="price" type="monotone" dataKey="sma2" name={`SMA ${sma2}`} dot={false} stroke="#a855f7" isAnimationActive={false} />
              </>
            )}
            {showEMA && (
              <>
                <Line yAxisId="price" type="monotone" dataKey="ema1" name={`EMA ${ema1}`} dot={false} stroke="#0ea5e9" isAnimationActive={false} />
                <Line yAxisId="price" type="monotone" dataKey="ema2" name={`EMA ${ema2}`} dot={false} stroke="#f59e0b" isAnimationActive={false} />
              </>
            )}
            {showVWAP && (
              <Line yAxisId="price" type="monotone" dataKey="vwap" name="VWAP" dot={false} stroke="#eab308" isAnimationActive={false} />
            )}

            {/* Session separator (optional) */}
            {/* <ReferenceLine x={someTs} stroke="#e5e7eb" /> */}

            <Brush dataKey="t" height={24} travellerWidth={6} tickFormatter={(t) => miniDate(t as number)} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** --------------------------- Custom Candle --------------------------- */
/**
 * Draws candle wicks + bodies using chart scales from Recharts context.
 * Assumes X is numeric time `t` and price axis id is provided.
 */
function CandleSeries(props: {
  up: string;
  down: string;
  yAxisId?: string | number;
  xKey?: string;
}) {
  // Recharts injects chart state into Customized child via props
  return (
    <Customized component={(chartProps: any) => {
      const { formattedGraphicalItems, xAxisMap, yAxisMap, offset } = chartProps;
      // grab our series data (same data passed to chart)
      const series = formattedGraphicalItems?.[0]?.props?.data ?? [];
      const xAxis = xAxisMap?.[Object.keys(xAxisMap)[0]];
      const yKey = props.yAxisId ?? "price";
      const yAxis = Object.values(yAxisMap).find((a: any) => (a?.yAxisId ?? a?.props?.yAxisId) === yKey) || Object.values(yAxisMap)[0];

     if (!xAxis || !yAxis) return null;

    const xScale = getScale(xAxis);
    const yScale = getScale(yAxis);

     if (!xScale || !yScale) return null;
      const xKey = props.xKey ?? "t";

      const ctx: JSX.Element[] = [];
      const bodyWidth = Math.max(3, Math.min(10, Math.floor((offset?.width ?? 800) / Math.max(50, series.length))));

      for (let i = 0; i < series.length; i++) {
        const d = series[i];
        const x = xScale(d[xKey]);
        const yO = yScale(d.open);
        const yH = yScale(d.high);
        const yL = yScale(d.low);
        const yC = yScale(d.close);
        const up = d.close >= d.open;
        const fill = up ? props.up : props.down;

        // wick
        ctx.push(
          <line key={`w-${i}`} x1={x} x2={x} y1={yH} y2={yL} stroke={fill} strokeWidth={1} />
        );

        // body
        const yTop = Math.min(yO, yC);
        const yBot = Math.max(yO, yC);
        const h = Math.max(1, yBot - yTop);
        ctx.push(
          <rect
            key={`b-${i}`}
            x={x - bodyWidth / 2}
            y={yTop}
            width={bodyWidth}
            height={h}
            fill={fill}
            opacity={0.85}
            stroke={fill}
          />
        );
      }
      return <g>{ctx}</g>;
    }} />
  );
}

/** ---------------------------- Indicators ---------------------------- */

function SMA(arr: number[], w: number): (number | null)[] {
  const out: (number | null)[] = new Array(arr.length).fill(null);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
    if (i >= w) sum -= arr[i - w];
    if (i >= w - 1) out[i] = sum / w;
  }
  return out;
}
function EMA(arr: number[], w: number): (number | null)[] {
  const out: (number | null)[] = new Array(arr.length).fill(null);
  if (!arr.length) return out;
  const k = 2 / (w + 1);
  let prev = arr[0];
  out[0] = prev;
  for (let i = 1; i < arr.length; i++) {
    prev = arr[i] * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}
function VWAP(highs: number[], lows: number[], closes: number[], vols: number[]) {
  const out: (number | null)[] = new Array(closes.length).fill(null);
  let pv = 0, vv = 0;
  for (let i = 0; i < closes.length; i++) {
    const tp = (highs[i] + lows[i] + closes[i]) / 3;
    pv += tp * (vols[i] ?? 0);
    vv += (vols[i] ?? 0);
    out[i] = vv ? pv / vv : null;
  }
  return out;
}

/** ------------------------------ Helpers ------------------------------ */
function fmt(x: number) {
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 6 }); } catch { return String(x); }
}
function miniDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

/** Optional: custom tooltip showing OHLC nicely */
function OHLCtooltip({ active, payload, label, currency = "" }: any) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload as Candle & any;
  return (
    <div className="rounded-md border bg-white dark:bg-gray-900 dark:border-gray-800 p-2 text-xs">
      <div className="font-medium mb-1">{new Date(d.t).toLocaleString()}</div>
      <div>Open: <b>{fmt(d.open)}</b></div>
      <div>High: <b>{fmt(d.high)}</b></div>
      <div>Low: <b>{fmt(d.low)}</b></div>
      <div>Close: <b>{fmt(d.close)}</b></div>
      {"volume" in d ? <div>Vol: <b>{fmt(d.volume)}</b></div> : null}
      {d.sma1 != null && <div>SMA: <b>{fmt(d.sma1)}</b></div>}
      {d.ema1 != null && <div>EMA: <b>{fmt(d.ema1)}</b></div>}
      {d.vwap != null && <div>VWAP: <b>{fmt(d.vwap)}</b></div>}
    </div>
  );
}
type ScaleFn = (v: any) => number;

function getScale(axis: any): ScaleFn | null {
  if (!axis) return null;

  // Common patterns across chart libs
  const raw =
    (axis as any).scale ??
    (axis as any).props?.scale ??
    (typeof (axis as any).getScale === "function" ? (axis as any).getScale() : undefined);

  // If it's a function (d3-scale, recharts internal), use it
  if (typeof raw === "function") return raw as ScaleFn;

  // Some libs keep the function under `.scale()` (method returning a fn)
  if (typeof (axis as any).scale === "function") {
    const maybe = (axis as any).scale();
    if (typeof maybe === "function") return maybe as ScaleFn;
  }

  return null;
}