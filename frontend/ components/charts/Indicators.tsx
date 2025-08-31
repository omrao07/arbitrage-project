// frontend/components/Indicators.tsx
import React, { useEffect, useMemo, useState } from "react";
import { ResponsiveContainer, LineChart, Line, YAxis, Tooltip } from "recharts";

/** ------------------------------- Types ------------------------------- */
type Candle = {
  time: string | number;
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
  height?: number;             // sparkline height
  colorUp?: string;
  colorDown?: string;
}

/** ------------------------------ Component --------------------------- */
export default function Indicators({
  symbol = "BTC/USDT",
  candles,
  endpoint = "/api/candles",
  height = 60,
  colorUp = "#16a34a",
  colorDown = "#ef4444",
}: Props) {
  const [data, setData] = useState<Candle[] | null>(candles ?? null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (candles) { setData(candles); return; }
    (async () => {
      try {
        const q = symbol ? `?symbol=${encodeURIComponent(symbol)}` : "";
        const res = await fetch(`${endpoint}${q}`);
        const json = (await res.json()) as Candle[];
        setData(Array.isArray(json) ? json : []);
      } catch (e: any) {
        setErr(e?.message || "Failed to load candles");
        setData([]);
      }
    })();
  }, [candles, endpoint, symbol]);

  const ind = useMemo(() => {
    if (!data || data.length < 5) return null;

    const closes = data.map(d => d.close);
    const highs  = data.map(d => d.high);
    const lows   = data.map(d => d.low);
    const vols   = data.map(d => d.volume ?? 0);

    const sma20 = SMA(closes, 20);
    const sma50 = SMA(closes, 50);
    const ema20 = EMA(closes, 20);
    const ema50 = EMA(closes, 50);
    const rsi14 = RSI(closes, 14);
    const macd  = MACD(closes, 12, 26, 9);        // returns { macd, signal, hist } arrays (number|null)[]
    const stoch = Stochastic(highs, lows, closes, 14, 3); // { k, d }
    const atr14 = ATR(highs, lows, closes, 14);
    const vwap  = VWAP(highs, lows, closes, vols);

    const last = data.length - 1;
    return {
      latest: {
        close: closes[last],
        sma20: lastVal(sma20), sma50: lastVal(sma50),
        ema20: lastVal(ema20), ema50: lastVal(ema50),
        rsi14: lastVal(rsi14),
        macd: lastVal(macd.macd), signal: lastVal(macd.signal), hist: lastVal(macd.hist),
        stochK: lastVal(stoch.k), stochD: lastVal(stoch.d),
        atr14: lastVal(atr14), vwap: lastVal(vwap),
      },
      series: { sma20, sma50, ema20, ema50, rsi14, macd, stoch, atr14, vwap },
    };
  }, [data]);

  const makeSpark = (series?: (number | null)[]) =>
    (data || []).map((_, i) => ({ i, v: series?.[i] ?? null }));

  if (err) return <div className="rounded-xl border p-3 text-sm text-red-600">{err}</div>;
  if (!data || !ind) return <div className="rounded-xl border p-3 text-sm opacity-70">Loading indicators…</div>;

  const { latest, series } = ind;
  const trendUp = latest.close >= (series.ema20.at(-1) ?? latest.close);

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Indicators — {symbol}</h2>
          <div className="text-xs opacity-70">
            Close: <b>{fmt(latest.close)}</b> •{" "}
            Trend: <b style={{ color: trendUp ? colorUp : colorDown }}>{trendUp ? "UP" : "DOWN"}</b>
          </div>
        </div>
      </header>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Card title="SMA 20 / 50" badge={badgeText(latest.sma20, latest.sma50)}>
          <Spark
            lines={[
              { data: makeSpark(series.sma20), color: "#3b82f6" },
              { data: makeSpark(series.sma50), color: "#a855f7" },
            ]}
            height={height}
          />
        </Card>

        <Card title="EMA 20 / 50" badge={badgeText(latest.ema20, latest.ema50)}>
          <Spark
            lines={[
              { data: makeSpark(series.ema20), color: "#0ea5e9" },
              { data: makeSpark(series.ema50), color: "#f59e0b" },
            ]}
            height={height}
          />
        </Card>

        <Card title="RSI 14" value={nfmt(latest.rsi14)}>
          <Spark lines={[{ data: makeSpark(series.rsi14), color: "#10b981" }]} height={height} yDomain={[0, 100]} />
          <div className="text-[11px] opacity-70 mt-1">30 oversold · 70 overbought</div>
        </Card>

        <Card
          title="MACD (12,26,9)"
          value={`MACD ${nfmt(latest.macd)} • Sig ${nfmt(latest.signal)} • Hist ${nfmt(latest.hist)}`}
        >
          <Spark
            lines={[
              { data: makeSpark(series.macd.macd), color: "#22d3ee" },
              { data: makeSpark(series.macd.signal), color: "#ef4444" },
            ]}
            height={height}
          />
        </Card>

        <Card title="Stochastic %K / %D" value={`${nfmt(latest.stochK)} / ${nfmt(latest.stochD)}`}>
          <Spark
            lines={[
              { data: makeSpark(series.stoch.k), color: "#22c55e" },
              { data: makeSpark(series.stoch.d), color: "#f97316" },
            ]}
            height={height}
            yDomain={[0, 100]}
          />
        </Card>

        <Card title="ATR 14" value={fmt(latest.atr14)}>
          <Spark lines={[{ data: makeSpark(series.atr14), color: "#9ca3af" }]} height={height} />
        </Card>

        <Card title="VWAP" value={fmt(latest.vwap)}>
          <Spark lines={[{ data: makeSpark(series.vwap), color: "#eab308" }]} height={height} />
        </Card>
      </div>
    </div>
  );
}

/** --------------------------- Indicator math --------------------------- */
/** These accept and/or return (number|null)[] safely. */

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

function EMA(arr: (number | null)[], w: number): (number | null)[] {
  const out: (number | null)[] = new Array(arr.length).fill(null);
  const k = 2 / (w + 1);
  // seed with first non-null
  let prevIdx = arr.findIndex(v => v !== null && v !== undefined);
  if (prevIdx === -1) return out;
  let prev = arr[prevIdx] as number;
  out[prevIdx] = prev;
  for (let i = prevIdx + 1; i < arr.length; i++) {
    const v = arr[i];
    if (v === null || v === undefined) { out[i] = out[i - 1]; continue; }
    prev = v * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

function RSI(closes: number[], w = 14): (number | null)[] {
  const out: (number | null)[] = new Array(closes.length).fill(null);
  if (closes.length < w + 1) return out;
  let gains = 0, losses = 0;
  for (let i = 1; i <= w; i++) {
    const ch = closes[i] - closes[i - 1];
    if (ch > 0) gains += ch; else losses -= ch;
  }
  let avgG = gains / w, avgL = losses / w;
  out[w] = 100 - 100 / (1 + (avgG / (avgL || 1e-12)));
  for (let i = w + 1; i < closes.length; i++) {
    const ch = closes[i] - closes[i - 1];
    const g = ch > 0 ? ch : 0;
    const l = ch < 0 ? -ch : 0;
    avgG = (avgG * (w - 1) + g) / w;
    avgL = (avgL * (w - 1) + l) / w;
    out[i] = 100 - 100 / (1 + (avgG / (avgL || 1e-12)));
  }
  return out;
}

function MACD(closes: number[], fast = 12, slow = 26, signal = 9) {
  const fastE = EMA(closes, fast) as number[];  // after warm-up, entries are numbers
  const slowE = EMA(closes, slow) as number[];
  const macd: (number | null)[] = closes.map((_, i) =>
    fastE[i] !== null && slowE[i] !== null ? (fastE[i] as number) - (slowE[i] as number) : null
  );
  const sig = EMA(macd, signal);
  const hist: (number | null)[] = macd.map((v, i) =>
    v !== null && sig[i] !== null ? (v as number) - (sig[i] as number) : null
  );
  return { macd, signal: sig, hist };
}

function Stochastic(highs: number[], lows: number[], closes: number[], kPeriod = 14, dPeriod = 3) {
  const k: (number | null)[] = new Array(closes.length).fill(null);
  for (let i = 0; i < closes.length; i++) {
    if (i < kPeriod - 1) continue;
    let hh = -Infinity, ll = Infinity;
    for (let j = i - kPeriod + 1; j <= i; j++) {
      if (highs[j] > hh) hh = highs[j];
      if (lows[j] < ll) ll = lows[j];
    }
    k[i] = ((closes[i] - ll) / Math.max(hh - ll, 1e-12)) * 100;
  }
  const d = SMA(k.map(v => v ?? 0), dPeriod);
  return { k, d };
}

function ATR(highs: number[], lows: number[], closes: number[], w = 14): (number | null)[] {
  const tr: number[] = new Array(closes.length).fill(0);
  for (let i = 0; i < closes.length; i++) {
    if (i === 0) { tr[i] = highs[i] - lows[i]; continue; }
    const h_l = highs[i] - lows[i];
    const h_pc = Math.abs(highs[i] - closes[i - 1]);
    const l_pc = Math.abs(lows[i] - closes[i - 1]);
    tr[i] = Math.max(h_l, h_pc, l_pc);
  }
  return EMA(tr, w);
}

function VWAP(highs: number[], lows: number[], closes: number[], vols: number[]): (number | null)[] {
  const out: (number | null)[] = new Array(closes.length).fill(null);
  let pvSum = 0, vSum = 0;
  for (let i = 0; i < closes.length; i++) {
    const tp = (highs[i] + lows[i] + closes[i]) / 3;
    pvSum += tp * (vols[i] ?? 0);
    vSum  += (vols[i] ?? 0);
    out[i] = vSum ? pvSum / vSum : null;
  }
  return out;
}

/** ------------------------------- UI -------------------------------- */

function Card({
  title, value, badge, children,
}: { title: string; value?: string | number; badge?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border dark:border-gray-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-semibold">{title}</div>
        <div className="flex items-center gap-2">
          {badge && <span className="text-[11px] px-2 py-0.5 rounded-md bg-gray-100 dark:bg-gray-800">{badge}</span>}
          {value !== undefined && <div className="text-sm font-medium">{value}</div>}
        </div>
      </div>
      {children}
    </div>
  );
}

function Spark({
  lines, height = 60, yDomain,
}: {
  lines: { data: { i: number; v: number | null }[]; color: string }[];
  height?: number;
  yDomain?: [number, number];
}) {
  const vals = lines.flatMap(l => l.data.map(d => d.v).filter((x): x is number => Number.isFinite(x as number)));
  const domain: [number, number] = yDomain ?? (vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1]);
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <YAxis domain={domain} hide />
          <Tooltip formatter={(v: any) => (typeof v === "number" ? v.toFixed(2) : v)} labelFormatter={() => ""} />
          {lines.map((l, idx) => (
            <Line
              key={idx}
              type="monotone"
              data={l.data}
              dataKey="v"
              stroke={l.color}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** --------------------------- helpers --------------------------- */

function lastVal(arr?: (number | null)[] | null) {
  if (!arr || !arr.length) return null;
  for (let i = arr.length - 1; i >= 0; i--) {
    const v = arr[i];
    if (v !== null && v !== undefined) return v;
  }
  return null;
}

function fmt(x?: number | null) {
  if (x === null || x === undefined) return "—";
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 4 }); } catch { return String(x); }
}
function nfmt(x?: number | null) {
  if (x === null || x === undefined) return "—";
  return Number(x).toFixed(2);
}

/** Small label comparing two lines (e.g., 20 vs 50 cross) */
function badgeText(a?: number | null, b?: number | null) {
  if (a == null || b == null) return "—";
  if (a > b) return "↑ above";
  if (a < b) return "↓ below";
  return "≈ equal";
}