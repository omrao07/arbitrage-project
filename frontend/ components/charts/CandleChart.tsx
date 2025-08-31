// frontend/components/CandleChart.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  Time,
} from "lightweight-charts";

type Candle = {
  time: string | number; // "YYYY-MM-DD" or unix ms/s
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

interface Props {
  symbol?: string;
  candles?: Candle[];          // if given, skip fetch
  endpoint?: string;           // GET -> Candle[]
  wsUrl?: string;              // optional WS streaming updates
  height?: number;
  showVolume?: boolean;
  showEMA?: boolean;
  emaWindows?: number[];       // e.g. [20, 50]
  dark?: boolean;              // force theme (else follow OS)
}

export default function CandleChart({
  symbol = "BTC/USDT",
  candles,
  endpoint = "/api/candles",
  wsUrl,
  height = 420,
  showVolume = true,
  showEMA = true,
  emaWindows = [20, 50],
  dark,
}: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const emaRefs = useRef<ISeriesApi<"Line">[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const [data, setData] = useState<Candle[] | null>(candles ?? null);

  // Fetch if not provided
  useEffect(() => {
    if (candles) { setData(candles); return; }
    (async () => {
      try {
        const q = symbol ? `?symbol=${encodeURIComponent(symbol)}` : "";
        const res = await fetch(`${endpoint}${q}`);
        const json = (await res.json()) as Candle[];
        setData(Array.isArray(json) ? json : []);
      } catch (e) {
        console.error("Candle fetch failed:", e);
        setData([]);
      }
    })();
  }, [candles, endpoint, symbol]);

  const isDark = useMemo(() => {
    if (typeof dark === "boolean") return dark;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  }, [dark]);

  // Build chart
  useEffect(() => {
    if (!hostRef.current || !data) return;

    // cleanup prior instance
    chartRef.current?.remove();
    chartRef.current = null;
    candleRef.current = null;
    volRef.current = null;
    emaRefs.current = [];

    const chart = createChart(hostRef.current, {
      autoSize: true,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: isDark ? "#D1D5DB" : "#374151",
      },
      grid: {
        vertLines: { color: isDark ? "#1F2937" : "#E5E7EB" },
        horzLines: { color: isDark ? "#1F2937" : "#E5E7EB" },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, rightOffset: 6, barSpacing: 6 },
      crosshair: { mode: 1 },
    });
    chartRef.current = chart;

    // Candles
    const candlesSeries = chart.addCandlestickSeries();
    candlesSeries.applyOptions({
      upColor: "#16a34a",
      borderUpColor: "#16a34a",
      wickUpColor: "#16a34a",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      wickDownColor: "#ef4444",
    });
    candlesSeries.setData(data.map(mapCandle));
    candleRef.current = candlesSeries;

    // Volume (bottom)
    if (showVolume) {
      const vol = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" } });
      vol.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
      vol.setData(
        data.map(d => ({
          time: toTime(d.time),
          value: d.volume ?? 0,
          color: d.close >= d.open ? "#16a34a" : "#ef4444",
        }))
      );
      volRef.current = vol;
    }

    // EMAs
    if (showEMA && emaWindows.length) {
      emaRefs.current = emaWindows.map((w, i) => {
        const color = i === 0 ? "#3b82f6" : i === 1 ? "#f59e0b" : "#a855f7";
        const s = chart.addLineSeries({ color, lineWidth: 2, priceLineVisible: false });
        s.setData(ema(data, w).map(p => ({ time: toTime(p.time), value: p.value })));
        return s;
      });
    }

    const onResize = () => chart.applyOptions({ width: hostRef.current!.clientWidth });
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.remove(); };
  }, [data, height, isDark, showVolume, showEMA, emaWindows]);

  // Optional WS updates
  useEffect(() => {
    if (!wsUrl || !candleRef.current) return;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "subscribe", channel: "candles", symbol }));
    ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m && typeof m.time !== "undefined") {
          candleRef.current!.update(mapCandle(m));
          if (showVolume && volRef.current) {
            volRef.current.update({
              time: toTime(m.time),
              value: m.volume ?? 0,
              color: m.close >= m.open ? "#16a34a" : "#ef4444",
            });
          }
          // (EMA incremental update omitted for brevity)
        }
      } catch {}
    };
    ws.onclose = () => (wsRef.current = null);
    ws.onerror = (e) => console.error("WS error:", e);
    return () => ws.close();
  }, [wsUrl, symbol, showVolume]);

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Candlestick — {symbol}</h2>
        <span className="text-xs opacity-70">
          {showEMA ? `EMA ${emaWindows.join(", ")}` : "EMA off"}
        </span>
      </div>
      <div ref={hostRef} style={{ height }} />
    </div>
  );
}

/* ---------------- helpers ---------------- */
function toTime(t: string | number): Time {
  if (typeof t === "number") return (String(t).length > 10 ? Math.floor(t / 1000) : t) as Time;
  return t as Time;
}
function mapCandle(d: Candle): CandlestickData<Time> {
  return { time: toTime(d.time), open: d.open, high: d.high, low: d.low, close: d.close };
}
function ema(data: Candle[], win: number): { time: Candle["time"]; value: number }[] {
  if (!data.length || win <= 1) return [];
  const k = 2 / (win + 1);
  let prev = data[0].close;
  const out = [{ time: data[0].time, value: prev }];
  for (let i = 1; i < data.length; i++) {
    const v = data[i].close * k + prev * (1 - k);
    out.push({ time: data[i].time, value: v });
    prev = v;
  }
  return out;
}