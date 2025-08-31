// frontend/components/SentimentRadar.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

type RadarPoint = { dim: string; score: number };   // score in [-1, 1]
type SeriesPoint = { t: string; score: number };
type Headline = { id: string; ts: number; source: string; title: string; url?: string; sentiment?: number };

interface Props {
  symbol?: string;                // optional: filter endpoints by ticker
  radarEndpoint?: string;         // GET -> RadarPoint[]
  seriesEndpoint?: string;        // GET -> SeriesPoint[]
  newsEndpoint?: string;          // GET -> Headline[]
}

export default function SentimentRadar({
  symbol,
  radarEndpoint = "/api/sentiment/radar",
  seriesEndpoint = "/api/sentiment/series",
  newsEndpoint = "/api/sentiment/news",
}: Props) {
  const [radar, setRadar] = useState<RadarPoint[]>([]);
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [news, setNews] = useState<Headline[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const q = symbol ? `?symbol=${encodeURIComponent(symbol)}` : "";
        const [r, s, n] = await Promise.all([
          fetch(`${radarEndpoint}${q}`).then((res) => res.json()),
          fetch(`${seriesEndpoint}${q}`).then((res) => res.json()),
          fetch(`${newsEndpoint}${q}`).then((res) => res.json()),
        ]);
        setRadar(sanitizeRadar(r));
        setSeries(Array.isArray(s) ? s : []);
        setNews(Array.isArray(n) ? n : []);
      } catch (e: any) {
        setErr(e?.message || "Sentiment API error");
      } finally {
        setLoading(false);
      }
    })();
  }, [symbol, radarEndpoint, seriesEndpoint, newsEndpoint]);

  const composite = useMemo(() => {
    if (!series.length) return 0;
    const last = series[series.length - 1]?.score ?? 0;
    return clamp(last, -1, 1);
  }, [series]);

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Sentiment Radar{symbol ? ` — ${symbol}` : ""}</h2>
          <p className="text-sm opacity-70">Composite sentiment by source/sector + trend & headlines</p>
        </div>
        <CompositeBadge value={composite} />
      </header>

      {loading && <div className="text-sm opacity-70">Loading sentiment…</div>}
      {err && !loading && <div className="text-sm text-red-500">Error: {err}</div>}

      {!loading && !err && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Radar */}
          <section className="lg:col-span-1">
            <h3 className="text-lg font-medium mb-2">Radar (−1 to +1)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radar}>
                <PolarGrid />
                <PolarAngleAxis dataKey="dim" />
                <PolarRadiusAxis domain={[-1, 1]} tickCount={5} />
                <Radar name="Sentiment" dataKey="score" stroke="#0A84FF" fill="#0A84FF" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          </section>

          {/* Series */}
          <section className="lg:col-span-2">
            <h3 className="text-lg font-medium mb-2">Composite Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={series}>
                <XAxis dataKey="t" hide />
                <YAxis domain={[-1, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="score" stroke="#34C759" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </section>

          {/* Headlines */}
          <section className="lg:col-span-3">
            <h3 className="text-lg font-medium mb-2">Top Headlines</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
              {news.map((h) => (
                <a
                  key={h.id}
                  href={h.url || "#"}
                  target="_blank"
                  rel="noreferrer"
                  className="p-3 rounded-xl border dark:border-gray-700 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-sm font-semibold line-clamp-2">{h.title}</div>
                    <span className="text-xs opacity-70 ml-2">{h.source}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs opacity-80">
                    <span>{new Date(h.ts).toLocaleString()}</span>
                    {typeof h.sentiment === "number" && (
                      <span className={h.sentiment >= 0 ? "text-green-600" : "text-red-500"}>
                        {h.sentiment >= 0 ? "↑" : "↓"} {Math.abs(h.sentiment).toFixed(2)}
                      </span>
                    )}
                  </div>
                </a>
              ))}
              {news.length === 0 && <div className="text-sm opacity-60">No headlines.</div>}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

/* ------------------------------ Small bits ------------------------------ */

function CompositeBadge({ value }: { value: number }) {
  const text =
    value > 0.4 ? "Bullish"
    : value < -0.4 ? "Bearish"
    : "Neutral";

  const cls =
    value > 0.4 ? "bg-green-100 text-green-700"
    : value < -0.4 ? "bg-red-100 text-red-700"
    : "bg-gray-100 text-gray-700";

  return (
    <div className={`px-3 py-1 rounded-lg text-sm font-medium ${cls}`}>
      {text} ({value >= 0 ? "+" : ""}{value.toFixed(2)})
    </div>
  );
}

function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

function sanitizeRadar(raw: any): RadarPoint[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((r) => ({
    dim: String(r.dim ?? r.dimension ?? r.name ?? "?"),
    score: clamp(Number(r.score ?? r.value ?? 0), -1, 1),
  }));
}