"use client";

import React from "react";

/** shape expected from your API */
type NewsItem = {
  id: string;
  title: string;
  url: string;
  source?: string;          // e.g., Bloomberg, FT
  tickers?: string[];       // ["AAPL","MSFT"]
  publishedAt: string;      // ISO timestamp
  sentiment?: "pos" | "neg" | "neu";
  summary?: string;
  image?: string;
};

type Props = {
  /** free-text search sent to API */
  query?: string;
  /** limit number of rows (after filters) */
  limit?: number;
  /** filter: symbols to include (comma-separated on API) */
  tickers?: string[];
  /** filter: news sources */
  sources?: string[];
  /** show compact summaries under titles */
  showSummaries?: boolean;
  /** API endpoint; default local route */
  endpoint?: string;
  /** called when a headline is clicked */
  onOpen?: (item: NewsItem) => void;
  /** optional header title */
  title?: string;
  /** poll interval in ms (0 = no polling) */
  refreshMs?: number;
};

export default function NewsHeadlinesMini({
  query = "",
  limit = 15,
  tickers,
  sources,
  showSummaries = false,
  endpoint = "/api/news/headlines",
  onOpen,
  title = "Top Headlines",
  refreshMs = 0,
}: Props) {
  const [items, setItems] = React.useState<NewsItem[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);

  async function load() {
    setLoading(true);
    setErr(null);
    try {
      const base = typeof window !== "undefined" ? window.location.origin : "http://localhost";
      const url = new URL(endpoint, base);
      if (query) url.searchParams.set("q", query);
      if (tickers?.length) url.searchParams.set("tickers", tickers.join(","));
      if (sources?.length) url.searchParams.set("sources", sources.join(","));
      url.searchParams.set("limit", String(limit));

      const res = await fetch(url.toString(), { cache: "no-store" });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

      const data = await res.json();
      const raw: NewsItem[] = Array.isArray(data) ? data : data.items ?? [];
      setItems(normalize(raw).slice(0, limit));
    } catch (e: any) {
      setErr(e?.message || "Failed to load headlines");
    } finally {
      setLoading(false);
    }
  }

  React.useEffect(() => {
    let timer: any;
    load();
    if (refreshMs && refreshMs > 0) {
      timer = setInterval(load, refreshMs);
    }
    return () => timer && clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, limit, endpoint, tickers?.join("|"), sources?.join("|"), refreshMs]);

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-3 py-2 border-b border-[#222] flex items-center justify-between">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">{loading ? "loading…" : timeNow()}</div>
      </div>

      {/* body */}
      {loading ? (
        <SkeletonList />
      ) : err ? (
        <div className="p-3 text-xs text-red-400">{err}</div>
      ) : items.length === 0 ? (
        <div className="p-3 text-xs text-gray-400">No headlines.</div>
      ) : (
        <ul className="divide-y divide-[#1f1f1f]">
          {items.map((n) => (
            <li key={n.id} className="group">
              <a
                href={n.url}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => {
                  onOpen?.(n);
                }}
                className="block px-3 py-2 hover:bg-[#111] focus:outline-none"
              >
                <div className="flex items-start gap-3">
                  {/* optional image thumbnail */}
                  {n.image ? (
                    <div className="w-16 h-12 flex-shrink-0 rounded overflow-hidden bg-[#1a1a1a]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={n.image}
                        alt=""
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    </div>
                  ) : null}

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-100 line-clamp-2">{n.title}</span>
                      {n.sentiment ? <SentimentBadge s={n.sentiment} /> : null}
                    </div>

                    <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-gray-500">
                      {n.source ? <span className="text-gray-400">{n.source}</span> : null}
                      <span>• {timeAgo(n.publishedAt)}</span>
                      {n.tickers?.slice(0, 5).map((t) => (
                        <span
                          key={t}
                          className="px-1.5 py-[1px] rounded border border-[#2a2a2a] text-gray-300"
                          title={t}
                        >
                          {t}
                        </span>
                      ))}
                      {n.tickers && n.tickers.length > 5 ? (
                        <span className="text-gray-500">+{n.tickers.length - 5}</span>
                      ) : null}
                    </div>

                    {showSummaries && n.summary ? (
                      <div className="mt-1 text-[12px] text-gray-300 line-clamp-2">{n.summary}</div>
                    ) : null}
                  </div>
                </div>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* --------------- helpers --------------- */

function normalize(arr: NewsItem[]): NewsItem[] {
  return arr
    .filter((x) => x && x.id && x.title && x.url && x.publishedAt)
    .map((x) => ({
      ...x,
      id: String(x.id),
      title: String(x.title),
      url: String(x.url),
      source: x.source ? String(x.source) : undefined,
      tickers: Array.isArray(x.tickers) ? x.tickers.map(String) : undefined,
      publishedAt: new Date(x.publishedAt).toISOString(),
      sentiment: x.sentiment as any,
      summary: x.summary,
      image: x.image,
    }));
}

function timeAgo(iso: string): string {
  const t = new Date(iso).getTime();
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function timeNow(): string {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

function SentimentBadge({ s }: { s: "pos" | "neg" | "neu" }) {
  const map: Record<string, string> = {
    pos: "bg-emerald-600/20 text-emerald-300 border border-emerald-700/50",
    neg: "bg-red-600/20 text-red-300 border border-red-700/50",
    neu: "bg-gray-600/20 text-gray-300 border border-gray-700/50",
  };
  const label = s === "pos" ? "Bullish" : s === "neg" ? "Bearish" : "Neutral";
  return <span className={`text-[10px] px-1.5 py-[1px] rounded ${map[s]}`}>{label}</span>;
}

function SkeletonList() {
  return (
    <ul className="divide-y divide-[#1f1f1f] animate-pulse">
      {Array.from({ length: 6 }).map((_, i) => (
        <li key={i} className="px-3 py-2">
          <div className="flex items-start gap-3">
            <div className="w-16 h-12 rounded bg-[#1a1a1a]" />
            <div className="flex-1">
              <div className="h-4 bg-[#1a1a1a] rounded w-5/6 mb-2" />
              <div className="h-3 bg-[#1a1a1a] rounded w-2/3" />
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}