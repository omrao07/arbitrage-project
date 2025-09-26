"use client";

/**
 * equitynews.tsx
 * Zero-import, self-contained news pane for an equity symbol.
 *
 * - Accepts pre-fetched news items via props (no imports, no data fetching)
 * - Filters: time range, sentiment, search
 * - Groups articles by date, compact cards with source/time
 * - Keyboard: "/" focuses search, "Esc" clears search
 * - Tailwind + inline SVG only
 */

export type EquityNewsItem = {
  id: string | number;
  title: string;
  source: string;          // e.g., "Bloomberg"
  url?: string;            // optional external link
  publishedAt: string;     // ISO timestamp
  summary?: string;
  sentiment?: "bullish" | "bearish" | "neutral";
  tickers?: string[];      // related tickers
  imageUrl?: string;       // optional thumbnail
};

export default function EquityNewsPane({
  symbol,
  items = [],
  className = "",
  showSummary = true,
  initialDays = 7,
}: {
  symbol: string;
  items?: EquityNewsItem[];
  className?: string;
  showSummary?: boolean;
  initialDays?: number;
}) {
  const [q, setQ] = useState("");
  const [days, setDays] = useState(initialDays);
  const [sent, setSent] = useState<"all" | "bullish" | "bearish" | "neutral">("all");
  const [w, setW] = useState(900);
  const hostRef = useRef<any>(null);

  // responsive width (no external libs)
  useEffect(() => {
    const el = hostRef.current as HTMLElement | null;
    if (!el) return;
    const apply = () => setW(Math.max(360, Math.floor(el.clientWidth)));
    apply();
    // @ts-ignore
    const ro: any = typeof ResizeObserver !== "undefined" ? new ResizeObserver(apply) : null;
    ro?.observe(el);
    return () => ro?.disconnect?.();
  }, []);

  // keyboard
  const searchRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (!inField && e.key === "/") {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (inField && e.key === "Escape") {
        (e.target as HTMLInputElement).blur();
        setQ("");
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const now = Date.now();
  const cutoff = now - days * 24 * 60 * 60 * 1000;

  // filter + sort
  const filtered = useMemo(() => {
    const qlc = q.trim().toLowerCase();
    const byTime = (it: EquityNewsItem) => +new Date(it.publishedAt);
    let arr = items
      .filter((it) => byTime(it) >= cutoff)
      .filter((it) => (sent === "all" ? true : (it.sentiment ?? "neutral") === sent))
      .filter((it) =>
        !qlc
          ? true
          : [it.title, it.summary, it.source, ...(it.tickers ?? [])]
              .filter(Boolean)
              .join(" ")
              .toLowerCase()
              .includes(qlc)
      )
      .sort((a, b) => +new Date(b.publishedAt) - +new Date(a.publishedAt));

    // optional: cap to 200 for perf if someone dumps a huge list
    if (arr.length > 200) arr = arr.slice(0, 200);
    return arr;
  }, [items, q, sent, cutoff]);

  // group by date (YYYY-MM-DD)
  const groups = useMemo(() => {
    const g = new Map<string, EquityNewsItem[]>();
    for (const it of filtered) {
      const k = new Date(it.publishedAt).toISOString().slice(0, 10);
      const list = g.get(k) ?? [];
      list.push(it);
      g.set(k, list);
    }
    // sort keys desc
    return Array.from(g.entries()).sort((a, b) => (a[0] < b[0] ? 1 : -1));
  }, [filtered]);

  return (
    <div ref={hostRef} className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header / controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{symbol} News</h3>
          <p className="text-xs text-neutral-400">
            {items.length} total · showing {filtered.length} in last {days}d
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <SearchBox
            value={q}
            onChange={setQ}
            placeholder="Search titles, sources, tickers…  (press /)"
            inputRef={searchRef}
            className="w-64"
          />
          <Select
            label="Sentiment"
            value={sent}
            onChange={(v) => setSent(v as any)}
            options={[
              { value: "all", label: "All" },
              { value: "bullish", label: "Bullish" },
              { value: "neutral", label: "Neutral" },
              { value: "bearish", label: "Bearish" },
            ]}
          />
          <Select
            label="Range"
            value={String(days)}
            onChange={(v) => setDays(parseInt(v) || 7)}
            options={[
              { value: "1", label: "1d" },
              { value: "3", label: "3d" },
              { value: "7", label: "7d" },
              { value: "14", label: "14d" },
              { value: "30", label: "30d" },
            ]}
          />
        </div>
      </div>

      {/* Groups & cards */}
      <div className="px-3 py-3">
        {groups.length === 0 && (
          <div className="px-2 py-8 text-center text-sm text-neutral-400">No articles match your filters.</div>
        )}
        {groups.map(([day, rows]) => (
          <section key={day} className="mb-4">
            <h4 className="mb-2 px-1 text-xs font-medium tracking-wide text-neutral-400">{prettyDay(day)}</h4>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {rows.map((it) => (
                <ArticleCard key={it.id} item={it} showSummary={showSummary} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

/* --------------------------------- Cards --------------------------------- */

function ArticleCard({ item, showSummary }: { item: EquityNewsItem; showSummary: boolean }) {
  const s = item.sentiment ?? "neutral";
  const chip = s === "bullish" ? "text-emerald-300 bg-emerald-600/15" :
              s === "bearish" ? "text-rose-300 bg-rose-600/15" :
              "text-amber-300 bg-amber-600/15";

  return (
    <div className="flex gap-3 overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 p-3 hover:bg-neutral-900/70">
      {item.imageUrl ? (
        <div className="relative hidden h-20 w-28 flex-none overflow-hidden rounded-md sm:block">
          <img
            src={item.imageUrl}
            alt=""
            className="h-full w-full object-cover"
            loading="lazy"
            referrerPolicy="no-referrer"
          />
        </div>
      ) : null}

      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-start justify-between gap-2">
          <h5 className="min-w-0 text-[13px] font-semibold leading-5 text-neutral-100">
            {item.title}
          </h5>
          <span className={`ml-2 inline-flex shrink-0 items-center rounded px-2 py-0.5 text-[10px] ${chip}`}>
            {s[0].toUpperCase() + s.slice(1)}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-neutral-400">
          <span className="truncate">{item.source}</span>
          <span>•</span>
          <span title={new Date(item.publishedAt).toLocaleString()}>
            {timeAgo(item.publishedAt)}
          </span>
          {item.tickers && item.tickers.length > 0 && (
            <>
              <span>•</span>
              <span className="text-neutral-500">
                {item.tickers.slice(0, 4).join(", ")}
                {item.tickers.length > 4 ? " +" + (item.tickers.length - 4) : ""}
              </span>
            </>
          )}
        </div>

        {showSummary && item.summary && (
          <p className="mt-1 line-clamp-3 text-[12px] leading-5 text-neutral-300">
            {item.summary}
          </p>
        )}

        <div className="mt-2 flex items-center gap-2">
          {item.url ? (
            <button
              onClick={() => safeOpen(item.url!)}
              className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-[11px] text-neutral-200 hover:bg-neutral-800"
            >
              Open
            </button>
          ) : null}
          <button
            onClick={() => copyText(item.title + (item.url ? ` — ${item.url}` : ""))}
            className="rounded-md border border-neutral-800 px-2 py-1 text-[11px] text-neutral-400 hover:bg-neutral-800"
            title="Copy title & link"
          >
            Copy
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------- Controls -------------------------------- */

function SearchBox({
  value,
  onChange,
  placeholder,
  className = "",
  inputRef,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  inputRef?: any;
}) {
  return (
    <label className={`relative ${className}`}>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md border border-neutral-700 bg-neutral-950 pl-8 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
      />
      <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
        <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </label>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="text-neutral-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-200"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/* --------------------------------- Utils --------------------------------- */

function safeOpen(url: string) {
  try {
    if (typeof window !== "undefined") window.open(url, "_blank", "noopener,noreferrer");
  } catch {}
}
function copyText(s: string) {
  try {
    if (typeof navigator !== "undefined" && (navigator as any).clipboard) {
      (navigator as any).clipboard.writeText(s);
    } else if (typeof document !== "undefined") {
      const ta = document.createElement("textarea");
      ta.value = s;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  } catch {}
}
function timeAgo(iso: string) {
  const ms = Date.now() - +new Date(iso);
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const d = Math.floor(hr / 24);
  if (d < 7) return `${d}d ago`;
  const dt = new Date(iso);
  return `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`;
}
function prettyDay(isoDay: string) {
  const today = new Date().toISOString().slice(0, 10);
  const yd = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  if (isoDay === today) return "Today";
  if (isoDay === yd) return "Yesterday";
  const d = new Date(isoDay);
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

/* --------------- Ambient React (keep zero imports; no imports) --------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useRef<T>(v: T | null): { current: T | null };

/* ------------------------------- Example --------------------------------
import { fetchEquitySnap } from "@/server/fetchequitysnap.server";
// Then pass articles to <EquityNewsPane symbol="AAPL" items={articles} />
--------------------------------------------------------------------------- */