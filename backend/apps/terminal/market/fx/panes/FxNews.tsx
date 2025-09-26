"use client";

/**
 * fx news.tsx
 * Zero-import, self-contained FX news list for a dark dashboard.
 *
 * Features
 * - Search (headline/source/body)
 * - Filter by pair (EURUSD, USDJPY, …) and sentiment
 * - Sort by time (newest/oldest) or source (A→Z)
 * - Compact cards with time-ago, source chip, sentiment chip, pair tags
 * - Expand/collapse long summaries
 * - Pagination (client-side) + CSV export (current view)
 *
 * Tailwind only. No imports, no external links required.
 */

export type FxNewsItem = {
  id: string;
  title: string;
  source: string;
  published: string;    // ISO datetime
  summary?: string;
  pairs?: string[];     // e.g., ["EURUSD","USDJPY"]
  sentiment?: "pos" | "neg" | "neu";
};

type SortKey = "new" | "old" | "source";

export default function FxNews({
  items = [],
  title = "FX News",
  className = "",
  pageSize = 10,
  defaultPair = "All",
  defaultSort: initialSort = "new",
}: {
  items: FxNewsItem[];
  title?: string;
  className?: string;
  pageSize?: number;
  defaultPair?: string;        // "All" or "EURUSD" etc.
  defaultSort?: SortKey;
}) {
  /* ------------------------------- Controls -------------------------------- */
  const [q, setQ] = useState("");
  const allPairs = useMemo(() => {
    const s = new Set<string>();
    for (const it of items) for (const p of (it.pairs || [])) s.add(normPair(p));
    return ["All", ...Array.from(s).sort()];
  }, [items]);
  const [pair, setPair] = useState<string>(allPairs.includes(defaultPair) ? defaultPair : "All");
  const [sent, setSent] = useState<"all" | "pos" | "neg" | "neu">("all");
  const [sort, setSort] = useState<SortKey>(initialSort);
  const [page, setPage] = useState(1);

  /* ------------------------------- Filtering -------------------------------- */
  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    const want = items.filter((it) => {
      if (pair !== "All") {
        const ps = (it.pairs || []).map(normPair);
        if (!ps.includes(pair)) return false;
      }
      if (sent !== "all" && (it.sentiment || "neu") !== sent) return false;
      if (!term) return true;
      const hay = (it.title + " " + (it.source || "") + " " + (it.summary || "")).toLowerCase();
      return hay.includes(term);
    });
    want.sort((a, b) => {
      if (sort === "new") return +new Date(b.published) - +new Date(a.published);
      if (sort === "old") return +new Date(a.published) - +new Date(b.published);
      return (a.source || "").localeCompare(b.source || "");
    });
    return want;
  }, [items, q, pair, sent, sort]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageItems = useMemo(() => filtered.slice((page - 1) * pageSize, page * pageSize), [filtered, page, pageSize]);

  useEffect(() => { setPage(1); }, [q, pair, sent, sort, pageSize]);

  /* -------------------------------- Render --------------------------------- */
  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">{filtered.length} headline{filtered.length!==1?"s":""}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="relative">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search headlines…"
              className="w-60 rounded-md border border-neutral-700 bg-neutral-950 pl-7 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
            />
            <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
              <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </label>

          <Select label="Pair" value={pair} onChange={setPair} options={allPairs.map((p) => ({ value: p, label: p }))} />
          <Select
            label="Sentiment"
            value={sent}
            onChange={(v) => setSent(v as any)}
            options={[
              { value: "all", label: "All" },
              { value: "pos", label: "Positive" },
              { value: "neu", label: "Neutral" },
              { value: "neg", label: "Negative" },
            ]}
          />
          <Select
            label="Sort"
            value={sort}
            onChange={(v) => setSort(v as SortKey)}
            options={[
              { value: "new", label: "Newest" },
              { value: "old", label: "Oldest" },
              { value: "source", label: "Source (A–Z)" },
            ]}
          />
          <button
            onClick={() => copyCSV(filtered)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 hover:bg-neutral-800"
          >
            Copy CSV
          </button>
        </div>
      </div>

      {/* List */}
      <div className="divide-y divide-neutral-800">
        {pageItems.length === 0 && (
          <div className="px-4 py-10 text-center text-xs text-neutral-500">No news matches your filters.</div>
        )}

        {pageItems.map((it) => (
          <NewsCard key={it.id} item={it} query={q} />
        ))}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between gap-2 border-t border-neutral-800 px-4 py-2 text-xs">
        <span className="text-neutral-400">
          Page {page} of {totalPages}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 disabled:opacity-50 hover:bg-neutral-800"
          >
            Prev
          </button>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 disabled:opacity-50 hover:bg-neutral-800"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

/* --------------------------------- Card --------------------------------- */

function NewsCard({ item, query }: { item: FxNewsItem; query: string }) {
  const [open, setOpen] = useState(false);
  const time = timeAgo(item.published);
  const pairs = (item.pairs || []).map(normPair);
  const sent = item.sentiment || "neu";
  const sentChip =
    sent === "pos" ? "bg-emerald-600/20 text-emerald-300 border-emerald-700/40" :
    sent === "neg" ? "bg-rose-600/20 text-rose-300 border-rose-700/40" :
    "bg-sky-600/20 text-sky-300 border-sky-700/40";

  const summary = item.summary || "";
  const short = summary.length > 220 && !open ? summary.slice(0, 220) + "…" : summary;

  return (
    <article className="px-4 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h4 className="text-sm font-semibold text-neutral-100" dangerouslySetInnerHTML={{ __html: highlight(item.title, query) }} />
          <p className="text-xs text-neutral-300" dangerouslySetInnerHTML={{ __html: highlight(short, query) }} />
          {summary.length > 220 && (
            <button
              onClick={() => setOpen((o) => !o)}
              className="mt-0.5 text-[11px] text-neutral-400 hover:text-neutral-200"
            >
              {open ? "Show less" : "Read more"}
            </button>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
            <span className="rounded border border-neutral-700 bg-neutral-950 px-1.5 py-0.5 text-neutral-400">{item.source}</span>
            <span className="text-neutral-500">•</span>
            <span className="text-neutral-400">{time}</span>
            {pairs.length > 0 && (
              <>
                <span className="text-neutral-500">•</span>
                <div className="flex flex-wrap items-center gap-1">
                  {pairs.map((p) => (
                    <span key={p} className="rounded bg-neutral-800 px-1.5 py-0.5 text-neutral-300">{prettyPair(p)}</span>
                  ))}
                </div>
              </>
            )}
            <span className="text-neutral-500">•</span>
            <span className={`rounded border px-1.5 py-0.5 ${sentChip}`}>{sentimentLabel(sent)}</span>
          </div>
        </div>

        {/* Mini sentinel icon */}
        <div className="shrink-0 rounded-md border border-neutral-800 p-2 text-neutral-400">
          <svg width="18" height="18" viewBox="0 0 24 24">
            <path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
      </div>
    </article>
  );
}

/* -------------------------------- Controls -------------------------------- */

function Select({
  label, value, onChange, options,
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
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

/* --------------------------------- Helpers -------------------------------- */

function highlight(text: string, q: string) {
  if (!q) return esc(text);
  const re = new RegExp(`(${escapeReg(q)})`, "ig");
  return esc(text).replace(re, "<mark class='bg-amber-300/30 text-amber-200 rounded px-0.5'>$1</mark>");
}

function timeAgo(iso: string) {
  const t = +new Date(iso);
  if (!Number.isFinite(t)) return "unknown";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  const m = s / 60;
  if (m < 60) return `${Math.floor(m)}m ago`;
  const h = m / 60;
  if (h < 24) return `${Math.floor(h)}h ago`;
  const d = h / 24;
  if (d < 7) return `${Math.floor(d)}d ago`;
  const dt = new Date(t);
  return dt.toISOString().slice(0, 10);
}

function sentimentLabel(s: "pos" | "neg" | "neu") {
  return s === "pos" ? "Positive" : s === "neg" ? "Negative" : "Neutral";
}

function normPair(p: string) {
  const s = (p || "").toUpperCase().replace(/[^A-Z]/g, "");
  return s.length === 6 ? s : "";
}
function prettyPair(p: string) {
  const n = normPair(p) || p;
  return `${n.slice(0,3)}/${n.slice(3)}`;
}

function copyCSV(rows: FxNewsItem[]) {
  const head = ["id", "title", "source", "published", "sentiment", "pairs", "summary"];
  const lines = [head.join(",")];
  for (const r of rows) {
    lines.push([
      r.id,
      r.title,
      r.source,
      r.published,
      r.sentiment || "neu",
      (r.pairs || []).join("|"),
      r.summary || ""
    ].map(csv).join(","));
  }
  const csvStr = lines.join("\n");
  try { (navigator as any).clipboard?.writeText(csvStr); } catch {}
}

function csv(x: any) {
  if (x == null) return "";
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
}

function esc(s: string) {
  return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]!));
}
function escapeReg(s: string) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;

/* --------------------------------- Example ---------------------------------
const demo: FxNewsItem[] = [
  { id: "1", title: "Dollar eases as traders eye US data", source: "Reuters", published: new Date().toISOString(), sentiment: "neu", pairs: ["EURUSD","USDJPY"], summary: "The dollar slipped from recent highs ahead of jobs data…" },
  { id: "2", title: "Yen rallies on BOJ chatter", source: "Bloomberg", published: new Date(Date.now()-3600e3).toISOString(), sentiment: "pos", pairs: ["USDJPY"], summary: "Speculation over policy tightening lifted the yen…" },
];
<FxNews items={demo} />
-------------------------------------------------------------------------------- */