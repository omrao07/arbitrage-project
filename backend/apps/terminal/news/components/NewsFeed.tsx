// components/newsfeed.tsx (or app/market/newsfeed.tsx)
// A dependency-free, client-side News Feed with search, filters, auto-refresh, and CSV/JSON export.
// Drop-in ready. No external UI libs.

"use client";

import React from "react";

type Item = {
  id: string;
  title: string;
  summary?: string;
  url?: string;
  source: string;
  category: "Equities" | "FX" | "Fixed Income" | "Derivatives" | "Macro";
  symbols?: string[];
  sentiment?: -1 | 0 | 1;
  ts: number; // epoch ms
};

// ------------------------ MOCK FETCHER (replace with your API) ------------------------
async function fetchNewsMock(): Promise<Item[]> {
  // Simulate latency
  await new Promise((r) => setTimeout(r, 250));
  const now = Date.now();
  const base: Item[] = [
    {
      id: "1",
      title: "RBI seen holding rates; commentary hints at durable pause",
      summary:
        "Economists expect the Reserve Bank of India to keep the policy rate unchanged while maintaining a cautious tone on inflation.",
      url: "#",
      source: "Economic Times",
      category: "Macro",
      symbols: ["USD/INR", "NIFTY"],
      sentiment: 0,
      ts: now - 6 * 60 * 1000,
    },
    {
      id: "2",
      title: "Infosys rallies on strong deal wins; guidance intact",
      summary:
        "Large deal pipeline remains healthy; margins guided to improve amid utilization gains.",
      url: "#",
      source: "Mint",
      category: "Equities",
      symbols: ["INFY", "NIFTY IT"],
      sentiment: 1,
      ts: now - 18 * 60 * 1000,
    },
    {
      id: "3",
      title: "USD/JPY slips as UST yields ease; risk appetite improves",
      url: "#",
      source: "FXStreet",
      category: "FX",
      symbols: ["USD/JPY", "DXY"],
      sentiment: 0,
      ts: now - 34 * 60 * 1000,
    },
    {
      id: "4",
      title: "India 10Y yields steady; demand strong at auction",
      summary:
        "Primary dealers report robust bidding amid steady global rates; supply well absorbed.",
      url: "#",
      source: "Bloomberg",
      category: "Fixed Income",
      symbols: ["IN10Y"],
      sentiment: 1,
      ts: now - 55 * 60 * 1000,
    },
    {
      id: "5",
      title: "Crude spikes; OPEC+ chatter revives supply risk premium",
      summary:
        "Brent pushes higher as traders price tighter balances into Q4.",
      url: "#",
      source: "Reuters",
      category: "Derivatives",
      symbols: ["BRENT", "CRUDEOIL"],
      sentiment: -1,
      ts: now - 85 * 60 * 1000,
    },
  ];

  // create a few varied duplicates with different timestamps to emulate a feed
  const extra = Array.from({ length: 15 }).map((_, i) => {
    const b = base[i % base.length];
    return {
      ...b,
      id: `${b.id}-${i + 10}`,
      ts: now - (i + 2) * 7 * 60 * 1000,
      title:
        i % 3 === 0
          ? b.title
          : b.title.replace(/seen|rallies|slips|steady|spikes/i, (m) => m + " again"),
    } as Item;
  });

  return [...base, ...extra].sort((a, b) => b.ts - a.ts);
}

// ------------------------ UTILITIES ------------------------
function timeAgo(ts: number) {
  const s = Math.max(1, Math.floor((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function download(filename: string, contents: string, mime = "text/plain") {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function toCSV(items: Item[]) {
  const esc = (v: any) =>
    `"${String(v ?? "").replace(/"/g, '""').replace(/\n/g, " ")}"`;
  const header = [
    "id",
    "title",
    "summary",
    "url",
    "source",
    "category",
    "symbols",
    "sentiment",
    "timestamp",
  ].join(",");
  const rows = items.map((i) =>
    [
      i.id,
      i.title,
      i.summary ?? "",
      i.url ?? "",
      i.source,
      i.category,
      (i.symbols ?? []).join("|"),
      i.sentiment ?? "",
      new Date(i.ts).toISOString(),
    ]
      .map(esc)
      .join(","),
  );
  return [header, ...rows].join("\n");
}

// ------------------------ COMPONENT ------------------------
export default function NewsFeed({
  autoRefreshMs = 60_000,
  initialCategory = "All",
}: {
  autoRefreshMs?: number;
  initialCategory?: "All" | Item["category"];
}) {
  const [items, setItems] = React.useState<Item[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [query, setQuery] = React.useState("");
  const [category, setCategory] =
    React.useState<"All" | Item["category"]>(initialCategory);
  const [show, setShow] = React.useState(12);
  const [auto, setAuto] = React.useState(autoRefreshMs > 0);
  const searchRef = React.useRef<HTMLInputElement | null>(null);

  async function load() {
    try {
      setLoading(true);
      const data = await fetchNewsMock(); // swap with your real fetch
      setItems(data);
    } finally {
      setLoading(false);
    }
  }

  React.useEffect(() => {
    load();
  }, []);

  React.useEffect(() => {
    if (!auto || autoRefreshMs <= 0) return;
    const id = setInterval(load, autoRefreshMs);
    return () => clearInterval(id);
  }, [auto, autoRefreshMs]);

  // keyboard shortcuts
  React.useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "/") {
        e.preventDefault();
        searchRef.current?.focus();
      } else if (e.key.toLowerCase() === "r") {
        load();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((i) => {
      const catOk = category === "All" || i.category === category;
      if (!catOk) return false;
      if (!q) return true;
      const hay =
        `${i.title} ${i.summary ?? ""} ${i.source} ${(i.symbols ?? []).join(" ")}`.toLowerCase();
      return hay.includes(q);
    });
  }, [items, query, category]);

  const visible = filtered.slice(0, show);

  return (
    <section style={wrap}>
      <style>{css}</style>

      {/* Controls */}
      <div style={controls}>
        <div style={leftControls}>
          <div style={searchWrap}>
            <span style={searchIcon}>⌕</span>
            <input
              ref={searchRef}
              placeholder="Search title, source, symbols…  (Press / to focus, R to refresh)"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setShow(12);
              }}
              style={searchInput}
            />
          </div>

          <div style={tabs}>
            {(["All", "Equities", "FX", "Fixed Income", "Derivatives", "Macro"] as const).map(
              (c) => (
                <button
                  key={c}
                  onClick={() => {
                    setCategory(c);
                    setShow(12);
                  }}
                  style={{
                    ...tab,
                    ...(category === c ? tabActive : null),
                  }}
                >
                  {c}
                </button>
              ),
            )}
          </div>
        </div>

        <div style={rightControls}>
          <label style={switchRow} title="Auto-refresh">
            <input
              type="checkbox"
              checked={auto}
              onChange={(e) => setAuto(e.target.checked)}
            />
            <span>Auto</span>
          </label>
          <button style={btn} onClick={() => load()} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              style={btnAlt}
              onClick={() => download("news.json", JSON.stringify(filtered, null, 2), "application/json")}
              title="Export filtered as JSON"
            >
              JSON
            </button>
            <button
              style={btnAlt}
              onClick={() => download("news.csv", toCSV(filtered), "text/csv")}
              title="Export filtered as CSV"
            >
              CSV
            </button>
          </div>
        </div>
      </div>

      {/* Feed */}
      {loading && items.length === 0 ? (
        <Skeleton />
      ) : visible.length === 0 ? (
        <EmptyState query={query} />
      ) : (
        <div style={list}>
          {visible.map((it) => (
            <article key={it.id} style={card}>
              <header style={cardHead}>
                <a
                  href={it.url || "#"}
                  target="_blank"
                  rel="noreferrer"
                  style={titleLink}
                >
                  {it.title}
                </a>
                <div style={chipRow}>
                  <Chip
                    label={it.category}
                    bg="#eef2ff"
                    fg="#4338ca"
                  />
                  <Chip
                    label={it.source}
                    bg="#f5f5f5"
                    fg="#111"
                  />
                  {typeof it.sentiment !== "undefined" ? (
                    <Chip
                      label={
                        it.sentiment > 0 ? "Bullish" : it.sentiment < 0 ? "Bearish" : "Neutral"
                      }
                      bg={it.sentiment > 0 ? "#ecfdf5" : it.sentiment < 0 ? "#fef2f2" : "#f4f4f5"}
                      fg={it.sentiment > 0 ? "#067647" : it.sentiment < 0 ? "#b42318" : "#111"}
                    />
                  ) : null}
                </div>
              </header>

              {it.summary ? <p style={summary}>{it.summary}</p> : null}

              <footer style={metaRow}>
                <div style={metaLeft}>
                  <span style={timeText}>{timeAgo(it.ts)}</span>
                  {it.symbols && it.symbols.length > 0 ? (
                    <span style={dot}>•</span>
                  ) : null}
                  {it.symbols && it.symbols.length > 0 ? (
                    <span style={syms}>
                      {it.symbols.map((s) => (
                        <code key={s} style={symCode}>{s}</code>
                      ))}
                    </span>
                  ) : null}
                </div>
                {it.url ? (
                  <a
                    href={it.url}
                    target="_blank"
                    rel="noreferrer"
                    style={openLink}
                  >
                    Open ↗
                  </a>
                ) : null}
              </footer>
            </article>
          ))}
        </div>
      )}

      {/* Load more */}
      {visible.length < filtered.length ? (
        <div style={moreRow}>
          <button style={btn} onClick={() => setShow((s) => s + 12)}>
            Load more ({filtered.length - visible.length} more)
          </button>
        </div>
      ) : null}
    </section>
  );
}

// ------------------------ PRESENTATIONAL SUB-COMPS ------------------------
function Chip({ label, bg, fg }: { label: string; bg: string; fg: string }) {
  return (
    <span
      style={{
        background: bg,
        color: fg,
        borderRadius: 999,
        fontSize: 12,
        padding: "2px 8px",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

function Skeleton() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
      <style>{shimmerCss}</style>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} style={card}>
          <div
            className="shimmer"
            style={{ height: 16, width: "70%", borderRadius: 8, marginBottom: 8 }}
          />
          <div
            className="shimmer"
            style={{ height: 12, width: "95%", borderRadius: 8, marginBottom: 6 }}
          />
          <div
            className="shimmer"
            style={{ height: 12, width: "80%", borderRadius: 8 }}
          />
          <div
            className="shimmer"
            style={{ height: 10, width: "30%", borderRadius: 8, marginTop: 10 }}
          />
        </div>
      ))}
    </div>
  );
}

function EmptyState({ query }: { query: string }) {
  return (
    <div style={empty}>
      <div style={emptyIcon}>📰</div>
      <div style={{ fontWeight: 600 }}>No news found</div>
      <div style={{ color: "#666", fontSize: 13 }}>
        {query ? <>Try a different search term.</> : <>Feed will appear here once data loads.</>}
      </div>
    </div>
  );
}

// ------------------------ STYLES (inline + tiny CSS) ------------------------
const wrap: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
  padding: 16,
};

const controls: React.CSSProperties = {
  display: "flex",
  gap: 12,
  justifyContent: "space-between",
  alignItems: "center",
  flexWrap: "wrap",
};

const leftControls: React.CSSProperties = { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" };
const rightControls: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };

const searchWrap: React.CSSProperties = {
  position: "relative",
  display: "flex",
  alignItems: "center",
  width: 380,
  maxWidth: "90vw",
};

const searchIcon: React.CSSProperties = {
  position: "absolute",
  left: 10,
  fontSize: 12,
  color: "#777",
};

const searchInput: React.CSSProperties = {
  width: "100%",
  height: 34,
  padding: "6px 10px 6px 26px",
  borderRadius: 10,
  border: "1px solid #ddd",
  outline: "none",
} as const;

const tabs: React.CSSProperties = { display: "flex", gap: 6, flexWrap: "wrap" };
const tab: React.CSSProperties = {
  appearance: "none",
  border: "1px solid #e5e7eb",
  background: "#f5f5f7",
  color: "#111",
  borderRadius: 999,
  padding: "6px 10px",
  cursor: "pointer",
  fontSize: 12.5,
};
const tabActive: React.CSSProperties = { background: "#111", color: "#fff", borderColor: "#111" };

const btn: React.CSSProperties = {
  appearance: "none",
  border: "1px solid #111",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
  background: "#111",
  color: "#fff",
};
const btnAlt: React.CSSProperties = {
  appearance: "none",
  border: "1px solid #d4d4d8",
  borderRadius: 10,
  padding: "6px 10px",
  cursor: "pointer",
  fontSize: 13,
  background: "#f4f4f5",
  color: "#111",
};

const switchRow: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  fontSize: 12.5,
  color: "#444",
};

const list: React.CSSProperties = { display: "grid", gridTemplateColumns: "1fr", gap: 12 };
const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 14,
};

const cardHead: React.CSSProperties = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 };
const chipRow: React.CSSProperties = { display: "flex", gap: 6, flexWrap: "wrap" };

const titleLink: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 700,
  lineHeight: "22px",
  textDecoration: "none",
  color: "#111",
};

const summary: React.CSSProperties = { margin: "8px 0 6px", color: "#444", lineHeight: "20px" };
const metaRow: React.CSSProperties = { display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 6 };
const metaLeft: React.CSSProperties = { display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" };
const timeText: React.CSSProperties = { color: "#666", fontSize: 12.5 };
const syms: React.CSSProperties = { display: "inline-flex", gap: 6, flexWrap: "wrap" };
const symCode: React.CSSProperties = { background: "#f5f5f7", padding: "1px 6px", borderRadius: 6, fontSize: 12 };
const dot: React.CSSProperties = { color: "#bbb" };

const openLink: React.CSSProperties = { color: "#0f62fe", textDecoration: "none", fontSize: 13 };

const moreRow: React.CSSProperties = { display: "flex", justifyContent: "center", marginTop: 8 };

const empty: React.CSSProperties = {
  display: "grid",
  placeItems: "center",
  padding: 40,
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  gap: 6,
};
const emptyIcon: React.CSSProperties = { fontSize: 28 };

// tiny CSS for shimmer + responsive grid
const shimmerCss = `
  @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
  .shimmer { background: linear-gradient(90deg, #f2f2f2 25%, #e9e9e9 37%, #f2f2f2 63%); background-size: 400px 100%; animation: shimmer 1.2s infinite linear; }
  @media (prefers-color-scheme: dark) {
    .shimmer { background: linear-gradient(90deg, #1d1d1f 25%, #2a2a2e 37%, #1d1d1f 63%); }
    a { color: #9ecaff; }
  }
`;

const css = `
  @media (min-width: 900px) {
    /* turn list into 2 columns on wide screens */
    section:has(.news-grid) {}
    /* (we're not using :has in practice; grid remains 1col for simplicity) */
  }
  ${shimmerCss}
`;
