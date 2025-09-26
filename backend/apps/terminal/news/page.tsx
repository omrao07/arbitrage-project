// app/news/page.tsx
// No imports. Server component that fetches & renders a News list.
// - Reads filters from query string (?q=&cat=&from=&to=&syms=&limit=)
// - Pulls from JSON endpoints in env: NEWS_ENDPOINTS="https://api.example.com/news,https://api2.example.com/feed"
// - If no endpoints are set (or fail), falls back to a small mocked feed.

export const dynamic = "force-dynamic";

/* ---------------- types ---------------- */
type Category = "All" | "Equities" | "FX" | "Fixed Income" | "Derivatives" | "Macro";
type Item = {
  id: string;
  title: string;
  summary?: string;
  url?: string;
  source: string;
  category: Exclude<Category, "All">;
  symbols?: string[];
  sentiment?: -1 | 0 | 1;
  ts: number; // epoch ms
};

type QP = {
  q: string;
  cat: Category;
  from: number; // 0 = unset
  to: number;   // 0 = unset
  syms: string[];
  limit: number;
};

/* ---------------- page ---------------- */
export default async function NewsPage({
  searchParams,
}: {
  searchParams?: Record<string, string | string[] | undefined>;
}) {
  const qp = normalizeQP(searchParams);
  const endpoints = getEnvEndpoints();

  let items: Item[] = [];
  try {
    items = endpoints.length
      ? (await Promise.all(endpoints.map((u) => fetchJsonSafe(u, 4000).then((j) => normalizePayload(j, u)).catch(() => []))))
          .flat()
      : mockFeed();
  } catch {
    items = mockFeed();
  }

  const filtered = applyFilters(items, qp);
  const deduped = dedupe(filtered).sort((a, b) => b.ts - a.ts);
  const limited = qp.limit > 0 ? deduped.slice(0, qp.limit) : deduped;

  const topSources = topCounts(limited.map((i) => i.source));
  const CATS: Category[] = ["All", "Equities", "FX", "Fixed Income", "Derivatives", "Macro"];

  return (
    <main style={wrap}>
      <style>{css}</style>

      <header style={header}>
        <h1 style={h1}>News</h1>
        <p style={sub}>Latest market headlines across Equities, FX, Fixed Income, Derivatives, and Macro.</p>
      </header>

      {/* Controls (server-side form; no client JS) */}
      <form method="get" style={controls}>
        <div style={searchWrap}>
          <span style={searchIcon}>⌕</span>
          <input
            name="q"
            defaultValue={qp.q}
            placeholder="Search title, source, symbols…"
            style={searchInput}
          />
        </div>

        <select name="cat" defaultValue={qp.cat} style={select} title="Category">
          {CATS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>

        <div style={rangeWrap}>
          <label style={rangeLab}>From</label>
          <input type="datetime-local" name="from" defaultValue={qp.from ? toLocal(qp.from) : ""} style={dt} />
          <label style={rangeLab}>To</label>
          <input type="datetime-local" name="to" defaultValue={qp.to ? toLocal(qp.to) : ""} style={dt} />
        </div>

        <input
          name="syms"
          defaultValue={qp.syms.join(" ")}
          placeholder="Symbols (space/comma separated)"
          style={symInput}
          title="Symbols"
        />

        <input
          name="limit"
          type="number"
          min={1}
          max={500}
          defaultValue={qp.limit}
          style={limitInput}
          title="Max items"
        />

        <button type="submit" style={primaryBtn}>Apply</button>

        {/* Quick category tabs (links so it's all server-rendered) */}
        <nav style={tabs}>
          {CATS.map((c) => (
            <a
              key={c}
              href={`?${qs({ ...qp, cat: c })}`}
              style={{ ...tab, ...(qp.cat === c ? tabActive : null) }}
            >
              {c}
            </a>
          ))}
        </nav>
      </form>

      {/* Meta */}
      <section style={metaRow}>
        <div>Showing <strong>{limited.length}</strong> of {deduped.length} results</div>
        {topSources.length ? (
          <div style={chipsRow}>
            {topSources.slice(0, 6).map(([src, n]) => (
              <span key={src} style={chip}>{src} · {n}</span>
            ))}
          </div>
        ) : null}
      </section>

      {/* List */}
      {limited.length === 0 ? (
        <div style={empty}>
          <div style={emptyIcon}>📰</div>
          <div style={{ fontWeight: 600 }}>No news found</div>
          <div style={{ color: "#666", fontSize: 13 }}>Try adjusting your filters.</div>
        </div>
      ) : (
        <section style={list}>
          {limited.map((it) => (
            <article key={it.id} style={card}>
              <header style={cardHead}>
                <a href={it.url || "#"} target={it.url ? "_blank" : "_self"} rel="noreferrer" style={titleLink}>
                  {it.title}
                </a>
                <div style={chipRow}>
                  <Chip label={it.category} bg="#eef2ff" fg="#4338ca" />
                  <Chip label={it.source} bg="#f4f4f5" fg="#111" />
                  {it.symbols?.length ? <Chip label={it.symbols.join(" ")} bg="#ecfeff" fg="#0e7490" /> : null}
                  {typeof it.sentiment !== "undefined" ? (
                    <Chip
                      label={it.sentiment > 0 ? "Bullish" : it.sentiment < 0 ? "Bearish" : "Neutral"}
                      bg={it.sentiment > 0 ? "#ecfdf5" : it.sentiment < 0 ? "#fef2f2" : "#f4f4f5"}
                      fg={it.sentiment > 0 ? "#067647" : it.sentiment < 0 ? "#b42318" : "#111"}
                    />
                  ) : null}
                </div>
              </header>

              {it.summary ? <p style={summary}>{it.summary}</p> : null}

              <footer style={metaItem}>
                <span style={timeText}>{timeAgo(it.ts)}</span>
                {it.url ? (
                  <a href={it.url} target="_blank" rel="noreferrer" style={openLink}>
                    Open ↗
                  </a>
                ) : null}
              </footer>
            </article>
          ))}
        </section>
      )}
    </main>
  );
}

/* ---------------- tiny presentational ---------------- */
function Chip({ label, bg, fg }: { label: string; bg: string; fg: string }) {
  return <span style={{ background: bg, color: fg, borderRadius: 999, fontSize: 12, padding: "2px 8px", whiteSpace: "nowrap" }}>{label}</span>;
}

/* ---------------- helpers ---------------- */
function normalizeQP(sp?: Record<string, string | string[] | undefined>): QP {
  const get = (k: string) => (Array.isArray(sp?.[k]) ? (sp?.[k] as string[])[0] : (sp?.[k] as string | undefined)) || "";
  const q = (get("q") || "").trim();
  const cat = (get("cat") || "All") as Category;
  const from = parseDT(get("from"));
  const to = parseDT(get("to"));
  const syms = splitSyms(get("syms"));
  const limit = clamp(parseInt(get("limit") || "100", 10), 1, 500);
  return { q, cat, from, to, syms, limit };
}

function parseDT(v?: string): number {
  if (!v) return 0;
  const t = Date.parse(v);
  return Number.isNaN(t) ? 0 : t;
}

function getEnvEndpoints(): string[] {
  try {
    // @ts-ignore
    const raw: string = (typeof process !== "undefined" && process.env && process.env.NEWS_ENDPOINTS) || "";
    return raw.split(",").map((s) => s.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

async function fetchJsonSafe(url: string, timeoutMs = 4000): Promise<any> {
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : (null as any);
  const id = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  try {
    const res = await fetch(url, { signal: ctrl?.signal });
    if (!res.ok) throw new Error(String(res.status));
    const ct = res.headers.get("content-type") || "";
    if (/json/i.test(ct)) return await res.json();
    const txt = await res.text();
    try { return JSON.parse(txt); } catch { return []; }
  } finally {
    if (id) clearTimeout(id as any);
  }
}

function normalizePayload(payload: any, url: string): Item[] {
  const arr = Array.isArray(payload) ? payload : Array.isArray(payload?.items) ? payload.items : [];
  const host = hostOf(url) || "source";
  const out: Item[] = [];
  for (const r of arr) {
    const title = str(r.title) || str(r.headline) || str(r.name) || "";
    if (!title) continue;
    const link = str(r.url) || str(r.link);
    const source = str(r.source) || str(r.publisher) || (link ? hostOf(link) : host) || host;
    const summary = str(r.summary) || str(r.description);
    const category =
      mapCategory(str(r.category)) ||
      mapCategory(str(r.section)) ||
      "Macro";
    const syms = (Array.isArray(r.symbols) ? r.symbols : Array.isArray(r.tickers) ? r.tickers : [])
      .map((s: any) => String(s).toUpperCase().trim())
      .filter(Boolean);
    const ts =
      num(r.ts) ||
      num(r.time) ||
      num(r.published) ||
      (str(r.published_at) ? Date.parse(String(r.published_at)) : undefined) ||
      (str(r.pubDate) ? Date.parse(String(r.pubDate)) : undefined) ||
      Date.now();
    const sentiment: -1 | 0 | 1 =
      (num(r.sentiment) as any) ??
      quickSentiment(`${title} ${summary || ""}`);

    out.push({
      id: str(r.id) || hashId(title + (link || "") + source + String(ts)),
      title,
      summary,
      url: link,
      source,
      category,
      symbols: syms.length ? syms : undefined,
      sentiment,
      ts: ts || Date.now(),
    });
  }
  return out;
}

function applyFilters(items: Item[], qp: QP): Item[] {
  const q = qp.q.toLowerCase();
  return items.filter((i) => {
    if (qp.cat !== "All" && i.category !== qp.cat) return false;
    if (qp.from && i.ts < qp.from) return false;
    if (qp.to && i.ts > qp.to) return false;
    if (qp.syms.length) {
      const has = (i.symbols || []).some((s) => qp.syms.includes(s.toUpperCase()));
      if (!has) return false;
    }
    if (q) {
      const hay = `${i.title} ${i.summary || ""} ${i.source} ${(i.symbols || []).join(" ")}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function dedupe(items: Item[]): Item[] {
  const seen = new Set<string>();
  const out: Item[] = [];
  for (const it of items) {
    const k = (it.id || "") + "|" + (it.url || "") + "|" + it.title;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(it);
  }
  return out;
}

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

function topCounts(xs: string[]): [string, number][] {
  const m = new Map<string, number>();
  xs.forEach((x) => m.set(x, (m.get(x) || 0) + 1));
  return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
}

/* mappers */
function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}
function num(v: any): number | undefined {
  if (v == null || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
function splitSyms(s?: string): string[] {
  if (!s) return [];
  return s.split(/[\s,|]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
}
function mapCategory(s?: string): Exclude<Category, "All"> | undefined {
  if (!s) return undefined;
  const t = s.toLowerCase();
  if (/equit|stock|share/.test(t)) return "Equities";
  if (/\bfx\b|forex|currency|usd|eur|inr|jpy|gbp|cny/.test(t)) return "FX";
  if (/fixed|bond|yield|gilt|10y|sovereign/.test(t)) return "Fixed Income";
  if (/deriv|option|futures|opec|brent|crude|hedge/.test(t)) return "Derivatives";
  return "Macro";
}
function quickSentiment(text: string): -1 | 0 | 1 {
  const s = text.toLowerCase();
  let score = 0;
  ["beats", "surge", "rally", "growth", "upgrade", "strong", "improve"].forEach((w) => (score += s.includes(w) ? 1 : 0));
  ["miss", "cuts", "drop", "fall", "downgrade", "weak", "risk"].forEach((w) => (score -= s.includes(w) ? 1 : 0));
  return score > 0 ? 1 : score < 0 ? -1 : 0;
}
function hashId(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return "n_" + (h >>> 0).toString(36);
}
function hostOf(u?: string) {
  try {
    return u ? new URL(u).hostname.replace(/^www\./, "") : undefined;
  } catch {
    return undefined;
  }
}
function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}
function toLocal(ts: number): string {
  const d = new Date(ts);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}
function qs(qp: QP): string {
  const p = new URLSearchParams();
  if (qp.q) p.set("q", qp.q);
  if (qp.cat) p.set("cat", qp.cat);
  if (qp.from) p.set("from", String(qp.from));
  if (qp.to) p.set("to", String(qp.to));
  if (qp.syms.length) p.set("syms", qp.syms.join(","));
  if (qp.limit) p.set("limit", String(qp.limit));
  return p.toString();
}

/* ---------------- fallback feed ---------------- */
function mockFeed(): Item[] {
  const now = Date.now();
  const mk = (
    id: string,
    title: string,
    source: string,
    category: Exclude<Category, "All">,
    minsAgo: number,
    summary?: string,
    symbols?: string[],
    url?: string,
  ): Item => ({
    id, title, source, category, summary, symbols, url,
    sentiment: quickSentiment(title + " " + (summary || "")),
    ts: now - minsAgo * 60_000,
  });
  return [
    mk("m1", "RBI seen holding rates; commentary hints at durable pause", "Economic Times", "Macro", 8, "Cautious tone on inflation."),
    mk("m2", "Infosys rallies on strong deal wins; guidance intact", "Mint", "Equities", 16, "Margins guided to improve.", ["INFY"]),
    mk("m3", "USD/JPY slips as UST yields ease; risk appetite improves", "FXStreet", "FX", 34, undefined, ["USD/JPY"]),
    mk("m4", "India 10Y yields steady; auction demand robust", "Bloomberg", "Fixed Income", 52, "Supply well absorbed."),
    mk("m5", "Brent spikes as OPEC+ chatter revives risk premium", "Reuters", "Derivatives", 77, "Tighter balances into Q4.", ["BRENT"]),
  ];
}

/* ---------------- styles ---------------- */
const wrap: any = { display: "flex", flexDirection: "column", gap: 12, padding: 16 };
const header: any = { marginBottom: 2 };
const h1: any = { margin: 0, fontSize: 22, lineHeight: "28px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const controls: any = {
  display: "flex",
  gap: 8,
  alignItems: "center",
  flexWrap: "wrap",
  marginBottom: 8,
};

const searchWrap: any = { position: "relative", width: 360, maxWidth: "90vw" };
const searchIcon: any = { position: "absolute", left: 10, top: 8, fontSize: 12, color: "#777" };
const searchInput: any = { width: "100%", height: 34, padding: "6px 10px 6px 26px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const select: any = { height: 34, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };
const rangeWrap: any = { display: "inline-flex", alignItems: "center", gap: 6 };
const rangeLab: any = { fontSize: 12, color: "#555" };
const dt: any = { height: 34, padding: "4px 8px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };
const symInput: any = { height: 34, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", outline: "none", minWidth: 200 };
const limitInput: any = { height: 34, width: 80, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const primaryBtn: any = {
  appearance: "none",
  border: "1px solid #111",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
  background: "#111",
  color: "#fff",
};

const tabs: any = { display: "flex", gap: 6, flexWrap: "wrap", marginLeft: "auto" };
const tab: any = {
  display: "inline-block",
  padding: "6px 10px",
  borderRadius: 999,
  border: "1px solid #e5e7eb",
  background: "#f5f5f7",
  color: "#111",
  textDecoration: "none",
  fontSize: 12.5,
};
const tabActive: any = { background: "#111", color: "#fff", borderColor: "#111" };

const metaRow: any = { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 };

const chipsRow: any = { display: "flex", gap: 6, flexWrap: "wrap" };
const chip: any = { background: "#f5f5f7", border: "1px solid #e5e7eb", borderRadius: 999, padding: "2px 8px", fontSize: 12 };

const list: any = { display: "grid", gridTemplateColumns: "1fr", gap: 12 };

const card: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 14,
};

const cardHead: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 };
const chipRow: any = { display: "flex", gap: 6, flexWrap: "wrap" };

const titleLink: any = { fontSize: 16, fontWeight: 700, lineHeight: "22px", textDecoration: "none", color: "#111" };

const summary: any = { margin: "8px 0 6px", color: "#444", lineHeight: "20px" };
const metaItem: any = { display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 6 };
const timeText: any = { color: "#666", fontSize: 12.5 };
const openLink: any = { color: "#0f62fe", textDecoration: "none", fontSize: 13 };

const empty: any = {
  display: "grid",
  placeItems: "center",
  padding: 40,
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  gap: 6,
};
const emptyIcon: any = { fontSize: 28 };

const css = `
  @media (prefers-color-scheme: dark) {
    a { color: #9ecaff; }
    article[style] {
      background: #0b0b0c !important;
      border-color: rgba(255,255,255,0.08) !important;
      box-shadow: 0 6px 24px rgba(0,0,0,0.6) !important;
    }
  }
`;
