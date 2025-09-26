// app/market/newsfilter.tsx
// No imports. No hooks. Pure client-side <form> that calls onChange(filters)
// whenever fields change. Uses DOM values (uncontrolled inputs).

"use client";

export type Filters = {
  q: string;
  category: "All" | "Equities" | "FX" | "Fixed Income" | "Derivatives" | "Macro";
  source: "All" | string;
  sentiment: "Any" | "Bullish" | "Neutral" | "Bearish";
  from?: number; // epoch ms
  to?: number;   // epoch ms
  symbols: string[];
};

type Props = {
  onChange: (f: Filters) => void;
  categories?: Filters["category"][];
  sources?: string[];
  initial?: Partial<Filters>;
  showExportLink?: boolean;
};

export default function NewsFilter({
  onChange,
  categories = ["All", "Equities", "FX", "Fixed Income", "Derivatives", "Macro"],
  sources = [],
  initial,
  showExportLink = false,
}: Props) {
  // ---- event helpers (no hooks) ----
  function readFilters(form: HTMLFormElement): Filters {
    const fd = new FormData(form);
    const q = (fd.get("q") as string) || "";
    const category = ((fd.get("category") as string) || "All") as Filters["category"];
    const source = ((fd.get("source") as string) || "All") as Filters["source"];
    const sentiment = ((fd.get("sentiment") as string) || "Any") as Filters["sentiment"];
    const symbols = tokenizeSymbols((fd.get("symbols") as string) || "");
    const fromStr = (fd.get("from") as string) || "";
    const toStr = (fd.get("to") as string) || "";
    const from = fromStr ? Date.parse(fromStr) : undefined;
    const to = toStr ? Date.parse(toStr) : undefined;
    return { q, category, source, sentiment, from, to, symbols };
  }

  function handleFormChange(e: any) {
    const form = e.currentTarget as HTMLFormElement;
    const f = readFilters(form);
    if (showExportLink) {
      const a = document.getElementById("nf-export-link") as HTMLAnchorElement | null;
      if (a) a.href = `?${toQuery(f)}`;
    }
    onChange(f);
  }

  function quick(hours: number, ev: any) {
    ev.preventDefault();
    const btn = ev.currentTarget as HTMLButtonElement;
    const form = btn.form as HTMLFormElement | null;
    if (!form) return;
    const now = Date.now();
    const start = now - hours * 3600_000;
    const fromEl = form.elements.namedItem("from") as HTMLInputElement;
    const toEl = form.elements.namedItem("to") as HTMLInputElement;
    fromEl.value = toLocalInput(start);
    toEl.value = toLocalInput(now);
    handleFormChange({ currentTarget: form });
  }

  function clearAll(ev: any) {
    ev.preventDefault();
    const btn = ev.currentTarget as HTMLButtonElement;
    const form = btn.form as HTMLFormElement | null;
    if (!form) return;
    form.reset();
    // ensure selects default back to All/Any even if browser kept history
    (form.elements.namedItem("category") as HTMLSelectElement).value = "All";
    (form.elements.namedItem("source") as HTMLSelectElement).value = "All";
    (form.elements.namedItem("sentiment") as HTMLSelectElement).value = "Any";
    (form.elements.namedItem("symbols") as HTMLInputElement).value = "";
    (form.elements.namedItem("q") as HTMLInputElement).value = "";
    (form.elements.namedItem("from") as HTMLInputElement).value = "";
    (form.elements.namedItem("to") as HTMLInputElement).value = "";
    handleFormChange({ currentTarget: form });
  }

  const initFrom = initial?.from ? toLocalInput(initial.from) : "";
  const initTo = initial?.to ? toLocalInput(initial.to) : "";

  return (
    <section style={wrap}>
      <style>{css}</style>

      <form onChange={handleFormChange} onInput={handleFormChange} style={formRow}>
        {/* Row 1 */}
        <div style={row}>
          <div style={searchWrap}>
            <span style={searchIcon}>⌕</span>
            <input
              name="q"
              defaultValue={initial?.q ?? ""}
              placeholder="Search title, source, symbols…"
              style={searchInput}
            />
          </div>

          <select name="category" defaultValue={initial?.category ?? "All"} style={select} title="Category">
            {categories.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <select name="source" defaultValue={initial?.source ?? "All"} style={select} title="Source">
            <option value="All">All sources</option>
            {sources.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>

          <select name="sentiment" defaultValue={initial?.sentiment ?? "Any"} style={select} title="Sentiment">
            <option>Any</option>
            <option>Bullish</option>
            <option>Neutral</option>
            <option>Bearish</option>
          </select>
        </div>

        {/* Row 2 */}
        <div style={row}>
          <div style={symWrap}>
            <span style={symLabel}>Symbols</span>
            <input
              name="symbols"
              defaultValue={(initial?.symbols ?? []).join(" ")}
              placeholder="INFY TCS USD/INR …"
              style={symInput}
            />
          </div>

          <div style={rangeWrap}>
            <label style={rangeLab}>From</label>
            <input type="datetime-local" name="from" defaultValue={initFrom} style={dt} />
            <label style={rangeLab}>To</label>
            <input type="datetime-local" name="to" defaultValue={initTo} style={dt} />
          </div>

          <div style={quickRow}>
            <button style={chipBtn} onClick={(e) => quick(1, e)}>1h</button>
            <button style={chipBtn} onClick={(e) => quick(3, e)}>3h</button>
            <button style={chipBtn} onClick={(e) => quick(6, e)}>6h</button>
            <button style={chipBtn} onClick={(e) => quick(24, e)}>24h</button>
          </div>

          <button style={clearBtn} onClick={clearAll}>Clear</button>
        </div>

        {/* Link reflecting current filters (no state) */}
        {showExportLink ? (
          <div style={linkRow}>
            <a id="nf-export-link" style={linkBtn} href={`?${toQuery({
              q: initial?.q ?? "",
              category: initial?.category ?? "All",
              source: initial?.source ?? "All",
              sentiment: initial?.sentiment ?? "Any",
              from: initial?.from,
              to: initial?.to,
              symbols: initial?.symbols ?? [],
            })}`}>
              Link to this view ↗
            </a>
          </div>
        ) : null}
      </form>
    </section>
  );
}

/* ---------------- utilities ---------------- */
function tokenizeSymbols(s: string): string[] {
  return s.split(/[\s,|]+/).map((t) => t.trim().toUpperCase()).filter(Boolean).slice(0, 24);
}
function toLocalInput(ts: number): string {
  const d = new Date(ts);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}
function toQuery(f: Partial<Filters>): string {
  const p = new URLSearchParams();
  if (f.q) p.set("q", f.q);
  if (f.category && f.category !== "All") p.set("cat", f.category);
  if (f.source && f.source !== "All") p.set("src", f.source);
  if (f.sentiment && f.sentiment !== "Any") p.set("sent", f.sentiment);
  if (f.from) p.set("from", String(f.from));
  if (f.to) p.set("to", String(f.to));
  if (f.symbols && f.symbols.length) p.set("syms", f.symbols.join(","));
  return p.toString();
}

/* ---------------- styles ---------------- */
const wrap: any = { display: "flex", flexDirection: "column", gap: 10, padding: 8 };
const formRow: any = { display: "block" };
const row: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };

const searchWrap: any = { position: "relative", width: 360, maxWidth: "90vw" };
const searchIcon: any = { position: "absolute", left: 10, top: 8, fontSize: 12, color: "#777" };
const searchInput: any = { width: "100%", height: 32, padding: "6px 10px 6px 26px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const select: any = { height: 32, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const symWrap: any = { display: "flex", alignItems: "center", gap: 6, minWidth: 260, flex: "1 1 260px" };
const symLabel: any = { fontSize: 12, color: "#555" };
const symInput: any = { flex: 1, height: 32, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const rangeWrap: any = { display: "inline-flex", alignItems: "center", gap: 6 };
const rangeLab: any = { fontSize: 12, color: "#555" };
const dt: any = { height: 32, padding: "4px 8px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const quickRow: any = { display: "inline-flex", gap: 6 };
const chipBtn: any = { border: "1px solid #e5e7eb", background: "#f5f5f7", borderRadius: 999, padding: "4px 10px", cursor: "pointer", fontSize: 12 };

const clearBtn: any = { border: "1px solid #ef4444", background: "#fee2e2", color: "#b42318", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 12 };

const linkRow: any = { marginTop: 4 };
const linkBtn: any = { textDecoration: "none", border: "1px solid #d4d4d8", background: "#fafafa", color: "#111", borderRadius: 10, padding: "6px 10px", fontSize: 12 };

const css = `
  @media (prefers-color-scheme: dark) {
    a { color: #9ecaff; }
  }
`;
