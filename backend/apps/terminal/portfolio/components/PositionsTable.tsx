// app/components/positionstable.tsx
// No imports. No hooks. Self-contained positions table with DOM-driven sort/filter/export.
// - Click column headers to sort (cycles asc/desc)
// - Type in search to filter by symbol/name/sector
// - Export visible rows to CSV
// - Computed columns: Value, Cost, P&L, P&L%, Weight
// - Optional row click callback via data-onclick attribute (see onRowClick prop)

"use client";

type Position = {
  symbol: string;
  name?: string;
  sector?: string;
  qty: number;      // current quantity (>= 0)
  price: number;    // last price (> 0)
  avgCost?: number; // average cost per unit (>= 0)
  dayPct?: number;  // optional daily change in %
  currency?: string;
};

type Props = {
  rows: Position[];
  unit?: string;                    // currency symbol/prefix (e.g., "$", "₹")
  title?: string;
  note?: string;
  showSector?: boolean;             // default true if any sector present
  onRowClick?: (symbol: string) => void | Promise<void>;
  dense?: boolean;                  // tighter row height
};

export default function PositionsTable({
  rows,
  unit = "",
  title = "Positions",
  note,
  showSector,
  onRowClick,
  dense = false,
}: Props) {
  const anySector = rows.some((r) => !!r.sector);
  const showSec = showSector ?? anySector;

  // Pre-compute enriched rows for initial render only (no hooks)
  const enriched = rows.map((r) => enrich(r));
  const totals = computeTotals(enriched);

  // After hydration: wire up sorting / filtering / export
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("pt-root");
      if (!root) return;
      const table = root.querySelector("table")!;
      const search = root.querySelector<HTMLInputElement>('input[name="pt-search"]')!;
      const exportBtn = root.querySelector<HTMLButtonElement>('#pt-export')!;
      const thead = table.tHead!;
      const tbody = table.tBodies[0];

      // Sort handler
      thead.addEventListener("click", (e) => {
        const th = (e.target as HTMLElement).closest("th[data-key]") as HTMLTableHeaderCellElement | null;
        if (!th) return;
        const key = th.dataset.key!;
        const type = th.dataset.type || "num"; // "num" | "str"
        const current = th.getAttribute("aria-sort") as "ascending" | "descending" | null;
        const next: "ascending" | "descending" = current === "ascending" ? "descending" : "ascending";

        // clear others
        thead.querySelectorAll("th[aria-sort]").forEach((el) => el.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", next);

        const rowsEls = Array.from(tbody.querySelectorAll("tr[data-row]"));
        rowsEls.sort((a, b) => {
          const va = a.querySelector<HTMLElement>(`[data-k="${key}"]`)?.dataset.value ?? "";
          const vb = b.querySelector<HTMLElement>(`[data-k="${key}"]`)?.dataset.value ?? "";
          let cmp = 0;
          if (type === "num") {
            cmp = (Number(va) || 0) - (Number(vb) || 0);
          } else {
            cmp = String(va).localeCompare(String(vb));
          }
          return next === "ascending" ? cmp : -cmp;
        });
        rowsEls.forEach((tr) => tbody.appendChild(tr));
      });

      // Filter handler
      const applyFilter = () => {
        const q = (search.value || "").trim().toLowerCase();
        const trs = Array.from(tbody.querySelectorAll<HTMLTableRowElement>("tr[data-row]"));
        trs.forEach((tr) => {
          const hay = (tr.dataset.hay || "").toLowerCase();
          tr.style.display = hay.includes(q) ? "" : "none";
        });
        // Update footer counts
        const vis = trs.filter((tr) => tr.style.display !== "none");
        const countEl = root.querySelector("#pt-count")!;
        countEl.textContent = String(vis.length);
      };
      search.addEventListener("input", applyFilter);
      applyFilter();

      // Export visible rows
      exportBtn.addEventListener("click", () => {
        const vis = Array.from(tbody.querySelectorAll<HTMLTableRowElement>('tr[data-row]')).filter((tr) => tr.style.display !== "none");
        const cols = Array.from(thead.querySelectorAll("th[data-key]")).map((th) => th.textContent?.trim() || "");
        const csvRows = vis.map((tr) =>
          Array.from(tr.querySelectorAll<HTMLElement>("[data-k]")).map((td) => td.dataset.csv ?? ""),
        );
        const data = [cols, ...csvRows].map((r) => r.map(csvEsc).join(",")).join("\n");
        const blob = new Blob(["\uFEFF" + data], { type: "text/csv;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `positions_${fmtTs(new Date())}.csv`;
        a.click();
        URL.revokeObjectURL(url);
      });

      // Row click
      if (onRowClick) {
        tbody.addEventListener("click", async (e) => {
          const tr = (e.target as HTMLElement).closest("tr[data-row]") as HTMLTableRowElement | null;
          if (!tr) return;
          const sym = tr.getAttribute("data-symbol") || "";
          if (!sym) return;
          try {
            tr.setAttribute("data-busy", "1");
            await onRowClick(sym);
          } finally {
            tr.removeAttribute("data-busy");
          }
        });
      }
    });
  }

  return (
    <section id="pt-root" style={wrap}>
      <style>{css}</style>

      <header style={head}>
        <div>
          <h3 style={h3}>{title}</h3>
          {note ? <p style={sub}>{note}</p> : null}
        </div>
        <div style={ctrls}>
          <div style={searchWrap}>
            <span style={searchIcon}>⌕</span>
            <input
              name="pt-search"
              placeholder="Search symbol, name, sector…"
              style={searchInput}
              aria-label="Search positions"
            />
          </div>
          <button id="pt-export" style={btn}>Export CSV</button>
        </div>
      </header>

      <div style={{ overflow: "auto" }}>
        <table style={{ ...table, ...(dense ? { fontSize: 12 } : null) }}>
          <thead style={theadStyle}>
            <tr>
              <TH label="Symbol" k="symbol" type="str" />
              {showSec ? <TH label="Sector" k="sector" type="str" /> : null}
              <TH label="Qty" k="qty" />
              <TH label="Price" k="price" />
              <TH label="Value" k="value" />
              <TH label="Avg Cost" k="avg" />
              <TH label="Cost" k="cost" />
              <TH label="P&L" k="pnl" />
              <TH label="P&L%" k="pnlpct" />
              <TH label="Day%" k="daypct" />
              <TH label="Weight" k="weight" />
            </tr>
          </thead>
          <tbody>
            {enriched.map((r) => (
              <tr
                key={r.symbol}
                data-row
                data-symbol={r.symbol}
                data-hay={(r.symbol + " " + (r.name || "") + " " + (r.sector || "")).trim()}
                style={tr}
                title="Click for details"
              >
                <TD k="symbol" type="str" value={r.symbol} csv={r.symbol}>
                  <div style={{ display: "grid" }}>
                    <span style={{ fontWeight: 700 }}>{r.symbol}</span>
                    {r.name ? <span style={{ color: "#6b7280", fontSize: 11 }}>{r.name}</span> : null}
                  </div>
                </TD>

                {showSec ? (
                  <TD k="sector" type="str" value={r.sector || ""} csv={r.sector || ""}>
                    {r.sector || "—"}
                  </TD>
                ) : null}

                <TD k="qty" value={r.qty} csv={num(r.qty)}>{num(r.qty)}</TD>
                <TD k="price" value={r.price} csv={money(r.price, unit)}>{money(r.price, unit)}</TD>
                <TD k="value" value={r.value} csv={money(r.value, unit)}>{money(r.value, unit)}</TD>
                <TD k="avg" value={r.avg} csv={money(r.avg, unit)}>{money(r.avg, unit)}</TD>
                <TD k="cost" value={r.cost} csv={money(r.cost, unit)}>{money(r.cost, unit)}</TD>

                <TD k="pnl" value={r.pnl} csv={money(r.pnl, unit)}>
                  <span style={{ color: r.pnl >= 0 ? "#067647" : "#b42318", fontWeight: 600 }}>
                    {money(r.pnl, unit)}
                  </span>
                </TD>

                <TD k="pnlpct" value={r.pnlPct} csv={pct(r.pnlPct)}>
                  <span style={{ color: r.pnlPct >= 0 ? "#067647" : "#b42318" }}>{pct(r.pnlPct)}</span>
                </TD>

                <TD k="daypct" value={r.dayPct} csv={pct(r.dayPct)}>{pct(r.dayPct)}</TD>
                <TD k="weight" value={r.weight} csv={pct(r.weight)}>{pct(r.weight)}</TD>
              </tr>
            ))}
          </tbody>

          {/* Footer totals */}
          <tfoot>
            <tr style={tfootRow}>
              <td style={tfLabel} colSpan={showSec ? 4 : 3}>Totals</td>
              <td style={tfNum}>{money(totals.value, unit)}</td>
              <td style={tfNum}></td>
              <td style={tfNum}>{money(totals.cost, unit)}</td>
              <td style={tfNum}>
                <span style={{ color: totals.pnl >= 0 ? "#067647" : "#b42318", fontWeight: 700 }}>
                  {money(totals.pnl, unit)}
                </span>
              </td>
              <td style={tfNum}>
                <span style={{ color: totals.pnlPct >= 0 ? "#067647" : "#b42318" }}>
                  {pct(totals.pnlPct)}
                </span>
              </td>
              <td style={tfNum}></td>
              <td style={tfNum}>{pct(1)}</td>
            </tr>
            <tr>
              <td colSpan={showSec ? 11 : 10} style={tfNote}>
                Showing <span id="pt-count">{enriched.length}</span> of {enriched.length} positions
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}

/* ---------------- tiny components (no imports) ---------------- */
function TH({ label, k, type = "num" }: { label: string; k: string; type?: "num" | "str" }) {
  return (
    <th
      scope="col"
      data-key={k}
      data-type={type}
      style={th}
      title="Click to sort"
    >
      <span>{label}</span>
      <span aria-hidden="true" style={sortIcon}>↕</span>
    </th>
  );
}

function TD(props: { k: string; value: any; csv: string; children: any; type?: "num" | "str" }) {
  const { k, value, csv, children } = props;
  return (
    <td data-k={k} data-value={String(value ?? "")} data-csv={csv} style={td}>
      {children}
    </td>
  );
}

/* ---------------- calculations ---------------- */
function enrich(p: Position) {
  const qty = Math.max(0, Number(p.qty) || 0);
  const price = Math.max(0, Number(p.price) || 0);
  const avg = Math.max(0, Number(p.avgCost || 0));
  const value = qty * price;
  const cost = qty * avg;
  const pnl = value - cost;
  const pnlPct = cost > 0 ? pnl / cost : 0;
  const dayPct = Number.isFinite(p.dayPct as number) ? (p.dayPct as number) : 0;
  return {
    symbol: p.symbol.toUpperCase(),
    name: p.name,
    sector: p.sector,
    qty, price, value, avg, cost, pnl, pnlPct, dayPct,
    weight: 0, // filled later
  };
}

function computeTotals(rows: ReturnType<typeof enrich>[]) {
  const value = rows.reduce((s, r) => s + r.value, 0);
  const cost = rows.reduce((s, r) => s + r.cost, 0);
  const pnl = value - cost;
  const pnlPct = cost > 0 ? pnl / cost : 0;
  // also fill weight on the same object (mutate once; no hooks)
  rows.forEach((r) => (r.weight = value > 0 ? r.value / value : 0));
  return { value, cost, pnl, pnlPct };
}

/* ---------------- formatters ---------------- */
function money(n: number, unit: string) {
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  return `${sign}${unit}${compact(v, 2)}`;
}
function pct(x: number) {
  if (!Number.isFinite(x)) return "—";
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}
function num(x: number) {
  return (Number(x) || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function compact(n: number, d = 2) {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(d) + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(d) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(d) + "k";
  return n.toFixed(d);
}
function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${s.replace(/"/g, '""')}"` : s;
}
function fmtTs(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/* ---------------- styles ---------------- */
const wrap: any = { display: "flex", flexDirection: "column", gap: 10, padding: 12 };
const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const ctrls: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const btn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const searchWrap: any = { position: "relative" };
const searchIcon: any = { position: "absolute", left: 8, top: 6, fontSize: 12, color: "#777" };
const searchInput: any = { width: 220, height: 30, padding: "4px 8px 4px 24px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const table: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, minWidth: 760 };
const theadStyle: any = { position: "sticky", top: 0, zIndex: 1, background: "#fff" };
const th: any = {
  position: "sticky",
  top: 0,
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "1px solid #e5e7eb",
  background: "#fff",
  fontSize: 12,
  color: "#6b7280",
  cursor: "pointer",
  userSelect: "none",
};
const sortIcon: any = { marginLeft: 6, fontSize: 11, opacity: 0.7 };

const td: any = { padding: "10px", borderBottom: "1px solid #f0f0f1", whiteSpace: "nowrap" };
const tr: any = { background: "#fff" };

const tfootRow: any = { background: "#fafafa", borderTop: "2px solid #e5e7eb" };
const tfLabel: any = { textAlign: "right", padding: "10px", fontWeight: 700 };
const tfNum: any = { padding: "10px", fontWeight: 700 };
const tfNote: any = { padding: "6px 10px", color: "#6b7280", fontSize: 12 };

const css = `
  th[aria-sort="ascending"] span:last-child { transform: rotate(180deg); display:inline-block; }
  tr[data-row][data-busy="1"] { opacity: .6; }
  tr[data-row]:hover { background: #f9fafb; }
  @media (prefers-color-scheme: dark) {
    thead[style], th[style], td[style], tr[style], table[style] { color: #e5e7eb !important; }
    thead[style] { background: #0b0b0c !important; }
    th[style] { background: #0b0b0c !important; border-color: rgba(255,255,255,.08) !important; }
    td[style] { border-color: rgba(255,255,255,.06) !important; }
    tr[data-row]:hover { background: #111214 !important; }
    tfoot tr[style] { background: #0f0f11 !important; }
    input[name="pt-search"] { background: #0b0b0c; border-color: rgba(255,255,255,.12); color:#e5e7eb; }
    button { color: #fff; }
  }
`;
