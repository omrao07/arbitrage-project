// app/portfolio/page.tsx
// Server page with zero imports. Renders a client-side positions table.
// - Accepts ?positions=<json>&prices=<json>&cash=<number> (all optional)
// - Falls back to a small demo portfolio if nothing is provided
// - Inline components below; no external imports

export const dynamic = "force-dynamic";

type PositionIn = { symbol: string; qty: number; avgCost?: number; name?: string; sector?: string; currency?: string };
type Snapshot = {
  rows: Array<{
    symbol: string;
    name?: string;
    sector?: string;
    currency?: string;
    qty: number;
    price: number;
    avg: number;
    value: number;
    cost: number;
    pnl: number;
    pnlPct: number;
    weight: number;
  }>;
  cash: number;
  totals: { value: number; cost: number; pnl: number; pnlPct: number; gross: number };
};

export default async function Page({ searchParams }: { searchParams?: Record<string, string | string[]> }) {
  const demo = buildDemo();
  const parsed = parseParams(searchParams || {});
  const snap = buildSnapshot(parsed.positions || demo.positions, parsed.prices || demo.prices, parsed.cash ?? demo.cash);

  return (
    <section style={wrap}>
      <style>{css}</style>

      <header style={head}>
        <div>
          <h1 style={h1}>Portfolio</h1>
          <p style={sub}>
            Equity MV <strong>{money(snap.totals.value, "₹")}</strong> · Cash <strong>{money(snap.cash, "₹")}</strong> · Gross{" "}
            <strong>{money(snap.totals.gross, "₹")}</strong>
          </p>
        </div>
        <div style={pillRow}>
          <div style={pill}>
            <span style={pillLbl}>P&amp;L</span>
            <span style={{ color: snap.totals.pnl >= 0 ? "#067647" : "#b42318", fontWeight: 700 }}>
              {money(snap.totals.pnl, "₹")}
            </span>
          </div>
          <div style={pill}>
            <span style={pillLbl}>P&amp;L%</span>
            <span style={{ color: snap.totals.pnlPct >= 0 ? "#067647" : "#b42318" }}>
              {pct(snap.totals.pnlPct)}
            </span>
          </div>
        </div>
      </header>

      <PortfolioClient title="Positions" unit="₹" rows={snap.rows} note="Click headers to sort. Search to filter." />
    </section>
  );
}

/* ----------------- Server helpers (no imports) ----------------- */

function parseParams(q: Record<string, string | string[]>): {
  positions?: PositionIn[];
  prices?: Record<string, number>;
  cash?: number;
} {
  function get(k: string): string | undefined {
    const v = q[k];
    if (Array.isArray(v)) return v[0];
    return typeof v === "string" ? v : undefined;
  }
  const positions = parseJson<PositionIn[]>(get("positions"));
  const prices = parseJson<Record<string, number>>(get("prices"));
  const cash = parseNum(get("cash"));
  return { positions: positions || undefined, prices: prices || undefined, cash: Number.isFinite(cash) ? cash : undefined };
}

function buildSnapshot(positions: PositionIn[], prices: Record<string, number>, cash = 0): Snapshot {
  const priceU = upperKeys(prices || {});
  const rows = (positions || []).map((p) => {
    const sym = (p.symbol || "").trim().toUpperCase();
    const qty = Math.max(0, Number(p.qty) || 0);
    const price = Math.max(0, Number(priceU[sym]) || 0);
    const avg = Math.max(0, Number(p.avgCost) || 0);
    const value = qty * price;
    const cost = qty * avg;
    const pnl = value - cost;
    const pnlPct = cost > 0 ? pnl / cost : 0;
    return { symbol: sym, name: p.name, sector: p.sector, currency: p.currency, qty, price, avg, value, cost, pnl, pnlPct, weight: 0 };
  });
  const totalValue = rows.reduce((s, r) => s + r.value, 0);
  const totalCost = rows.reduce((s, r) => s + r.cost, 0);
  rows.forEach((r) => (r.weight = totalValue > 0 ? r.value / totalValue : 0));
  const pnl = totalValue - totalCost;
  const pnlPct = totalCost > 0 ? pnl / totalCost : 0;
  return { rows, cash, totals: { value: totalValue, cost: totalCost, pnl, pnlPct, gross: totalValue + cash } };
}

function buildDemo() {
  const positions: PositionIn[] = [
    { symbol: "INFY", name: "Infosys Ltd", sector: "IT", qty: 120, avgCost: 1420, currency: "INR" },
    { symbol: "TCS", name: "Tata Consultancy Services", sector: "IT", qty: 80, avgCost: 3650, currency: "INR" },
    { symbol: "HDFCBANK", name: "HDFC Bank", sector: "Banks", qty: 150, avgCost: 1540, currency: "INR" },
  ];
  const prices = { INFY: 1490.5, TCS: 3920, HDFCBANK: 1632.2 };
  const cash = 25000;
  return { positions, prices, cash };
}

function parseJson<T>(s?: string): T | undefined {
  if (!s) return undefined;
  try { return JSON.parse(s) as T; } catch { return undefined; }
}
function parseNum(s?: string) {
  if (s == null || s === "") return NaN;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}
function upperKeys<T extends Record<string, any>>(obj: T) {
  const out: any = {};
  for (const k of Object.keys(obj || {})) out[String(k).toUpperCase()] = obj[k];
  return out as T;
}
function money(n: number, unit: string) {
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  if (v >= 1_000_000_000) return `${sign}${unit}${(v / 1_000_000_000).toFixed(2)}B`;
  if (v >= 1_000_000) return `${sign}${unit}${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${sign}${unit}${(v / 1_000).toFixed(2)}k`;
  return `${sign}${unit}${v.toFixed(2)}`;
}
function pct(x: number) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}

/* ----------------- Inline client table (no imports) ----------------- */

function TH({ label, k, type = "num" }: { label: string; k: string; type?: "num" | "str" }) {
  return (
    <th scope="col" data-key={k} data-type={type} style={th} title="Sort">
      <span>{label}</span><span aria-hidden="true" style={sortIcon}>↕</span>
    </th>
  );
}
function TD(props: { k: string; value: any; csv: string; children: any }) {
  const { k, value, csv, children } = props;
  return <td data-k={k} data-value={String(value ?? "")} data-csv={csv} style={td}>{children}</td>;
}

function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${s.replace(/"/g, '""')}"` : s;
}

function fmtTs(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function num(x: number) { return (Number(x) || 0).toLocaleString(undefined, { maximumFractionDigits: 0 }); }

function moneyCell(n: number, unit: string) {
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  return `${sign}${unit}${v.toFixed(2)}`;
}

function pctCell(x: number) {
  if (!Number.isFinite(x)) return "—";
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}

function weightPct(w: number) { return pctCell(w); }

export function PortfolioClient({
  rows,
  unit = "",
  title = "Positions",
  note,
}: {
  rows: Snapshot["rows"];
  unit?: string;
  title?: string;
  note?: string;
}) {
  "use client";

  const totals = (() => {
    const value = rows.reduce((s, r) => s + r.value, 0);
    const cost = rows.reduce((s, r) => s + r.cost, 0);
    const pnl = value - cost;
    const pnlPct = cost > 0 ? pnl / cost : 0;
    return { value, cost, pnl, pnlPct };
  })();

  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("pt-root");
      if (!root) return;
      const table = root.querySelector("table")!;
      const thead = table.tHead!;
      const tbody = table.tBodies[0];
      const search = root.querySelector<HTMLInputElement>('input[name="pt-search"]')!;
      const exportBtn = root.querySelector<HTMLButtonElement>('#pt-export')!;

      // sort
      thead.addEventListener("click", (e) => {
        const th = (e.target as HTMLElement).closest("th[data-key]") as HTMLTableHeaderCellElement | null;
        if (!th) return;
        const key = th.dataset.key!;
        const type = th.dataset.type || "num";
        const cur = th.getAttribute("aria-sort") as "ascending" | "descending" | null;
        const next: "ascending" | "descending" = cur === "ascending" ? "descending" : "ascending";
        thead.querySelectorAll("th[aria-sort]").forEach((el) => el.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", next);

        const rowsEls = Array.from(tbody.querySelectorAll("tr[data-row]"));
        rowsEls.sort((a, b) => {
          const va = a.querySelector<HTMLElement>(`[data-k="${key}"]`)?.dataset.value ?? "";
          const vb = b.querySelector<HTMLElement>(`[data-k="${key}"]`)?.dataset.value ?? "";
          let cmp = 0;
          if (type === "num") cmp = (Number(va) || 0) - (Number(vb) || 0);
          else cmp = String(va).localeCompare(String(vb));
          return next === "ascending" ? cmp : -cmp;
        });
        rowsEls.forEach((tr) => tbody.appendChild(tr));
      });

      // filter
      const applyFilter = () => {
        const q = (search.value || "").trim().toLowerCase();
        const trs = Array.from(tbody.querySelectorAll<HTMLTableRowElement>("tr[data-row]"));
        trs.forEach((tr) => {
          const hay = (tr.dataset.hay || "").toLowerCase();
          tr.style.display = hay.includes(q) ? "" : "none";
        });
        const vis = trs.filter((tr) => tr.style.display !== "none");
        root.querySelector("#pt-count")!.textContent = String(vis.length);
      };
      search.addEventListener("input", applyFilter);
      applyFilter();

      // export
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
    });
  }

  return (
    <section id="pt-root" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <header style={headRow}>
        <div>
          <h3 style={h3}>{title}</h3>
          {note ? <p style={noteTxt}>{note}</p> : null}
        </div>
        <div style={ctrlsRow}>
          <div style={searchWrap}>
            <span style={searchIcon}>⌕</span>
            <input name="pt-search" placeholder="Search…" style={searchInput} />
          </div>
          <button id="pt-export" style={btn}>Export CSV</button>
        </div>
      </header>

      <div style={{ overflow: "auto" }}>
        <table style={table}>
          <thead style={theadStyle}>
            <tr>
              <TH label="Symbol" k="symbol" type="str" />
              <TH label="Sector" k="sector" type="str" />
              <TH label="Qty" k="qty" />
              <TH label="Price" k="price" />
              <TH label="Value" k="value" />
              <TH label="Avg Cost" k="avg" />
              <TH label="Cost" k="cost" />
              <TH label="P&L" k="pnl" />
              <TH label="P&L%" k="pnlpct" />
              <TH label="Weight" k="weight" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol} data-row data-hay={(r.symbol + " " + (r.name || "") + " " + (r.sector || "")).trim()} style={tr}>
                <TD k="symbol" value={r.symbol} csv={r.symbol}>
                  <div style={{ display: "grid" }}>
                    <span style={{ fontWeight: 700 }}>{r.symbol}</span>
                    {r.name ? <span style={{ color: "#6b7280", fontSize: 11 }}>{r.name}</span> : null}
                  </div>
                </TD>
                <TD k="sector" value={r.sector || ""} csv={r.sector || ""}>{r.sector || "—"}</TD>
                <TD k="qty" value={r.qty} csv={num(r.qty)}>{num(r.qty)}</TD>
                <TD k="price" value={r.price} csv={moneyCell(r.price, unit)}>{moneyCell(r.price, unit)}</TD>
                <TD k="value" value={r.value} csv={moneyCell(r.value, unit)}>{moneyCell(r.value, unit)}</TD>
                <TD k="avg" value={r.avg} csv={moneyCell(r.avg, unit)}>{moneyCell(r.avg, unit)}</TD>
                <TD k="cost" value={r.cost} csv={moneyCell(r.cost, unit)}>{moneyCell(r.cost, unit)}</TD>
                <TD k="pnl" value={r.pnl} csv={moneyCell(r.pnl, unit)}>
                  <span style={{ color: r.pnl >= 0 ? "#067647" : "#b42318", fontWeight: 600 }}>
                    {moneyCell(r.pnl, unit)}
                  </span>
                </TD>
                <TD k="pnlpct" value={r.pnlPct} csv={pctCell(r.pnlPct)}>
                  <span style={{ color: r.pnlPct >= 0 ? "#067647" : "#b42318" }}>{pctCell(r.pnlPct)}</span>
                </TD>
                <TD k="weight" value={r.weight} csv={weightPct(r.weight)}>{weightPct(r.weight)}</TD>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr style={tfootRow}>
              <td style={tfLabel} colSpan={4}>Totals</td>
              <td style={tfNum}>{moneyCell(totals.value, unit)}</td>
              <td />
              <td style={tfNum}>{moneyCell(totals.cost, unit)}</td>
              <td style={tfNum}>
                <span style={{ color: totals.pnl >= 0 ? "#067647" : "#b42318", fontWeight: 700 }}>
                  {moneyCell(totals.pnl, unit)}
                </span>
              </td>
              <td style={tfNum}>
                <span style={{ color: totals.pnlPct >= 0 ? "#067647" : "#b42318" }}>{pctCell(totals.pnlPct)}</span>
              </td>
              <td style={tfNum}>{pctCell(1)}</td>
            </tr>
            <tr>
              <td colSpan={10} style={tfNote}>
                Showing <span id="pt-count">{rows.length}</span> of {rows.length} positions
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}

/* ----------------- styles ----------------- */
const wrap: any = { padding: 12, display: "grid", gap: 12 };
const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h1: any = { margin: 0, fontSize: 20 };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const pillRow: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const pill: any = { display: "grid", gap: 2, border: "1px solid #e5e7eb", background: "#fff", borderRadius: 10, padding: "6px 10px", minWidth: 110, textAlign: "right" };
const pillLbl: any = { color: "#6b7280", fontSize: 11 };

const headRow: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18 };
const noteTxt: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };
const ctrlsRow: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };

const searchWrap: any = { position: "relative" };
const searchIcon: any = { position: "absolute", left: 8, top: 6, fontSize: 12, color: "#777" };
const searchInput: any = { width: 220, height: 30, padding: "4px 8px 4px 24px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };

const btn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const table: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, minWidth: 760 };
const theadStyle: any = { position: "sticky", top: 0, zIndex: 1, background: "#fff" };
const th: any = { position: "sticky", top: 0, textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #e5e7eb", background: "#fff", fontSize: 12, color: "#6b7280", cursor: "pointer", userSelect: "none" };
const sortIcon: any = { marginLeft: 6, fontSize: 11, opacity: 0.7 };

const td: any = { padding: "10px", borderBottom: "1px solid #f0f0f1", whiteSpace: "nowrap" };
const tr: any = { background: "#fff" };

const tfootRow: any = { background: "#fafafa", borderTop: "2px solid #e5e7eb" };
const tfLabel: any = { textAlign: "right", padding: "10px", fontWeight: 700 };
const tfNum: any = { padding: "10px", fontWeight: 700 };
const tfNote: any = { padding: "6px 10px", color: "#6b7280", fontSize: 12 };

const css = `
  th[aria-sort="ascending"] span:last-child { transform: rotate(180deg); display:inline-block; }
  tr[data-row]:hover { background: #f9fafb; }
  @media (prefers-color-scheme: dark) {
    section, table, th, td { color: #e5e7eb !important; }
    th { background: #0b0b0c !important; border-color: rgba(255,255,255,.08) !important; }
    td { border-color: rgba(255,255,255,.06) !important; }
    tr[data-row]:hover { background: #111214 !important; }
    input[name="pt-search"] { background: #0b0b0c; border-color: rgba(255,255,255,.12); color:#e5e7eb; }
    .pill { background: #0b0b0c; border-color: rgba(255,255,255,.12); }
    button { color: #fff; }
  }
`;
