// app/components/breachestable.tsx
// No imports. No hooks. Self-contained "Breaches" table with sort/filter/export + inline actions.
// - Click headers to sort (asc/desc)
// - Search (rule/scope/metric/assignee)
// - Filter by status & severity
// - Export visible rows to CSV
// - Optional callbacks: onAck(id), onResolve(id), onAssign(id, assignee)
// - Inline styles; dark-mode friendly

"use client";

type Severity = "low" | "medium" | "high" | "critical";
type Status = "open" | "ack" | "resolved";
type UnitKind = "pct" | "currency" | "raw";

type Breach = {
  id: string;
  occurredAt: string;         // ISO
  scope: "account" | "household" | "strategy" | "sleeve" | "symbol";
  scopeId?: string;
  rule: string;               // e.g., "Max Position Weight %"
  metric?: string;            // e.g., symbol "AAPL", sector, etc.
  actual: number;             // if unit=pct -> percent units (e.g., 7 means 7%)
  limit?: number;             // same unit as `actual`
  unit?: UnitKind;            // default "raw"
  currency?: string;          // when unit="currency", e.g., "₹" or "$"
  severity?: Severity;        // default "low"
  status?: Status;            // default "open"
  assignee?: string;
  note?: string;
};

type Props = {
  title?: string;
  note?: string;
  items: Breach[];
  defaultCurrency?: string; // fallback for currency unit (e.g., "₹")
  dense?: boolean;

  onAck?: (id: string) => void | Promise<void>;
  onResolve?: (id: string) => void | Promise<void>;
  onAssign?: (id: string, assignee: string) => void | Promise<void>;
};

export default function BreachesTable({
  title = "Breaches",
  note,
  items,
  defaultCurrency = "",
  dense = false,
  onAck,
  onResolve,
  onAssign,
}: Props) {
  // Normalize once for initial render
  const rows = (items || []).map((b) => norm(b));

  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("br-root");
      if (!root) return;

      const table = root.querySelector("table")!;
      const thead = table.tHead!;
      const tbody = table.tBodies[0];
      const search = root.querySelector<HTMLInputElement>('input[name="br-search"]')!;
      const sStatus = root.querySelector<HTMLSelectElement>('select[name="br-status"]')!;
      const sSev = root.querySelector<HTMLSelectElement>('select[name="br-sev"]')!;
      const exportBtn = root.querySelector<HTMLButtonElement>("#br-export")!;
      const bulkAck = root.querySelector<HTMLButtonElement>("#br-bulk-ack")!;
      const bulkRes = root.querySelector<HTMLButtonElement>("#br-bulk-res")!;

      // Sorting
      thead.addEventListener("click", (e) => {
        const th = (e.target as HTMLElement).closest("th[data-key]") as HTMLTableHeaderCellElement | null;
        if (!th) return;
        const key = th.dataset.key!;
        const type = th.dataset.type || "num"; // "num" | "str" | "ts"
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
          else if (type === "ts") cmp = (Number(va) || 0) - (Number(vb) || 0);
          else cmp = String(va).localeCompare(String(vb));
          return next === "ascending" ? cmp : -cmp;
        });
        rowsEls.forEach((tr) => tbody.appendChild(tr));
      });

      // Filters
      const applyFilter = () => {
        const q = (search.value || "").trim().toLowerCase();
        const status = (sStatus.value || "any").toLowerCase();
        const sev = (sSev.value || "any").toLowerCase();
        const trs = Array.from(tbody.querySelectorAll<HTMLTableRowElement>("tr[data-row]"));
        let visCount = 0;
        trs.forEach((tr) => {
          const hay = (tr.dataset.hay || "").toLowerCase();
          const st = (tr.dataset.status || "").toLowerCase();
          const sv = (tr.dataset.sev || "").toLowerCase();
          const okQ = !q || hay.includes(q);
          const okS = status === "any" || st === status;
          const okV = sev === "any" || sv === sev;
          const show = okQ && okS && okV;
          tr.style.display = show ? "" : "none";
          if (show) visCount++;
        });
        root.querySelector("#br-count")!.textContent = String(visCount);
      };
      search.addEventListener("input", applyFilter);
      sStatus.addEventListener("change", applyFilter);
      sSev.addEventListener("change", applyFilter);
      applyFilter();

      // Export
      exportBtn.addEventListener("click", () => exportCSV(root));

      // Row actions: Ack / Resolve / Assign
      tbody.addEventListener("click", async (e) => {
        const btnAck = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-ack]');
        const btnRes = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-res]');
        const btnAssign = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-assign]');
        const row = (e.target as HTMLElement).closest<HTMLTableRowElement>('tr[data-row]');
        if (!row) return;
        const id = row.dataset.id || "";

        if (btnAck) {
          await updateStatus(row, "ack", onAck);
        } else if (btnRes) {
          await updateStatus(row, "resolved", onResolve);
        } else if (btnAssign) {
          const who = prompt("Assign to (name or email):", row.dataset.assignee || "") || "";
          if (!who) return;
          try {
            row.setAttribute("data-busy", "1");
            if (onAssign) await onAssign(id, who);
            row.dataset.assignee = who;
            const cell = row.querySelector<HTMLElement>('[data-k="assignee"] ._v');
            if (cell) cell.textContent = who;
            toast(root, `Assigned to ${who}`);
          } catch {
            toast(root, "Assign failed");
          } finally {
            row.removeAttribute("data-busy");
          }
        }
      });

      // Bulk actions on visible rows
      bulkAck.addEventListener("click", async () => {
        const vis = visibleRows(tbody);
        for (const r of vis) await updateStatus(r, "ack", onAck);
      });
      bulkRes.addEventListener("click", async () => {
        const vis = visibleRows(tbody);
        for (const r of vis) await updateStatus(r, "resolved", onResolve);
      });
    });
  }

  return (
    <section id="br-root" style={wrap}>
      <style>{css}</style>

      <header style={head}>
        <div>
          <h3 style={h3}>{title}</h3>
          {note ? <p style={sub}>{note}</p> : null}
        </div>
        <div style={ctrls}>
          <div style={searchWrap}>
            <span style={searchIcon}>⌕</span>
            <input name="br-search" placeholder="Search rule, scope, metric, assignee…" style={searchInput} />
          </div>
          <select name="br-status" style={select}>
            <option value="any">Any status</option>
            <option value="open">Open</option>
            <option value="ack">Acknowledged</option>
            <option value="resolved">Resolved</option>
          </select>
          <select name="br-sev" style={select}>
            <option value="any">Any severity</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <button id="br-bulk-ack" style={btnGhost} title="Acknowledge all visible">Bulk Ack</button>
          <button id="br-bulk-res" style={btnGhost} title="Resolve all visible">Bulk Resolve</button>
          <button id="br-export" style={btn}>Export CSV</button>
        </div>
      </header>

      <div style={{ overflow: "auto" }}>
        <table style={{ ...table, ...(dense ? { fontSize: 12 } : null) }}>
          <thead style={theadStyle}>
            <tr>
              <TH label="Time" k="ts" type="ts" />
              <TH label="Scope" k="scope" type="str" />
              <TH label="Rule" k="rule" type="str" />
              <TH label="Metric" k="metric" type="str" />
              <TH label="Actual" k="actual" />
              <TH label="Limit" k="limit" />
              <TH label="Δ vs limit" k="delta" />
              <TH label="Severity" k="sevRank" />
              <TH label="Status" k="statRank" />
              <TH label="Assignee" k="assignee" type="str" />
              <th style={thStatic}>Actions</th>
            </tr>
          </thead>

          <tbody>
            {rows.map((r) => (
              <tr
                key={r.id}
                data-row
                data-id={r.id}
                data-status={r.status}
                data-sev={r.severity}
                data-assignee={r.assignee || ""}
                data-hay={r.hay}
                style={tr}
                title={r.note || r.rule}
              >
                <TD k="ts" value={r.ts} csv={new Date(r.ts).toISOString()}>
                  <div style={{ display: "grid" }}>
                    <span style={{ fontWeight: 600 }}>{fmtTime(r.ts)}</span>
                    <span style={{ color: "#6b7280", fontSize: 11 }}>{ago(r.ts)}</span>
                  </div>
                </TD>

                <TD k="scope" value={r.scopeCsv} csv={`${r.scope.toUpperCase()}${r.scopeId ? `:${r.scopeId}` : ""}`}>
                  <div style={{ display: "grid" }}>
                    <span style={{ fontWeight: 600, letterSpacing: 0.2 }}>{r.scope.toUpperCase()}</span>
                    {r.scopeId ? <span style={{ color: "#6b7280", fontSize: 11 }}>{r.scopeId}</span> : null}
                  </div>
                </TD>

                <TD k="rule" value={r.rule} csv={r.rule}>
                  <span style={{ fontWeight: 600 }}>{r.rule}</span>
                </TD>

                <TD k="metric" value={r.metric || ""} csv={r.metric || ""}>
                  {r.metric || "—"}
                </TD>

                <TD k="actual" value={r.actualVal} csv={r.actualStr}>
                  {r.actualStr}
                </TD>

                <TD k="limit" value={r.limitVal} csv={r.limitStr}>
                  {r.limitStr || "—"}
                </TD>

                <TD k="delta" value={r.deltaVal} csv={r.deltaStr}>
                  <span style={{ color: r.deltaVal >= 0 ? "#b42318" : "#067647", fontWeight: 600 }}>
                    {r.deltaStr}
                  </span>
                </TD>

                <TD k="sevRank" value={r.sevRank} csv={cap(r.severity)}>
                  <span style={{ ...sevPill, ...sevStyle(r.severity) }}>{cap(r.severity)}</span>
                </TD>

                <TD k="statRank" value={r.statRank} csv={cap(r.status)}>
                  <span style={{ ...statPill, ...statusStyle(r.status) }}>{cap(r.status)}</span>
                </TD>

                <TD k="assignee" value={r.assignee || ""} csv={r.assignee || ""}>
                  <span className="_v">{r.assignee || "—"}</span>
                </TD>

                <td style={td}>
                  <div style={actRow}>
                    <button data-ack style={chipBtn} disabled={r.status !== "open"}>Ack</button>
                    <button data-res style={chipBtn} disabled={r.status === "resolved"}>Resolve</button>
                    <button data-assign style={chipBtn}>Assign</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>

          <tfoot>
            <tr>
              <td colSpan={11} style={tfNote}>
                Showing <span id="br-count">{rows.length}</span> of {rows.length} breaches
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div id="br-toast" style={toastStyle} />
    </section>
  );
}

/* ---------------- small components ---------------- */

function TH({ label, k, type = "num" }: { label: string; k: string; type?: "num" | "str" | "ts" }) {
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

/* ---------------- normalization / render helpers ---------------- */

const sevOrder: Record<Severity, number> = { critical: 0, high: 1, medium: 2, low: 3 };
const statOrder: Record<Status, number> = { open: 0, ack: 1, resolved: 2 };

function norm(b: Breach) {
  const unit = (b.unit || "raw") as UnitKind;
  const currency = unit === "currency" ? (b.currency || "") : "";
  const status = (b.status || "open") as Status;
  const severity = (b.severity || "low") as Severity;

  const ts = Date.parse(b.occurredAt || "") || Date.now();
  const actualStr = fmt(unit, b.actual, currency);
  const limitStr = typeof b.limit === "number" ? fmt(unit, b.limit, currency) : "";
  const deltaVal = typeof b.limit === "number" ? b.actual - b.limit : b.actual - 0;
  const deltaStr = fmt(unit, deltaVal, currency, true);
  const hay = [
    b.rule, b.scope, b.scopeId, b.metric, b.assignee, b.note,
    severity, status,
    actualStr, limitStr, deltaStr,
    new Date(ts).toISOString(),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return {
    id: b.id,
    ts,
    scope: b.scope,
    scopeId: b.scopeId,
    scopeCsv: `${b.scope.toUpperCase()}${b.scopeId ? `:${b.scopeId}` : ""}`,
    rule: b.rule,
    metric: b.metric,
    unit,
    currency,
    actualVal: b.actual,
    actualStr,
    limitVal: typeof b.limit === "number" ? b.limit : Number.NaN,
    limitStr,
    deltaVal,
    deltaStr,
    severity,
    status,
    sevRank: sevOrder[severity],
    statRank: statOrder[status],
    assignee: b.assignee,
    note: b.note,
    hay,
  };
}

function fmt(unit: UnitKind, v: number, curr: string, showSign = false) {
  const sgn = v < 0 ? "-" : showSign && v > 0 ? "+" : "";
  const abs = Math.abs(v);
  if (unit === "pct") return `${sgn}${abs.toFixed(2)}%`;
  if (unit === "currency") return `${sgn}${curr}${compact(abs, 2)}`;
  return `${sgn}${abs.toFixed(2)}`;
}

function compact(n: number, d = 2) {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(d) + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(d) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(d) + "k";
  return n.toFixed(d);
}

function cap(s: string) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function fmtTime(ts: number) {
  const d = new Date(ts);
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

function ago(ts: number) {
  const ms = Date.now() - ts;
  const mins = Math.round(ms / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  return `${days}d ago`;
}

/* ---------------- DOM helpers ---------------- */

async function updateStatus(row: HTMLTableRowElement, to: Status, cb?: (id: string) => void | Promise<void>) {
  const root = row.closest<HTMLElement>("#br-root")!;
  const id = row.dataset.id || "";
  const from = (row.dataset.status || "open") as Status;
  if (from === to) return;

  try {
    row.setAttribute("data-busy", "1");
    if (cb) await cb(id);
    // reflect status locally
    row.dataset.status = to;
    // update pill cell
    const pill = row.querySelector<HTMLElement>('[data-k="statRank"] span');
    if (pill) {
      const txt = cap(to);
      pill.textContent = txt;
      Object.assign(pill.style, statusStyle(to));
    }
    // set sort value
    const td = row.querySelector<HTMLElement>('[data-k="statRank"]');
    if (td) td.dataset.value = String(statOrder[to]);
    // enable/disable buttons
    const ack = row.querySelector<HTMLButtonElement>('button[data-ack]');
    const res = row.querySelector<HTMLButtonElement>('button[data-res]');
    if (ack) ack.disabled = to !== "open";
    if (res) res.disabled = to === "resolved";

    toast(root, `Marked as ${to}`);
  } catch {
    toast(root, "Action failed");
  } finally {
    row.removeAttribute("data-busy");
  }
}

function visibleRows(tbody: HTMLTableSectionElement) {
  return Array.from(tbody.querySelectorAll<HTMLTableRowElement>('tr[data-row]')).filter((tr) => tr.style.display !== "none");
}

function exportCSV(root: HTMLElement) {
  const thead = root.querySelector("thead")!;
  const headers = Array.from(thead.querySelectorAll("th")).map((th) => (th as HTMLElement).innerText.trim());
  const vis = visibleRows(root.querySelector("tbody")!);
  const rows = vis.map((tr) => Array.from(tr.querySelectorAll<HTMLElement>("[data-k]")).map((td) => td.dataset.csv || ""));
  const data = [headers, ...rows].map((r) => r.map(csvEsc).join(",")).join("\n");
  const blob = new Blob(["\uFEFF" + data], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `breaches_${ts(new Date())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
function ts(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function toast(root: HTMLElement, msg: string) {
  const el = root.querySelector("#br-toast") as HTMLElement | null;
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

/* ---------------- styles ---------------- */

const wrap: any = { display: "flex", flexDirection: "column", gap: 10, padding: 12 };

const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const ctrls: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const btn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };
const btnGhost: any = { border: "1px solid #e5e7eb", background: "#fff", color: "#111", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };
const chipBtn: any = { border: "1px solid #d4d4d8", background: "#fff", borderRadius: 999, padding: "2px 8px", cursor: "pointer", fontSize: 12 };

const searchWrap: any = { position: "relative" };
const searchIcon: any = { position: "absolute", left: 8, top: 6, fontSize: 12, color: "#777" };
const searchInput: any = { width: 260, height: 30, padding: "4px 8px 4px 24px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };
const select: any = { height: 30, padding: "4px 8px", borderRadius: 10, border: "1px solid #ddd", outline: "none", background: "#fff" };

const table: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, minWidth: 1000 };
const theadStyle: any = { position: "sticky", top: 0, zIndex: 1, background: "#fff" };
const th: any = { position: "sticky", top: 0, textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #e5e7eb", background: "#fff", fontSize: 12, color: "#6b7280", cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" };
const thStatic: any = { ...th, cursor: "default" as const };
const sortIcon: any = { marginLeft: 6, fontSize: 11, opacity: 0.7 };

const td: any = { padding: "10px", borderBottom: "1px solid #f0f0f1", whiteSpace: "nowrap", verticalAlign: "top" };
const tr: any = { background: "#fff", transition: "background .15s ease" };

const actRow: any = { display: "flex", gap: 6, justifyContent: "flex-end" };

const tfNote: any = { padding: "8px 10px", color: "#6b7280", fontSize: 12 };

const sevPill: any = { display: "inline-block", padding: "2px 8px", borderRadius: 999, fontSize: 12, fontWeight: 600 };
const statPill: any = { display: "inline-block", padding: "2px 8px", borderRadius: 999, fontSize: 12, fontWeight: 600 };

function sevStyle(s: Severity) {
  if (s === "critical") return { background: "#fef2f2", color: "#b42318", border: "1px solid #fee2e2" };
  if (s === "high") return { background: "#fff7ed", color: "#b45309", border: "1px solid #ffedd5" };
  if (s === "medium") return { background: "#fffbeb", color: "#92400e", border: "1px solid #fef3c7" };
  return { background: "#ecfeff", color: "#155e75", border: "1px solid #cffafe" };
}
function statusStyle(s: Status) {
  if (s === "open") return { background: "#eef2ff", color: "#3730a3", border: "1px solid #e0e7ff" };
  if (s === "ack") return { background: "#f5f3ff", color: "#6d28d9", border: "1px solid #ede9fe" };
  return { background: "#ecfdf5", color: "#065f46", border: "1px solid #d1fae5" };
}

const toastStyle: any = {
  position: "fixed",
  right: 16,
  bottom: 16,
  background: "#111",
  color: "#fff",
  padding: "8px 12px",
  borderRadius: 10,
  opacity: 0,
  transition: "opacity .25s ease",
  pointerEvents: "none",
  zIndex: 60,
};

const css = `
  th[aria-sort="ascending"] span:last-child { transform: rotate(180deg); display:inline-block; }
  tr[data-row][data-busy="1"] { opacity: .6; pointer-events: none; }
  tr[data-row]:hover { background: #f9fafb; }
  @media (prefers-color-scheme: dark) {
    table, th, td { color: #e5e7eb !important; }
    th { background: #0b0b0c !important; border-color: rgba(255,255,255,.08) !important; }
    td { border-color: rgba(255,255,255,.06) !important; }
    tr[data-row]:hover { background: #111214 !important; }
    input[name="br-search"], select { background: #0b0b0c; border-color: rgba(255,255,255,.12); color:#e5e7eb; }
    button { color: inherit; }
  }
`;

/* ---------------- utils ---------------- */

function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${s.replace(/"/g, '""')}"` : s;
}
