// app/components/sleevestress.tsx
// No imports. No hooks. Self-contained “Sleeve Stress” tool.
// - Shows a hierarchical sleeves table (like a tree) with indentation
// - Edit shock % for leaf sleeves; parents aggregate automatically
// - Columns: Base Value, Shock %, P&L, New Value, Contribution %
// - Global controls: set all shocks, reset, export CSV
// - Inline styles; dark-mode friendly

"use client";

type Sleeve = {
  id: string;
  name: string;
  mv?: number;              // base market value (absolute). If missing, sum(children.mv)
  children?: Sleeve[];
  note?: string;
  color?: string;
};

type Props = {
  sleeves: Sleeve[];        // forest
  unit?: string;            // currency prefix (e.g., "$", "₹")
  title?: string;
  note?: string;
  defaultShockPct?: number; // initial shock for leaves, in % (e.g., -3). Default 0.
  dense?: boolean;
};

export default function SleevesStress({
  sleeves,
  unit = "",
  title = "Sleeve Stress",
  note,
  defaultShockPct = 0,
  dense = false,
}: Props) {
  // ---------- build a computed forest once (no hooks) ----------
  const forest = sleeves.map((s) => computeNode(s));
  const totalMV = forest.reduce((s, n) => s + n.mv, 0);
  const flat = flattenForest(forest); // depth-first

  // ---------- wire DOM after hydration ----------
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("sst-root");
      if (!root) return;

      // caret toggles (collapse/expand)
      root.addEventListener("click", (e) => {
        const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-caret]');
        if (!btn) return;
        const tr = btn.closest<HTMLTableRowElement>('tr[data-id]')!;
        const isOpen = tr.getAttribute("data-open") === "1";
        if (isOpen) tr.removeAttribute("data-open"); else tr.setAttribute("data-open", "1");
        applyVisibility(root);
      });

      // global shock setter
      const g = root.querySelector<HTMLInputElement>('input[name="globalShock"]');
      g?.addEventListener("input", () => {
        const v = clampNum(parseFloat(g.value), -1000, 1000);
        root.querySelectorAll<HTMLInputElement>('input[data-leaf="1"]').forEach((inp) => (inp.value = String(v)));
        recompute(root, unit);
      });

      // per-leaf edits
      root.querySelectorAll<HTMLInputElement>('input[data-leaf="1"]').forEach((inp) => {
        inp.addEventListener("input", () => recompute(root, unit));
      });

      // reset and export
      root.querySelector<HTMLButtonElement>("#sst-reset")?.addEventListener("click", () => {
        const z = "0";
        if (g) g.value = z;
        root.querySelectorAll<HTMLInputElement>('input[data-leaf="1"]').forEach((inp) => (inp.value = z));
        recompute(root, unit);
      });
      root.querySelector<HTMLButtonElement>("#sst-export")?.addEventListener("click", () => exportCSV(root));

      // initial state
      applyVisibility(root);
      recompute(root, unit);
    });
  }

  return (
    <section id="sst-root" style={wrap}>
      <style>{css}</style>

      <header style={head}>
        <div>
          <h3 style={h3}>{title}</h3>
          {note ? <p style={sub}>{note}</p> : null}
        </div>

        <div style={ctrls}>
          <div style={globalRow}>
            <span style={{ color: "#6b7280", fontSize: 12 }}>Global shock</span>
            <input
              name="globalShock"
              type="number"
              step="0.1"
              defaultValue={String(defaultShockPct)}
              style={numInput}
              aria-label="Global shock percent"
            />
            <span style={{ fontSize: 12, color: "#6b7280" }}>%</span>
            <button id="sst-reset" style={ghostBtn} title="Reset shocks">Reset</button>
            <button id="sst-export" style={btn} title="Export CSV">Export</button>
          </div>
        </div>
      </header>

      <div style={{ overflow: "auto" }}>
        <table style={{ ...table, ...(dense ? { fontSize: 12 } : null) }}>
          <thead>
            <tr>
              <th style={thLeft}>Sleeve</th>
              <th style={thRight}>Base Value</th>
              <th style={thRight}>Shock %</th>
              <th style={thRight}>P&amp;L</th>
              <th style={thRight}>New Value</th>
              <th style={thRight}>Contrib%</th>
            </tr>
          </thead>
          <tbody>
            {flat.map((n) => {
              const isLeaf = !n.children || n.children.length === 0;
              const indent = 10 + n.depth * 16;
              return (
                <tr
                  key={n.id}
                  data-id={n.id}
                  data-parent={n.parentId || ""}
                  data-depth={n.depth}
                  data-open={n.depth === 0 ? "1" : undefined}
                  data-leaf={isLeaf ? "1" : "0"}
                  data-mv={n.mv}
                  style={tr}
                >
                  <td style={{ ...td, ...nameCell }}>
                    <div style={{ ...nameWrap, paddingLeft: indent }}>
                      {n.children && n.children.length > 0 ? (
                        <button data-caret style={caretBtn} aria-label="toggle">
                          <span style={caretIcon}>▸</span>
                        </button>
                      ) : (
                        <span style={{ width: 16 }} />
                      )}
                      <span style={{ ...chip, background: n.color || "#eef2ff" }} />
                      <div style={{ display: "grid" }}>
                        <span style={{ fontWeight: 600 }}>{n.name}</span>
                        {n.note ? <span style={{ color: "#6b7280", fontSize: 11 }}>{n.note}</span> : null}
                      </div>
                    </div>
                  </td>

                  <td style={tdNum} data-k="base" data-csv={money(n.mv, unit)}>{money(n.mv, unit)}</td>

                  <td style={tdNum}>
                    {isLeaf ? (
                      <span>
                        <input
                          data-leaf="1"
                          type="number"
                          step="0.1"
                          defaultValue={String(defaultShockPct)}
                          style={{ ...numInput, width: 82 }}
                          aria-label="Leaf shock percent"
                        />
                        <span style={{ marginLeft: 4, color: "#6b7280" }}>%</span>
                      </span>
                    ) : (
                      <span data-shock id={`shock-${n.id}`}>—</span>
                    )}
                  </td>

                  <td style={tdNum}>
                    <span id={`pnl-${n.id}`} data-csv="">{money(0, unit)}</span>
                  </td>

                  <td style={tdNum}>
                    <span id={`new-${n.id}`} data-csv="">{money(n.mv, unit)}</span>
                  </td>

                  <td style={tdNum}>
                    <span id={`ctr-${n.id}`} data-csv="">{pct(0)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>

          <tfoot>
            <tr style={tfootRow}>
              <td style={tfLeft}>Total</td>
              <td style={tfRight}>{money(totalMV, unit)}</td>
              <td style={tfRight}>—</td>
              <td style={tfRight}><span id="sst-total-pnl">{money(0, unit)}</span></td>
              <td style={tfRight}><span id="sst-total-new">{money(totalMV, unit)}</span></td>
              <td style={tfRight}>{pct(0)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}

/* ====================== compute helpers (no imports) ====================== */

type CNode = Sleeve & { mv: number; depth: number; parentId?: string };

function computeNode(node: Sleeve, depth = 0, parentId?: string): CNode {
  const kids = (node.children || []).map((c) => computeNode(c, depth + 1, node.id));
  let sumMV = 0;
  for (const k of kids) sumMV += k.mv;
  const mv = Number.isFinite(node.mv as number) ? (node.mv as number) : sumMV;
  return { ...node, children: kids, mv, depth, parentId };
}

function flattenForest(forest: CNode[]): CNode[] {
  const out: CNode[] = [];
  const walk = (n: CNode) => {
    out.push(n);
    const kids = n.children || [];
    for (const k of kids) walk
  };
  for (const n of forest) walk(n);
  return out;
}

/* ====================== DOM recompute / visibility ====================== */

function recompute(root: HTMLElement, unit: string) {
  // sort rows by depth desc so we can aggregate parents after children
  const rows = Array.from(root.querySelectorAll<HTMLTableRowElement>('tbody tr[data-id]')).sort(
    (a, b) => Number(b.dataset.depth || "0") - Number(a.dataset.depth || "0"),
  );

  const baseTotal = rows
    .filter((r) => Number(r.dataset.depth || "0") === 0)
    .reduce((s, r) => s + numAttr(r, "mv"), 0);

  const pnlMap = new Map<string, number>();
  const avgShockMap = new Map<string, number>(); // parent shock = pnl / base

  // pass 1: leaves from inputs
  for (const r of rows) {
    const id = r.dataset.id || "";
    const mv = numAttr(r, "mv");
    const isLeaf = r.dataset.leaf === "1";

    if (isLeaf) {
      const inp = r.querySelector<HTMLInputElement>('input[data-leaf="1"]');
      const shockPct = clampNum(parseFloat(inp?.value || "0"), -1000, 1000) / 100; // frac
      const pnl = mv * shockPct;
      pnlMap.set(id, pnl);
      avgShockMap.set(id, shockPct);
    }
  }

  // pass 2: parents aggregate their children
  // (rows is sorted deepest -> shallowest)
  for (const r of rows) {
    const id = r.dataset.id || "";
    if (r.dataset.leaf === "1") continue;

    const kids = Array.from(root.querySelectorAll<HTMLTableRowElement>(`tr[data-parent="${id}"]`));
    let sumPnl = 0;
    let sumMv = 0;
    for (const k of kids) {
      const kidId = k.dataset.id || "";
      const kidMv = numAttr(k, "mv");
      const kidPnl = pnlMap.get(kidId) || 0;
      sumPnl += kidPnl;
      sumMv += kidMv;
    }
    pnlMap.set(id, sumPnl);
    const shock = sumMv > 0 ? sumPnl / sumMv : 0;
    avgShockMap.set(id, shock);
    const shockEl = root.querySelector<HTMLSpanElement>(`#shock-${id}`);
    if (shockEl) shockEl.textContent = pct(shock);
  }

  // pass 3: write numbers, compute totals & contributions
  let totalPnl = 0;
  for (const r of rows) {
    const id = r.dataset.id || "";
    const mv = numAttr(r, "mv");
    const pnl = pnlMap.get(id) || 0;
    totalPnl += Number(r.dataset.depth || "0") === 0 ? pnl : 0;

    const newVal = mv + pnl;
    const ctr = baseTotal > 0 ? pnl / baseTotal : 0;

    const pnlEl = root.querySelector<HTMLSpanElement>(`#pnl-${id}`);
    const newEl = root.querySelector<HTMLSpanElement>(`#new-${id}`);
    const ctrEl = root.querySelector<HTMLSpanElement>(`#ctr-${id}`);

    if (pnlEl) { pnlEl.textContent = money(pnl, unit); pnlEl.dataset.csv = money(pnl, unit); }
    if (newEl) { newEl.textContent = money(newVal, unit); newEl.dataset.csv = money(newVal, unit); }
    if (ctrEl) { ctrEl.textContent = pct(ctr); ctrEl.dataset.csv = pct(ctr); }
  }

  const tP = root.querySelector<HTMLSpanElement>("#sst-total-pnl");
  const tN = root.querySelector<HTMLSpanElement>("#sst-total-new");
  if (tP) tP.textContent = money(totalPnl, unit);
  if (tN) tN.textContent = money(baseTotal + totalPnl, unit);
}

function applyVisibility(root: HTMLElement) {
  const rows = Array.from(root.querySelectorAll<HTMLTableRowElement>('tbody tr[data-id]'));
  const byId = new Map(rows.map((r) => [r.dataset.id!, r]));
  const visible = (tr: HTMLTableRowElement): boolean => {
    const depth = Number(tr.dataset.depth || "0");
    if (depth === 0) return true;
    const pid = tr.dataset.parent || "";
    const parent = byId.get(pid);
    if (!parent) return true;
    if (parent.getAttribute("data-open") !== "1") return false;
    return visible(parent);
  };
  for (const tr of rows) {
    tr.style.display = visible(tr) ? "" : "none";
    const icon = tr.querySelector<HTMLElement>('button[data-caret] span');
    if (icon) icon.style.transform = tr.getAttribute("data-open") === "1" ? "rotate(90deg)" : "rotate(0deg)";
  }
}

/* ====================== small utils ====================== */

function numAttr(el: Element, k: string): number {
  const v = (el as HTMLElement).dataset[k] || "0";
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : 0;
}
function clampNum(n: number, lo: number, hi: number) {
  if (!Number.isFinite(n)) return 0;
  return Math.max(lo, Math.min(hi, n));
}
function money(n: number, unit: string) {
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  return `${sign}${unit}${compact(v, 2)}`;
}
function pct(x: number) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}
function compact(n: number, d = 2) {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(d) + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(d) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(d) + "k";
  return n.toFixed(d);
}

function exportCSV(root: HTMLElement) {
  const headers = Array.from(root.querySelectorAll<HTMLTableElement>("table thead th")).map(
    (th) => th.textContent?.trim() || "",
  );
  const vis = Array.from(root.querySelectorAll<HTMLTableRowElement>('tbody tr[data-id]')).filter(
    (tr) => tr.style.display !== "none",
  );

  const rows = vis.map((tr) => {
    const name = tr.querySelector("td:first-child")?.textContent?.trim() || "";
    const base = tr.querySelector<HTMLElement>('td[data-k="base"]')?.dataset.csv || "";
    const shock =
      tr.querySelector<HTMLInputElement>('input[data-leaf="1"]')?.value ??
      tr.querySelector<HTMLElement>('[data-shock]')?.textContent ??
      "";
    const pnl = tr.querySelector<HTMLElement>('span[id^="pnl-"]')?.dataset.csv || "";
    const nval = tr.querySelector<HTMLElement>('span[id^="new-"]')?.dataset.csv || "";
    const ctr = tr.querySelector<HTMLElement>('span[id^="ctr-"]')?.dataset.csv || "";
    return [name, base, (shock ? (/\%$/.test(shock) ? shock : `${shock}%`) : ""), pnl, nval, ctr];
  });

  const csv = [headers, ...rows].map((r) => r.map(csvEsc).join(",")).join("\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sleeve_stress_${ts(new Date())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${s.replace(/"/g, '""')}"` : s;
}
function ts(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/* ====================== styles ====================== */

const wrap: any = { display: "flex", flexDirection: "column", gap: 10, padding: 12 };

const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const ctrls: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const globalRow: any = { display: "inline-flex", alignItems: "center", gap: 8 };
const numInput: any = { width: 64, height: 30, padding: "4px 6px", borderRadius: 8, border: "1px solid #e5e7eb", outline: "none", textAlign: "right" };
const btn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };
const ghostBtn: any = { border: "1px solid #e5e7eb", background: "#fff", color: "#111", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const table: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, minWidth: 760 };
const thLeft: any = { textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #e5e7eb", color: "#6b7280", fontSize: 12 };
const thRight: any = { ...thLeft, textAlign: "right" };

const tr: any = { background: "#fff", transition: "background .15s ease" };
const td: any = { padding: "8px 10px", borderBottom: "1px solid #f0f0f1", whiteSpace: "nowrap" };
const tdNum: any = { ...td, textAlign: "right", fontVariantNumeric: "tabular-nums" };

const nameCell: any = { width: "40%" };
const nameWrap: any = { display: "flex", alignItems: "center", gap: 8, minHeight: 28 };
const caretBtn: any = { width: 16, height: 16, border: "none", background: "transparent", padding: 0, cursor: "pointer" as const };
const caretIcon: any = { display: "inline-block", transformOrigin: "center", transition: "transform .15s ease" };
const chip: any = { width: 8, height: 8, borderRadius: 999, display: "inline-block" };

const tfootRow: any = { background: "#fafafa", borderTop: "2px solid #e5e7eb" };
const tfLeft: any = { textAlign: "left", padding: "10px", fontWeight: 700 };
const tfRight: any = { textAlign: "right", padding: "10px", fontWeight: 700 };

const css = `
  tr[data-id]:hover { background: #f9fafb; }
  /* hide children when ancestor is closed (managed via JS by toggling display) */
  @media (prefers-color-scheme: dark) {
    thead th { color: #a3a3a3 !important; border-color: rgba(255,255,255,.08) !important; }
    table, tr, td { color: #e5e7eb !important; }
    td { border-color: rgba(255,255,255,.06) !important; }
    tr[data-id]:hover { background: #111214 !important; }
    input, button { color: inherit; }
  }
`;
