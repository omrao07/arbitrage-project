// app/components/stresstester.tsx
// No imports. Self-contained client UI to run sleeve stress tests.
// - Paste/edit Sleeves JSON and Leaf Shocks JSON
// - Optional Global Shock % and Scenario label
// - Click "Run Stress" to compute P&L up the tree (leaves first, parents aggregate)
// - Sortable, searchable results table; export CSV/JSON; sample data loader
// - Dark-mode friendly; inline styles only

"use client";

type Sleeve = {
  id: string;
  name: string;
  mv?: number;                // base market value (absolute). If missing, summed from children
  children?: Sleeve[];
  note?: string;
  color?: string;
};

type StressRow = {
  id: string;
  name: string;
  depth: number;
  parentId?: string;
  baseMV: number;
  shockPct: number;           // resulting percent (parents derived from children)
  pnl: number;
  newMV: number;
  contrib: number;            // pnl / totalBaseMV (roots total)
  note?: string;
  color?: string;
};

type StressResult = {
  scenario: string;
  updatedAt: string;
  totals: { baseMV: number; pnl: number; newMV: number };
  rows: StressRow[];          // flattened, depth-first
  errors: string[];
  csv?: string;
};

type Props = {
  title?: string;
  sleeves?: Sleeve[];
  shocks?: Record<string, number>; // id -> percent (e.g., -3 for -3%)
  globalShockPct?: number;
  scenario?: string;
  dense?: boolean;
};

export default function StressTester({
  title = "Stress Tester",
  sleeves,
  shocks,
  globalShockPct = 0,
  scenario = "Ad-hoc",
  dense = false,
}: Props) {
  // Wire up events after hydration. No hooks used.
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("st-root");
      if (!root) return;

      // Seed editors
      const slTA = root.querySelector<HTMLTextAreaElement>("#st-sleeves");
      const shTA = root.querySelector<HTMLTextAreaElement>("#st-shocks");
      const gInp = root.querySelector<HTMLInputElement>('#st-global');
      const scInp = root.querySelector<HTMLInputElement>('#st-scenario');

      if (slTA && !slTA.value.trim()) slTA.value = pretty(sleeves || demoSleeves());
      if (shTA && !shTA.value.trim()) shTA.value = pretty(shocks || {});
      if (gInp && gInp.value === "") gInp.value = String(globalShockPct || 0);
      if (scInp && scInp.value === "") scInp.value = scenario || "Ad-hoc";

      // Buttons
      root.querySelector<HTMLButtonElement>("#st-run")?.addEventListener("click", () => run(root));
      root.querySelector<HTMLButtonElement>("#st-sample")?.addEventListener("click", () => {
        const ta = root.querySelector<HTMLTextAreaElement>("#st-sleeves");
        if (ta) ta.value = pretty(demoSleeves());
        const sh = root.querySelector<HTMLTextAreaElement>("#st-shocks");
        if (sh) sh.value = pretty({ "TECH": -3, "BANKS": -2, "HEALTH": -1 });
        toast("Loaded sample data");
      });
      root.querySelector<HTMLButtonElement>("#st-reset")?.addEventListener("click", () => {
        const ta = root.querySelector<HTMLTextAreaElement>("#st-sleeves"); if (ta) ta.value = "";
        const sh = root.querySelector<HTMLTextAreaElement>("#st-shocks"); if (sh) sh.value = "";
        const g = root.querySelector<HTMLInputElement>('#st-global'); if (g) g.value = "0";
        const sc = root.querySelector<HTMLInputElement>('#st-scenario'); if (sc) sc.value = "Ad-hoc";
        clearResult(root);
        toast("Cleared");
      });
      root.querySelector<HTMLButtonElement>("#st-export-csv")?.addEventListener("click", () => exportCSV(root));
      root.querySelector<HTMLButtonElement>("#st-export-json")?.addEventListener("click", () => exportJSON(root));

      // Table sort & filter
      const table = root.querySelector("table")!;
      const thead = table.tHead!;
      const tbody = table.tBodies[0];
      const search = root.querySelector<HTMLInputElement>('input[name="st-search"]')!;
      const chkLeaves = root.querySelector<HTMLInputElement>('#st-leaves-only')!;

      thead.addEventListener("click", (e) => {
        const th = (e.target as HTMLElement).closest("th[data-key]") as HTMLTableHeaderCellElement | null;
        if (!th) return;
        const key = th.dataset.key!;
        const type = th.dataset.type || "num"; // "num" | "str"
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

      const applyFilter = () => {
        const q = (search.value || "").trim().toLowerCase();
        const leavesOnly = chkLeaves.checked;
        const trs = Array.from(tbody.querySelectorAll<HTMLTableRowElement>("tr[data-row]"));
        let vis = 0;
        trs.forEach((tr) => {
          const hay = (tr.dataset.hay || "");
          const leaf = tr.dataset.leaf === "1";
          const show = (!q || hay.includes(q)) && (!leavesOnly || leaf);
          tr.style.display = show ? "" : "none";
          if (show) vis++;
        });
        root.querySelector("#st-count")!.textContent = String(vis);
      };
      search.addEventListener("input", applyFilter);
      chkLeaves.addEventListener("change", applyFilter);

      // Initial run with seeded content
      if (slTA?.value.trim()) run(root);
    });
  }

  return (
    <section id="st-root" style={wrap} data-busy="0">
      <style>{css}</style>
      <div id="st-toast" style={toastStyle} />

      <header style={head}>
        <div>
          <h3 style={h3}>{title}</h3>
          <p style={sub}>Provide a sleeves tree and per-leaf shocks (percent). Parents aggregate from children.</p>
        </div>
        <div style={ctrls}>
          <button id="st-run" style={btn}>Run Stress</button>
          <button id="st-export-csv" style={btnGhost}>Export CSV</button>
          <button id="st-export-json" style={btnGhost}>Export JSON</button>
          <button id="st-sample" style={btnGhost}>Load Sample</button>
          <button id="st-reset" style={btnGhost}>Reset</button>
        </div>
      </header>

      <section style={grid2}>
        {/* Inputs */}
        <div style={card}>
          <h4 style={h4}>Inputs</h4>
          <div style={grid2small}>
            <div>
              <label style={lbl}>Global Shock % (fallback)</label>
              <input id="st-global" type="number" step="0.01" defaultValue={String(globalShockPct || 0)} style={input} />
            </div>
            <div>
              <label style={lbl}>Scenario</label>
              <input id="st-scenario" defaultValue={scenario || "Ad-hoc"} placeholder="e.g., Macro -2%" style={input} />
            </div>
          </div>
          <div>
            <label style={lbl}>Sleeves JSON</label>
            <textarea id="st-sleeves" rows={10} style={ta} placeholder='[{ "id":"ROOT", "name":"Portfolio", "children":[...]}]' />
          </div>
          <div>
            <label style={lbl}>Leaf Shocks JSON (id → %)</label>
            <textarea id="st-shocks" rows={6} style={ta} placeholder='{"TECH": -3, "BANKS": -2}' />
          </div>
          <p style={hint}>Tip: You can omit <code>mv</code> on parents — their base MV is the sum of children.</p>
        </div>

        {/* Results */}
        <div style={card}>
          <h4 style={h4}>Results</h4>
          <div id="st-summary" style={sumRow}>
            <div style={pill}><span style={pillLbl}>Base MV</span><b id="st-bmv">—</b></div>
            <div style={pill}><span style={pillLbl}>P&amp;L</span><b id="st-pnl">—</b></div>
            <div style={pill}><span style={pillLbl}>New MV</span><b id="st-nmv">—</b></div>
            <div style={pill}><span style={pillLbl}>Updated</span><b id="st-upd">—</b></div>
          </div>

          <div style={listHead}>
            <div style={searchWrap}>
              <span style={searchIcon}>⌕</span>
              <input name="st-search" placeholder="Search name/id…" style={searchInput} />
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <input id="st-leaves-only" type="checkbox" /> Leaves only
            </label>
          </div>

          <div style={{ overflow: "auto", border: "1px solid var(--b)", borderRadius: 12 }}>
            <table style={{ ...table, ...(dense ? { fontSize: 12 } : null) }}>
              <thead style={theadStyle}>
                <tr>
                  <TH label="Name" k="name" type="str" />
                  <TH label="Base MV" k="base" />
                  <TH label="Shock %" k="shock" />
                  <TH label="P&L" k="pnl" />
                  <TH label="New MV" k="new" />
                  <TH label="Contrib %" k="contrib" />
                </tr>
              </thead>
              <tbody id="st-body" />
              <tfoot>
                <tr>
                  <td colSpan={6} style={tfNote}>
                    Showing <span id="st-count">0</span> rows
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </section>
    </section>
  );
}

/* ----------------------------- Tiny Components ----------------------------- */

function TH({ label, k, type = "num" }: { label: string; k: string; type?: "num" | "str" }) {
  return (
    <th scope="col" data-key={k} data-type={type} style={th} title="Sort">
      <span>{label}</span><span aria-hidden="true" style={sortIcon}>↕</span>
    </th>
  );
}

/* --------------------------------- Actions --------------------------------- */

function run(root: HTMLElement) {
  try {
    root.setAttribute("data-busy", "1");
    const sl = readJson<Sleeve[]>(getVal(root, "#st-sleeves")) || [];
    const shocks = readJson<Record<string, number>>(getVal(root, "#st-shocks")) || {};
    const g = Number((root.querySelector<HTMLInputElement>("#st-global")?.value || "0").trim()) || 0;
    const sc = (root.querySelector<HTMLInputElement>("#st-scenario")?.value || "Stress").trim() || "Stress";

    const res = computeStress({ sleeves: sl, leafShocks: shocks, globalShockPct: g, scenario: sc });
    (root as any)._stRes = res; // stash

    renderResult(root, res);
    toast("Stress complete");
  } catch (e) {
    console.error(e);
    toast("Invalid inputs");
  } finally {
    root.removeAttribute("data-busy");
  }
}

function clearResult(root: HTMLElement) {
  const body = root.querySelector("#st-body") as HTMLElement | null;
  if (body) body.innerHTML = "";
  setTxt(root, "#st-bmv", "—");
  setTxt(root, "#st-pnl", "—");
  setTxt(root, "#st-nmv", "—");
  setTxt(root, "#st-upd", "—");
  (root as any)._stRes = undefined;
}

/* -------------------------------- Rendering -------------------------------- */

function renderResult(root: HTMLElement, res: StressResult) {
  setTxt(root, "#st-bmv", money(res.totals.baseMV, "₹"));
  setTxt(root, "#st-pnl", colorNum(money(res.totals.pnl, "₹"), res.totals.pnl));
  setTxt(root, "#st-nmv", money(res.totals.newMV, "₹"));
  setTxt(root, "#st-upd", new Date(res.updatedAt).toLocaleString());

  const body = root.querySelector("#st-body") as HTMLElement | null;
  if (!body) return;
  body.innerHTML = "";

  for (const r of res.rows) {
    const tr = document.createElement("tr");
    tr.setAttribute("data-row", "1");
    tr.dataset.hay = `${r.id} ${r.name}`.toLowerCase();
    tr.dataset.leaf = r.depth > 0 && !res.rows.some((x) => x.parentId === r.id) ? "1" : "0";
    tr.innerHTML = `
      <td style="${tdCss()}">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="opacity:.6;font-size:11;width:${r.depth * 12}px;flex:0 0 ${r.depth * 12}px"></span>
          <div style="display:grid">
            <span style="font-weight:700">${esc(r.name)}</span>
            <span style="color:#6b7280;font-size:11">${esc(r.id)}</span>
          </div>
        </div>
      </td>
      <td data-k="base" data-value="${r.baseMV}" style="${tdNumCss()}">${money(r.baseMV, "₹")}</td>
      <td data-k="shock" data-value="${r.shockPct}" style="${tdNumCss()}">${pct(r.shockPct)}</td>
      <td data-k="pnl" data-value="${r.pnl}" style="${tdNumCss()}"><span style="font-weight:600;${r.pnl>=0 ? 'color:#067647' : 'color:#b42318'}">${money(r.pnl, "₹")}</span></td>
      <td data-k="new" data-value="${r.newMV}" style="${tdNumCss()}">${money(r.newMV, "₹")}</td>
      <td data-k="contrib" data-value="${r.contrib}" style="${tdNumCss()}">${pct(r.contrib)}</td>
    `;
    body.appendChild(tr);
  }
  // trigger current filter to update visible count
  const inp = root.querySelector<HTMLInputElement>('input[name="st-search"]');
  if (inp) inp.dispatchEvent(new Event("input"));
}

function exportCSV(root: HTMLElement) {
  const res: StressResult | undefined = (root as any)._stRes;
  if (!res) return toast("Run stress first");
  const head = ["Depth", "ID", "Name", "Base", "Shock%", "P&L", "New", "Contrib%", "ParentID"];
  const rows = res.rows.map((r) => [
    String(r.depth),
    r.id,
    r.name,
    money(r.baseMV, ""),
    pct(r.shockPct),
    money(r.pnl, ""),
    money(r.newMV, ""),
    pct(r.contrib),
    r.parentId || "",
  ]);
  const csv = [head, ...rows].map((r) => r.map(csvEsc).join(",")).join("\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `stress_${stamp(new Date())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportJSON(root: HTMLElement) {
  const res: StressResult | undefined = (root as any)._stRes;
  if (!res) return toast("Run stress first");
  const blob = new Blob([JSON.stringify(res, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `stress_${stamp(new Date())}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------- Computation ------------------------------- */

function computeStress(input: {
  sleeves: Sleeve[];
  leafShocks?: Record<string, number>;
  globalShockPct?: number;
  scenario?: string;
}): StressResult {
  const errors: string[] = [];
  const sleeves = Array.isArray(input.sleeves) ? input.sleeves : [];
  const leafShocks = input.leafShocks || {};
  const globalShockPct = Number(input.globalShockPct || 0);
  const scenario = input.scenario || "Stress";

  // Build computed forest (resolve base market values)
  const forest = sleeves.map((s) => computeNode(s));
  const flat = flatten(forest);
  const rootIds = new Set(flat.filter((r) => r.depth === 0).map((r) => r.id));

  // Apply shocks to leaves
  const map = new Map<string, { base: number; newV: number; pnl: number; shock: number }>();
  for (const n of flat) {
    const isLeaf = !flat.some((k) => k.parentId === n.id);
    if (isLeaf) {
      const pct = pctToFrac(leafShocks[n.id] ?? globalShockPct);
      const newV = n.baseMV * (1 + pct);
      map.set(n.id, { base: n.baseMV, newV, pnl: newV - n.baseMV, shock: pct });
    }
  }

  // Aggregate parents (deepest -> shallowest)
  for (const n of [...flat].sort((a, b) => b.depth - a.depth)) {
    const hasKids = flat.some((k) => k.parentId === n.id);
    if (!hasKids) continue; // leaf already handled
    let base = 0, newV = 0;
    for (const k of flat.filter((x) => x.parentId === n.id)) {
      const m = map.get(k.id);
      if (m) { base += m.base; newV += m.newV; }
    }
    if (base === 0) { base = n.baseMV; newV = n.baseMV; }
    const pnl = newV - base;
    const shock = base > 0 ? pnl / base : 0;
    map.set(n.id, { base, newV, pnl, shock });
  }

  // Totals consider roots
  let tBase = 0, tNew = 0, tPnL = 0;
  const rows: StressRow[] = flat.map((n) => {
    const m = map.get(n.id) || { base: n.baseMV, newV: n.baseMV, pnl: 0, shock: 0 };
    if (rootIds.has(n.id)) { tBase += m.base; tNew += m.newV; tPnL += m.pnl; }
    return {
      id: n.id,
      name: n.name,
      depth: n.depth,
      parentId: n.parentId,
      baseMV: fix(m.base),
      shockPct: fix(m.shock),
      pnl: fix(m.pnl),
      newMV: fix(m.newV),
      contrib: 0, // fill later
      note: n.note,
      color: n.color,
    };
  });

  const totalBase = tBase || rows.filter(r => r.depth === 0).reduce((s, r) => s + r.baseMV, 0);
  for (const r of rows) r.contrib = totalBase > 0 ? fix(r.pnl / totalBase) : 0;

  return {
    scenario,
    updatedAt: new Date().toISOString(),
    totals: { baseMV: fix(totalBase), pnl: fix(tPnL), newMV: fix(tNew) },
    rows,
    errors,
  };
}

function computeNode(n: Sleeve, depth = 0, parentId?: string): { id: string; name: string; baseMV: number; depth: number; parentId?: string; note?: string; color?: string } {
  const kids = (n.children || []).map((c) => computeNode(c, depth + 1, n.id));
  const sumKids = kids.reduce((s, k) => s + k.baseMV, 0);
  const baseMV = Number.isFinite(n.mv as number) ? Number(n.mv) : sumKids;
  return {
    id: n.id,
    name: n.name,
    baseMV: Math.max(0, Number(baseMV) || 0),
    depth,
    parentId,
    note: n.note,
    color: n.color,
  };
}

function flatten(forest: ReturnType<typeof computeNode>[]): ReturnType<typeof computeNode>[] {
  const out: any[] = [];
  const byParent = new Map<string | undefined, any[]>();
  for (const node of forest) {
    if (!byParent.has(node.parentId)) byParent.set(node.parentId, []);
    byParent.get(node.parentId)!.push(node);
  }
  const dfs = (node: any) => {
    out.push(node);
    const children = byParent.get(node.id) || [];
    children.forEach(dfs);
  };
  (byParent.get(undefined) || forest).forEach(dfs);
  return out;
}

/* --------------------------------- Utils --------------------------------- */

function getVal(root: HTMLElement, sel: string) {
  const el = root.querySelector<HTMLInputElement | HTMLTextAreaElement>(sel);
  return el ? el.value : "";
}
function setTxt(root: HTMLElement, sel: string, v: string) {
  const el = root.querySelector<HTMLElement>(sel);
  if (el) el.textContent = v;
}
function readJson<T>(s: string): T | undefined {
  const t = (s || "").trim();
  if (!t) return undefined;
  try { return JSON.parse(t) as T; } catch { return undefined; }
}
function pctToFrac(pct: number): number {
  const n = Number(pct);
  return Number.isFinite(n) ? n / 100 : 0;
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
function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${String(s).replace(/"/g, '""')}"` : String(s);
}
function stamp(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function fix(n: number) { return Math.round(n * 1e10) / 1e10; }
function pretty(x: any) { return JSON.stringify(x, null, 2); }
function esc(s: string) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function colorNum(s: string, n: number) { return `<span style="${n>=0?'color:#067647':'color:#b42318'};font-weight:700">${s}</span>`; }
function tdCss() { return "padding:10px;border-bottom:1px solid var(--rb);white-space:nowrap"; }
function tdNumCss() { return "padding:10px;border-bottom:1px solid var(--rb);white-space:nowrap;text-align:right"; }
function toast(msg: string) {
  const el = document.getElementById("st-toast");
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

/* --------------------------------- Sample --------------------------------- */

function demoSleeves(): Sleeve[] {
  return [
    {
      id: "ROOT",
      name: "Total Portfolio",
      children: [
        {
          id: "TECH",
          name: "Technology",
          children: [
            { id: "INFY", name: "Infosys", mv: 178_860 },
            { id: "TCS", name: "TCS", mv: 313_600 },
          ],
        },
        {
          id: "BANKS",
          name: "Banks",
          children: [
            { id: "HDFCBANK", name: "HDFC Bank", mv: 244_830 },
            { id: "ICICIBANK", name: "ICICI Bank", mv: 201_200 },
          ],
        },
        {
          id: "HEALTH",
          name: "Healthcare",
          children: [
            { id: "SUNPHARMA", name: "Sun Pharma", mv: 92_500 },
            { id: "DRREDDY", name: "Dr. Reddy's", mv: 85_100 },
          ],
        },
      ],
    },
  ];
}

/* ---------------------------------- Styles --------------------------------- */

const wrap: any = { display: "grid", gap: 12, padding: 12 };
const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18 };
const h4: any = { margin: 0, fontSize: 16 };
const sub: any = { margin: "4px 0 0", color: "#6b7280", fontSize: 12.5 };

const ctrls: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const btn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };
const btnGhost: any = { border: "1px solid #e5e7eb", background: "#fff", color: "#111", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const grid2: any = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 };
const grid2small: any = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 };

const card: any = { border: "1px solid var(--b)", borderRadius: 14, background: "var(--bg)", padding: 12, display: "grid", gap: 10 };

const lbl: any = { fontSize: 12, color: "#6b7280" };
const input: any = { width: "100%", height: 32, padding: "4px 8px", borderRadius: 10, border: "1px solid var(--b)", outline: "none", background: "#fff" };
const ta: any = { width: "100%", border: "1px solid var(--b)", borderRadius: 10, padding: 8, minHeight: 120, fontFamily: "ui-monospace,Menlo,monospace", fontSize: 12.5 };
const hint: any = { margin: 0, color: "#6b7280", fontSize: 12 };

const sumRow: any = { display: "flex", gap: 8, flexWrap: "wrap" };
const pill: any = { display: "grid", gap: 2, border: "1px solid #e5e7eb", background: "#fff", borderRadius: 10, padding: "6px 10px", minWidth: 140, textAlign: "right" };
const pillLbl: any = { color: "#6b7280", fontSize: 11 };

const listHead: any = { display: "flex", alignItems: "center", justifyContent: "space-between" };
const searchWrap: any = { position: "relative" };
const searchIcon: any = { position: "absolute", left: 8, top: 6, fontSize: 12, color: "#777" };
const searchInput: any = { width: 220, height: 30, padding: "4px 8px 4px 24px", borderRadius: 10, border: "1px solid var(--b)", outline: "none", background: "#fff" };

const table: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, minWidth: 720, background: "var(--bg)" };
const theadStyle: any = { position: "sticky", top: 0, zIndex: 1, background: "#fff" };
const th: any = { position: "sticky", top: 0, textAlign: "left", padding: "8px 10px", borderBottom: "1px solid var(--b)", background: "#fff", fontSize: 12, color: "#6b7280", cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" };
const sortIcon: any = { marginLeft: 6, fontSize: 11, opacity: 0.7 };
const tfNote: any = { padding: "8px 10px", color: "#6b7280", fontSize: 12, textAlign: "right" };

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
  :root { --b:#e5e7eb; --rb:#f0f0f1; --bg:#fff; }
  #st-root[data-busy="1"] { opacity:.6; pointer-events:none; }

  th[aria-sort="ascending"] span:last-child { transform: rotate(180deg); display:inline-block; }
  tr[data-row]:hover { background: #f9fafb; }

  @media (max-width: 1000px) { .grid2 { grid-template-columns: 1fr !important; } .grid2small { grid-template-columns: 1fr !important; } }

  @media (prefers-color-scheme: dark) {
    :root { --b:rgba(255,255,255,.12); --rb:rgba(255,255,255,.06); --bg:#0b0b0c; }
    section, table, th, td, input, textarea { color:#e5e7eb !important; }
    th { background:#0b0b0c !important; border-color:var(--b) !important; }
    td { border-color: var(--rb) !important; }
    tr[data-row]:hover { background: #111214 !important; }
    input, textarea { background:#0b0b0c !important; border-color: var(--b) !important; }
    .pill { background:#0b0b0c; border-color: var(--b); }
    button { color: inherit; }
  }

  #st-toast[data-show="1"] { opacity: 1; }
`;
