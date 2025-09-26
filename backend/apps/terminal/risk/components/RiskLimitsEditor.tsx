// app/components/risklimitseditor.tsx
// No imports. No hooks. Self-contained Risk Limits Editor.
// - Matches the shape used by updatelimits.action.ts
// - Sections: Weights, Exposures, Risk, Trading, Cash, Per-symbol Caps
// - Edit → Validate → Save (via onSubmit) or Export JSON
// - Paste JSON to load, Compare vs Previous, Reset
// - Inline styles; dark-mode friendly

"use client";

type Scope = "account" | "household" | "strategy" | "sleeve" | "symbol";

type NormalizedLimits = {
  weights?: { maxPositionWeightPct?: number; maxSectorWeightPct?: number };
  exposures?: { grossExposurePct?: number; netExposurePct?: number; leverageMax?: number };
  risk?: { dailyLossLimitPct?: number; trailingDrawdownPct?: number; varPct?: number };
  trading?: { turnoverCapPct?: number; minTradeValue?: number };
  cash?: { minCashPct?: number; maxCashPct?: number };
  perSymbolCaps?: Record<
    string,
    { maxQty?: number; maxNotional?: number; maxWeightPct?: number }
  >;
};

type Props = {
  title?: string;
  scope?: Scope;
  scopeId?: string;
  effectiveAt?: string; // ISO
  notes?: string;
  initial?: NormalizedLimits;
  prev?: NormalizedLimits;
  dense?: boolean;
  onSubmit?: (payload: {
    scope?: Scope;
    scopeId?: string;
    effectiveAt?: string;
    notes?: string;
    limits: NormalizedLimits;
    prev?: NormalizedLimits;
  }) => void | Promise<void>;
};

export default function RiskLimitsEditor({
  title = "Risk Limits",
  scope,
  scopeId,
  effectiveAt,
  notes,
  initial,
  prev,
  dense = false,
  onSubmit,
}: Props) {
  // Wire DOM after hydration
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("rle-root");
      if (!root) return;

      // Populate initial & prev JSON areas
      const initTA = root.querySelector<HTMLTextAreaElement>("#rle-json");
      if (initTA) initTA.value = pretty({ scope, scopeId, effectiveAt, notes, limits: initial || {} });
      const prevTA = root.querySelector<HTMLTextAreaElement>("#rle-prev");
      if (prevTA) prevTA.value = prev ? pretty(prev) : "";

      // Add row
      root.querySelector<HTMLButtonElement>("#rle-add-row")?.addEventListener("click", () => {
        addPSCRow(root, { symbol: "", maxQty: "", maxNotional: "", maxWeightPct: "" });
      });

      // Clear PSC
      root.querySelector<HTMLButtonElement>("#rle-clear-psc")?.addEventListener("click", () => {
        const body = root.querySelector("#rle-psc-body");
        if (body) body.innerHTML = "";
        markDirty(root);
      });

      // Load from JSON
      root.querySelector<HTMLButtonElement>("#rle-load")?.addEventListener("click", () => {
        const ta = root.querySelector<HTMLTextAreaElement>("#rle-json");
        if (!ta) return;
        try {
          const obj = JSON.parse(ta.value || "{}");
          applyModel(root, obj);
          toast("Loaded from JSON");
        } catch {
          toast("Invalid JSON");
        }
      });

      // Apply prev JSON into PREV box (for diffing only)
      root.querySelector<HTMLButtonElement>("#rle-set-prev")?.addEventListener("click", () => {
        const ta = root.querySelector<HTMLTextAreaElement>("#rle-json");
        const pv = root.querySelector<HTMLTextAreaElement>("#rle-prev");
        if (!ta || !pv) return;
        pv.value = tryPretty(JSON.parse(ta.value || "{}").limits || {});
        toast("Set current limits as Previous for diff");
      });

      // Compare vs previous
      root.querySelector<HTMLButtonElement>("#rle-compare")?.addEventListener("click", () => {
        const p = readPrev(root);
        const cur = buildPayload(root).limits;
        const diff = computeDiff(p || {}, cur);
        renderDiff(root, diff);
      });

      // Export JSON
      root.querySelector<HTMLButtonElement>("#rle-export")?.addEventListener("click", () => {
        const data = buildPayload(root);
        download(`risk_limits_${stamp(new Date())}.json`, pretty(data));
      });

      // Reset
      root.querySelector<HTMLButtonElement>("#rle-reset")?.addEventListener("click", () => {
        applyModel(root, { scope, scopeId, effectiveAt, notes, limits: initial || {} });
        toast("Reset to initial");
      });

      // Save
      root.querySelector<HTMLButtonElement>("#rle-save")?.addEventListener("click", async () => {
        const payload = buildPayload(root);
        const { errors, warnings } = validate(payload.limits);
        renderMessages(root, errors, warnings);
        if (errors.length) {
          toast("Fix errors before saving");
          return;
        }
        try {
          root.setAttribute("data-busy", "1");
         
          toast("Saved");
          markClean(root);
        } catch {
          toast("Save failed");
        } finally {
          root.removeAttribute("data-busy");
        }
      });

      // Track dirtiness
      root.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("input, textarea").forEach((el) => {
        el.addEventListener("input", () => markDirty(root));
      });
    });
  }

  // Render
  const L = initial || {};
  const PSC = L.perSymbolCaps || {};
  const rows = Object.keys(PSC).map((sym) => ({
    symbol: sym,
    maxQty: safeNum(PSC[sym]?.maxQty),
    maxNotional: safeNum(PSC[sym]?.maxNotional),
    maxWeightPct: safeNum(PSC[sym]?.maxWeightPct),
  }));

  return (
    <section id="rle-root" style={wrap} data-dirty="0">
      <style>{css}</style>
      <div id="rle-toast" style={toastStyle} />

      <header style={head}>
        <div>
          <h3 style={h3}>{title}</h3>
          <p style={sub}>Percent fields use PERCENT UNITS (e.g., <b>5</b> → 5%).</p>
        </div>
        <div style={ctrls}>
          <button id="rle-save" style={btn}>Save</button>
          <button id="rle-export" style={btnGhost}>Export JSON</button>
          <button id="rle-compare" style={btnGhost}>Compare vs Previous</button>
          <button id="rle-reset" style={btnGhost}>Reset</button>
        </div>
      </header>

      {/* Meta */}
      <section style={card}>
        <h4 style={h4}>Meta</h4>
        <div style={grid3}>
          <div>
            <label style={lbl}>Scope</label>
            <select name="scope" defaultValue={scope || ""} style={input}>
              <option value="">(none)</option>
              <option value="account">Account</option>
              <option value="household">Household</option>
              <option value="strategy">Strategy</option>
              <option value="sleeve">Sleeve</option>
              <option value="symbol">Symbol</option>
            </select>
          </div>
          <div>
            <label style={lbl}>Scope ID</label>
            <input name="scopeId" defaultValue={scopeId || ""} placeholder="e.g., ACC-123" style={input} />
          </div>
          <div>
            <label style={lbl}>Effective At</label>
            <input
              name="effectiveAt"
              type="datetime-local"
              defaultValue={toLocalDT(effectiveAt)}
              style={input}
            />
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <label style={lbl}>Notes</label>
          <input name="notes" defaultValue={notes || ""} placeholder="Optional note…" style={input} />
        </div>
      </section>

      {/* Limits sections */}
      <section style={grid2}>
        <div style={card}>
          <h4 style={h4}>Weights</h4>
          <div style={row}>
            <Field name="weights.maxPositionWeightPct" label="Max Position Weight %" def={L.weights?.maxPositionWeightPct} />
            <Field name="weights.maxSectorWeightPct" label="Max Sector Weight %" def={L.weights?.maxSectorWeightPct} />
          </div>
        </div>

        <div style={card}>
          <h4 style={h4}>Exposures</h4>
          <div style={row}>
            <Field name="exposures.grossExposurePct" label="Gross Exposure %" def={L.exposures?.grossExposurePct} />
            <Field name="exposures.netExposurePct" label="Net Exposure %" def={L.exposures?.netExposurePct} />
            <Field name="exposures.leverageMax" label="Leverage Max (x)" def={L.exposures?.leverageMax} step="0.1" />
          </div>
        </div>

        <div style={card}>
          <h4 style={h4}>Risk</h4>
          <div style={row}>
            <Field name="risk.dailyLossLimitPct" label="Daily Loss Limit %" def={L.risk?.dailyLossLimitPct} />
            <Field name="risk.trailingDrawdownPct" label="Trailing Drawdown %" def={L.risk?.trailingDrawdownPct} />
            <Field name="risk.varPct" label="VaR %" def={L.risk?.varPct} />
          </div>
        </div>

        <div style={card}>
          <h4 style={h4}>Trading</h4>
          <div style={row}>
            <Field name="trading.turnoverCapPct" label="Turnover Cap %" def={L.trading?.turnoverCapPct} />
            <Field name="trading.minTradeValue" label="Min Trade Value" def={L.trading?.minTradeValue} step="1" />
          </div>
        </div>

        <div style={card}>
          <h4 style={h4}>Cash</h4>
          <div style={row}>
            <Field name="cash.minCashPct" label="Min Cash %" def={L.cash?.minCashPct} />
            <Field name="cash.maxCashPct" label="Max Cash %" def={L.cash?.maxCashPct} />
          </div>
        </div>

        <div style={{ ...card, gridColumn: "1 / -1" }}>
          <header style={pscHead}>
            <h4 style={h4}>Per-symbol Caps</h4>
            <div style={{ display: "flex", gap: 8 }}>
              <button id="rle-add-row" style={btnGhost}>Add row</button>
              <button id="rle-clear-psc" style={btnGhost}>Clear</button>
            </div>
          </header>

          <div style={{ overflow: "auto", border: "1px solid var(--b)", borderRadius: 10 }}>
            <table style={pscTable}>
              <thead>
                <tr>
                  <th style={th}>Symbol</th>
                  <th style={th}>Max Qty</th>
                  <th style={th}>Max Notional</th>
                  <th style={th}>Max Weight %</th>
                  <th style={th} />
                </tr>
              </thead>
              <tbody id="rle-psc-body">
                {rows.map((r, i) => (
                  <tr key={i} data-row="1">
                    <td style={td}><input data-k="symbol" placeholder="AAPL" defaultValue={r.symbol} style={input} /></td>
                    <td style={td}><input data-k="maxQty" type="number" step="1" placeholder="—" defaultValue={r.maxQty} style={input} /></td>
                    <td style={td}><input data-k="maxNotional" type="number" step="1" placeholder="—" defaultValue={r.maxNotional} style={input} /></td>
                    <td style={td}><input data-k="maxWeightPct" type="number" step="0.01" placeholder="—" defaultValue={r.maxWeightPct} style={input} /></td>
                    
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Messages */}
      <section id="rle-msgs" style={{ ...card, ...(dense ? { fontSize: 12 } : null) }}>
        <h4 style={h4}>Validation</h4>
        <div id="rle-errs" style={{ display: "grid", gap: 4 }} />
        <div id="rle-warns" style={{ display: "grid", gap: 4, marginTop: 6 }} />
      </section>

      {/* JSON helpers */}
      <section style={{ ...card, display: "grid", gap: 8 }}>
        <h4 style={h4}>JSON</h4>
        <label style={lbl}>Current (editable)</label>
        <textarea id="rle-json" rows={8} style={ta} placeholder='{"scope":"account","limits":{...}}' />
        <div style={{ display: "flex", gap: 8 }}>
          <button id="rle-load" style={btnGhost}>Load From JSON</button>
          <button id="rle-set-prev" style={btnGhost}>Use Current Limits as Previous</button>
        </div>

        <label style={{ ...lbl, marginTop: 8 }}>Previous (for diff)</label>
        <textarea id="rle-prev" rows={6} style={ta} placeholder="Paste previous limits JSON…" />

        <div id="rle-diff" style={{ marginTop: 8, display: "grid", gap: 6 }} />
      </section>
    </section>
  );
}

/* -------------------------- Small field component -------------------------- */

function Field({ name, label, def, step = "0.01" }: { name: string; label: string; def?: number; step?: string }) {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <label style={lbl}>{label}</label>
      <input name={name} type="number" step={step} defaultValue={safeNum(def)} placeholder="—" style={input} />
    </div>
  );
}

/* ------------------------------- Build model ------------------------------- */

function buildPayload(root: HTMLElement) {
  const get = (sel: string) =>
    (root.querySelector(`[name="${sel}"]`) as HTMLInputElement | null)?.value ?? "";
  const num = (sel: string): number | undefined => {
    const v = get(sel).trim();
    if (v === "") return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };

  const model: NormalizedLimits = {};

  // weights
  const w1 = num("weights.maxPositionWeightPct");
  const w2 = num("weights.maxSectorWeightPct");
  if (w1 != null || w2 != null) model.weights = {};
  if (w1 != null) model.weights!.maxPositionWeightPct = w1;
  if (w2 != null) model.weights!.maxSectorWeightPct = w2;

  // exposures
  const e1 = num("exposures.grossExposurePct");
  const e2 = num("exposures.netExposurePct");
  const e3 = num("exposures.leverageMax");
  if (e1 != null || e2 != null || e3 != null) model.exposures = {};
  if (e1 != null) model.exposures!.grossExposurePct = e1;
  if (e2 != null) model.exposures!.netExposurePct = e2;
  if (e3 != null) model.exposures!.leverageMax = e3;

  // risk
  const r1 = num("risk.dailyLossLimitPct");
  const r2 = num("risk.trailingDrawdownPct");
  const r3 = num("risk.varPct");
  if (r1 != null || r2 != null || r3 != null) model.risk = {};
  if (r1 != null) model.risk!.dailyLossLimitPct = r1;
  if (r2 != null) model.risk!.trailingDrawdownPct = r2;
  if (r3 != null) model.risk!.varPct = r3;

  // trading
  const t1 = num("trading.turnoverCapPct");
  const t2 = num("trading.minTradeValue");
  if (t1 != null || t2 != null) model.trading = {};
  if (t1 != null) model.trading!.turnoverCapPct = t1;
  if (t2 != null) model.trading!.minTradeValue = t2;

  // cash
  const c1 = num("cash.minCashPct");
  const c2 = num("cash.maxCashPct");
  if (c1 != null || c2 != null) model.cash = {};
  if (c1 != null) model.cash!.minCashPct = c1;
  if (c2 != null) model.cash!.maxCashPct = c2;

  // perSymbolCaps
  const body = root.querySelector("#rle-psc-body");
  if (body) {
    const out: Record<string, any> = {};
    body.querySelectorAll("tr[data-row]").forEach((tr) => {
      const sym = (tr.querySelector<HTMLInputElement>('input[data-k="symbol"]')?.value || "").trim().toUpperCase();
      if (!sym) return;
      const maxQty = (tr.querySelector<HTMLInputElement>('input[data-k="maxQty"]')?.value || "").trim();
      const maxNotional = (tr.querySelector<HTMLInputElement>('input[data-k="maxNotional"]')?.value || "").trim();
      const maxWeightPct = (tr.querySelector<HTMLInputElement>('input[data-k="maxWeightPct"]')?.value || "").trim();
      const row: any = {};
      if (maxQty !== "") row.maxQty = Number(maxQty);
      if (maxNotional !== "") row.maxNotional = Number(maxNotional);
      if (maxWeightPct !== "") row.maxWeightPct = Number(maxWeightPct);
      if (Object.keys(row).length) out[sym] = row;
    });
    if (Object.keys(out).length) model.perSymbolCaps = out;
  }

  const payload = {
    scope: valStr((root.querySelector('[name="scope"]') as HTMLSelectElement | null)?.value),
    scopeId: valStr(get("scopeId")),
    effectiveAt: toIso((root.querySelector('[name="effectiveAt"]') as HTMLInputElement | null)?.value),
    notes: valStr(get("notes")),
    limits: model,
    prev: readPrev(root) || undefined,
  };
  return payload;
}

/* ------------------------------- Apply model ------------------------------- */

function applyModel(root: HTMLElement, obj: any) {
  const L = (obj?.limits || {}) as NormalizedLimits;

  setVal(root, 'select[name="scope"]', obj?.scope || "");
  setVal(root, 'input[name="scopeId"]', obj?.scopeId || "");
  setVal(root, 'input[name="effectiveAt"]', toLocalDT(obj?.effectiveAt));
  setVal(root, 'input[name="notes"]', obj?.notes || "");

  setVal(root, 'input[name="weights.maxPositionWeightPct"]', safeNum(L.weights?.maxPositionWeightPct));
  setVal(root, 'input[name="weights.maxSectorWeightPct"]', safeNum(L.weights?.maxSectorWeightPct));

  setVal(root, 'input[name="exposures.grossExposurePct"]', safeNum(L.exposures?.grossExposurePct));
  setVal(root, 'input[name="exposures.netExposurePct"]', safeNum(L.exposures?.netExposurePct));
  setVal(root, 'input[name="exposures.leverageMax"]', safeNum(L.exposures?.leverageMax));

  setVal(root, 'input[name="risk.dailyLossLimitPct"]', safeNum(L.risk?.dailyLossLimitPct));
  setVal(root, 'input[name="risk.trailingDrawdownPct"]', safeNum(L.risk?.trailingDrawdownPct));
  setVal(root, 'input[name="risk.varPct"]', safeNum(L.risk?.varPct));

  setVal(root, 'input[name="trading.turnoverCapPct"]', safeNum(L.trading?.turnoverCapPct));
  setVal(root, 'input[name="trading.minTradeValue"]', safeNum(L.trading?.minTradeValue));

  setVal(root, 'input[name="cash.minCashPct"]', safeNum(L.cash?.minCashPct));
  setVal(root, 'input[name="cash.maxCashPct"]', safeNum(L.cash?.maxCashPct));

  // PSC rows
  const body = root.querySelector("#rle-psc-body");
  if (body) {
    body.innerHTML = "";
    const PSC = L.perSymbolCaps || {};
    Object.keys(PSC).forEach((sym) => {
      addPSCRow(root, {
        symbol: sym,
        maxQty: safeNum(PSC[sym]?.maxQty),
        maxNotional: safeNum(PSC[sym]?.maxNotional),
        maxWeightPct: safeNum(PSC[sym]?.maxWeightPct),
      });
    });
  }

  markDirty(root);
}

/* --------------------------------- Validate -------------------------------- */

function validate(lim: NormalizedLimits) {
  const errors: string[] = [];
  const warnings: string[] = [];

  const pct = (path: string, v?: number, lo = 0, hi = 1000) => {
    if (v == null) return;
    if (!Number.isFinite(v)) errors.push(`${path} must be a number`);
    else if (v < lo || v > hi) warnings.push(`${path} is outside typical range (${lo}..${hi})`);
  };
  const nonNeg = (path: string, v?: number) => {
    if (v == null) return;
    if (!Number.isFinite(v)) errors.push(`${path} must be a number`);
    else if (v < 0) errors.push(`${path} cannot be negative`);
  };

  if (lim.weights) {
    pct("weights.maxPositionWeightPct", lim.weights.maxPositionWeightPct, 0, 100);
    pct("weights.maxSectorWeightPct", lim.weights.maxSectorWeightPct, 0, 100);
  }
  if (lim.exposures) {
    pct("exposures.grossExposurePct", lim.exposures.grossExposurePct, 0, 500);
    pct("exposures.netExposurePct", lim.exposures.netExposurePct, 0, 500);
    nonNeg("exposures.leverageMax", lim.exposures.leverageMax);
  }
  if (lim.risk) {
    pct("risk.dailyLossLimitPct", lim.risk.dailyLossLimitPct, 0, 100);
    pct("risk.trailingDrawdownPct", lim.risk.trailingDrawdownPct, 0, 100);
    pct("risk.varPct", lim.risk.varPct, 0, 100);
  }
  if (lim.trading) {
    pct("trading.turnoverCapPct", lim.trading.turnoverCapPct, 0, 1000);
    nonNeg("trading.minTradeValue", lim.trading.minTradeValue);
  }
  if (lim.cash) {
    pct("cash.minCashPct", lim.cash.minCashPct, 0, 100);
    pct("cash.maxCashPct", lim.cash.maxCashPct, 0, 100);
    if ((lim.cash.minCashPct ?? -1) > (lim.cash.maxCashPct ?? 99999)) {
      errors.push("cash.minCashPct cannot be greater than cash.maxCashPct");
    }
  }
  if (lim.perSymbolCaps) {
    for (const sym of Object.keys(lim.perSymbolCaps)) {
      const r = lim.perSymbolCaps[sym]!;
      if (r.maxQty != null && r.maxQty < 0) errors.push(`perSymbolCaps.${sym}.maxQty cannot be negative`);
      if (r.maxNotional != null && r.maxNotional < 0) errors.push(`perSymbolCaps.${sym}.maxNotional cannot be negative`);
      if (r.maxWeightPct != null && (r.maxWeightPct < 0 || r.maxWeightPct > 100)) warnings.push(`perSymbolCaps.${sym}.maxWeightPct is outside 0..100`);
    }
  }

  return { errors, warnings };
}

/* ----------------------------------- Diff ---------------------------------- */

function computeDiff(prev: NormalizedLimits, next: NormalizedLimits) {
  const rows: Array<{ path: string; from: any; to: any }> = [];
  const keys = new Set<string>();
  const walk = (obj: any, base: string) => {
    for (const k of Object.keys(obj || {})) {
      const path = base ? `${base}.${k}` : k;
      keys.add(path);
      if (obj[k] && typeof obj[k] === "object" && !Array.isArray(obj[k])) walk(obj[k], path);
    }
  };
  walk(prev || {}, "");
  walk(next || {}, "");
  for (const p of keys) {
    const a = get(prev, p);
    const b = get(next, p);
    if (!deepEq(a, b)) rows.push({ path: p, from: a, to: b });
  }
  return rows.sort((x, y) => x.path.localeCompare(y.path));
}

/* ---------------------------------- UI fx ---------------------------------- */

function renderMessages(root: HTMLElement, errs: string[], warns: string[]) {
  const e = root.querySelector("#rle-errs") as HTMLElement;
  const w = root.querySelector("#rle-warns") as HTMLElement;
  if (e) e.innerHTML = errs.length ? errs.map((s) => `<div style="color:#b42318">• ${esc(s)}</div>`).join("") : `<div style="color:#6b7280">No errors.</div>`;
  if (w) w.innerHTML = warns.length ? warns.map((s) => `<div style="color:#a16207">• ${esc(s)}</div>`).join("") : `<div style="color:#6b7280">No warnings.</div>`;
}

function renderDiff(root: HTMLElement, diff: Array<{ path: string; from: any; to: any }>) {
  const box = root.querySelector("#rle-diff") as HTMLElement | null;
  if (!box) return;
  if (!diff.length) {
    box.innerHTML = `<div style="color:#6b7280">No differences.</div>`;
    return;
  }
  box.innerHTML = diff
    .map(
      (d) => `<div style="font-family: ui-monospace, Menlo, monospace; font-size:12.5px;">
        <b>${esc(d.path)}</b>: <span style="color:#6b7280">${esc(JSON.stringify(d.from))}</span> → <span style="color:#111">${esc(JSON.stringify(d.to))}</span>
      </div>`,
    )
    .join("");
}

function addPSCRow(
  root: HTMLElement,
  row: { symbol: string; maxQty: string | number; maxNotional: string | number; maxWeightPct: string | number },
) {
  const body = root.querySelector("#rle-psc-body");
  if (!body) return;
  const tr = document.createElement("tr");
  tr.setAttribute("data-row", "1");
  tr.innerHTML = `
    <td style="padding:8px;border-bottom:1px solid var(--rb);"><input data-k="symbol" placeholder="AAPL" value="${esc(String(row.symbol || ""))}" style="${inlineInput()}"/></td>
    <td style="padding:8px;border-bottom:1px solid var(--rb);"><input data-k="maxQty" type="number" step="1" placeholder="—" value="${esc(String(row.maxQty || ""))}" style="${inlineInput()}"/></td>
    <td style="padding:8px;border-bottom:1px solid var(--rb);"><input data-k="maxNotional" type="number" step="1" placeholder="—" value="${esc(String(row.maxNotional || ""))}" style="${inlineInput()}"/></td>
    <td style="padding:8px;border-bottom:1px solid var(--rb);"><input data-k="maxWeightPct" type="number" step="0.01" placeholder="—" value="${esc(String(row.maxWeightPct || ""))}" style="${inlineInput()}"/></td>
    <td style="padding:8px;border-bottom:1px solid var(--rb); text-align:right;"><button style="${inlineChipBtn()}" onclick="this.closest('tr').remove(); document.getElementById('rle-root').setAttribute('data-dirty','1');">Remove</button></td>
  `;
  body.appendChild(tr);
  markDirty(root);
}

function download(filename: string, text: string) {
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function readPrev(root: HTMLElement): NormalizedLimits | null {
  try {
    const ta = root.querySelector<HTMLTextAreaElement>("#rle-prev");
    if (!ta) return null;
    const obj = JSON.parse(ta.value || "{}");
    // allow either full payload or just limits
    return (obj?.limits as NormalizedLimits) || (obj as NormalizedLimits) || null;
  } catch {
    return null;
  }
}

/* ---------------------------------- Utils ---------------------------------- */

function setVal(root: HTMLElement, sel: string, v: any) {
  const el = root.querySelector<HTMLInputElement | HTMLSelectElement>(sel);
  if (!el) return;
  (el as any).value = v ?? "";
}

function safeNum(n?: number) {
  return Number.isFinite(n as number) ? String(n) : "";
}

function pretty(x: any) {
  return JSON.stringify(x, null, 2);
}
function tryPretty(x: any) {
  try { return JSON.stringify(x, null, 2); } catch { return ""; }
}

function valStr(v?: string) {
  const s = (v || "").trim();
  return s ? s : undefined;
}
function toIso(local?: string) {
  if (!local) return undefined;
  try {
    // local datetime-local -> ISO
    const d = new Date(local);
    return isNaN(d.getTime()) ? undefined : d.toISOString();
  } catch {
    return undefined;
  }
}
function toLocalDT(iso?: string) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch { return ""; }
}

function get(obj: any, path: string): any {
  return path.split(".").reduce((acc, k) => (acc ? acc[k] : undefined), obj);
}
function deepEq(a: any, b: any): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a && b && typeof a === "object") {
    const ak = Object.keys(a), bk = Object.keys(b);
    if (ak.length !== bk.length) return false;
    for (const k of ak) if (!deepEq(a[k], b[k])) return false;
    return true;
  }
  return false;
}

function stamp(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function esc(s: string) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function toast(msg: string) {
  const el = document.getElementById("rle-toast");
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

function markDirty(root: HTMLElement) { root.setAttribute("data-dirty", "1"); }
function markClean(root: HTMLElement) { root.setAttribute("data-dirty", "0"); }

function inlineInput() {
  return "width:100%;height:30px;padding:4px 8px;border-radius:10px;border:1px solid var(--b);outline:none;background:#fff";
}
function inlineChipBtn() {
  return "border:1px solid #d4d4d8;background:#fff;border-radius:999px;padding:2px 8px;cursor:pointer;font-size:12px";
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

const card: any = { border: "1px solid var(--b)", borderRadius: 14, background: "var(--bg)", padding: 12, display: "grid", gap: 10 };
const grid2: any = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 };
const grid3: any = { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 };
const row: any = { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 };

const lbl: any = { fontSize: 12, color: "#6b7280" };
const input: any = { width: "100%", height: 32, padding: "4px 8px", borderRadius: 10, border: "1px solid var(--b)", outline: "none", background: "#fff" };
const ta: any = { width: "100%", border: "1px solid var(--b)", borderRadius: 10, padding: 8, minHeight: 120, fontFamily: "ui-monospace,Menlo,monospace", fontSize: 12.5 };

const pscHead: any = { display: "flex", alignItems: "center", justifyContent: "space-between" };
const pscTable: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, background: "var(--bg)" };
const th: any = { textAlign: "left", padding: "8px 10px", borderBottom: "1px solid var(--b)", color: "#6b7280", fontSize: 12, whiteSpace: "nowrap" };
const td: any = { padding: "8px", borderBottom: "1px solid var(--rb)", whiteSpace: "nowrap" };
const tdAct: any = { ...td, textAlign: "right" };
const chipBtn: any = { border: "1px solid #d4d4d8", background: "#fff", borderRadius: 999, padding: "2px 8px", cursor: "pointer", fontSize: 12 };

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
  #rle-root[data-busy="1"] { opacity:.6; pointer-events:none; }
  #rle-root[data-dirty="1"] h3::after { content:" • unsaved"; color:#b45309; font-weight:600; font-size:12px; margin-left:6px; }

  @media (max-width: 900px) { .grid2, .grid3, .row { grid-template-columns: 1fr !important; } }

  @media (prefers-color-scheme: dark) {
    :root { --b:rgba(255,255,255,.12); --rb:rgba(255,255,255,.06); --bg:#0b0b0c; }
    section, table, th, td, input, select, textarea { color:#e5e7eb !important; }
    input, select, textarea { background:#0b0b0c !important; border-color:var(--b) !important; }
  }

  #rle-toast[data-show="1"] { opacity: 1; }
`;
