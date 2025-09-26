// app/components/rebalancemodeal.tsx
// No imports. No hooks. Self-contained modal to collect rebalance parameters.
// - Optional trigger button opens the modal; overlay click & Esc close it
// - Can either POST to a passed Server Action `action` OR call `onRun(cfg)`
// - Textareas accept JSON for positions/prices/targets/lot sizes; blacklist as CSV
// - Inline styles only; dark-mode friendly basics

"use client";

type Position = { symbol: string; qty: number };
type PriceMap = Record<string, number>;
type Target = { symbol: string; weight: number };
type LotSizeMap = Record<string, number>;

type SidePref = "neutral" | "sellFirst" | "buyFirst";

type Props = {
  triggerLabel?: string;                 // default "Rebalance…"
  showTrigger?: boolean;                 // default true
  action?: (fd: FormData) => Promise<any>; // Next.js Server Action passed from parent
  onRun?: (cfg: {
    positions: Position[];
    prices: PriceMap;
    cash: number;
    targets: Target[];
    feeBps: number;
    minTradeValue: number;
    lotSizeMap: LotSizeMap;
    tolerancePct: number;
    maxTurnoverPct: number;
    reserveCashPct: number;
    blacklist: string[];
    sidePreference: SidePref;
    notes?: string;
  }) => void | Promise<void>;
  onClose?: () => void | Promise<void>;

  // Initial values (optional)
  positions?: Position[];
  prices?: PriceMap;
  cash?: number;
  targets?: Target[];
  feeBps?: number;
  minTradeValue?: number;
  lotSizeMap?: LotSizeMap;
  tolerancePct?: number;
  maxTurnoverPct?: number;
  reserveCashPct?: number;
  blacklist?: string[];
  sidePreference?: SidePref;
  notes?: string;
};

export default function RebalanceModal({
  triggerLabel = "Rebalance…",
  showTrigger = true,
  action,
  onRun,
  onClose,

  positions = [],
  prices = {},
  cash = 0,
  targets = [],
  feeBps = 5,
  minTradeValue = 0,
  lotSizeMap = {},
  tolerancePct = 0.1,
  maxTurnoverPct = 100,
  reserveCashPct = 0,
  blacklist = [],
  sidePreference = "sellFirst",
  notes = "",
}: Props) {
  // ---------- wiring without hooks ----------
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("rbm-root");
      if (!root) return;

      const openBtn = root.querySelector<HTMLButtonElement>("#rbm-open");
      const closeBtn = root.querySelector<HTMLButtonElement>("#rbm-close");
      const overlay = root.querySelector<HTMLDivElement>("#rbm-ovl");
      const panel = root.querySelector<HTMLDivElement>("#rbm-panel");
      const form = root.querySelector<HTMLFormElement>("#rbm-form");

      const open = () => {
        root.setAttribute("data-open", "1");
        setTimeout(() => {
          form?.querySelector<HTMLTextAreaElement>('textarea[name="targets"]')?.focus();
        }, 0);
      };
      const close = async () => {
        root.removeAttribute("data-open");
        if (onClose) await onClose();
      };

      openBtn?.addEventListener("click", open);
      closeBtn?.addEventListener("click", close);
      overlay?.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).id === "rbm-ovl") close();
      });

      document.addEventListener("keydown", (e) => {
        if (root.getAttribute("data-open") === "1" && e.key === "Escape") {
          e.preventDefault();
          close();
        }
      });

      // Quick chips
      root.querySelectorAll<HTMLButtonElement>('button[data-quick]').forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          const t = btn.dataset.quick || "";
          const input = form?.elements.namedItem(btn.dataset.name || "") as HTMLInputElement | null;
          if (!input) return;
          input.value = t;
        });
      });

      // Paste JSON helpers
      root.querySelectorAll<HTMLButtonElement>('button[data-paste]').forEach((btn) => {
        btn.addEventListener("click", async (e) => {
          e.preventDefault();
          try {
            const txt = await navigator.clipboard.readText();
            const ta = form?.elements.namedItem(btn.dataset.paste || "") as HTMLTextAreaElement | null;
            if (ta) ta.value = txt;
            toast("Pasted from clipboard");
          } catch {
            toast("Clipboard unavailable");
          }
        });
      });

      // Submit handling when no Server Action is provided
      if (!action && onRun && form) {
        form.addEventListener("submit", async (e) => {
          e.preventDefault();
          const fd = new FormData(form);
          const payload = parseForm(fd);
          if (!payload.ok) {
            toast(payload.error || "Invalid input");
            return;
          }
          try {
            root.setAttribute("data-busy", "1");
            await onRun(payload.data);
            toast("Rebalance submitted");
            close();
          } catch {
            toast("Run failed");
          } finally {
            root.removeAttribute("data-busy");
          }
        });
      }
    });
  }

  // ---------- prefilled strings ----------
  const sPositions = pretty(positions.length ? positions : samplePositions());
  const sPrices = pretty(Object.keys(prices).length ? prices : samplePrices(positions));
  const sTargets = pretty(targets.length ? targets : sampleTargets(positions));
  const sLots = pretty(Object.keys(lotSizeMap).length ? lotSizeMap : sampleLots(positions));
  const sBlacklist = (blacklist || []).join(", ");

  return (
    <section id="rbm-root" style={wrap} aria-label="Rebalance Modal Root">
      <style>{css}</style>
      <div id="rbm-toast" style={toastStyle} />

      {showTrigger ? (
        <button id="rbm-open" style={triggerBtn} type="button" aria-haspopup="dialog" aria-controls="rbm-panel">
          {triggerLabel}
        </button>
      ) : null}

      {/* Overlay + Panel */}
      <div id="rbm-ovl" style={overlay} role="presentation">
        <div id="rbm-panel" role="dialog" aria-modal="true" aria-labelledby="rbm-title" style={panel}>
          <header style={head}>
            <div>
              <h3 id="rbm-title" style={h3}>Rebalance</h3>
              <p style={sub}>Configure tolerance, fees, turnover cap, lots, and targets. Paste JSON or edit inline.</p>
            </div>
            <button id="rbm-close" type="button" style={closeBtn} aria-label="Close">✕</button>
          </header>

          <div style={twoCol}>
            {/* Left: JSON inputs */}
            <div style={col}>
              <Field label="Positions (JSON array)" hint='[{ "symbol": "INFY", "qty": 120 } …]'>
                <textarea name="positions" defaultValue={sPositions} style={ta} spellCheck={false} />
                <div style={rowSmall}><button data-paste="positions" style={ghostBtn}>Paste</button></div>
              </Field>

              <Field label="Prices (JSON map)" hint='{ "INFY": 1490.5, "TCS": 3920 }'>
                <textarea name="prices" defaultValue={sPrices} style={ta} spellCheck={false} />
                <div style={rowSmall}><button data-paste="prices" style={ghostBtn}>Paste</button></div>
              </Field>

              <Field label="Targets (JSON array)" hint='[{ "symbol": "INFY", "weight": 0.20 } …] (sum ≤ 1)'>
                <textarea name="targets" defaultValue={sTargets} style={ta} spellCheck={false} />
                <div style={rowSmall}><button data-paste="targets" style={ghostBtn}>Paste</button></div>
              </Field>
            </div>

            {/* Right: Parameters */}
            <div style={col}>
              <Field label="Cash" hint="Available cash for buys">
                <input name="cash" type="number" step="0.01" defaultValue={cash} style={input} />
              </Field>

              <div style={grid2}>
                <Field label="Fee (bps)">
                  <input name="feeBps" type="number" step="1" defaultValue={feeBps} style={input} />
                  <div style={rowSmall}>
                    <button data-quick="0" data-name="feeBps" style={chipBtn}>0</button>
                    <button data-quick="5" data-name="feeBps" style={chipBtn}>5</button>
                    <button data-quick="10" data-name="feeBps" style={chipBtn}>10</button>
                  </div>
                </Field>

                <Field label="Min trade value">
                  <input name="minTradeValue" type="number" step="1" defaultValue={minTradeValue} style={input} />
                  <div style={rowSmall}>
                    <button data-quick="0" data-name="minTradeValue" style={chipBtn}>0</button>
                    <button data-quick="1000" data-name="minTradeValue" style={chipBtn}>1k</button>
                    <button data-quick="5000" data-name="minTradeValue" style={chipBtn}>5k</button>
                  </div>
                </Field>

                <Field label="Tolerance (%)" hint="Drift threshold per symbol">
                  <input name="tolerancePct" type="number" step="0.01" defaultValue={tolerancePct} style={input} />
                  <div style={rowSmall}>
                    <button data-quick="0.1" data-name="tolerancePct" style={chipBtn}>0.10</button>
                    <button data-quick="0.5" data-name="tolerancePct" style={chipBtn}>0.50</button>
                    <button data-quick="1" data-name="tolerancePct" style={chipBtn}>1.00</button>
                  </div>
                </Field>

                <Field label="Turnover cap (%)" hint="Max gross turnover as % of starting MV">
                  <input name="maxTurnoverPct" type="number" step="1" defaultValue={maxTurnoverPct} style={input} />
                  <div style={rowSmall}>
                    <button data-quick="20" data-name="maxTurnoverPct" style={chipBtn}>20</button>
                    <button data-quick="50" data-name="maxTurnoverPct" style={chipBtn}>50</button>
                    <button data-quick="100" data-name="maxTurnoverPct" style={chipBtn}>100</button>
                  </div>
                </Field>

                <Field label="Reserve cash (%)" hint="Keeps this % in cash after rebalance">
                  <input name="reserveCashPct" type="number" step="0.1" defaultValue={reserveCashPct} style={input} />
                  <div style={rowSmall}>
                    <button data-quick="0" data-name="reserveCashPct" style={chipBtn}>0</button>
                    <button data-quick="2" data-name="reserveCashPct" style={chipBtn}>2</button>
                    <button data-quick="5" data-name="reserveCashPct" style={chipBtn}>5</button>
                  </div>
                </Field>
              </div>

              <Field label="Lot sizes (JSON map)" hint='{ "INFY": 1, "NIFTY": 25 }'>
                <textarea name="lotSizeMap" defaultValue={sLots} style={ta} spellCheck={false} />
              </Field>

              <Field label="Blacklist (CSV symbols)" hint='e.g. "XYZ,ABC"'>
                <input name="blacklist" defaultValue={sBlacklist} style={input} />
              </Field>

              <Field label="Side preference">
                <select name="sidePreference" defaultValue={sidePreference} style={input}>
                  <option value="sellFirst">Sell first</option>
                  <option value="buyFirst">Buy first</option>
                  <option value="neutral">Neutral</option>
                </select>
              </Field>

              <Field label="Notes">
                <textarea name="notes" defaultValue={notes} style={taSmall} />
              </Field>
            </div>
          </div>

          {/* Submit */}
          <footer style={actions}>
            {action ? (
              <form id="rbm-form" method="post" action={action as any} style={formInline}>
                {/* mirrored inputs for server action post */}
                <Mirror name="positions" value={sPositions} />
                <Mirror name="prices" value={sPrices} />
                <Mirror name="targets" value={sTargets} />
                <Mirror name="lotSizeMap" value={sLots} />
                <Mirror name="blacklist" value={sBlacklist} />

                <input name="cash" type="hidden" defaultValue={String(cash)} />
                <input name="feeBps" type="hidden" defaultValue={String(feeBps)} />
                <input name="minTradeValue" type="hidden" defaultValue={String(minTradeValue)} />
                <input name="tolerancePct" type="hidden" defaultValue={String(tolerancePct)} />
                <input name="maxTurnoverPct" type="hidden" defaultValue={String(maxTurnoverPct)} />
                <input name="reserveCashPct" type="hidden" defaultValue={String(reserveCashPct)} />
                <input name="sidePreference" type="hidden" defaultValue={sidePreference} />
                <input name="notes" type="hidden" defaultValue={notes} />

                <button type="submit" style={primaryBtn}>Run Rebalance</button>
              </form>
            ) : (
              <form id="rbm-form" style={formInline}>
                <button type="submit" style={primaryBtn}>Run Rebalance</button>
              </form>
            )}
            <button id="rbm-close" type="button" style={ghostBtn}>Cancel</button>
          </footer>
        </div>
      </div>
    </section>
  );
}

/* ---------------- tiny components/helpers ---------------- */
function Field(props: { label: string; hint?: string; children: any }) {
  return (
    <label style={field}>
      <div style={labRow}>
        <span style={lab}>{props.label}</span>
        {props.hint ? <span style={hint}>{props.hint}</span> : null}
      </div>
      <div>{props.children}</div>
    </label>
  );
}

function Mirror({ name, value }: { name: string; value: string }) {
  return <input type="hidden" name={name} defaultValue={value} />;
}

function parseForm(fd: FormData): { ok: true; data: any } | { ok: false; error: string } {
  const positions = readJson<Position[]>(fd.get("positions")) ?? [];
  const prices = readJson<PriceMap>(fd.get("prices")) ?? {};
  const targets = readJson<Target[]>(fd.get("targets")) ?? [];
  const lotSizeMap = readJson<LotSizeMap>(fd.get("lotSizeMap")) ?? {};
  const blacklist = readCsv(fd.get("blacklist"));

  if (!Object.keys(prices).length) return { ok: false, error: "Prices are required" };

  const data = {
    positions: positions.map((p) => ({ symbol: String(p.symbol).toUpperCase(), qty: num(p.qty) || 0 })),
    prices: upperKeys(prices),
    cash: num(fd.get("cash")) ?? 0,
    targets: (targets || []).map((t) => ({ symbol: String(t.symbol).toUpperCase(), weight: num(t.weight) || 0 })),
    feeBps: num(fd.get("feeBps")) ?? 5,
    minTradeValue: num(fd.get("minTradeValue")) ?? 0,
    lotSizeMap: upperKeys(lotSizeMap),
    tolerancePct: num(fd.get("tolerancePct")) ?? 0.1,
    maxTurnoverPct: num(fd.get("maxTurnoverPct")) ?? 100,
    reserveCashPct: num(fd.get("reserveCashPct")) ?? 0,
    blacklist,
    sidePreference: (str(fd.get("sidePreference")) as any) || "sellFirst",
    notes: str(fd.get("notes")),
  };
  return { ok: true, data };
}

function pretty(x: any) {
  try { return JSON.stringify(x, null, 2); } catch { return ""; }
}
function readJson<T>(v: any): T | undefined {
  const s = str(v);
  if (!s) return undefined;
  try { return JSON.parse(s) as T; } catch { return undefined; }
}
function readCsv(v: any): string[] {
  const s = str(v);
  if (!s) return [];
  return s.split(/[,\s]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
}
function upperKeys<T extends Record<string, any>>(obj: T) {
  const out: any = {};
  for (const k of Object.keys(obj || {})) out[String(k).toUpperCase()] = obj[k];
  return out as T;
}
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

function samplePositions(): Position[] {
  return [
    { symbol: "INFY", qty: 120 },
    { symbol: "TCS", qty: 80 },
  ];
}
function samplePrices(pos: Position[]): PriceMap {
  const out: PriceMap = {};
  (pos.length ? pos : samplePositions()).forEach((p) => {
    out[p.symbol] = p.symbol === "INFY" ? 1490.5 : p.symbol === "TCS" ? 3920 : 100;
  });
  return out;
}
function sampleTargets(pos: Position[]): Target[] {
  const syms = (pos.length ? pos : samplePositions()).map((p) => p.symbol);
  const w = 1 / Math.max(1, syms.length);
  return syms.map((s) => ({ symbol: s, weight: Number(w.toFixed(2)) }));
}
function sampleLots(pos: Position[]): LotSizeMap {
  const out: LotSizeMap = {};
  (pos.length ? pos : samplePositions()).forEach((p) => (out[p.symbol] = 1));
  return out;
}

function toast(msg: string) {
  const el = document.getElementById("rbm-toast");
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

/* ---------------- styles ---------------- */
const wrap: any = { position: "relative", display: "inline-block" };
const triggerBtn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "8px 12px", cursor: "pointer", fontSize: 14 };

const overlay: any = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.32)",
  display: "none",
  alignItems: "center",
  justifyContent: "center",
  padding: 10,
  zIndex: 50,
};
const panel: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 12px 48px rgba(0,0,0,0.18)",
  width: "min(980px, 96vw)",
  maxHeight: "92vh",
  overflow: "auto",
  padding: 14,
  display: "grid",
  gap: 12,
};

const head: any = { display: "flex", alignItems: "start", justifyContent: "space-between", gap: 8 };
const h3: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };
const closeBtn: any = { border: "1px solid #e5e7eb", background: "#fff", borderRadius: 10, padding: "4px 8px", cursor: "pointer" };

const twoCol: any = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 };
const col: any = { display: "grid", gap: 10 };

const field: any = { display: "grid", gap: 6 };
const labRow: any = { display: "flex", alignItems: "baseline", justifyContent: "space-between" };
const lab: any = { fontWeight: 600, fontSize: 13 };
const hint: any = { color: "#6b7280", fontSize: 12 };

const input: any = { width: "100%", height: 34, padding: "6px 8px", borderRadius: 10, border: "1px solid #e5e7eb", outline: "none" };
const ta: any = { width: "100%", minHeight: 120, fontFamily: "ui-monospace, Menlo, Monaco, monospace", fontSize: 12.5, padding: 10, borderRadius: 10, border: "1px solid #e5e7eb", outline: "none", whiteSpace: "pre" };
const taSmall: any = { ...ta, minHeight: 70 };

const grid2: any = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 };
const rowSmall: any = { display: "flex", gap: 6, marginTop: 4 };
const chipBtn: any = { border: "1px solid #e5e7eb", background: "#f5f5f7", borderRadius: 999, padding: "2px 8px", cursor: "pointer", fontSize: 12 };
const ghostBtn: any = { border: "1px solid #d4d4d8", background: "#fff", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const actions: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginTop: 4 };
const primaryBtn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "8px 12px", cursor: "pointer", fontSize: 14 };
const formInline: any = { display: "inline" };

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
  #rbm-root[data-open="1"] #rbm-ovl { display: flex; }
  #rbm-root[data-busy="1"] { opacity: .7; pointer-events: none; }
  #rbm-toast[data-show="1"] { opacity: 1 !important; }
  @media (prefers-color-scheme: dark) {
    #rbm-ovl { background: rgba(0,0,0,0.6); }
    #rbm-panel[role="dialog"] {
      background: #0b0b0c !important;
      border-color: rgba(255,255,255,0.1) !important;
      box-shadow: 0 12px 48px rgba(0,0,0,0.8) !important;
      color: #e5e7eb;
    }
    textarea, input, select { background: #111214; border-color: rgba(255,255,255,.15); color: #e5e7eb; }
    button { color: inherit; }
  }
`;
// ---------- comparison snippets from recently edited files ----------