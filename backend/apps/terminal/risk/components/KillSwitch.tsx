// app/components/killswitch.tsx
// No imports. No hooks. Self-contained, client-only Kill Switch.
// - States: SAFE → ARMED → KILLED (with cooldown)
// - Requires reason (and optional PIN) to KILL
// - Press “Arm”, then “Kill” (confirm modal). “Disarm” returns to SAFE.
// - Cooldown blocks Disarm after KILL until timer elapses
// - Persists state + audit log in localStorage
// - Exposes optional callbacks: onArm, onKill({reason}), onDisarm
// - Export audit log to CSV
//
// Usage:
//   <KillSwitch title="Execution Kill Switch" scope="Household H-1023" pin="1234" cooldownSec={90} />

"use client";

type KSState = "safe" | "armed" | "killed";

type LogRow = {
  ts: string;            // ISO time
  actor: string;         // best-effort user (from browser)
  action: "ARM" | "DISARM" | "KILL";
  reason?: string;
  scope?: string;
};

type Props = {
  title?: string;
  scope?: string;            // what this switch controls (display-only)
  requireReason?: boolean;   // default true
  pin?: string;              // optional PIN code required to KILL
  cooldownSec?: number;      // default 60
  storageKey?: string;       // default "killswitch_v1"
  onArm?: () => void | Promise<void>;
  onKill?: (payload: { reason: string }) => void | Promise<void>;
  onDisarm?: () => void | Promise<void>;
  note?: string;             // small helper text
};

export default function KillSwitch({
  title = "Kill Switch",
  scope,
  requireReason = true,
  pin,
  cooldownSec = 60,
  storageKey = "killswitch_v1",
  onArm,
  onKill,
  onDisarm,
  note,
}: Props) {
  const state = loadState(storageKey, { state: "safe" });

  // Paint once; wire events after hydration
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("ks-root");
      if (!root) return;

      // Initial UI reflect
      applyState(root, state.state as KSState, state.killedAt, cooldownSec);

      // Buttons
      root.querySelector<HTMLButtonElement>("#ks-arm")?.addEventListener("click", () => arm(root, storageKey, onArm));
      root.querySelector<HTMLButtonElement>("#ks-disarm")?.addEventListener("click", () => disarm(root, storageKey, onDisarm, cooldownSec));
      root.querySelector<HTMLButtonElement>("#ks-kill")?.addEventListener("click", () => openModal(root));
      root.querySelector<HTMLButtonElement>("#ks-export")?.addEventListener("click", () => exportLog(storageKey));

      // Modal confirm
      root.querySelector<HTMLButtonElement>("#ks-confirm")?.addEventListener("click", async () => {
        const r = (root.querySelector<HTMLTextAreaElement>("#ks-reason")?.value || "").trim();
        const p = (root.querySelector<HTMLInputElement>("#ks-pin")?.value || "").trim();

        if (pin && p !== pin) return toast(root, "Incorrect PIN");
        if (requireReason && !r) return toast(root, "Reason is required");

        await kill(root, storageKey, r, onKill);
        closeModal(root);
      });
      root.querySelector<HTMLButtonElement>("#ks-cancel")?.addEventListener("click", () => closeModal(root));

      // Countdown tick
      startTicker(root, cooldownSec);

      // Searchable audit filter
      root.querySelector<HTMLInputElement>('input[name="ks-search"]')?.addEventListener("input", () => filterLog(root, storageKey));

      // Hydrate audit log list
      renderLog(root, storageKey);
    });
  }

  const isKilled = state.state === "killed";

  return (
    <section id="ks-root" style={wrap} data-state={state.state} data-killed-at={state.killedAt || ""}>
      <style>{css}</style>
      <div id="ks-toast" style={toastStyle} />

      <header style={head}>
        <div style={{ display: "grid", gap: 4 }}>
          <h3 style={h3}>{title}</h3>
          {scope ? <p style={sub}><strong>Scope:</strong> {scope}</p> : null}
          {note ? <p style={{ ...sub, marginTop: 0 }}>{note}</p> : null}
        </div>

        <div style={statusWrap}>
          <span style={{ ...dot, ...(isKilled ? dotRed : state.state === "armed" ? dotAmber : dotGreen) }} />
          <span style={{ fontWeight: 700 }}>{labelFor(state.state as KSState)}</span>
          {isKilled ? <span id="ks-cd" style={cdLbl}></span> : null}
        </div>
      </header>

      <div style={card}>
        <div style={btnRow}>
          <button id="ks-arm" style={{ ...btn, ...btnWarn }} title="Arm the switch">Arm</button>
          <button id="ks-kill" style={{ ...btn, ...btnDanger }} title="Kill (requires confirm)">Kill</button>
          <button id="ks-disarm" style={{ ...btn, ...btnGhost }} title="Disarm to SAFE">Disarm</button>
          <button id="ks-export" style={{ ...btn, ...btnGhost }} title="Export audit CSV">Export Log</button>
        </div>
        <p style={help}>
          Flow: <b>Arm</b> → <b>Kill</b> (confirm) → <b>Disarm</b> after cooldown. Killing requires {requireReason ? "a reason" : "confirmation"}{pin ? " and PIN" : ""}.
        </p>
      </div>

      {/* Audit Log */}
      <section style={logWrap}>
        <header style={logHead}>
          <h4 style={h4}>Audit Log</h4>
          <div style={searchWrap}>
            <span style={searchIcon}>⌕</span>
            <input name="ks-search" placeholder="Filter by action, actor, reason…" style={searchInput} />
          </div>
        </header>
        <div style={{ overflow: "hidden", border: "1px solid var(--b)", borderRadius: 12 }}>
          <table style={logTable}>
            <thead>
              <tr>
                <th style={th}>Time</th>
                <th style={th}>Action</th>
                <th style={th}>Actor</th>
                <th style={th}>Scope</th>
                <th style={th}>Reason</th>
              </tr>
            </thead>
            <tbody id="ks-log-body" />
          </table>
        </div>
      </section>

      {/* Modal */}
      <div id="ks-modal" style={modalWrap} aria-hidden="true">
        <div style={modalMask} />
        <div style={modalCard} role="dialog" aria-modal="true" aria-label="Confirm kill">
          <header style={modalHead}>
            <span style={boom}>🛑</span>
            <div>
              <h4 style={h4}>Confirm Kill</h4>
              <p style={sub}>This immediately disables all actions for the configured scope.</p>
            </div>
          </header>
          <div style={{ display: "grid", gap: 8 }}>
            {requireReason ? (
              <div>
                <label style={lbl}>Reason</label>
                <textarea id="ks-reason" rows={3} placeholder="Write a short reason…" style={ta} />
              </div>
            ) : null}
            {pin ? (
              <div>
                <label style={lbl}>PIN</label>
                <input id="ks-pin" type="password" inputMode="numeric" style={pinInput} placeholder="Enter PIN" />
              </div>
            ) : null}
          </div>
          <footer style={modalBtns}>
            <button id="ks-cancel" style={{ ...btn, ...btnGhost }}>Cancel</button>
            <button id="ks-confirm" style={{ ...btn, ...btnDanger }}>Confirm Kill</button>
          </footer>
        </div>
      </div>
    </section>
  );
}

/* ========================== State & Persistence ========================== */

function loadState(key: string, fallback: any) {
  try {
    const j = localStorage.getItem(key);
    if (!j) return fallback;
    const obj = JSON.parse(j);
    return { ...fallback, ...obj };
  } catch {
    return fallback;
  }
}
function saveState(key: string, obj: any) {
  try { localStorage.setItem(key, JSON.stringify(obj)); } catch {}
}
function addLog(key: string, row: LogRow) {
  try {
    const obj = loadState(key, {});
    const log: LogRow[] = Array.isArray(obj.log) ? obj.log : [];
    log.unshift(row);
    obj.log = log.slice(0, 500); // keep last 500
    saveState(key, obj);
  } catch {}
}

/* ========================== Actions ========================== */

async function arm(root: HTMLElement, key: string, cb?: () => void | Promise<void>) {
  if (root.dataset.state === "killed") return toast(root, "Cannot arm while KILLED");
  try {
    root.setAttribute("data-busy", "1");
    const obj = loadState(key, {});
    obj.state = "armed";
    delete obj.killedAt;
    saveState(key, obj);
    addLog(key, { ts: new Date().toISOString(), actor: who(), action: "ARM", scope: obj.scope });
    applyState(root, "armed");
    renderLog(root, key);
    if (cb) await cb();
    toast(root, "Armed");
  } finally {
    root.removeAttribute("data-busy");
  }
}

async function disarm(root: HTMLElement, key: string, cb?: () => void | Promise<void>, cooldownSec = 60) {
  const st = (root.dataset.state || "safe") as KSState;
  if (st === "killed") {
    // enforce cooldown
    const killedAt = Date.parse(root.dataset.killedAt || "");
    const remain = timeLeft(killedAt, cooldownSec);
    if (remain > 0) return toast(root, `Cooldown: ${remain}s remaining`);
  }
  try {
    root.setAttribute("data-busy", "1");
    const obj = loadState(key, {});
    obj.state = "safe";
    delete obj.killedAt;
    saveState(key, obj);
    addLog(key, { ts: new Date().toISOString(), actor: who(), action: "DISARM", scope: obj.scope });
    applyState(root, "safe");
    renderLog(root, key);
    if (cb) await cb();
    toast(root, "Disarmed");
  } finally {
    root.removeAttribute("data-busy");
  }
}

async function kill(root: HTMLElement, key: string, reason: string, cb?: (payload: { reason: string }) => void | Promise<void>) {
  try {
    root.setAttribute("data-busy", "1");
    const obj = loadState(key, {});
    obj.state = "killed";
    obj.killedAt = new Date().toISOString();
    saveState(key, obj);
    addLog(key, { ts: obj.killedAt, actor: who(), action: "KILL", reason, scope: obj.scope });
    applyState(root, "killed", obj.killedAt);
    renderLog(root, key);
    if (cb) await cb({ reason });
    toast(root, "KILLED");
  } finally {
    root.removeAttribute("data-busy");
  }
}

/* ========================== UI Helpers ========================== */

function labelFor(s: KSState) {
  if (s === "killed") return "KILLED";
  if (s === "armed") return "ARMED";
  return "SAFE";
}

function applyState(root: HTMLElement, s: KSState, killedAt?: string, cooldownSec?: number) {
  root.dataset.state = s;
  if (killedAt) root.dataset.killedAt = killedAt;
  const arm = root.querySelector<HTMLButtonElement>("#ks-arm");
  const dis = root.querySelector<HTMLButtonElement>("#ks-disarm");
  const kil = root.querySelector<HTMLButtonElement>("#ks-kill");

  if (s === "safe") {
    arm && (arm.disabled = false);
    kil && (kil.disabled = true);
    dis && (dis.disabled = true);
  } else if (s === "armed") {
    arm && (arm.disabled = true);
    kil && (kil.disabled = false);
    dis && (dis.disabled = false);
  } else {
    // killed
    arm && (arm.disabled = true);
    kil && (kil.disabled = true);
    dis && (dis.disabled = false);
  }

  // update label & dot via CSS only
  tickCountdown(root, cooldownSec || 60);
}

function openModal(root: HTMLElement) {
  if ((root.dataset.state || "safe") !== "armed") return toast(root, "Arm first");
  const m = root.querySelector<HTMLElement>("#ks-modal");
  if (!m) return;
  m.setAttribute("data-open", "1");
  m.setAttribute("aria-hidden", "false");
  root.querySelector<HTMLTextAreaElement>("#ks-reason")?.focus();
}
function closeModal(root: HTMLElement) {
  const m = root.querySelector<HTMLElement>("#ks-modal");
  if (!m) return;
  m.removeAttribute("data-open");
  m.setAttribute("aria-hidden", "true");
}

function who() {
  try {
    const n = (navigator as any)?.userAgentData?.platform || navigator.platform || "web";
    const u = (navigator as any)?.user?.name || (navigator as any)?.user?.email || "";
    return (u || "").toString() || n.toString();
  } catch {
    return "web";
  }
}

function toast(root: HTMLElement, msg: string) {
  const el = root.querySelector("#ks-toast") as HTMLElement | null;
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

/* -------- audit log -------- */

function renderLog(root: HTMLElement, key: string) {
  const body = root.querySelector("#ks-log-body") as HTMLElement | null;
  if (!body) return;
  const obj = loadState(key, {});
  const log: LogRow[] = Array.isArray(obj.log) ? obj.log : [];
  body.innerHTML = "";
  for (const r of log) {
    const tr = document.createElement("tr");
    tr.setAttribute("data-row", "1");
    tr.setAttribute("data-hay", `${r.action} ${r.actor || ""} ${r.reason || ""}`.toLowerCase());
    tr.innerHTML = `
      <td style="padding:10px;border-bottom:1px solid var(--rb);white-space:nowrap;">${fmtDT(r.ts)}</td>
      <td style="padding:10px;border-bottom:1px solid var(--rb);"><b>${r.action}</b></td>
      <td style="padding:10px;border-bottom:1px solid var(--rb);">${r.actor || "—"}</td>
      <td style="padding:10px;border-bottom:1px solid var(--rb);">${r.scope || "—"}</td>
      <td style="padding:10px;border-bottom:1px solid var(--rb);max-width:520px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${esc(r.reason || "")}">${esc(r.reason || "—")}</td>
    `;
    body.appendChild(tr);
  }
  filterLog(root, key); // respect current filter
}

function filterLog(root: HTMLElement, key: string) {
  const q = (root.querySelector<HTMLInputElement>('input[name="ks-search"]')?.value || "").trim().toLowerCase();
  const rows = Array.from(root.querySelectorAll<HTMLTableRowElement>('#ks-log-body tr[data-row]'));
  rows.forEach((tr) => {
    const hay = (tr.dataset.hay || "");
    tr.style.display = !q || hay.includes(q) ? "" : "none";
  });
}

function exportLog(key: string) {
  const obj = loadState(key, {});
  const log: LogRow[] = Array.isArray(obj.log) ? obj.log : [];
  const head = ["Time", "Action", "Actor", "Scope", "Reason"];
  const data = log.map((r) => [r.ts, r.action, r.actor || "", r.scope || "", r.reason || ""]);
  const csv = [head, ...data].map((row) => row.map(csvEsc).join(",")).join("\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `killswitch_audit_${stamp(new Date())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* -------- countdown -------- */

function timeLeft(killedAtMs: number, cooldownSec: number) {
  if (!Number.isFinite(killedAtMs)) return 0;
  const end = killedAtMs + cooldownSec * 1000;
  const rem = Math.ceil((end - Date.now()) / 1000);
  return Math.max(0, rem);
}

function startTicker(root: HTMLElement, cooldownSec: number) {
  tickCountdown(root, cooldownSec);
  (root as any)._ksTimer && clearInterval((root as any)._ksTimer);
  (root as any)._ksTimer = setInterval(() => tickCountdown(root, cooldownSec), 1000);
}

function tickCountdown(root: HTMLElement, cooldownSec: number) {
  if ((root.dataset.state || "safe") !== "killed") {
    const el = root.querySelector("#ks-cd") as HTMLElement | null;
    if (el) el.textContent = "";
    return;
  }
  const killedAt = Date.parse(root.dataset.killedAt || "");
  const rem = timeLeft(killedAt, cooldownSec);
  const el = root.querySelector("#ks-cd") as HTMLElement | null;
  if (el) el.textContent = rem > 0 ? `· Cooldown ${rem}s` : "· Cooldown over";
}

/* ========================== Misc utils ========================== */

function fmtDT(iso: string) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}
function stamp(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${String(s).replace(/"/g, '""')}"` : String(s);
}
function esc(s: string) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* =============================== Styles =============================== */

const wrap: any = { display: "grid", gap: 12, padding: 12 };
const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const h4: any = { margin: 0, fontSize: 16 };
const sub: any = { margin: "2px 0 0", color: "#6b7280", fontSize: 13 };
const card: any = { border: "1px solid var(--b)", background: "var(--bg)", borderRadius: 14, padding: 12, display: "grid", gap: 8 };
const btnRow: any = { display: "flex", gap: 8, flexWrap: "wrap" };
const help: any = { margin: 0, color: "#6b7280", fontSize: 12 };

const btn: any = { borderRadius: 12, padding: "8px 12px", cursor: "pointer", fontSize: 13, border: "1px solid", background: "#fff" };
const btnWarn: any = { borderColor: "#fde68a", background: "#fffbeb", color: "#92400e" };
const btnDanger: any = { borderColor: "#fecaca", background: "#fef2f2", color: "#b42318", fontWeight: 700 };
const btnGhost: any = { borderColor: "#e5e7eb", background: "#fff", color: "#111" };

const statusWrap: any = { display: "flex", alignItems: "center", gap: 8, fontSize: 14 };
const dot: any = { width: 10, height: 10, borderRadius: 999, display: "inline-block", border: "1px solid rgba(0,0,0,.08)" };
const dotGreen: any = { background: "#dcfce7", borderColor: "#bbf7d0" };
const dotAmber: any = { background: "#fef3c7", borderColor: "#fde68a" };
const dotRed: any = { background: "#fee2e2", borderColor: "#fecaca" };
const cdLbl: any = { color: "#6b7280" };

const logWrap: any = { display: "grid", gap: 8 };
const logHead: any = { display: "flex", alignItems: "center", justifyContent: "space-between" };
const logTable: any = { width: "100%", borderCollapse: "separate", borderSpacing: 0, background: "var(--bg)" };
const th: any = { textAlign: "left", padding: "8px 10px", borderBottom: "1px solid var(--rb)", color: "#6b7280", fontSize: 12 };

const searchWrap: any = { position: "relative" };
const searchIcon: any = { position: "absolute", left: 8, top: 6, fontSize: 12, color: "#777" };
const searchInput: any = { width: 260, height: 30, padding: "4px 8px 4px 24px", borderRadius: 10, border: "1px solid var(--b)", outline: "none", background: "#fff" };

const modalWrap: any = { position: "fixed", inset: 0, display: "none", alignItems: "center", justifyContent: "center", zIndex: 60 };
const modalMask: any = { position: "absolute", inset: 0, background: "rgba(0,0,0,.4)" };
const modalCard: any = { position: "relative", width: "min(640px, 96vw)", background: "#fff", borderRadius: 16, border: "1px solid var(--b)", boxShadow: "0 20px 60px rgba(0,0,0,.2)", padding: 14, display: "grid", gap: 10 };
const modalHead: any = { display: "flex", alignItems: "center", gap: 10 };
const boom: any = { fontSize: 22 };
const lbl: any = { display: "block", fontSize: 12, color: "#6b7280", marginBottom: 4 };
const ta: any = { width: "100%", border: "1px solid var(--b)", borderRadius: 10, padding: 8, minHeight: 72, resize: "vertical" as const };
const pinInput: any = { width: 140, border: "1px solid var(--b)", borderRadius: 10, padding: "6px 8px" };
const modalBtns: any = { display: "flex", justifyContent: "flex-end", gap: 8 };

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
  zIndex: 80,
};

const css = `
  :root { --b:#e5e7eb; --rb:#f0f0f1; --bg:#fff; }
  #ks-root[data-busy="1"] { opacity:.6; pointer-events:none; }

  #ks-modal[data-open="1"] { display:flex; }

  @media (prefers-color-scheme: dark) {
    :root { --b:rgba(255,255,255,.12); --rb:rgba(255,255,255,.06); --bg:#0b0b0c; }
    #ks-root, table, th, td { color:#e5e7eb !important; }
    #ks-modal > div[role="dialog"] { background:#0b0b0c !important; }
    textarea, input[type="password"], input[name="ks-search"] { background:#0b0b0c; color:#e5e7eb; border-color:var(--b); }
    button { color:#e5e7eb; }
  }

  /* state-driven hint colors on header dot (already inline for base) */
`;

