// app/market/alertsconfig.tsx
// No imports. Self-contained server component + inline Server Actions.
// (Added missing `muted` style and silenced TS noise on <form action={fn}>.)

export const dynamic = "force-dynamic";

/* ---------------- Types ---------------- */
type Condition = ">" | "<" | ">=" | "<=" | "==" | "crosses_up" | "crosses_down";
type Channel = "ui" | "email" | "sms" | "webhook";
type Recurrence = "once" | "every_time" | "once_per_day";

type Alert = {
  id: string;
  userId: string;
  symbol: string;
  condition: Condition;
  value: number;
  note?: string;
  channel: Channel;
  webhookUrl?: string;
  expiresAt?: string; // ISO
  recurrence: Recurrence;
  status: "active" | "paused" | "expired";
  createdAt: string; // ISO
  updatedAt: string; // ISO
  metadata?: Record<string, any>;
};

/* ---------------- In-memory store ---------------- */
declare global {
  // eslint-disable-next-line no-var
  var __ALERT_STORE: Map<string, Alert[]> | undefined;
}
if (!globalThis.__ALERT_STORE) globalThis.__ALERT_STORE = new Map();
const STORE = globalThis.__ALERT_STORE!;

/* ---------------- Utilities ---------------- */
const rid = () =>
  globalThis.crypto?.randomUUID?.() ??
  "alrt_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

const normSym = (s: string) => s.trim().toUpperCase();
const parseISO = (v?: string) => (v ? (isNaN(Date.parse(v)) ? undefined : new Date(v).toISOString()) : undefined);
const tryJson = (s?: string) => {
  if (!s) return undefined;
  try {
    const o = JSON.parse(s);
    return typeof o === "object" && o ? o : undefined;
  } catch {
    return undefined;
  }
};

/* ---------------- Top-level Server Actions (no imports) ---------------- */
export async function saveAlert(formData: FormData) {
  "use server";
  const input = {
    userId: String(formData.get("userId") || "anon"),
    symbol: normSym(String(formData.get("symbol") || "")),
    condition: String(formData.get("condition") || "") as Condition,
    value: Number(formData.get("value")),
    note: formData.get("note") ? String(formData.get("note")) : undefined,
    channel: (String(formData.get("channel") || "ui") as Channel) || "ui",
    webhookUrl: formData.get("webhookUrl") ? String(formData.get("webhookUrl")) : undefined,
    expiresAt: parseISO(formData.get("expiresAt") ? String(formData.get("expiresAt")) : undefined),
    recurrence: (String(formData.get("recurrence") || "once") as Recurrence) || "once",
    metadata: tryJson(formData.get("metadata") ? String(formData.get("metadata")) : undefined),
  };

  // minimal validation (keep it lightweight)
  if (!input.symbol) throw new Error("Symbol is required");
  if (![">", "<", ">=", "<=", "==", "crosses_up", "crosses_down"].includes(input.condition))
    throw new Error("Invalid condition");
  if (!Number.isFinite(input.value)) throw new Error("Value must be a number");

  const now = new Date().toISOString();
  const alert: Alert = {
    id: rid(),
    userId: input.userId,
    symbol: input.symbol,
    condition: input.condition,
    value: input.value,
    note: input.note?.slice(0, 280),
    channel: input.channel,
    webhookUrl: input.webhookUrl,
    expiresAt: input.expiresAt,
    recurrence: input.recurrence,
    status: "active",
    createdAt: now,
    updatedAt: now,
    metadata: input.metadata,
  };

  const list = STORE.get(alert.userId) ?? [];
  list.push(alert);
  STORE.set(alert.userId, list);
}

export async function pauseAlert(fd: FormData) {
  "use server";
  const id = String(fd.get("id") || "");
  const userId = "anon";
  const list = STORE.get(userId) ?? [];
  const i = list.findIndex((a) => a.id === id);
  if (i >= 0) list[i] = { ...list[i], status: "paused", updatedAt: new Date().toISOString() };
  STORE.set(userId, list);
}

export async function resumeAlert(fd: FormData) {
  "use server";
  const id = String(fd.get("id") || "");
  const userId = "anon";
  const list = STORE.get(userId) ?? [];
  const i = list.findIndex((a) => a.id === id);
  if (i >= 0) list[i] = { ...list[i], status: "active", updatedAt: new Date().toISOString() };
  STORE.set(userId, list);
}

export async function deleteAlert(fd: FormData) {
  "use server";
  const id = String(fd.get("id") || "");
  const userId = "anon";
  const list = STORE.get(userId) ?? [];
  STORE.set(userId, list.filter((a) => a.id !== id));
}

/* ---------------- Server component ---------------- */
export default async function AlertsConfig() {
  const alerts = STORE.get("anon") ?? [];

  return (
    <section style={wrap}>
      <header style={header}>
        <h2 style={h2}>Alerts</h2>
        <p style={sub}>Create threshold or crossover alerts and choose delivery channel.</p>
      </header>

      {/* Create alert */}
      <div style={card}>
        <h3 style={h3}>New Alert</h3>
        {/* @ts-ignore — Server Action on form is valid in Next.js */}
        <form action={saveAlert} style={formRow}>
          <input type="hidden" name="userId" value="anon" />
          <div style={grid}>
            <div style={field}>
              <label style={label}>Symbol</label>
              <input name="symbol" placeholder="RELIANCE / NIFTY / USD/INR" required style={input} />
            </div>

            <div style={field}>
              <label style={label}>Condition</label>
              <select name="condition" required style={select}>
                <option value=">">{">"}</option>
                <option value="<">{"<"}</option>
                <option value=">=">{">="}</option>
                <option value="<=">{"<="}</option>
                <option value="==">{"=="}</option>
                <option value="crosses_up">crosses_up</option>
                <option value="crosses_down">crosses_down</option>
              </select>
            </div>

            <div style={field}>
              <label style={label}>Value</label>
              <input name="value" type="number" step="any" required style={input} />
            </div>

            <div style={field}>
              <label style={label}>Channel</label>
              <select name="channel" defaultValue="ui" style={select}>
                <option value="ui">UI</option>
                <option value="email">Email</option>
                <option value="sms">SMS</option>
                <option value="webhook">Webhook</option>
              </select>
            </div>

            <div style={field}>
              <label style={label}>Webhook URL (if channel=webhook)</label>
              <input name="webhookUrl" placeholder="https://example.com/hook" style={input} />
            </div>

            <div style={field}>
              <label style={label}>Expires At (ISO)</label>
              <input name="expiresAt" type="datetime-local" style={input} />
            </div>

            <div style={{ ...field, gridColumn: "1 / -1" }}>
              <label style={label}>Note</label>
              <input name="note" maxLength={280} placeholder="Optional context…" style={input} />
            </div>

            <div style={field}>
              <label style={label}>Recurrence</label>
              <select name="recurrence" defaultValue="once" style={select}>
                <option value="once">once</option>
                <option value="every_time">every_time</option>
                <option value="once_per_day">once_per_day</option>
              </select>
            </div>

            <div style={{ ...field, alignSelf: "end" }}>
              <button type="submit" style={primaryBtn}>Save Alert</button>
            </div>
          </div>
        </form>
      </div>

      {/* Existing alerts */}
      <div style={card}>
        <h3 style={h3}>My Alerts</h3>
        {alerts.length === 0 ? (
          <p style={muted}>No alerts yet.</p>
        ) : (
          <div style={table}>
            <div style={thead}>
              <div style={{ ...th, flex: 1.1 }}>Symbol</div>
              <div style={{ ...th, flex: 1 }}>Condition</div>
              <div style={{ ...th, flex: 1 }}>Value</div>
              <div style={{ ...th, flex: 2 }}>Note</div>
              <div style={{ ...th, flex: 1 }}>Channel</div>
              <div style={{ ...th, flex: 1 }}>Status</div>
              <div style={{ ...th, flex: 1.8, textAlign: "right" }}>Actions</div>
            </div>

            {alerts.map((a) => (
              <div key={a.id} style={tr}>
                <div style={{ ...td, flex: 1.1, fontWeight: 600 }}>{a.symbol}</div>
                <div style={{ ...td, flex: 1 }}>{a.condition}</div>
                <div style={{ ...td, flex: 1 }}>{a.value.toLocaleString(undefined, { maximumFractionDigits: 4 })}</div>
                <div style={{ ...td, flex: 2, color: "#555" }}>{a.note || "—"}</div>
                <div style={{ ...td, flex: 1 }}>{a.channel}</div>
                <div style={{ ...td, flex: 1 }}>
                  <span
                    style={{
                      background: a.status === "active" ? "#ecfdf5" : a.status === "paused" ? "#f5f5f5" : "#fef2f2",
                      color: a.status === "active" ? "#067647" : a.status === "paused" ? "#444" : "#b42318",
                      borderRadius: 999,
                      fontSize: 12,
                      padding: "2px 8px",
                    }}
                  >
                    {a.status[0].toUpperCase() + a.status.slice(1)}
                  </span>
                </div>

                <div style={{ ...td, flex: 1.8 }}>
                  <div style={actions}>
                    {/* @ts-ignore — Server Action on form is valid in Next.js */}
                    <form action={a.status === "active" ? pauseAlert : resumeAlert}>
                      <input type="hidden" name="id" value={a.id} />
                      <button style={secondaryBtn}>{a.status === "active" ? "Pause" : "Resume"}</button>
                    </form>
                    {/* @ts-ignore — Server Action on form is valid in Next.js */}
                    <form action={deleteAlert}>
                      <input type="hidden" name="id" value={a.id} />
                      <button style={dangerBtn}>Delete</button>
                    </form>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

/* ---------------- Inline styles ---------------- */
const wrap = { display: "flex", flexDirection: "column", gap: 16, padding: 16 } as const;
const header = { marginBottom: 4 } as const;
const h2 = { margin: 0, fontSize: 20, lineHeight: "26px" } as const;
const h3 = { margin: "0 0 10px 0", fontSize: 16 } as const;
const sub = { margin: "4px 0 0", color: "#555", fontSize: 13 } as const;
const muted = { color: "#666", fontSize: 13 } as const;

const card = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 16,
} as const;

const formRow = { display: "block" } as const;
const grid = { display: "grid", gridTemplateColumns: "1fr", gap: 12 } as const;
const field = { display: "flex", flexDirection: "column", gap: 6 } as const;
const label = { fontSize: 12, color: "#555" } as const;
const input = { height: 34, padding: "6px 10px", borderRadius: 8, border: "1px solid #ddd", outline: "none" } as const;
const select = { ...input } as const;

const primaryBtn = {
  appearance: "none",
  border: "1px solid #111",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
  background: "#111",
  color: "#fff",
} as const;
const secondaryBtn = {
  appearance: "none",
  border: "1px solid #d4d4d8",
  borderRadius: 10,
  padding: "6px 10px",
  cursor: "pointer",
  fontSize: 13,
  background: "#f4f4f5",
  color: "#111",
} as const;
const dangerBtn = {
  appearance: "none",
  border: "1px solid #ef4444",
  borderRadius: 10,
  padding: "6px 10px",
  cursor: "pointer",
  fontSize: 13,
  background: "#fee2e2",
  color: "#b42318",
} as const;

const table = { display: "flex", flexDirection: "column", gap: 8 } as const;
const thead = { display: "flex", gap: 10, padding: "6px 0", borderBottom: "1px solid #eee", color: "#666", fontSize: 12 } as const;
const th = {} as const;
const tr = { display: "flex", gap: 10, padding: "10px 0", borderBottom: "1px dashed #eee" } as const;
const td = { fontSize: 13 } as const;
const actions = { display: "flex", gap: 8, alignItems: "center", justifyContent: "flex-end", flexWrap: "wrap" } as const;
