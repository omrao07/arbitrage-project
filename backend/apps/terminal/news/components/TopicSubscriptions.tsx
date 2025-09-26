// app/market/topicsubscription.tsx
// No imports. No hooks. Pure client component.
// Lets users pick preset topics, add custom topics, choose frequency & channel,
// and saves to localStorage (and/or calls onSave) on submit.

"use client";

type Config = {
  topics: string[];                 // preset + custom
  frequency: "realtime" | "hourly" | "daily" | "weekly";
  channel: "ui" | "email" | "sms" | "webhook";
  webhookUrl?: string;
};

type Props = {
  presets?: string[];               // default topic checklist
  storageKey?: string;              // localStorage key
  initial?: Partial<Config> & { custom?: string[] };
  onSave?: (cfg: Config) => void;   // optional callback
};

export default function TopicSubscription({
  presets = [
    "RBI", "Fed", "Inflation", "GDP", "Budget",
    "Crude", "Brent", "Gold", "USD/INR", "DXY",
    "NIFTY", "BANKNIFTY", "IT sector", "Autos",
    "Earnings", "M&A", "IPOs",
  ],
  storageKey = "topic_subs_v1",
  initial,
  onSave,
}: Props) {
  // ---- initial values (no hooks) ----
  const initFromStore = (() => {
    try {
      if (typeof window !== "undefined") {
        const s = localStorage.getItem(storageKey);
        if (s) return JSON.parse(s) as Config;
      }
    } catch {}
    return null;
  })();

  const init: Config = {
    topics: [
      ...(initial?.custom ?? []),
      ...(initial?.topics ?? [] as any),
      ...(initFromStore?.topics ?? []),
    ].filter(Boolean).map(normalize).filter(unique),
    frequency: (initial?.frequency ?? initFromStore?.frequency ?? "realtime") as Config["frequency"],
    channel: (initial?.channel ?? initFromStore?.channel ?? "ui") as Config["channel"],
    webhookUrl: initial?.webhookUrl ?? initFromStore?.webhookUrl,
  };

  // ---- DOM helpers (no hooks/state) ----
  function normalize(s: string) {
    return s.trim().replace(/\s+/g, " ");
  }
  function unique<T>(v: T, i: number, a: T[]) {
    return a.indexOf(v) === i;
  }

  function readConfig(form: HTMLFormElement): Config {
    const fd = new FormData(form);
    const chosenPresets = Array.from(form.querySelectorAll<HTMLInputElement>('input[name="preset"]:checked'))
      .map((el) => normalize(el.value));
    const customHidden = fd.get("customTopics") as string | null;
    const custom = (customHidden ? customHidden.split(",") : []).map(normalize).filter(Boolean);
    const frequency = (fd.get("frequency") as Config["frequency"]) || "realtime";
    const channel = (fd.get("channel") as Config["channel"]) || "ui";
    const webhookUrl = (fd.get("webhookUrl") as string) || undefined;

    return {
      topics: [...chosenPresets, ...custom].filter(unique),
      frequency,
      channel,
      webhookUrl: channel === "webhook" ? webhookUrl : undefined,
    };
  }

  function showWebhookIfNeeded(selectEl: HTMLSelectElement) {
    const group = document.getElementById("webhook-group");
    if (!group) return;
    group.style.display = selectEl.value === "webhook" ? "flex" : "none";
  }

  function addCustomTopic(e: any) {
    e.preventDefault();
    const btn = e.currentTarget as HTMLButtonElement;
    const form = btn.form as HTMLFormElement;
    const input = form.querySelector<HTMLInputElement>('input[name="newTopic"]')!;
    const val = normalize(input.value || "");
    if (!val) return;
    addPill(form, val);
    input.value = "";
  }

  function onNewTopicKey(e: any) {
    const el = e.currentTarget as HTMLInputElement;
    if (e.key === "Enter" || e.key === "," ) {
      e.preventDefault();
      const val = normalize(el.value || "");
      if (!val) return;
      addPill((el.form as HTMLFormElement), val);
      el.value = "";
    }
  }

  function addPill(form: HTMLFormElement, val: string) {
    const list = form.querySelector<HTMLDivElement>("#pill-list")!;
    // avoid dupes against existing pills
    const exists = Array.from(list.querySelectorAll("span[data-topic]")).some((n) => (n as HTMLElement).dataset.topic === val);
    if (exists) return;

    const span = document.createElement("span");
    span.setAttribute("data-topic", val);
    Object.assign(span.style, pill);
    span.textContent = val + " ";
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "×";
    Object.assign(x.style, pillX);
    x.onclick = () => {
      span.remove();
      syncHidden(form);
    };
    span.appendChild(x);
    list.appendChild(span);
    syncHidden(form);
  }

  function syncHidden(form: HTMLFormElement) {
    const pills = Array.from(form.querySelectorAll("span[data-topic]")).map((n) => (n as HTMLElement).dataset.topic || "");
    const hidden = form.querySelector<HTMLInputElement>('input[name="customTopics"]')!;
    hidden.value = pills.join(",");
  }

  function loadInitialPills(form: HTMLFormElement) {
    const existing = init.topics.filter((t) => !presets.includes(t));
    existing.forEach((t) => addPill(form, t));
    // check any preset topics that were in init
    init.topics.forEach((t) => {
      if (presets.includes(t)) {
        const box = form.querySelector<HTMLInputElement>(`input[type="checkbox"][value="${cssEscape(t)}"]`);
        if (box) box.checked = true;
      }
    });
    // frequency / channel defaults
    (form.elements.namedItem("frequency") as HTMLSelectElement).value = init.frequency;
    const chan = form.elements.namedItem("channel") as HTMLSelectElement;
    chan.value = init.channel;
    showWebhookIfNeeded(chan);
    (form.elements.namedItem("webhookUrl") as HTMLInputElement).value = init.webhookUrl ?? "";
    syncHidden(form);
  }

  function cssEscape(s: string) {
    // safe enough for our simple value selector
    return s.replace(/"/g, '\\"');
  }

  function handleSubmit(e: any) {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const cfg = readConfig(form);
    try {
      localStorage.setItem(storageKey, JSON.stringify(cfg));
    } catch {}
    if (onSave) onSave(cfg);
    flash("Saved!");
  }

  function flash(msg: string) {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = msg;
    el.style.opacity = "1";
    setTimeout(() => (el.style.opacity = "0"), 1200);
  }

  // on first client paint, populate pills & defaults
  if (typeof window !== "undefined") {
    // queue after hydration
    queueMicrotask(() => {
      const form = document.getElementById("topic-form") as HTMLFormElement | null;
      if (form) loadInitialPills(form);
    });
  }

  return (
    <section style={wrap}>
      <div id="toast" style={toast} />

      <header style={header}>
        <h2 style={h2}>Topic Subscriptions</h2>
        <p style={sub}>Pick topics to follow, add your own keywords, and choose how you receive updates.</p>
      </header>

      <form id="topic-form" onSubmit={handleSubmit} style={card}>
        {/* Presets */}
        <div style={block}>
          <div style={blockTitle}>Preset topics</div>
          <div style={grid}>
            {presets.map((t) => (
              <label key={t} style={chk}>
                <input name="preset" type="checkbox" value={t} />
                <span>{t}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Custom topics */}
        <div style={block}>
          <div style={blockTitle}>Custom topics / symbols</div>
          <div style={pillRow}>
            <input
              name="newTopic"
              placeholder="Type a topic/symbol and press Enter"
              onKeyDown={onNewTopicKey}
              style={input}
            />
            <button type="button" onClick={addCustomTopic} style={secondaryBtn}>Add</button>
          </div>
          <div id="pill-list" style={pillList} />
          <input type="hidden" name="customTopics" defaultValue="" />
        </div>

        {/* Delivery */}
        <div style={block}>
          <div style={blockTitle}>Delivery</div>
          <div style={row}>
            <label style={field}>
              <span style={label}>Frequency</span>
              <select name="frequency" defaultValue={init.frequency} style={select}>
                <option value="realtime">Realtime</option>
                <option value="hourly">Hourly digest</option>
                <option value="daily">Daily digest</option>
                <option value="weekly">Weekly digest</option>
              </select>
            </label>

            <label style={field}>
              <span style={label}>Channel</span>
              <select
                name="channel"
                defaultValue={init.channel}
                onChange={(e) => showWebhookIfNeeded(e.currentTarget)}
                style={select}
              >
                <option value="ui">UI</option>
                <option value="email">Email</option>
                <option value="sms">SMS</option>
                <option value="webhook">Webhook</option>
              </select>
            </label>

            <label id="webhook-group" style={{ ...field, display: "none" }}>
              <span style={label}>Webhook URL</span>
              <input name="webhookUrl" placeholder="https://example.com/hook" style={input} />
            </label>
          </div>
        </div>

        <div style={actions}>
          <button type="submit" style={primaryBtn}>Save Subscriptions</button>
          <button
            type="button"
            style={btnGhost}
            onClick={(e) => {
              const form = (e.currentTarget as HTMLButtonElement).form!;
              form.reset();
              (form.querySelector("#pill-list") as HTMLDivElement).innerHTML = "";
              (form.elements.namedItem("customTopics") as HTMLInputElement).value = "";
              flash("Cleared");
            }}
          >
            Clear
          </button>
        </div>
      </form>
    </section>
  );
}

/* ---------------- styles (no imports) ---------------- */
const wrap: any = { display: "flex", flexDirection: "column", gap: 12, padding: 16 };
const header: any = { marginBottom: 2 };
const h2: any = { margin: 0, fontSize: 20, lineHeight: "26px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const card: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 16,
  display: "flex",
  flexDirection: "column",
  gap: 14,
};

const block: any = { display: "flex", flexDirection: "column", gap: 8 };
const blockTitle: any = { fontWeight: 600, fontSize: 14 };

const grid: any = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
  gap: 8,
};

const chk: any = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  padding: "8px 10px",
  border: "1px solid #eee",
  borderRadius: 10,
  background: "#fafafa",
};

const pillRow: any = { display: "flex", gap: 8, flexWrap: "wrap" };
const pillList: any = { display: "flex", gap: 6, flexWrap: "wrap" };

const row: any = { display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" };
const field: any = { display: "flex", flexDirection: "column", gap: 6, minWidth: 220, flex: "1 1 220px" };
const label: any = { fontSize: 12, color: "#555" };
const input: any = { height: 34, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", outline: "none" };
const select: any = { ...input };

const pill: any = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  background: "#f5f5f7",
  border: "1px solid #e5e7eb",
  borderRadius: 999,
  padding: "2px 8px",
  fontSize: 12,
};
const pillX: any = {
  border: "none",
  background: "transparent",
  cursor: "pointer",
  fontWeight: 700,
  lineHeight: 1,
};

const actions: any = { display: "flex", gap: 8, alignItems: "center" };
const primaryBtn: any = {
  appearance: "none",
  border: "1px solid #111",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
  background: "#111",
  color: "#fff",
};
const secondaryBtn: any = {
  appearance: "none",
  border: "1px solid #d4d4d8",
  borderRadius: 10,
  padding: "6px 10px",
  cursor: "pointer",
  fontSize: 13,
  background: "#f4f4f5",
  color: "#111",
};
const btnGhost: any = {
  appearance: "none",
  border: "1px solid #e5e7eb",
  borderRadius: 10,
  padding: "6px 10px",
  cursor: "pointer",
  fontSize: 13,
  background: "#fff",
  color: "#111",
};

const toast: any = {
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
  zIndex: 50,
};
