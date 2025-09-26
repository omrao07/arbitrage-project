"use client";

/**
 * settings/page.tsx
 * Zero-import, fully self-contained Settings page with tabbed sections.
 *
 * What’s inside (no external imports):
 * - Tabs: Profile • Connections • Feature Flags • Hotkeys • Layouts
 * - Minimal but functional UIs for each section (localStorage persistence)
 * - Dark Tailwind styling, keyboard-friendly
 *
 * NOTE: This file does not import the standalone components you may have
 * in /profilesettings.tsx, /components/connections.tsx, etc. It’s a clean,
 * drop-in page that works by itself.
 */

export default function SettingsPage() {
  const [tab, setTab] = useState<TabKey>("profile");
  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      <header className="border-b border-neutral-800 px-6 py-4">
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="mt-1 text-sm text-neutral-400">
          Manage your profile, connections, flags, hotkeys, and layouts.
        </p>
      </header>

      <main className="mx-auto max-w-6xl p-6">
        <Tabs tab={tab} onChange={setTab} />
        <div className="mt-4">
          {tab === "profile" && <ProfileSection />}
          {tab === "connections" && <ConnectionsSection />}
          {tab === "flags" && <FlagsSection />}
          {tab === "hotkeys" && <HotkeysSection />}
          {tab === "layouts" && <LayoutsSection />}
        </div>
      </main>
    </div>
  );
}

/* --------------------------------- Tabs ---------------------------------- */

type TabKey = "profile" | "connections" | "flags" | "hotkeys" | "layouts";

function Tabs({ tab, onChange }: { tab: TabKey; onChange: (t: TabKey) => void }) {
  const items: { k: TabKey; label: string }[] = [
    { k: "profile", label: "Profile" },
    { k: "connections", label: "Connections" },
    { k: "flags", label: "Feature flags" },
    { k: "hotkeys", label: "Hotkeys" },
    { k: "layouts", label: "Layouts" },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((it) => (
        <button
          key={it.k}
          onClick={() => onChange(it.k)}
          className={`rounded-md border px-3 py-2 text-sm ${
            tab === it.k
              ? "border-sky-600 bg-sky-600/20 text-sky-300"
              : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:bg-neutral-800"
          }`}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------ Profile ---------------------------------- */

function ProfileSection() {
  const [state, setState] = useState(() => loadJSON("settings:profile", {
    name: "",
    handle: "",
    email: "",
    tz: guessTZ(),
    locale: guessLocale(),
    theme: "system",
    avatar: "",
  }));
  const [pwd, setPwd] = useState({ cur: "", next: "", conf: "" });
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => saveJSON("settings:profile", state), [state]);

  function toast(t: string) {
    setMsg(t);
    setTimeout(() => setMsg(null), 2000);
  }

  function onAvatar(file?: File | null) {
    if (!file) return;
    if (!/^image\//.test(file.type)) return toast("Choose an image file.");
    const reader = new FileReader();
    reader.onload = () => setState((s: any) => ({ ...s, avatar: String(reader.result || "") }));
    reader.readAsDataURL(file);
  }

  function save() {
    const errs = [];
    if (errs.length) return toast(errs[0]);
    toast("Saved.");
  }

  function changePwd() {
    if (!pwd.cur || !pwd.next || !pwd.conf) return toast("Fill all password fields.");
    if (pwd.next.length < 8) return toast("New password must be 8+ chars.");
    if (pwd.next !== pwd.conf) return toast("Passwords do not match.");
    setPwd({ cur: "", next: "", conf: "" });
    toast("Password updated.");
  }

  return (
    <div className="space-y-6">
      {msg && <div className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2 text-xs">{msg}</div>}

      <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
        <h2 className="text-sm font-semibold">Profile</h2>
        <p className="mb-3 mt-1 text-xs text-neutral-400">Avatar, name, handle, email.</p>

        <div className="mt-2 flex items-center gap-3">
          <div className="h-16 w-16 overflow-hidden rounded-full border border-neutral-700 bg-neutral-950">
            {state.avatar ? (
              <img src={state.avatar} className="h-full w-full object-cover" />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-xl">👤</div>
            )}
          </div>
          <div className="text-xs">
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">
              <input type="file" className="hidden" accept="image/*" onChange={(e) => onAvatar(e.target.files?.[0])} />
              Change…
            </label>
            {state.avatar && (
              <button
                onClick={() => setState((s: any) => ({ ...s, avatar: "" }))}
                className="ml-2 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800"
              >
                Remove
              </button>
            )}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="Full name">
            <input className="input" value={state.name} onChange={(e) => setState({ ...state, name: e.target.value })} />
          </Field>
          <Field label="Handle">
            <div className="flex items-center">
              <span className="rounded-l-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-400">@</span>
              <input
                className="input rounded-l-none"
                value={state.handle}
                onChange={(e) => setState({ ...state, handle: e.target.value.replace(/[^a-z0-9_\-\.]/gi, "").slice(0, 32) })}
              />
            </div>
          </Field>
          <Field label="Email">
            <input className="input" value={state.email} onChange={(e) => setState({ ...state, email: e.target.value })} />
          </Field>
          <Field label="Time zone">
            <select className="input" value={state.tz} onChange={(e) => setState({ ...state, tz: e.target.value })}>
              {TZS.map((z) => <option key={z} value={z}>{z}</option>)}
            </select>
          </Field>
          <Field label="Locale">
            <select className="input" value={state.locale} onChange={(e) => setState({ ...state, locale: e.target.value })}>
              {LOCALES.map((z) => <option key={z} value={z}>{z}</option>)}
            </select>
          </Field>
          <Field label="Theme">
            <div className="flex gap-2">
              {(["system","light","dark"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setState({ ...state, theme: t })}
                  className={`rounded-md border px-2 py-1 text-xs ${state.theme===t ? "border-sky-600 bg-sky-600/20 text-sky-300" : "border-neutral-700 bg-neutral-900 hover:bg-neutral-800"}`}
                >
                  {t}
                </button>
              ))}
            </div>
          </Field>
        </div>

        <div className="mt-4 text-right">
          <button onClick={save} className="rounded-md border border-emerald-700 bg-emerald-600/20 px-3 py-2 text-sm text-emerald-300 hover:bg-emerald-600/30">
            Save
          </button>
        </div>
      </section>

      <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
        <h2 className="text-sm font-semibold">Security</h2>
        <p className="mb-3 mt-1 text-xs text-neutral-400">Change your password.</p>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <Field label="Current">
            <input type="password" className="input" value={pwd.cur} onChange={(e) => setPwd({ ...pwd, cur: e.target.value })} />
          </Field>
          <Field label="New">
            <input type="password" className="input" value={pwd.next} onChange={(e) => setPwd({ ...pwd, next: e.target.value })} />
          </Field>
          <Field label="Confirm">
            <input type="password" className="input" value={pwd.conf} onChange={(e) => setPwd({ ...pwd, conf: e.target.value })} />
          </Field>
        </div>
        <div className="mt-3 text-right">
          <button onClick={changePwd} className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-xs hover:bg-neutral-800">
            Update password
          </button>
        </div>
      </section>
    </div>
  );
}

/* ----------------------------- Connections ------------------------------- */

function ConnectionsSection() {
  type Conn = { id: string; name: string; type: "API"|"DB"|"File"|"Other"; status: "connected"|"disconnected" };
  const [list, setList] = useState<Conn[]>(() => loadJSON("settings:connections", [
    { id: "c1", name: "Market Data (Demo)", type: "API", status: "connected" },
    { id: "c2", name: "Internal DB", type: "DB", status: "disconnected" },
  ]));

  useEffect(() => saveJSON("settings:connections", list), [list]);

  function toggle(id: string) {
    setList((L) => L.map((c) => c.id === id ? { ...c, status: c.status === "connected" ? "disconnected" : "connected" } : c));
  }
  function add() {
    const name = prompt("Connection name?")?.trim(); if (!name) return;
    const type = (prompt('Type (API, DB, File, Other)?') || "Other").toUpperCase() as any;
    setList((L) => [...L, { id: "c"+Date.now(), name, type: ["API","DB","File","Other"].includes(type) ? type : "Other", status: "disconnected" }]);
  }
  function remove(id: string) {
    if (!confirm("Remove connection?")) return;
    setList((L) => L.filter((c) => c.id !== id));
  }

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">Connections</h2>
          <p className="text-xs text-neutral-400">Connect APIs, databases, or files.</p>
        </div>
        <button onClick={add} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs hover:bg-neutral-800">
          + Add
        </button>
      </div>

      <ul className="space-y-2">
        {list.map((c) => (
          <li key={c.id} className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{c.name}</p>
              <p className="text-xs text-neutral-400">{c.type}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${c.status === "connected" ? "bg-emerald-400" : "bg-neutral-500"}`} />
              <button onClick={() => toggle(c.id)} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800">
                {c.status === "connected" ? "Disconnect" : "Connect"}
              </button>
              <button onClick={() => remove(c.id)} className="rounded-md border border-rose-900 bg-rose-950/30 px-2 py-1 text-xs text-rose-300 hover:bg-rose-900/40">
                Remove
              </button>
            </div>
          </li>
        ))}
        {list.length === 0 && <li className="text-xs text-neutral-500">No connections.</li>}
      </ul>
    </section>
  );
}

/* ----------------------------- Feature Flags ------------------------------ */

function FlagsSection() {
  type Flag = { key: string; label: string; cat?: string; on: boolean; def: boolean };
  const [flags, setFlags] = useState<Flag[]>(() => loadJSON("settings:flags", [
    { key: "new_dashboard", label: "New dashboard", cat: "UI", on: false, def: false },
    { key: "fast_search", label: "Fast search path", cat: "Search", on: true, def: true },
  ]));
  const [q, setQ] = useState("");

  useEffect(() => saveJSON("settings:flags", flags), [flags]);

  const rows = flags
    .filter(f => !q.trim() || (f.key + " " + f.label + " " + (f.cat || "")).toLowerCase().includes(q.trim().toLowerCase()))
    .sort((a,b)=> (a.cat||"").localeCompare(b.cat||"") || a.key.localeCompare(b.key));

  function setFlag(k: string, v: boolean) {
    setFlags((L) => L.map((f) => f.key === k ? { ...f, on: v } : f));
  }
  function add() {
    const key = prompt("Flag key (a-z,0-9,_):")?.trim() || "";
    if (!/^[a-z0-9_][a-z0-9_\-]*$/i.test(key)) return alert("Invalid key.");
    if (flags.some(f => f.key === key)) return alert("Exists.");
    const label = prompt("Label?")?.trim() || key;
    const cat = prompt("Category?")?.trim() || "General";
    setFlags([...flags, { key, label, cat, on: false, def: false }]);
  }
  function reset() {
    setFlags((L)=>L.map(f=>({ ...f, on: f.def })));
  }

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">Feature flags</h2>
          <p className="text-xs text-neutral-400">{rows.length} flags</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="Search…" className="w-40 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1" />
          <button onClick={add} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">+ Add</button>
          <button onClick={reset} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Reset</button>
        </div>
      </div>

      <ul className="divide-y divide-neutral-800 rounded-md border border-neutral-800">
        {rows.map((f)=>(
          <li key={f.key} className="grid grid-cols-12 items-center gap-3 px-3 py-2">
            <div className="col-span-6 min-w-0">
              <div className="flex items-center gap-2">
                <span className={`h-1.5 w-1.5 rounded-full ${f.on?"bg-emerald-400":"bg-neutral-500"}`} />
                <span className="truncate text-sm font-medium">{f.label}</span>
              </div>
              <div className="truncate text-xs text-neutral-400"><code className="rounded bg-neutral-950 px-1">{f.key}</code> · {f.cat || "General"}</div>
            </div>
            <div className="col-span-6 flex items-center justify-end gap-2">
              <span className="text-[11px] text-neutral-500">default: <b className="text-neutral-300">{f.def?"on":"off"}</b></span>
              <button onClick={()=>setFlag(f.key,false)} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800">Off</button>
              <button onClick={()=>setFlag(f.key,true)} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800">On</button>
              <button
                onClick={()=>setFlag(f.key,!f.on)}
                className={`relative h-6 w-11 rounded-full ${f.on?"bg-emerald-600/70":"bg-neutral-700"}`}
                role="switch" aria-checked={f.on}
              >
                <span className={`absolute top-0.5 h-5 w-5 transform rounded-full bg-neutral-100 ${f.on?"translate-x-6":"translate-x-0.5"}`} />
              </button>
            </div>
          </li>
        ))}
        {rows.length===0 && <li className="p-4 text-xs text-neutral-500">No flags.</li>}
      </ul>
    </section>
  );
}

/* -------------------------------- Hotkeys -------------------------------- */

function HotkeysSection() {
  type HK = { id: string; label: string; combo: string; cat?: string };
  const [list, setList] = useState<HK[]>(() => loadJSON("settings:hotkeys", [
    { id: "run", label: "Run Screen", combo: "Ctrl+Enter", cat: "Screener" },
    { id: "copycsv", label: "Copy CSV", combo: "Ctrl+Shift+C", cat: "Screener" },
    { id: "help", label: "Open Help", combo: "F1", cat: "Global" },
  ]));
  const [cap, setCap] = useState<string | null>(null);
  const [q, setQ] = useState("");

  useEffect(() => saveJSON("settings:hotkeys", list), [list]);

  useEffect(() => {
    if (!cap) return;
    const handler = (e: KeyboardEvent) => {
      e.preventDefault(); e.stopPropagation();
      const combo = comboFromEvent(e);
      setList((L) => L.map((x)=> x.id === cap ? { ...x, combo } : x));
      setCap(null);
      document.removeEventListener("keydown", handler, true);
    };
    document.addEventListener("keydown", handler, true);
    return () => document.removeEventListener("keydown", handler, true);
  }, [cap]);

  const filtered = list
    .filter(i => !q.trim() || (i.label + " " + i.combo + " " + (i.cat||"")).toLowerCase().includes(q.trim().toLowerCase()))
    .sort((a,b)=> (a.cat||"").localeCompare(b.cat||"") || a.label.localeCompare(b.label));

  const conflicts = computeHKConflicts(filtered);

  function add() {
    const id = "hk_" + Math.random().toString(36).slice(2,8);
    setList([...list, { id, label: "New action", combo: "Unassigned", cat: "Custom" }]);
  }
  function remove(id: string) {
    setList(list.filter(x => x.id !== id));
  }

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">Hotkeys</h2>
          <p className="text-xs text-neutral-400">{filtered.length} shortcuts · {Object.keys(conflicts).length} conflicts</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="Search…" className="w-44 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1" />
          <button onClick={add} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">+ Add</button>
        </div>
      </div>

      <ul className="divide-y divide-neutral-800 rounded-md border border-neutral-800">
        {filtered.map((it)=> {
          const dups = conflicts[it.combo] || [];
          const hasConflict = it.combo !== "Unassigned" && dups.length > 1;
          return (
            <li key={it.id} className="grid grid-cols-12 items-center gap-3 px-3 py-2">
              <div className="col-span-6 min-w-0">
                <input value={it.label} onChange={(e)=>setList(L=>L.map(x=>x.id===it.id?{...x,label:e.target.value}:x))}
                       className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm" />
                <div className="mt-1 flex items-center gap-2 text-xs">
                  <span className="text-neutral-500">Category</span>
                  <input value={it.cat||""} onChange={(e)=>setList(L=>L.map(x=>x.id===it.id?{...x,cat:e.target.value}:x))}
                         className="w-36 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1" />
                </div>
              </div>
              <div className="col-span-5">
                <div className={`flex items-center gap-2 rounded-md border px-2 py-1 ${hasConflict?"border-amber-600 bg-amber-600/10":"border-neutral-700 bg-neutral-950"}`}>
                  <kbd className="rounded bg-neutral-800 px-2 py-1 text-xs">{cap===it.id ? "Press keys…" : it.combo}</kbd>
                  <button onClick={()=>setCap(it.id)} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800">Rebind</button>
                  <button onClick={()=>setList(L=>L.map(x=>x.id===it.id?{...x,combo:"Unassigned"}:x))} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800">Clear</button>
                </div>
                {hasConflict && <p className="mt-1 text-[11px] text-amber-300">Conflict with {dups.filter(id=>id!==it.id).join(", ")}.</p>}
              </div>
              <div className="col-span-1 text-right">
                <button onClick={()=>remove(it.id)} className="rounded-md border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-300 hover:bg-neutral-800">×</button>
              </div>
            </li>
          );
        })}
        {filtered.length===0 && <li className="p-4 text-xs text-neutral-500">No hotkeys.</li>}
      </ul>
    </section>
  );
}

/* ------------------------------ Layouts ---------------------------------- */

function LayoutsSection() {
  type Preset = { id: string; name: string; icon?: string; created: string; layout: any };
  const [list, setList] = useState<Preset[]>(() => loadJSON("settings:layouts", [
    { id: "default", name: "Default layout", icon: "📊", created: new Date().toISOString(), layout: {} },
  ]));
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => saveJSON("settings:layouts", list), [list]);

  const rows = list.filter(p => !q.trim() || (p.name + " " + (p.icon||"")).toLowerCase().includes(q.trim().toLowerCase()));

  function add() {
    const name = prompt("Preset name?")?.trim(); if (!name) return;
    const icon = prompt("Icon emoji (optional)?")?.trim() || "🧩";
    setList([...list, { id: "p"+Date.now(), name, icon, created: new Date().toISOString(), layout: {} }]);
  }
  function rename(id: string) {
    const name = prompt("New name?")?.trim(); if (!name) return;
    setList(list.map(p => p.id===id?{...p,name}:p));
  }
  function remove(id: string) {
    if (!confirm("Delete preset?")) return;
    setList(list.filter(p=>p.id!==id));
  }
  function loadPreset(p: Preset) {
    setSel(p.id);
    // Hook your layout loader here
    try { (navigator as any).clipboard?.writeText(JSON.stringify(p, null, 2)); } catch {}
    alert(`Loaded preset "${p.name}". (Copied JSON to clipboard for demo)`);
  }

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold">Layout presets</h2>
          <p className="text-xs text-neutral-400">{rows.length} presets</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="Search…" className="w-44 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1" />
          <button onClick={add} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">+ Add</button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3">
        {rows.map((p)=>(
          <div
            key={p.id}
            onClick={()=>loadPreset(p)}
            className={`cursor-pointer rounded-lg border p-4 transition-colors ${sel===p.id?"border-sky-600 bg-sky-600/20":"border-neutral-800 bg-neutral-950 hover:bg-neutral-800/40"}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">{p.icon || "📐"}</span>
              <span className="text-[11px] text-neutral-500">{new Date(p.created).toLocaleDateString()}</span>
            </div>
            <h3 className="mt-2 truncate text-sm font-medium">{p.name}</h3>
            <div className="mt-3 flex gap-2 text-xs">
              <button onClick={(e)=>{e.stopPropagation(); rename(p.id);}} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">Rename</button>
              <button onClick={(e)=>{e.stopPropagation(); remove(p.id);}} className="rounded-md border border-rose-900 bg-rose-950/30 px-2 py-1 text-rose-300 hover:bg-rose-900/40">Delete</button>
            </div>
          </div>
        ))}
        {rows.length===0 && (
          <div className="col-span-full rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-center text-sm text-neutral-400">
            No presets match your search.
          </div>
        )}
      </div>
    </section>
  );
}

/* --------------------------------- Bits ---------------------------------- */

function Field({ label, children }: { label: string; children: any }) {
  return (
    <label className="block">
      <div className="mb-1 text-xs text-neutral-400">{label}</div>
      {children}
    </label>
  );
}

/* -------------------------------- Utils ---------------------------------- */

function loadJSON<T>(k: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(k);
    if (!raw) return fallback as T;
    return JSON.parse(raw) as T;
  } catch { return fallback as T; }
}
function saveJSON(k: string, v: any) {
  try { localStorage.setItem(k, JSON.stringify(v)); } catch {}
}
function guessTZ() { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; } }
function guessLocale() { try { return navigator.language || "en-US"; } catch { return "en-US"; } }
const TZS = ["UTC","Asia/Kolkata","America/New_York","Europe/London","Europe/Paris","Asia/Tokyo","Asia/Singapore","Australia/Sydney"];
const LOCALES = ["en-US","en-GB","en-IN","hi-IN","fr-FR","de-DE","ja-JP","zh-CN"];

/* --------------------------- Hotkey helpers ------------------------------- */

function comboFromEvent(e: KeyboardEvent) {
  const mods = [];
  
  let key = e.key;
  if (key.length === 1) key = /^[a-z]$/i.test(key) ? key.toUpperCase() : key;
  const map: Record<string,string> = { " ": "Space", "/": "Slash", "\\":"Backslash", ".":"Period", ",":"Comma", ";":"Semicolon", "'":"Quote", "`":"Backquote", "-":"Minus", "=":"Equal" };
  if (map[key]) key = map[key];
  if (["Control","Meta","Alt","Shift"].includes(key)) key = "Unassigned";
  const F = key.toUpperCase().match(/^F([1-9]|1[0-2])$/); if (F) key = `F${F[1]}`;
  return key === "Unassigned" ? "Unassigned" : [...mods, key].join("+");
}
function computeHKConflicts(items: { id: string; combo: string }[]) {
  const map: Record<string,string[]> = {};
  for (const it of items) {
    const k = it.combo || "Unassigned";
    if (k === "Unassigned") continue;
    (map[k] ||= []).push(it.id);
  }
  for (const k of Object.keys(map)) if (map[k].length < 2) delete map[k];
  return map;
}

/* -------------------------- Tiny CSS conveniences ------------------------- */
const style = (function(){
  const el = typeof document !== "undefined" ? document.createElement("style") : null;
  if (el) {
    el.textContent = `.input{border-radius:.375rem;border-color:#404040;background-color:#0a0a0a;padding:.375rem .5rem;font-size:.875rem;color:#e5e5e5}`;
    document.head.appendChild(el);
  }
  return el;
})();

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;