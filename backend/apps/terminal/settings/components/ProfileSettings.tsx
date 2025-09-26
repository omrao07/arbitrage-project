"use client";

/**
 * profilesettings.tsx
 * Zero-import, self-contained Profile Settings panel (Tailwind only).
 *
 * Features
 * - Avatar upload with instant preview (no uploads; local only)
 * - Name, handle, email (basic validation)
 * - Time zone + locale + theme (light/dark/system)
 * - Password change (client-side validation placeholders)
 * - API keys (masked, create/revoke, copy)
 * - Danger zone (delete account confirmation step)
 * - LocalStorage persistence with namespace
 *
 * Usage:
 * <ProfileSettings
 *   namespace="myapp"
 *   initial={{ name:"", handle:"", email:"", timezone:"Asia/Kolkata" }}
 *   onSave={(state)=>console.log(state)}
 * />
 */

export type ProfileState = {
  name: string;
  handle: string;
  email: string;
  avatarDataUrl?: string;
  timezone: string;
  locale: string;
  theme: "system" | "light" | "dark";
  // password change is ephemeral; not persisted
  apiKeys: { id: string; label: string; masked: string; created: string }[];
};

export default function ProfileSettings({
  namespace = "profile",
  title = "Profile settings",
  initial,
  onSave,
  className = "",
}: {
  namespace?: string;
  title?: string;
  initial?: Partial<ProfileState>;
  onSave?: (state: ProfileState) => void;
  className?: string;
}) {
  /* --------------------------------- State --------------------------------- */

  const [state, setState] = useState<ProfileState>(() => {
    const saved = load(namespace);
    return normalize({
      name: "",
      handle: "",
      email: "",
      avatarDataUrl: "",
      timezone: guessTimezone(),
      locale: guessLocale(),
      theme: "system",
      apiKeys: [],
      ...initial,
      ...saved,
    });
  });

  const [pwd, setPwd] = useState({ current: "", next: "", confirm: "" });
  const [message, setMessage] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    save(namespace, state);
  }, [state]);

  /* ------------------------------- Handlers -------------------------------- */

  function set<K extends keyof ProfileState>(k: K, v: ProfileState[K]) {
    setState((s) => ({ ...s, [k]: v }));
  }

  function onAvatarChange(file?: File | null) {
    if (!file) return;
    if (!/^image\//.test(file.type)) return alert("Please choose an image file.");
    const reader = new FileReader();
    reader.onload = () => set("avatarDataUrl", String(reader.result || ""));
    reader.readAsDataURL(file);
  }

  function genKey() {
    const id = "k_" + Math.random().toString(36).slice(2, 10);
    const secret = createSecret();
    const masked = maskSecret(secret);
    setState((s) => ({
      ...s,
      apiKeys: [...s.apiKeys, { id, label: "New key", masked, created: new Date().toISOString() }],
    }));
    try { (navigator as any).clipboard?.writeText(secret); } catch {}
    toast("New API key created and copied to clipboard.");
  }

  function revokeKey(id: string) {
    if (!confirm("Revoke this key?")) return;
    setState((s) => ({ ...s, apiKeys: s.apiKeys.filter((k) => k.id !== id) }));
  }

  function copyMasked(m: string) {
    try { (navigator as any).clipboard?.writeText(m.replaceAll("•", "")); } catch {}
    toast("Copied.");
  }

  function submitSave() {
    const errs = validate(state);
    if (errs.length) return toast(errs[0]);
    onSave?.(state);
    toast("Saved.");
  }

  function changePassword() {
    if (!pwd.current || !pwd.next || !pwd.confirm) return toast("Fill all password fields.");
    if (pwd.next.length < 8) return toast("New password must be at least 8 characters.");
    if (pwd.next !== pwd.confirm) return toast("Passwords do not match.");
    // NOTE: in real app, call server to change password
    setPwd({ current: "", next: "", confirm: "" });
    toast("Password updated.");
  }

  function deleteAccount() {
    if (!deleting) { setDeleting(true); return; }
    const phrase = prompt('Type "DELETE" to confirm account deletion:');
    if (phrase !== "DELETE") { toast("Deletion cancelled."); setDeleting(false); return; }
    // NOTE: call server in real app
    toast("Account deletion requested.");
    setDeleting(false);
  }

  function toast(t: string) {
    setMessage(t);
    setTimeout(() => setMessage(null), 2500);
  }

    function renameKey(id: string, value: string, setState: (v: ProfileState | ((p: ProfileState) => ProfileState)) => void): void {
        throw new Error("Function not implemented.");
    }

  /* -------------------------------- Render -------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-100 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h2 className="text-sm font-semibold">{title}</h2>
          <p className="text-xs text-neutral-400">
            Personal info, security, and preferences.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setState(normalize({ ...state, ...load(namespace) || {} }))}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 hover:bg-neutral-800"
          >
            Reset
          </button>
          <button
            onClick={submitSave}
            className="rounded-md border border-emerald-700 bg-emerald-600/20 px-3 py-2 font-medium text-emerald-300 hover:bg-emerald-600/30"
          >
            Save
          </button>
        </div>
      </div>

      {/* Toast */}
      {message && (
        <div className="px-4 pt-3 text-center text-xs text-neutral-300">
          {message}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 p-4 lg:grid-cols-3">
        {/* Profile */}
        <section className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
          <h3 className="text-sm font-semibold">Profile</h3>
          <p className="mb-3 mt-1 text-xs text-neutral-400">Avatar, name, handle, email.</p>

          <div className="mt-2 flex items-center gap-3">
            <div className="relative h-16 w-16 overflow-hidden rounded-full border border-neutral-700 bg-neutral-900">
              {state.avatarDataUrl ? (
                <img src={state.avatarDataUrl} alt="avatar" className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-xl">👤</div>
              )}
            </div>
            <div className="text-xs">
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => onAvatarChange(e.target.files?.[0])}
                />
                Change…
              </label>
              {state.avatarDataUrl && (
                <button
                  onClick={() => set("avatarDataUrl", "")}
                  className="ml-2 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800"
                >
                  Remove
                </button>
              )}
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3">
            <Labeled label="Full name">
              <input value={state.name} onChange={(e)=>set("name", e.target.value)} className="input" placeholder="Your name" />
            </Labeled>
            <Labeled label="Handle">
              <div className="flex items-center">
                <span className="rounded-l-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-400">@</span>
                <input
                  value={state.handle}
                  onChange={(e)=>set("handle", sanitizeHandle(e.target.value))}
                  className="input rounded-l-none"
                  placeholder="username"
                />
              </div>
            </Labeled>
            <Labeled label="Email">
              <input value={state.email} onChange={(e)=>set("email", e.target.value)} className="input" placeholder="you@example.com" />
            </Labeled>
          </div>
        </section>

        {/* Preferences */}
        <section className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
          <h3 className="text-sm font-semibold">Preferences</h3>
          <p className="mb-3 mt-1 text-xs text-neutral-400">Time zone, locale, and theme.</p>

          <div className="space-y-3">
            <Labeled label="Time zone">
              <select value={state.timezone} onChange={(e)=>set("timezone", e.target.value)} className="input">
                {TIMEZONES.map((z)=> <option key={z} value={z}>{z}</option>)}
              </select>
            </Labeled>
            <Labeled label="Locale">
              <select value={state.locale} onChange={(e)=>set("locale", e.target.value)} className="input">
                {LOCALES.map((z)=> <option key={z} value={z}>{z}</option>)}
              </select>
            </Labeled>
            <Labeled label="Theme">
              <div className="flex gap-2">
                {(["system","light","dark"] as const).map((t)=>(
                  <button
                    key={t}
                    onClick={()=>set("theme", t)}
                    className={`rounded-md border px-2 py-1 text-xs ${state.theme===t ? "border-sky-600 bg-sky-600/20 text-sky-300" : "border-neutral-700 bg-neutral-900 hover:bg-neutral-800"}`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </Labeled>
          </div>

          <div className="mt-4 rounded-md border border-neutral-800 bg-neutral-900 p-3 text-[11px] text-neutral-400">
            Tip: Your current time zone is <b className="text-neutral-300">{guessTimezone()}</b>.
          </div>
        </section>

        {/* Security */}
        <section className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
          <h3 className="text-sm font-semibold">Security</h3>
          <p className="mb-3 mt-1 text-xs text-neutral-400">Change password and manage API keys.</p>

          {/* Password */}
          <div className="space-y-3">
            <Labeled label="Current password">
              <input type="password" value={pwd.current} onChange={(e)=>setPwd({...pwd, current: e.target.value})} className="input" />
            </Labeled>
            <Labeled label="New password">
              <input type="password" value={pwd.next} onChange={(e)=>setPwd({...pwd, next: e.target.value})} className="input" />
            </Labeled>
            <Labeled label="Confirm new password">
              <input type="password" value={pwd.confirm} onChange={(e)=>setPwd({...pwd, confirm: e.target.value})} className="input" />
            </Labeled>
            <div className="text-right">
              <button onClick={changePassword} className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-xs hover:bg-neutral-800">Update password</button>
            </div>
          </div>

          {/* Keys */}
          <div className="mt-5">
            <div className="mb-2 flex items-center justify-between">
              <h4 className="text-xs font-semibold text-neutral-300">API keys</h4>
              <button onClick={genKey} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800">+ New key</button>
            </div>
            <ul className="divide-y divide-neutral-800 rounded-md border border-neutral-800">
              {state.apiKeys.length === 0 && (
                <li className="p-3 text-xs text-neutral-500">No API keys yet.</li>
              )}
              {state.apiKeys.map((k) => (
                <li key={k.id} className="grid grid-cols-12 items-center gap-2 px-3 py-2 hover:bg-neutral-800/40">
                  <div className="col-span-5 min-w-0">
                    <input
                      value={k.label}
                      onChange={(e)=>renameKey(k.id, e.target.value, setState)}
                      className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm"
                    />
                    <div className="mt-0.5 text-[11px] text-neutral-500">Created {new Date(k.created).toLocaleString()}</div>
                  </div>
                  <div className="col-span-5 min-w-0 font-mono text-sm">
                    <span className="rounded bg-neutral-800 px-2 py-1">{k.masked}</span>
                  </div>
                  <div className="col-span-2 flex items-center justify-end gap-2 text-xs">
                    <button onClick={()=>copyMasked(k.masked)} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">Copy</button>
                    <button onClick={()=>revokeKey(k.id)} className="rounded-md border border-rose-900 bg-rose-950/30 px-2 py-1 text-rose-300 hover:bg-rose-900/40">Revoke</button>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Danger zone */}
          <div className="mt-6 rounded-md border border-rose-900 bg-rose-950/40 p-3">
            <div className="mb-2 text-sm font-semibold text-rose-300">Danger zone</div>
            <p className="text-xs text-neutral-300">
              Delete your account and all associated data. This action is irreversible.
            </p>
            <div className="mt-2 text-right">
              <button
                onClick={deleteAccount}
                className={`rounded-md border px-3 py-2 text-xs ${deleting ? "border-rose-600 bg-rose-600/20 text-rose-200" : "border-rose-900 bg-rose-950/30 text-rose-300 hover:bg-rose-900/40"}`}
              >
                {deleting ? "Confirm…" : "Delete account"}
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/* --------------------------------- Bits ---------------------------------- */

function Labeled({ label, children }: { label: string; children: any }) {
  return (
    <label className="block">
      <div className="mb-1 text-xs text-neutral-400">{label}</div>
      {children}
    </label>
  );
}

/* --------------------------------- Utils --------------------------------- */

function normalize(s: any): ProfileState {
  return {
    name: String(s.name || ""),
    handle: sanitizeHandle(String(s.handle || "")),
    email: String(s.email || ""),
    avatarDataUrl: s.avatarDataUrl || "",
    timezone: s.timezone || guessTimezone(),
    locale: s.locale || guessLocale(),
    theme: (s.theme === "light" || s.theme === "dark" ? s.theme : "system") as ProfileState["theme"],
    apiKeys: Array.isArray(s.apiKeys) ? s.apiKeys.map(normalizeKey) : [],
  };
}
function normalizeKey(k: any) {
  return {
    id: String(k.id || ("k_" + Math.random().toString(36).slice(2, 10))),
    label: String(k.label || "Key"),
    masked: String(k.masked || "••••••••••"),
    created: k.created || new Date().toISOString(),
  };
}
function validate(s: ProfileState) {
  const errs: string[] = [];
  if (!s.name.trim()) errs.push("Name is required.");
  if (!/^[a-z0-9_][a-z0-9_\-\.]{1,31}$/i.test(s.handle || "")) errs.push("Handle must be 2–32 chars (letters, numbers, _.-).");
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(s.email || "")) errs.push("Invalid email.");
  return errs;
}
function sanitizeHandle(v: string) { return v.replace(/[^a-z0-9_\-\.]/gi, "").slice(0, 32); }

function maskSecret(secret: string) {
  if (!secret) return "••••••••••";
  return secret.slice(0, 4) + "•".repeat(Math.max(0, secret.length - 8)) + secret.slice(-4);
}
function createSecret() {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function storageKey(ns: string) { return `pf:${ns}`; }
function load(ns: string): Partial<ProfileState> | null {
  try {
    const raw = localStorage.getItem(storageKey(ns));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function save(ns: string, state: ProfileState) {
  try { localStorage.setItem(storageKey(ns), JSON.stringify(state)); } catch {}
}
function guessTimezone() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; }
}
function guessLocale() {
  try { return navigator.language || "en-US"; } catch { return "en-US"; }
}
const TIMEZONES = [
  "UTC","Asia/Kolkata","America/New_York","Europe/London","Europe/Paris","Asia/Tokyo","Asia/Singapore","Australia/Sydney"
];
const LOCALES = ["en-US","en-GB","en-IN","hi-IN","fr-FR","de-DE","ja-JP","zh-CN"];

/* ------------------------------- Styles ---------------------------------- */
/* small utility to avoid repeating classes */
const inputBase = "w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm text-neutral-100 placeholder:text-neutral-600";
const style = document.createElement("style");
style.textContent = `.input{${toCss({
  "border-radius":"0.375rem",
  "border-color":"#404040",
  "background-color":"#0a0a0a",
  "padding":"0.375rem 0.5rem",
  "font-size":"0.875rem",
  "color":"#e5e5e5"
})}}`;
try { document.head.appendChild(style); } catch {}
function toCss(map: Record<string,string>) { return Object.entries(map).map(([k,v])=>`${k}:${v}`).join(";"); }

/* ----------------------- Ambient React (no imports) ---------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;