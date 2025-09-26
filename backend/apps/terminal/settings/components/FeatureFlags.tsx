"use client";

/**
 * feautureflags.tsx
 * Zero-import, self-contained Feature Flags panel (Tailwind only, no links/imports).
 *
 * What it does
 * - List + toggle flags (with optional categories, descriptions, and environments)
 * - Local persistence (localStorage) with namespacing
 * - Search + quick filters (All / Enabled / Disabled / Changed)
 * - Bulk actions (Enable all, Disable all, Reset changes)
 * - Environment switcher (dev/stage/prod — custom list via props)
 * - JSON export / import of current env
 * - Keyboard: "/" focus search, "e" enable all, "d" disable all, "r" reset
 *
 * Drop-in usage
 * <FeatureFlags
 *   namespace="myapp"
 *   environments={["dev", "stage", "prod"]}
 *   initialFlags={[
 *     { key: "newDashboard", label: "New dashboard", default: false, category: "UI" },
 *     { key: "fastSearch", label: "Fast search path", default: true, category: "Search" },
 *   ]}
 *   onChange={(env, flags) => console.log(env, flags)}
 * />
 */

export type FlagDef = {
  key: string;
  label?: string;
  description?: string;
  category?: string;
  default: boolean;
};

export type FlagState = Record<string, boolean>;

export default function FeatureFlags({
  namespace = "flags",
  environments = ["dev", "stage", "prod"],
  initialEnv = "dev",
  initialFlags = [],
  onChange,
  className = "",
  title = "Feature flags",
}: {
  namespace?: string;                // storage namespace
  environments?: string[];
  initialEnv?: string;
  initialFlags: FlagDef[];
  onChange?: (env: string, flags: FlagState) => void;
  className?: string;
  title?: string;
}) {
  /* --------------------------------- State --------------------------------- */

  const [env, setEnv] = useState<string>(normalizeEnv(initialEnv, environments));
  const [defs, setDefs] = useState<FlagDef[]>(normalizeDefs(initialFlags));
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "on" | "off" | "changed">("all");
  const [flagsByEnv, setFlagsByEnv] = useState<Record<string, FlagState>>(() => {
    // load from storage or seed from defaults
    const loaded: Record<string, FlagState> = {};
    for (const e of environments) {
      loaded[e] = loadFlags(namespace, e) || seedFromDefaults(defs);
    }
    return loaded;
  });

  // Derived
  const flags = flagsByEnv[env] || {};
  const defaults = toDefaults(defs);
  const categories = Array.from(new Set(defs.map((d) => d.category || "General"))).sort();

  const query = search.trim().toLowerCase();
  const rows = defs
    .filter((d) => {
      if (query) {
        const hay = (d.key + " " + (d.label || "") + " " + (d.description || "") + " " + (d.category || "")).toLowerCase();
        if (!hay.includes(query)) return false;
      }
      if (filter === "on" && !flags[d.key]) return false;
      if (filter === "off" && flags[d.key]) return false;
      if (filter === "changed" && flags[d.key] === defaults[d.key]) return false;
      return true;
    })
    .sort((a, b) => (a.category || "").localeCompare(b.category || "") || a.key.localeCompare(b.key));

  useEffect(() => {
    onChange?.(env, flags);
  }, [env, flags]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey)) return; // ignore cmd combos
      if (e.key === "/") {
        e.preventDefault();
        focusSearch();
      } else if (e.key.toLowerCase() === "e") {
        e.preventDefault();
        bulkSet(true);
      } else if (e.key.toLowerCase() === "d") {
        e.preventDefault();
        bulkSet(false);
      } else if (e.key.toLowerCase() === "r") {
        e.preventDefault();
        resetChanges();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [env, defs, flagsByEnv]);

  /* ------------------------------- Actions -------------------------------- */

  function setFlag(k: string, v: boolean) {
    setFlagsByEnv((map) => {
      const next = { ...map, [env]: { ...(map[env] || {}), [k]: v } };
      saveFlags(namespace, env, next[env]);
      return next;
    });
  }

  function bulkSet(v: boolean) {
    setFlagsByEnv((map) => {
      const base = map[env] || {};
      const nextEnv: FlagState = {};
      for (const d of defs) nextEnv[d.key] = v;
      const next = { ...map, [env]: nextEnv };
      saveFlags(namespace, env, nextEnv);
      return next;
    });
  }

  function resetChanges() {
    setFlagsByEnv((map) => {
      const nextEnv = seedFromDefaults(defs);
      const next = { ...map, [env]: nextEnv };
      saveFlags(namespace, env, nextEnv);
      return next;
    });
  }

  function exportJSON() {
    const payload = {
      namespace,
      env,
      flags,
      timestamp: new Date().toISOString(),
    };
    try {
      (navigator as any).clipboard?.writeText(JSON.stringify(payload, null, 2));
    } catch {}
  }

  function importJSON(raw?: string) {
    let txt = raw;
    if (!txt) {
      txt = prompt("Paste JSON to import for this environment:") || "";
    }
    if (!txt.trim()) return;
    try {
      const obj = JSON.parse(txt);
      if (!obj || typeof obj !== "object" || !obj.flags) throw new Error("Invalid JSON payload");
      const imported = obj.flags as FlagState;
      // Only accept keys that exist in defs; ignore unknown
      const nextEnv: FlagState = {};
      for (const d of defs) {
        nextEnv[d.key] = typeof imported[d.key] === "boolean" ? imported[d.key] : (flags[d.key] ?? d.default);
      }
      setFlagsByEnv((map) => {
        const next = { ...map, [env]: nextEnv };
        saveFlags(namespace, env, nextEnv);
        return next;
      });
      alert("Imported feature flags for env: " + env);
    } catch (e: any) {
      alert("Import failed: " + (e?.message || String(e)));
    }
  }

  function addFlagInteractive() {
    const key = prompt("New flag key (a-z, 0-9, _):")?.trim() || "";
    if (!/^[a-z0-9_][a-z0-9_\-]*$/i.test(key)) return alert("Invalid key.");
    if (defs.some((d) => d.key === key)) return alert("Flag already exists.");
    const label = prompt("Label (optional):") || "";
    const category = prompt("Category (optional):") || "";
    const defVal = confirm("Enable by default?");
    const nextDef: FlagDef = { key, label, category, default: defVal };
    const nextDefs = [...defs, nextDef];
    setDefs(nextDefs);
    // Initialize value across envs
    setFlagsByEnv((map) => {
      const copy = { ...map };
      for (const e of environments) {
        copy[e] = { ...(copy[e] || {}), [key]: defVal };
        saveFlags(namespace, e, copy[e]);
      }
      return copy;
    });
  }

  /* -------------------------------- Render -------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-100 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h2 className="text-sm font-semibold">{title}</h2>
          <p className="text-xs text-neutral-400">
            {Object.keys(flags).length} flags · env <span className="text-neutral-200">{env}</span>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {/* Env picker */}
          <label className="flex items-center gap-2">
            <span className="text-neutral-400">Env</span>
            <select
              value={env}
              onChange={(e) => setEnv(e.target.value)}
              className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1"
              title="Environment"
            >
              {environments.map((e) => <option key={e} value={e}>{e}</option>)}
            </select>
          </label>

          {/* Filter */}
          <label className="flex items-center gap-2">
            <span className="text-neutral-400">View</span>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as any)}
              className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1"
              title="Filter"
            >
              <option value="all">All</option>
              <option value="on">Enabled</option>
              <option value="off">Disabled</option>
              <option value="changed">Changed</option>
            </select>
          </label>

          {/* Search */}
          <div className="relative">
            <input
              id="ff-search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search flags (/)"
              className="w-48 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-100 placeholder:text-neutral-600 focus:w-64 transition-all"
            />
            <kbd className="pointer-events-none absolute right-1 top-1/2 -translate-y-1/2 rounded border border-neutral-700 bg-neutral-900 px-1 text-[10px] text-neutral-400">/</kbd>
          </div>

          {/* Actions */}
          <button onClick={() => bulkSet(true)} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Enable all</button>
          <button onClick={() => bulkSet(false)} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Disable all</button>
          <button onClick={resetChanges} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Reset</button>

          {/* JSON */}
          <button onClick={exportJSON} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Copy JSON</button>
          <button onClick={() => importJSON()} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Import</button>

          {/* New flag */}
          <button onClick={addFlagInteractive} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">+ Flag</button>
        </div>
      </div>

      {/* Body */}
      <div className="p-4">
        {rows.length === 0 ? (
          <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-center text-sm text-neutral-400">
            No flags match. Try clearing search/filters.
          </div>
        ) : (
          <div className="overflow-hidden rounded-lg border border-neutral-800">
            {/* Grouped by category */}
            {categories.map((cat) => {
              const subset = rows.filter((r) => (r.category || "General") === cat);
              if (!subset.length) return null;
              return (
                <div key={cat} className="border-b border-neutral-800 last:border-b-0">
                  <div className="flex items-center justify-between bg-neutral-950 px-3 py-2 text-xs">
                    <span className="font-semibold text-neutral-300">{cat}</span>
                    <span className="text-neutral-500">{subset.length} items</span>
                  </div>
                  <ul className="divide-y divide-neutral-800">
                    {subset.map((d) => {
                      const val = !!flags[d.key];
                      const changed = val !== defaults[d.key];
                      return (
                        <li key={d.key} className="grid grid-cols-12 items-center gap-3 px-3 py-2 hover:bg-neutral-800/40">
                          <div className="col-span-7 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className={`inline-block h-1.5 w-1.5 rounded-full ${val ? "bg-emerald-400" : "bg-neutral-500"}`} />
                              <span className="truncate text-sm font-medium text-neutral-100">
                                {d.label || d.key}
                              </span>
                              {changed && (
                                <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                                  changed
                                </span>
                              )}
                            </div>
                            <div className="truncate text-xs text-neutral-400">
                              <code className="rounded bg-neutral-900 px-1 py-0.5 text-[10px] text-neutral-300">{d.key}</code>
                              {d.description ? <span className="ml-2">{d.description}</span> : null}
                            </div>
                          </div>

                          <div className="col-span-5 flex items-center justify-end gap-2">
                            {/* default hint */}
                            <span className="hidden text-[11px] text-neutral-500 sm:inline">
                              default: <b className="text-neutral-300">{defaults[d.key] ? "on" : "off"}</b>
                            </span>
                            {/* toggle */}
                            <button
                              onClick={() => setFlag(d.key, !val)}
                              className={`relative h-6 w-11 rounded-full transition-colors ${
                                val ? "bg-emerald-600/70" : "bg-neutral-700"
                              }`}
                              role="switch"
                              aria-checked={val}
                              title={val ? "Disable" : "Enable"}
                            >
                              <span
                                className={`absolute top-0.5 h-5 w-5 transform rounded-full bg-neutral-100 transition-transform ${
                                  val ? "translate-x-6" : "translate-x-0.5"
                                }`}
                              />
                            </button>
                            {/* set exact */}
                            <button
                              onClick={() => setFlag(d.key, false)}
                              className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs hover:bg-neutral-800"
                            >
                              Off
                            </button>
                            <button
                              onClick={() => setFlag(d.key, true)}
                              className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs hover:bg-neutral-800"
                            >
                              On
                            </button>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer tips */}
      <div className="border-t border-neutral-800 px-4 py-3 text-[11px] text-neutral-500">
        Shortcuts: <b>/</b> focus search · <b>E</b> enable all · <b>D</b> disable all · <b>R</b> reset.
      </div>
    </div>
  );
}

/* --------------------------------- Helpers -------------------------------- */

function normalizeDefs(defs: FlagDef[]): FlagDef[] {
  // de-dupe by key, ensure required shape
  const seen: Record<string, boolean> = {};
  const out: FlagDef[] = [];
  for (const d of defs) {
    const k = String(d.key || "").trim();
    if (!k || seen[k]) continue;
    seen[k] = true;
    out.push({
      key: k,
      label: d.label || k,
      description: d.description || "",
      category: d.category || "General",
      default: !!d.default,
    });
  }
  // sort stable for UX
  out.sort((a, b) => (a.category || "").localeCompare(b.category || "") || a.key.localeCompare(b.key));
  return out;
}

function toDefaults(defs: FlagDef[]): FlagState {
  const o: FlagState = {};
  for (const d of defs) o[d.key] = !!d.default;
  return o;
}

function seedFromDefaults(defs: FlagDef[]): FlagState {
  const o: FlagState = {};
  for (const d of defs) o[d.key] = !!d.default;
  return o;
}

function storageKey(ns: string, env: string) {
  return `ff:${ns}:${env}`;
}

function loadFlags(ns: string, env: string): FlagState | null {
  try {
    const raw = localStorage.getItem(storageKey(ns, env));
    return raw ? (JSON.parse(raw) as FlagState) : null;
  } catch { return null; }
}

function saveFlags(ns: string, env: string, flags: FlagState) {
  try { localStorage.setItem(storageKey(ns, env), JSON.stringify(flags)); } catch {}
}

function normalizeEnv(e: string, list: string[]) {
  return list.includes(e) ? e : (list[0] || "default");
}

function focusSearch() {
  try { (document.getElementById("ff-search") as any)?.focus(); } catch {}
}

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;