"use client";

/**
 * hotkeymappings.tsx
 * Zero-import, self-contained hotkey editor with conflict detection & persistence.
 *
 * Features
 * - View, add, edit, remove hotkey mappings (action ⇄ key combo)
 * - Capture combos via keydown (Ctrl/Cmd/Alt/Shift + key)
 * - Conflict highlighting with inline resolution tips
 * - Group by category, search, and sort
 * - LocalStorage persistence (namespace)
 * - Export / Import JSON, Reset to defaults
 *
 * Usage
 * <HotkeyMappings
 *   namespace="myapp"
 *   initial={[
 *     { id: "run", label: "Run Screen", combo: "Ctrl+Enter", category: "Screener" },
 *     { id: "search", label: "Focus Search", combo: "Slash", category: "Global" },
 *   ]}
 *   onChange={(list) => console.log(list)}
 * />
 */

export type HotkeyItem = {
  id: string;            // stable id for the action (unique)
  label: string;         // user-facing label
  combo: string;         // normalized combo string, e.g. "Ctrl+Shift+K", "Cmd+S", "Slash"
  category?: string;     // optional group (e.g., "Global", "Editor")
  description?: string;  // optional help text
};

export default function HotkeyMappings({
  namespace = "hotkeys",
  initial = DEFAULT_HOTKEYS,
  title = "Hotkeys",
  onChange,
  className = "",
}: {
  namespace?: string;
  initial?: HotkeyItem[];
  title?: string;
  onChange?: (items: HotkeyItem[]) => void;
  className?: string;
}) {
  /* --------------------------------- State --------------------------------- */

  const [query, setQuery] = useState("");
  const [items, setItems] = useState<HotkeyItem[]>(() => load(namespace) || normalize(initial));
  const [capturingId, setCapturingId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"category" | "label">("category");

  useEffect(() => {
    save(namespace, items);
    onChange?.(items);
  }, [items]);

  /* ------------------------------ Derivations ------------------------------ */

  const conflicts = useMemo(() => computeConflicts(items), [items]);
  const categories = useMemo(() => Array.from(new Set(items.map(i => i.category || "General"))).sort(), [items]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = items.filter(i => {
      if (!q) return true;
      const hay = `${i.label} ${i.id} ${i.combo} ${i.category || ""} ${i.description || ""}`.toLowerCase();
      return hay.includes(q);
    });
    if (sortBy === "label") {
      list = list.sort((a, b) => a.label.localeCompare(b.label));
    } else {
      list = list.sort(
        (a, b) =>
          (a.category || "General").localeCompare(b.category || "General") ||
          a.label.localeCompare(b.label)
      );
    }
    return list;
  }, [items, query, sortBy]);

  /* -------------------------------- Actions -------------------------------- */

  function startCapture(id: string) {
    setCapturingId(id);
    // Modal-less capture: focus body and listen for next keydown
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const combo = comboFromEvent(e);
      setItems(prev => prev.map(x => (x.id === id ? { ...x, combo } : x)));
      setCapturingId(null);
      document.removeEventListener("keydown", onKey, true);
    };
    document.addEventListener("keydown", onKey, true);
  }

  function updateItem(id: string, patch: Partial<HotkeyItem>) {
    setItems(prev => prev.map(x => (x.id === id ? { ...x, ...patch } : x)));
  }

  function removeItem(id: string) {
    setItems(prev => prev.filter(x => x.id !== id));
  }

  function addItem() {
    const nid = suggestId(items, "custom");
    const next: HotkeyItem = {
      id: nid,
      label: "New action",
      combo: "Unassigned",
      category: "Custom",
      description: "",
    };
    setItems(prev => [...prev, next]);
  }

  function resetAll() {
    setItems(normalize(initial));
  }

  function exportJSON() {
    const payload = { namespace, items, ts: new Date().toISOString() };
    try { (navigator as any).clipboard?.writeText(JSON.stringify(payload, null, 2)); } catch {}
  }

  function importJSON(txt?: string) {
    const raw = txt ?? prompt("Paste hotkeys JSON:") ?? "";
    if (!raw.trim()) return;
    try {
      const obj = JSON.parse(raw);
      if (!obj || !Array.isArray(obj.items)) throw new Error("Invalid payload");
      setItems(normalize(obj.items));
    } catch (e: any) {
      alert("Import failed: " + (e?.message || String(e)));
    }
  }

  /* -------------------------------- Render -------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-100 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h2 className="text-sm font-semibold">{title}</h2>
          <p className="text-xs text-neutral-400">
            {items.length} shortcuts · {Object.keys(conflicts).length} conflict{Object.keys(conflicts).length === 1 ? "" : "s"}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <div className="relative">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search hotkeys…"
              className="w-52 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-100 placeholder:text-neutral-600 focus:w-72 transition-all"
            />
          </div>
          <label className="flex items-center gap-2">
            <span className="text-neutral-500">Sort</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1"
            >
              <option value="category">Category</option>
              <option value="label">Label</option>
            </select>
          </label>
          <button onClick={addItem} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">+ Add</button>
          <button onClick={exportJSON} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Copy JSON</button>
          <button onClick={() => importJSON()} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Import</button>
          <button onClick={resetAll} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Reset</button>
        </div>
      </div>

      {/* Body */}
      <div className="p-4">
        {categories.map((cat) => {
          const group = filtered.filter(i => (i.category || "General") === cat);
          if (!group.length) return null;
          return (
            <div key={cat} className="mb-5 rounded-lg border border-neutral-800 overflow-hidden">
              <div className="flex items-center justify-between bg-neutral-950 px-3 py-2 text-xs">
                <span className="font-semibold text-neutral-300">{cat}</span>
                <span className="text-neutral-500">{group.length} item{group.length===1?"":"s"}</span>
              </div>
              <ul className="divide-y divide-neutral-800">
                {group.map((it) => {
                  const isCapturing = capturingId === it.id;
                  const dupIds = conflicts[it.combo] || [];
                  const hasConflict = it.combo !== "Unassigned" && dupIds && dupIds.length > 1;
                  return (
                    <li key={it.id} className="grid grid-cols-12 items-center gap-3 px-3 py-2 hover:bg-neutral-800/40">
                      {/* Label / description */}
                      <div className="col-span-5 min-w-0">
                        <input
                          value={it.label}
                          onChange={(e) => updateItem(it.id, { label: e.target.value })}
                          className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm"
                        />
                        <input
                          value={it.description || ""}
                          onChange={(e) => updateItem(it.id, { description: e.target.value })}
                          placeholder="Description (optional)"
                          className="mt-1 w-full rounded-md border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-300 placeholder:text-neutral-600"
                        />
                      </div>

                      {/* Category */}
                      <div className="col-span-2">
                        <select
                          value={it.category || "General"}
                          onChange={(e) => updateItem(it.id, { category: e.target.value })}
                          className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm"
                        >
                          {unique([...categories, "General"]).map(c => (
                            <option key={c} value={c}>{c}</option>
                          ))}
                        </select>
                      </div>

                      {/* Combo */}
                      <div className="col-span-4">
                        <div className={`flex items-center gap-2 rounded-md border px-2 py-1 ${hasConflict ? "border-amber-600 bg-amber-600/10" : "border-neutral-700 bg-neutral-950"}`}>
                          <kbd className="inline-block rounded bg-neutral-800 px-2 py-1 text-xs text-neutral-200">
                            {isCapturing ? "Press keys…" : prettyCombo(it.combo)}
                          </kbd>
                          <button
                            onClick={() => startCapture(it.id)}
                            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 hover:bg-neutral-800"
                            title="Rebind"
                          >
                            {isCapturing ? "Listening…" : "Rebind"}
                          </button>
                          <button
                            onClick={() => updateItem(it.id, { combo: "Unassigned" })}
                            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 hover:bg-neutral-800"
                            title="Clear"
                          >
                            Clear
                          </button>
                        </div>
                        {hasConflict && (
                          <p className="mt-1 text-[11px] text-amber-300">
                            Conflict with {dupIds.filter(x => x !== it.id).map((id) => labelFor(items, id)).join(", ")}. Change one of them.
                          </p>
                        )}
                      </div>

                      {/* Delete */}
                      <div className="col-span-1 text-right">
                        <button
                          onClick={() => removeItem(it.id)}
                          className="rounded-md border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
                          title="Remove"
                        >
                          ×
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}

        {filtered.length === 0 && (
          <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-center text-sm text-neutral-400">
            No hotkeys match your search.
          </div>
        )}
      </div>

      {/* Footer tips */}
      <div className="border-t border-neutral-800 px-4 py-3 text-[11px] text-neutral-500">
        Tip: use <b>Ctrl</b>/<b>Cmd</b>, <b>Alt</b>, <b>Shift</b> with a letter/number/function key. Conflicts are highlighted.
      </div>
    </div>
  );
}

/* --------------------------------- Helpers -------------------------------- */

function normalize(list: HotkeyItem[]): HotkeyItem[] {
  // Ensure unique ids and normalized combos
  const seen: Record<string, boolean> = {};
  const out: HotkeyItem[] = [];
  for (const it of list) {
    const id = String(it.id || "").trim() || suggestId(out, "action");
    if (seen[id]) continue;
    seen[id] = true;
    out.push({
      id,
      label: it.label?.trim() || id,
      combo: normalizeCombo(it.combo || "Unassigned"),
      category: it.category || "General",
      description: it.description || "",
    });
  }
  return out;
}

function computeConflicts(items: HotkeyItem[]): Record<string, string[]> {
  const map: Record<string, string[]> = {};
  for (const it of items) {
    const k = normalizeCombo(it.combo);
    if (k === "Unassigned") continue;
    (map[k] ||= []).push(it.id);
  }
  // keep only conflicting keys
  for (const k of Object.keys(map)) if (map[k].length < 2) delete map[k];
  return map;
}

function labelFor(items: HotkeyItem[], id: string) {
  return items.find(i => i.id === id)?.label || id;
}

function unique(arr: string[]) { return Array.from(new Set(arr)); }

function suggestId(items: HotkeyItem[] | string[], base: string) {
  const ids = Array.isArray(items) ? (items as any[]).map((x) => (typeof x === "string" ? x : x.id)) : [];
  let i = 1;
  while (ids.includes(`${base}${i}`)) i++;
  return `${base}${i}`;
}

function prettyCombo(combo: string) {
  const k = normalizeCombo(combo);
  if (k === "Unassigned") return "Unassigned";
  return k
    .replace(/\bCtrl\b/g, "Ctrl")
    .replace(/\bCmd\b/g, "Cmd")
    .replace(/\bAlt\b/g, "Alt")
    .replace(/\bShift\b/g, "Shift");
}

function normalizeCombo(combo: string): string {
  if (!combo) return "Unassigned";
  const low = combo.trim();
  if (!low || low.toLowerCase() === "unassigned") return "Unassigned";

  // Split any separators and normalize order: Ctrl/Cmd, Alt, Shift, Key
  const parts = low
    .split(/[\+\-\s]+/)
    .map((p) => p.trim())
    .filter(Boolean);

  let ctrl = false, cmd = false, alt = false, shift = false, key = "";
  for (const p of parts) {
    const up = p.toLowerCase();
    if (up === "ctrl" || up === "control") ctrl = true;
    else if (up === "cmd" || up === "meta" || up === "command") cmd = true;
    else if (up === "alt" || up === "option") alt = true;
    else if (up === "shift") shift = true;
    else key = normalizeKeyName(p);
  }
  if (!key) key = "Unassigned";
  const mods = [];
  
  return key === "Unassigned" ? "Unassigned" : [...mods, key].join("+");
}

function normalizeKeyName(k: string) {
  const up = k.trim();
  const map: Record<string, string> = {
    " ": "Space",
    "Space": "Space",
    "Spacebar": "Space",
    "/": "Slash",
    "\\": "Backslash",
    ".": "Period",
    ",": "Comma",
    ";": "Semicolon",
    "'": "Quote",
    "`": "Backquote",
    "-": "Minus",
    "=": "Equal",
    "[": "BracketLeft",
    "]": "BracketRight",
  };
  if (map[up]) return map[up];
  const F = up.toUpperCase().match(/^F([1-9]|1[0-2])$/);
  if (F) return `F${F[1]}`;
  if (/^[A-Za-z]$/.test(up)) return up.toUpperCase();
  if (/^\d$/.test(up)) return up;
  // fallback: title case
  return up.charAt(0).toUpperCase() + up.slice(1);
}

function comboFromEvent(e: KeyboardEvent) {
  const mods = [];
  

  let key = e.key;

  // Normalize key name
  if (key.length === 1) key = /^[a-z]$/i.test(key) ? key.toUpperCase() : key;
  const map: Record<string, string> = {
    " ": "Space",
    "/": "Slash",
    "\\": "Backslash",
    ".": "Period",
    ",": "Comma",
    ";": "Semicolon",
    "'": "Quote",
    "`": "Backquote",
    "-": "Minus",
    "=": "Equal",
    "Escape": "Esc",
    "ArrowUp": "ArrowUp",
    "ArrowDown": "ArrowDown",
    "ArrowLeft": "ArrowLeft",
    "ArrowRight": "ArrowRight",
  };
  if (map[key]) key = map[key];
  // Function keys
  const F = key.toUpperCase().match(/^F([1-9]|1[0-2])$/);
  if (F) key = `F${F[1]}`;

  // Disallow pure modifier keys
  if (["Control", "Meta", "Alt", "Shift"].includes(key)) key = "Unassigned";

  return key === "Unassigned" ? "Unassigned" : [...mods, key].join("+");
}

/* ------------------------------- Persistence ------------------------------- */

function storageKey(ns: string) { return `hk:${ns}`; }

function load(ns: string): HotkeyItem[] | null {
  try {
    const raw = localStorage.getItem(storageKey(ns));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return normalize(parsed);
  } catch { return null; }
}

function save(ns: string, items: HotkeyItem[]) {
  try { localStorage.setItem(storageKey(ns), JSON.stringify(items)); } catch {}
}

/* -------------------------------- Defaults -------------------------------- */

const DEFAULT_HOTKEYS: HotkeyItem[] = [
  { id: "run", label: "Run Screen", combo: "Ctrl+Enter", category: "Screener" },
  { id: "copyCsv", label: "Copy CSV", combo: "Ctrl+Shift+C", category: "Screener" },
  { id: "downloadCsv", label: "Download CSV", combo: "Ctrl+Shift+D", category: "Screener" },
  { id: "focusSearch", label: "Focus Search", combo: "Slash", category: "Global" },
  { id: "toggleTheme", label: "Toggle Theme", combo: "Ctrl+Shift+L", category: "Global" },
  { id: "openHelp", label: "Open Help", combo: "F1", category: "Global" },
];

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useMemo<T>(cb: () => T, deps: any[]): T;