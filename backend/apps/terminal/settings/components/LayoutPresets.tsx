"use client";

/**
 * layoutpresets.tsx
 * Zero-import, self-contained Layout Presets manager (Tailwind only).
 *
 * Features:
 * - Save, load, rename, delete dashboard layout presets
 * - LocalStorage persistence with namespace
 * - Export / Import JSON
 * - Grid list of presets with preview thumbnails (emoji placeholder)
 * - Default preset always available
 *
 * Usage:
 * <LayoutPresets
 *   namespace="myapp"
 *   onLoad={(preset) => console.log("Load layout:", preset)}
 * />
 */

export type LayoutPreset = {
  id: string;
  name: string;
  layout: any;        // replace with your actual layout type
  icon?: string;      // optional emoji or short label
  created: string;    // ISO date
};

export default function LayoutPresets({
  namespace = "layout",
  onLoad,
  initial = [],
  className = "",
  title = "Layout presets",
}: {
  namespace?: string;
  onLoad?: (p: LayoutPreset) => void;
  initial?: LayoutPreset[];
  className?: string;
  title?: string;
}) {
  const [presets, setPresets] = useState<LayoutPreset[]>(() => load(namespace) || seed(initial));
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    save(namespace, presets);
  }, [presets]);

  /* ------------------------------ Actions ------------------------------ */

  function addPreset() {
    const name = prompt("Preset name?")?.trim();
    if (!name) return;
    const icon = prompt("Icon emoji (optional)?")?.trim() || "📊";
    const preset: LayoutPreset = {
      id: `p${Date.now()}`,
      name,
      layout: { widgets: [] }, // replace with actual snapshot
      icon,
      created: new Date().toISOString(),
    };
    setPresets([...presets, preset]);
  }

  function renamePreset(id: string) {
    const name = prompt("New name?")?.trim();
    if (!name) return;
    setPresets(presets.map(p => (p.id === id ? { ...p, name } : p)));
  }

  function deletePreset(id: string) {
    if (!confirm("Delete this preset?")) return;
    setPresets(presets.filter(p => p.id !== id));
  }

  function loadPreset(p: LayoutPreset) {
    setSelected(p.id);
    onLoad?.(p);
  }

  function exportJSON() {
    const payload = { namespace, presets, ts: new Date().toISOString() };
    try { (navigator as any).clipboard?.writeText(JSON.stringify(payload, null, 2)); } catch {}
  }

  function importJSON(txt?: string) {
    const raw = txt ?? prompt("Paste presets JSON:") ?? "";
    if (!raw.trim()) return;
    try {
      const obj = JSON.parse(raw);
      if (!obj || !Array.isArray(obj.presets)) throw new Error("Invalid payload");
      setPresets(normalize(obj.presets));
    } catch (e: any) {
      alert("Import failed: " + (e?.message || String(e)));
    }
  }

  /* ------------------------------ Derived ------------------------------ */

  const filtered = presets.filter(p => {
    if (!query.trim()) return true;
    return (p.name + " " + (p.icon || "")).toLowerCase().includes(query.trim().toLowerCase());
  });

  /* ------------------------------ Render ------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-100 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h2 className="text-sm font-semibold">{title}</h2>
          <p className="text-xs text-neutral-400">{presets.length} presets</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search presets…"
            className="w-40 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-100 placeholder:text-neutral-600 focus:w-64 transition-all"
          />
          <button onClick={addPreset} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">+ Add</button>
          <button onClick={exportJSON} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Copy JSON</button>
          <button onClick={() => importJSON()} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Import</button>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 gap-3 p-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        {filtered.map(p => (
          <div
            key={p.id}
            className={`rounded-lg border p-4 cursor-pointer transition-colors ${
              selected === p.id
                ? "border-sky-600 bg-sky-600/20"
                : "border-neutral-800 bg-neutral-950 hover:bg-neutral-800/40"
            }`}
            onClick={() => loadPreset(p)}
          >
            <div className="flex items-center justify-between">
              <span className="text-2xl">{p.icon || "📊"}</span>
              <span className="text-[11px] text-neutral-500">{new Date(p.created).toLocaleDateString()}</span>
            </div>
            <h3 className="mt-2 truncate text-sm font-medium">{p.name}</h3>
            <div className="mt-3 flex gap-2 text-xs">
              <button
                onClick={(e) => { e.stopPropagation(); renamePreset(p.id); }}
                className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800"
              >
                Rename
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); deletePreset(p.id); }}
                className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-rose-300 hover:bg-rose-900/30"
              >
                Delete
              </button>
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="col-span-full rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-center text-sm text-neutral-400">
            No presets match your search.
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------- Helpers ------------------------------- */

function seed(initial: LayoutPreset[]): LayoutPreset[] {
  if (initial.length) return normalize(initial);
  return [
    { id: "default", name: "Default layout", layout: {}, icon: "📊", created: new Date().toISOString() },
  ];
}

function normalize(list: LayoutPreset[]): LayoutPreset[] {
  const seen: Record<string, boolean> = {};
  const out: LayoutPreset[] = [];
  for (const it of list) {
    const id = String(it.id || "").trim() || `p${Date.now()}`;
    if (seen[id]) continue;
    seen[id] = true;
    out.push({
      id,
      name: it.name?.trim() || id,
      layout: it.layout || {},
      icon: it.icon || "📊",
      created: it.created || new Date().toISOString(),
    });
  }
  return out;
}

function storageKey(ns: string) { return `lp:${ns}`; }

function load(ns: string): LayoutPreset[] | null {
  try {
    const raw = localStorage.getItem(storageKey(ns));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return normalize(parsed);
  } catch { return null; }
}

function save(ns: string, items: LayoutPreset[]) {
  try { localStorage.setItem(storageKey(ns), JSON.stringify(items)); } catch {}
}

/* ----------------------- Ambient React (no imports) ---------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;