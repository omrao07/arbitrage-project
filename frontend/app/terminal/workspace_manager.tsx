'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ======================= Types ======================= */
type PanelType =
  | 'Chart'
  | 'OrderBook'
  | 'TimeAndSales'
  | 'Screener'
  | 'News'
  | 'Risk'
  | 'PnL'
  | 'Custom';

type Workspace = {
  id: string;
  name: string;
  layout: LayoutKind;       // grid preset
  slots: (PanelType | null)[]; // size depends on layout preset
  createdAt: number;
  updatedAt: number;
};

type LayoutKind = '1x1' | '1x2' | '2x2' | '3x3';

type Props = {
  className?: string;
  // Optional: render an actual component for a panel type; fallbacks to placeholder tile.
  renderPanel?: (panel: PanelType) => React.ReactNode;
  // Optional: initial registry of allowed panels shown in the "Add panel" menu
  registry?: PanelType[];
  // Storage key if you want multiple independent managers on the same domain
  storageKey?: string;
};

/* ======================= Constants ======================= */
const DEFAULT_REGISTRY: PanelType[] = [
  'Chart',
  'OrderBook',
  'TimeAndSales',
  'Screener',
  'News',
  'Risk',
  'PnL',
];

const DEFAULT_WORKSPACE: Workspace = {
  id: uid(),
  name: 'Default',
  layout: '2x2',
  slots: Array.from({ length: slotsFor('2x2') }, () => null),
  createdAt: Date.now(),
  updatedAt: Date.now(),
};

const KB = (k: string) => `workspace_mgr:${k}`;

/* ======================= Utilities ======================= */
function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function slotsFor(layout: LayoutKind) {
  switch (layout) {
    case '1x1': return 1;
    case '1x2': return 2;
    case '2x2': return 4;
    case '3x3': return 9;
  }
}

function gridClasses(layout: LayoutKind) {
  switch (layout) {
    case '1x1': return 'grid-cols-1 grid-rows-1';
    case '1x2': return 'grid-cols-2 grid-rows-1';
    case '2x2': return 'grid-cols-2 grid-rows-2';
    case '3x3': return 'grid-cols-3 grid-rows-3';
  }
}

function saveAll(key: string, workspaces: Workspace[], activeId: string | null) {
  const payload = { v: 1, activeId, workspaces };
  try { localStorage.setItem(KB(key), JSON.stringify(payload)); } catch { /* ignore */ }
}

function loadAll(key: string): { workspaces: Workspace[]; activeId: string | null } {
  try {
    const raw = localStorage.getItem(KB(key));
    if (!raw) return { workspaces: [DEFAULT_WORKSPACE], activeId: DEFAULT_WORKSPACE.id };
    const parsed = JSON.parse(raw);
    const ws: Workspace[] = parsed.workspaces || [DEFAULT_WORKSPACE];
    const activeId: string | null = parsed.activeId ?? (ws[0]?.id || null);
    return { workspaces: ws, activeId };
  } catch {
    return { workspaces: [DEFAULT_WORKSPACE], activeId: DEFAULT_WORKSPACE.id };
  }
}

/* ======================= Component ======================= */
export default function WorkspaceManager({
  className = '',
  renderPanel,
  registry = DEFAULT_REGISTRY,
  storageKey = 'default',
}: Props) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([DEFAULT_WORKSPACE]);
  const [activeId, setActiveId] = useState<string | null>(DEFAULT_WORKSPACE.id);
  const [filter, setFilter] = useState('');

  // load on mount
  useEffect(() => {
    const { workspaces, activeId } = loadAll(storageKey);
    setWorkspaces(workspaces);
    setActiveId(activeId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // save on changes
  useEffect(() => {
    saveAll(storageKey, workspaces, activeId);
  }, [workspaces, activeId, storageKey]);

  const active = useMemo(
    () => workspaces.find(w => w.id === activeId) || null,
    [workspaces, activeId]
  );

  /* ------------ Workspace ops ------------ */
  function createWorkspace(name = `Workspace ${workspaces.length + 1}`, layout: LayoutKind = '2x2') {
    const ws: Workspace = {
      id: uid(),
      name,
      layout,
      slots: Array.from({ length: slotsFor(layout) }, () => null),
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setWorkspaces(prev => [...prev, ws]);
    setActiveId(ws.id);
  }

  function renameWorkspace(id: string, name: string) {
    setWorkspaces(prev => prev.map(w => w.id === id ? { ...w, name, updatedAt: Date.now() } : w));
  }

  function duplicateWorkspace(id: string) {
    const w = workspaces.find(x => x.id === id); if (!w) return;
    const copy: Workspace = {
      ...w,
      id: uid(),
      name: `${w.name} (copy)`,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setWorkspaces(prev => [...prev, copy]);
    setActiveId(copy.id);
  }

  function deleteWorkspace(id: string) {
    const next = workspaces.filter(w => w.id !== id);
    setWorkspaces(next.length ? next : [DEFAULT_WORKSPACE]);
    if (activeId === id) setActiveId(next[0]?.id ?? null);
  }

  function changeLayout(layout: LayoutKind) {
    if (!active) return;
    const n = slotsFor(layout);
    const slots = (active.slots || []).slice(0, n);
    while (slots.length < n) slots.push(null);
    updateActive({ layout, slots });
  }

  function updateActive(patch: Partial<Workspace>) {
    if (!active) return;
    setWorkspaces(prev =>
      prev.map(w => (w.id === active.id ? { ...w, ...patch, updatedAt: Date.now() } : w))
    );
  }

  /* ------------ Slot ops ------------ */
  function setSlot(i: number, panel: PanelType | null) {
    if (!active) return;
    const slots = active.slots.slice();
    slots[i] = panel;
    updateActive({ slots });
  }

  // drag & drop
  const dragIdx = useRef<number | null>(null);
  function onDragStart(i: number) { dragIdx.current = i; }
  function onDrop(i: number) {
    if (!active) return;
    const from = dragIdx.current; dragIdx.current = null;
    if (from == null || from === i) return;
    const slots = active.slots.slice();
    const tmp = slots[i]; slots[i] = slots[from]; slots[from] = tmp;
    updateActive({ slots });
  }

  /* ------------ Import/Export ------------ */
  function exportJSON() {
    const payload = JSON.stringify({ v: 1, activeId, workspaces }, null, 2);
    download('workspaces.json', payload, 'application/json');
  }
  function importJSON(file: File) {
    const fr = new FileReader();
    fr.onload = () => {
      try {
        const obj = JSON.parse(String(fr.result));
        if (!obj || !Array.isArray(obj.workspaces)) throw new Error('Invalid file');
        setWorkspaces(obj.workspaces);
        setActiveId(obj.activeId ?? obj.workspaces[0]?.id ?? null);
      } catch (e) {
        alert('Import failed: ' + (e as any)?.message);
      }
    };
    fr.readAsText(file);
  }

  const visibleWorkspaces = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return workspaces;
    return workspaces.filter(w => w.name.toLowerCase().includes(q));
  }, [filter, workspaces]);

  return (
    <div className={`flex h-full w-full gap-3 ${className}`}>
      {/* ===== Left rail (workspaces) ===== */}
      <aside className="w-64 min-w-[16rem] rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/80 dark:bg-neutral-900 p-3 flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-semibold">Workspaces</div>
          <button
            className="text-xs px-2 py-0.5 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
            onClick={() => createWorkspace()}
          >
            + New
          </button>
        </div>

        <input
          value={filter}
          onChange={(e)=>setFilter(e.target.value)}
          placeholder="Filter…"
          className="mb-2 w-full rounded border px-2 py-1 text-xs"
        />

        <div className="flex-1 overflow-auto">
          {visibleWorkspaces.map(w => (
            <div
              key={w.id}
              className={`px-2 py-1 rounded cursor-pointer text-sm flex items-center justify-between
                          ${w.id===activeId ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-200' : 'hover:bg-neutral-100 dark:hover:bg-neutral-800'}`}
              onClick={()=>setActiveId(w.id)}
            >
              <span className="truncate">{w.name}</span>
              <Menu>
                <MenuItem onClick={()=>renameWorkspacePrompt(w)}>Rename</MenuItem>
                <MenuItem onClick={()=>duplicateWorkspace(w.id)}>Duplicate</MenuItem>
                <MenuItem danger onClick={()=>deleteWorkspace(w.id)}>Delete</MenuItem>
              </Menu>
            </div>
          ))}
        </div>

        <div className="pt-2 mt-2 border-t border-neutral-200 dark:border-neutral-800">
          <button onClick={exportJSON} className="w-full text-xs mb-2 px-2 py-1 rounded border">Export JSON</button>
          <label className="w-full text-xs px-2 py-1 rounded border inline-flex items-center justify-center cursor-pointer">
            Import JSON
            <input type="file" accept="application/json" className="hidden" onChange={e=>{ const f=e.target.files?.[0]; if(f) importJSON(f); }} />
          </label>
        </div>
      </aside>

      {/* ===== Main area (active workspace) ===== */}
      <section className="flex-1 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/80 dark:bg-neutral-900 p-3 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="text-sm font-semibold">{active?.name || '—'}</div>
            <button className="text-[11px] px-2 py-0.5 rounded border" onClick={()=>active && renameWorkspacePrompt(active)}>Rename</button>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <LayoutPicker value={active?.layout || '2x2'} onChange={l=>changeLayout(l)} />
            <AddPanelMenu registry={registry} onAdd={p => addFirstEmpty(p)} />
            <button className="px-2 py-1 rounded border" onClick={()=>clearPanels()}>Clear</button>
          </div>
        </div>

        {/* Grid */}
        <div className={`grid gap-2 ${gridClasses(active?.layout || '2x2')} min-h-[400px]`}>
          {active?.slots.map((panel, i) => (
            <div
              key={i}
              className="relative rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 overflow-hidden"
              draggable={panel!=null}
              onDragStart={() => onDragStart(i)}
              onDragOver={(e)=>e.preventDefault()}
              onDrop={() => onDrop(i)}
            >
              <div className="absolute top-1 left-1 right-1 flex items-center justify-between gap-2">
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800">
                  Slot {i+1}
                </span>
                <div className="flex items-center gap-1">
                  <SlotMenu
                    panel={panel}
                    registry={registry}
                    onSet={(p)=>setSlot(i,p)}
                    onRemove={()=>setSlot(i,null)}
                  />
                </div>
              </div>

              <div className="h-full w-full p-2 pt-6">
                {panel
                  ? renderPanel?.(panel) ?? <Placeholder panel={panel}/>
                  : <EmptySlot onPick={(p)=>setSlot(i,p)} registry={registry}/>}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );

  /* ---- helpers bound to active ---- */
  function renameWorkspacePrompt(w: Workspace) {
    const name = prompt('Rename workspace', w.name);
    if (name && name.trim()) renameWorkspace(w.id, name.trim());
  }
  function addFirstEmpty(p: PanelType) {
    if (!active) return;
    const idx = active.slots.findIndex(x => x == null);
    if (idx === -1) return; // full
    setSlot(idx, p);
  }
  function clearPanels() {
    if (!active) return;
    updateActive({ slots: active.slots.map(()=>null) });
  }
}

/* ======================= Subcomponents ======================= */
function Placeholder({ panel }: { panel: PanelType }) {
  return (
    <div className="h-full w-full flex items-center justify-center rounded bg-neutral-50 dark:bg-neutral-900 border border-dashed border-neutral-300 dark:border-neutral-700 text-xs">
      {panel}
    </div>
  );
}

function EmptySlot({ registry, onPick }: { registry: PanelType[]; onPick: (p: PanelType)=>void }) {
  return (
    <div className="h-full w-full flex items-center justify-center">
      <div className="text-[11px] text-neutral-500">Empty slot</div>
      <div className="ml-2 relative">
        <Menu>
          {registry.map(p => <MenuItem key={p} onClick={()=>onPick(p)}>{p}</MenuItem>)}
        </Menu>
      </div>
    </div>
  );
}

function LayoutPicker({ value, onChange }: { value: LayoutKind; onChange: (v: LayoutKind)=>void }) {
  return (
    <div className="inline-flex items-center gap-1">
      <span className="text-[11px] text-neutral-500">Layout</span>
      <select
        className="text-xs rounded border px-2 py-1"
        value={value}
        onChange={e=>onChange(e.target.value as LayoutKind)}
      >
        <option value="1x1">1×1</option>
        <option value="1x2">1×2</option>
        <option value="2x2">2×2</option>
        <option value="3x3">3×3</option>
      </select>
    </div>
  );
}

function AddPanelMenu({ registry, onAdd }: { registry: PanelType[]; onAdd: (p: PanelType)=>void }) {
  return (
    <Menu label="Add panel">
      {registry.map(p => <MenuItem key={p} onClick={()=>onAdd(p)}>{p}</MenuItem>)}
    </Menu>
  );
}

function SlotMenu({
  panel,
  registry,
  onSet,
  onRemove,
}: {
  panel: PanelType | null;
  registry: PanelType[];
  onSet: (p: PanelType)=>void;
  onRemove: ()=>void;
}) {
  return (
    <Menu>
      {registry.map(p => <MenuItem key={p} onClick={()=>onSet(p)}>{p}</MenuItem>)}
      {panel && <MenuItem danger onClick={onRemove}>Remove</MenuItem>}
    </Menu>
  );
}

/* --- tiny headless menu (no deps) --- */
function Menu({ children, label }: { children?: React.ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(()=>{
    const onDoc = (e: MouseEvent) => { if (!ref.current?.contains(e.target as any)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return ()=>document.removeEventListener('mousedown', onDoc);
  }, []);
  return (
    <div ref={ref} className="relative inline-block">
      <button
        onClick={()=>setOpen(o=>!o)}
        className="text-[11px] px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800"
      >
        {label ?? '⋮'}
      </button>
      {open && (
        <div className="absolute right-0 mt-1 z-10 min-w-[10rem] rounded-md border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 shadow">
          <div className="py-1">{children}</div>
        </div>
      )}
    </div>
  );
}

function MenuItem({
  children,
  onClick,
  danger = false,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  danger?: boolean;
}) {
  return (
    <div
      onClick={() => { onClick?.(); }}
      className={`px-3 py-1.5 text-xs cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-800 ${danger ? 'text-rose-600' : ''}`}
    >
      {children}
    </div>
  );
}

/* ======================= Helpers ======================= */
function download(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}