// panel-grid.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * A dependency-free, responsive dashboard grid with:
 *  - CSS Grid (auto-fit) layout
 *  - Drag & drop reordering (HTML5 DnD)
 *  - Resize via handles (row/col span)
 *  - Keyboard a11y (arrow keys to resize, Enter to pick up/put down)
 *  - Persistent layout via localStorage
 *
 * Tailwind is optional; classNames assume a dark UI but you can swap them out.
 */

export type Panel = {
  id: string;
  title?: string;
  /** grid spans; 1..N. Interpreted relative to column/row sizes */
  w: number;
  h: number;
  /** Optional group to filter or segment panels */
  group?: string;
  /** Arbitrary payload your app uses to render content */
  payload?: any;
};

export interface PanelGridProps {
  panels: Panel[];
  /** Render the body of a panel */
  renderPanel: (p: Panel) => React.ReactNode;
  /** Min column size; grid auto-fits as viewport grows */
  minColPx?: number; // e.g., 260
  rowHeightPx?: number; // base row height
  gapPx?: number;
  /** Limits for spans */
  minSpan?: { w: number; h: number };
  maxSpan?: { w: number; h: number };
  /** Persist layout between reloads */
  persistKey?: string;
  /** Called when layout changes (order or spans) */
  onLayout?: (panels: Panel[]) => void;
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const cx = (...xs: (string | false | null | undefined)[]) => xs.filter(Boolean).join(" ");

const PanelGrid: React.FC<PanelGridProps> = ({
  panels,
  renderPanel,
  minColPx = 260,
  rowHeightPx = 120,
  gapPx = 12,
  minSpan = { w: 1, h: 1 },
  maxSpan = { w: 6, h: 6 },
  persistKey,
  onLayout,
}) => {
  // internal layout (order + spans)
  const [items, setItems] = useState<Panel[]>(() => panels);

  // sync when prop changes (first mount or external reset)
  useEffect(() => setItems(panels), [panels]);

  // persistence
  useEffect(() => {
    if (!persistKey) return;
    try {
      const raw = localStorage.getItem(persistKey);
      if (raw) {
        const stored = JSON.parse(raw) as Panel[];
        // merge spans/order onto incoming panels by id
        const byId = new Map(stored.map((p) => [p.id, p]));
        const merged = panels.map((p) => {
          const s = byId.get(p.id);
          return s ? { ...p, w: s.w, h: s.h } : p;
        });
        // order according to stored order; append new ones at end
        const orderIds = stored.map((p) => p.id);
        merged.sort((a, b) => {
          const ia = orderIds.indexOf(a.id);
          const ib = orderIds.indexOf(b.id);
          return (ia === -1 ? 1e9 : ia) - (ib === -1 ? 1e9 : ib);
        });
        setItems(merged);
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persistKey]);

  const save = useCallback(
    (arr: Panel[]) => {
      setItems(arr);
      onLayout?.(arr);
      if (persistKey) {
        try {
          localStorage.setItem(
            persistKey,
            JSON.stringify(arr.map(({ id, w, h, title, group, payload }) => ({ id, w, h, title, group, payload })))
          );
        } catch {}
      }
    },
    [persistKey, onLayout]
  );

  // DnD
  const dragId = useRef<string | null>(null);
  const onDragStart = (id: string) => (e: React.DragEvent) => {
    dragId.current = id;
    e.dataTransfer.setData("text/plain", id);
    e.dataTransfer.effectAllowed = "move";
  };
  const onDragOver = (id: string) => (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };
  const onDrop = (targetId: string) => (e: React.DragEvent) => {
    e.preventDefault();
    const srcId = dragId.current || e.dataTransfer.getData("text/plain");
    if (!srcId || srcId === targetId) return;
    const arr = [...items];
    const i = arr.findIndex((p) => p.id === srcId);
    const j = arr.findIndex((p) => p.id === targetId);
    if (i === -1 || j === -1) return;
    const [moved] = arr.splice(i, 1);
    arr.splice(j, 0, moved);
    dragId.current = null;
    save(arr);
  };

  // Resize
  const resize = (id: string, dw: number, dh: number) => {
    const arr = items.map((p) =>
      p.id === id
        ? {
            ...p,
            w: clamp(p.w + dw, minSpan.w, maxSpan.w),
            h: clamp(p.h + dh, minSpan.h, maxSpan.h),
          }
        : p
    );
    save(arr);
  };

  // Keyboard a11y: Enter toggles "pick up" panel, arrows resize, [ / ] change width, { / } change height
  const [kbdPick, setKbdPick] = useState<string | null>(null);
  const onKey = (id: string) => (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      setKbdPick((prev) => (prev === id ? null : id));
      e.preventDefault();
    } else if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "[", "]", "{", "}"].includes(e.key)) {
      e.preventDefault();
      const delta = { w: 0, h: 0 };
      if (e.key === "ArrowRight" || e.key === "]") delta.w = 1;
      if (e.key === "ArrowLeft" || e.key === "[") delta.w = -1;
      if (e.key === "ArrowDown" || e.key === "}") delta.h = 1;
      if (e.key === "ArrowUp" || e.key === "{") delta.h = -1;
      resize(id, delta.w, delta.h);
    } else if ((e.ctrlKey || e.metaKey) && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      // reorder with Ctrl/Cmd + Arrow
      const idx = items.findIndex((p) => p.id === id);
      if (idx === -1) return;
      const dir = e.key === "ArrowRight" ? 1 : -1;
      const j = clamp(idx + dir, 0, items.length - 1);
      if (j !== idx) {
        const arr = [...items];
        const [moved] = arr.splice(idx, 1);
        arr.splice(j, 0, moved);
        save(arr);
      }
    }
  };

  // Grid styles (CSS variables for sizing)
  const gridStyle: React.CSSProperties = useMemo(
    () => ({
      display: "grid",
      gridTemplateColumns: `repeat(auto-fit, minmax(${minColPx}px, 1fr))`,
      gridAutoRows: `${rowHeightPx}px`,
      gap: `${gapPx}px`,
      alignItems: "start",
    }),
    [minColPx, rowHeightPx, gapPx]
  );

  return (
    <div className="w-full">
      <div style={gridStyle}>
        {items.map((p) => (
          <div
            key={p.id}
            draggable
            onDragStart={onDragStart(p.id)}
            onDragOver={onDragOver(p.id)}
            onDrop={onDrop(p.id)}
            onKeyDown={onKey(p.id)}
            tabIndex={0}
            aria-grabbed={kbdPick === p.id}
            className={cx(
              "rounded-2xl border border-zinc-800 bg-zinc-900 shadow-sm outline-none focus:ring-2 focus:ring-amber-400/60",
              "flex flex-col overflow-hidden"
            )}
            style={{
              gridColumn: `span ${clamp(p.w, minSpan.w, maxSpan.w)} / span ${clamp(p.w, minSpan.w, maxSpan.w)}`,
              gridRow: `span ${clamp(p.h, minSpan.h, maxSpan.h)} / span ${clamp(p.h, minSpan.h, maxSpan.h)}`,
            }}
          >
            {/* Header */}
            <div
              className={cx(
                "flex items-center justify-between px-3 py-2 border-b border-zinc-800 select-none",
                "cursor-move"
              )}
              title="Drag to reorder. Enter to pick up; arrows to resize; Ctrl/Cmd+Arrows to move."
            >
              <div className="min-w-0 flex items-center gap-2">
                <span className="inline-flex h-5 w-5 items-center justify-center rounded bg-zinc-800 text-zinc-300">⋮⋮</span>
                <span className="truncate text-sm text-zinc-100">{p.title ?? p.id}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <span className="tabular-nums">{p.w}×{p.h}</span>
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 p-3">{renderPanel(p)}</div>

            {/* Resize handles */}
            <div className="absolute pointer-events-none" />
            <button
              className="absolute bottom-1 right-1 pointer-events-auto rounded-md bg-zinc-800/80 px-2 py-1 text-[11px] text-zinc-200 hover:bg-zinc-700"
              onClick={() => resize(p.id, 1, 0)}
              title="Wider"
            >
              ⇲
            </button>
            <button
              className="absolute bottom-1 right-10 pointer-events-auto rounded-md bg-zinc-800/80 px-2 py-1 text-[11px] text-zinc-200 hover:bg-zinc-700"
              onClick={() => resize(p.id, 0, 1)}
              title="Taller"
            >
              ⇳
            </button>
            <button
              className="absolute bottom-1 right-20 pointer-events-auto rounded-md bg-zinc-800/80 px-2 py-1 text-[11px] text-zinc-200 hover:bg-zinc-700"
              onClick={() => resize(p.id, -1, 0)}
              title="Narrower"
            >
              ⇱
            </button>
            <button
              className="absolute bottom-1 right-30 pointer-events-auto rounded-md bg-zinc-800/80 px-2 py-1 text-[11px] text-zinc-200 hover:bg-zinc-700"
              onClick={() => resize(p.id, 0, -1)}
              title="Shorter"
            >
              ⇱
            </button>
          </div>
        ))}
      </div>

      {/* tiny legend */}
      <div className="mt-3 text-xs text-zinc-400">
        Drag to reorder. <kbd className="border border-zinc-700 px-1 rounded">Enter</kbd> pick up,
        <kbd className="border border-zinc-700 px-1 rounded ml-1">Arrows</kbd> resize,
        <kbd className="border border-zinc-700 px-1 rounded ml-1">Ctrl/Cmd</kbd> + Arrows move.
      </div>
    </div>
  );
};

export default PanelGrid;