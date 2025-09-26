"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

/** Keep this aligned with your CatalogResults item shape */
export type CatalogItem = {
  id: string;
  platform: "Bloomberg" | "Koyfin" | "Hammer" | string;
  category: string;
  assetClass: string;
  subClass?: string;
  type: "function" | "feature" | "ticker" | "screen" | string;
  code: string;
  name: string;
  description: string;
  aliases?: string[];
  tags?: string[];
  region?: string;
  overlaps?: string[];
  source?: string;
};

export type CompareDrawerProps = {
  open: boolean;
  items: CatalogItem[];
  onClose: () => void;
  onRemove?: (id: string) => void;
  onReorder?: (from: number, to: number) => void;
};

/** utility: shallow “are equal” for primitives/arrays */
function same(a: any, b: any): boolean {
  if (Array.isArray(a) || Array.isArray(b)) {
    const A = Array.isArray(a) ? a : [];
    const B = Array.isArray(b) ? b : [];
    if (A.length !== B.length) return false;
    const aS = [...A].map(String).sort().join("|");
    const bS = [...B].map(String).sort().join("|");
    return aS === bS;
  }
  return String(a ?? "") === String(b ?? "");
}

/** fields to show in the grid */
const FIELDS: Array<{ key: keyof CatalogItem; label: string }> = [
  { key: "platform", label: "Platform" },
  { key: "category", label: "Category" },
  { key: "assetClass", label: "Asset Class" },
  { key: "subClass", label: "Sub-Class" },
  { key: "type", label: "Type" },
  { key: "code", label: "Code" },
  { key: "name", label: "Name" },
  { key: "description", label: "Description" },
  { key: "aliases", label: "Aliases" },
  { key: "tags", label: "Tags" },
  { key: "region", label: "Region" },
  { key: "overlaps", label: "Overlaps" },
  { key: "source", label: "Source" },
];

export default function CompareDrawer({
  open,
  items,
  onClose,
  onRemove,
  onReorder,
}: CompareDrawerProps) {
  const [width, setWidth] = useState<number>(Math.min(920, Math.max(520, typeof window !== "undefined" ? Math.floor(window.innerWidth * 0.65) : 720)));
  const startX = useRef<number | null>(null);
  const startW = useRef<number>(width);
  const dragRef = useRef<HTMLDivElement | null>(null);
  const shellRef = useRef<HTMLDivElement | null>(null);

  // close on ESC
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!open) return;
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // diff map: which cells differ per field
  const diff = useMemo(() => {
    const out: Record<string, boolean> = {};
    FIELDS.forEach(({ key }) => {
      const vals = items.map((it) => (it as any)[key]);
      const allSame = vals.every((v) => same(v, vals[0]));
      out[String(key)] = !allSame;
    });
    return out;
  }, [items]);

  // drag to resize
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (startX.current === null) return;
      const dx = startX.current - e.clientX; // dragging handle (left edge)
      const next = Math.min(1200, Math.max(480, startW.current + dx));
      setWidth(next);
    }
    function onUp() {
      startX.current = null;
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  // click outside to close (only left half/backdrop)
  useEffect(() => {
    function onDown(e: MouseEvent) {
      if (!open) return;
      if (!shellRef.current) return;
      // if click target is inside the panel, ignore
      if (shellRef.current.contains(e.target as Node)) return;
      onClose();
    }
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, onClose]);

  function copyJSON() {
    const minimal = items.map(({ id, platform, category, assetClass, type, code, name, description, tags }) => ({
      id, platform, category, assetClass, type, code, name, description, tags,
    }));
    const text = JSON.stringify(minimal, null, 2);
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
  }

  function move(from: number, dir: -1 | 1) {
    if (!onReorder) return;
    const to = from + dir;
    if (to < 0 || to >= items.length) return;
    onReorder(from, to);
  }

  return (
    <>
      {/* backdrop */}
      <div
        className={`fixed inset-0 bg-black/50 transition-opacity ${open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"}`}
        style={{ zIndex: 60 }}
      />

      {/* drawer shell */}
      <aside
        className={`fixed right-0 top-0 h-screen bg-[#0b0b0b] border-l border-[#222] shadow-2xl transition-transform`}
        style={{
          width,
          transform: open ? "translateX(0)" : `translateX(${width + 24}px)`,
          zIndex: 61,
        }}
        ref={shellRef}
      >
        {/* resize handle */}
        <div
          ref={dragRef}
          className="absolute left-0 top-0 h-full w-1 cursor-ew-resize bg-transparent"
          onMouseDown={(e) => {
            startX.current = e.clientX;
            startW.current = width;
          }}
          title="Drag to resize"
        />

        {/* header */}
        <div className="p-3 border-b border-[#222] flex items-center gap-2">
          <div className="text-sm font-semibold text-gray-100">Compare ({items.length})</div>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={copyJSON}
              className="text-xs px-2 py-1 rounded border border-[#333] text-gray-300 hover:bg-[#111]"
              title="Copy as JSON"
            >
              Copy JSON
            </button>
            <button
              onClick={onClose}
              className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100"
              title="Close (Esc)"
            >
              Close
            </button>
          </div>
        </div>

        {/* selected items header row */}
        <div className="overflow-auto">
          <div className="min-w-[640px]">
            {/* top item cards */}
            <div className="grid" style={{ gridTemplateColumns: `200px repeat(${items.length}, minmax(220px, 1fr))` }}>
              <div className="p-2 text-xs text-gray-400 border-b border-[#222]">Item</div>
              {items.map((it, idx) => (
                <div key={it.id} className="border-b border-[#222] p-2">
                  <div className="flex items-start gap-2">
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-gray-100 truncate" title={it.code}>
                        {it.code}
                      </div>
                      <div className="text-xs text-gray-300 truncate" title={it.name}>
                        {it.name}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        <Chip label={it.platform} tone={toneForPlatform(it.platform)} />
                        <Chip label={it.category} />
                        <Chip label={it.assetClass} />
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <div className="flex gap-1">
                        <IconButton title="Move left" onClick={() => move(idx, -1)} disabled={idx === 0}>
                          ←
                        </IconButton>
                        <IconButton title="Move right" onClick={() => move(idx, +1)} disabled={idx === items.length - 1}>
                          →
                        </IconButton>
                      </div>
                      {onRemove ? (
                        <button
                          onClick={() => onRemove?.(it.id)}
                          className="text-[11px] px-2 py-[2px] rounded border border-[#333] text-gray-300 hover:bg-[#111]"
                          title="Remove from compare"
                        >
                          Remove
                        </button>
                      ) : null}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* field-by-field comparison grid */}
            <div className="grid" style={{ gridTemplateColumns: `200px repeat(${items.length}, minmax(220px, 1fr))` }}>
              {FIELDS.map(({ key, label }) => (
                <React.Fragment key={String(key)}>
                  {/* left label column */}
                  <div className="p-2 text-xs text-gray-400 border-b border-[#1f1f1f] bg-[#0e0e0e] sticky left-0 z-10">
                    {label}
                  </div>

                  {/* value cells */}
                  {items.map((it) => {
                    const val = (it as any)[key];
                    const isDiff = diff[String(key)];
                    return (
                      <div
                        key={it.id + "_" + String(key)}
                        className={`p-2 border-b border-[#1f1f1f] text-sm ${
                          isDiff ? "bg-[#0f1210]" : "bg-transparent"
                        }`}
                      >
                        <ValueCell value={val} />
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

/* ---------- small presentational bits ---------- */

function Chip({ label, tone }: { label: string; tone?: "purple" | "blue" | "green" | "gray" }) {
  const map: Record<string, string> = {
    purple: "bg-purple-700/30 text-purple-200 border border-purple-700/60",
    blue: "bg-blue-700/30 text-blue-200 border border-blue-700/60",
    green: "bg-emerald-700/30 text-emerald-200 border border-emerald-700/60",
    gray: "bg-gray-700/30 text-gray-200 border border-gray-700/60",
  };
  const cls = map[tone || "gray"];
  return <span className={`text-[11px] px-2 py-[2px] rounded ${cls}`}>{label}</span>;
}

function toneForPlatform(p: string): "purple" | "blue" | "green" | "gray" {
  if (p === "Bloomberg") return "purple";
  if (p === "Koyfin") return "blue";
  if (p === "Hammer") return "green";
  return "gray";
}

function IconButton({
  children,
  onClick,
  disabled,
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      className={`text-[11px] w-6 h-6 rounded border ${
        disabled ? "border-[#222] text-[#444] cursor-not-allowed" : "border-[#333] text-gray-300 hover:bg-[#111]"
      }`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      {children}
    </button>
  );
}

function ValueCell({ value }: { value: any }) {
  if (value === null || value === undefined || value === "") {
    return <span className="text-gray-500 text-xs">—</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-gray-500 text-xs">—</span>;
    return (
      <div className="flex flex-wrap gap-1">
        {value.map((v, i) => (
          <span key={i} className="text-[11px] px-2 py-[1px] rounded border border-[#333] text-gray-300">
            {String(v)}
          </span>
        ))}
      </div>
    );
  }
  const s = String(value);
  // multi-line descriptions
  if (s.length > 140) {
    return <div className="text-xs text-gray-300 leading-snug whitespace-pre-wrap">{s}</div>;
  }
  return <span className="text-sm text-gray-200">{s}</span>;
}