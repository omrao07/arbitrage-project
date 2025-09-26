"use client";

import React, { useMemo, useState } from "react";

/** Keep in sync with your CatalogResults item shape */
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
  overlaps?: string[];   // ids it overlaps with (empty/undefined => unique)
  source?: string;       // URL or ref
};

export default function ItemDetails({
  item,
  onAddToCompare,
  onClose,
}: {
  item: CatalogItem | null;
  onAddToCompare?: (id: string) => void;
  onClose?: () => void;
}) {
  const [showJSON, setShowJSON] = useState(false);

  const isUnique = useMemo(
    () => !item?.overlaps || item.overlaps.length === 0,
    [item]
  );

  if (!item) {
    return (
      <div className="p-6 text-sm text-gray-400 bg-[#0b0b0b] rounded-lg border border-[#222]">
        Select an item to see details.
      </div>
    );
  }

  function copyText(text: string) {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
  }

  const miniJSON = useMemo(() => {
    const { id, platform, category, assetClass, subClass, type, code, name, description, tags } = item;
    return { id, platform, category, assetClass, subClass, type, code, name, description, tags };
  }, [item]);

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-[#222] flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="text-sm text-gray-400 truncate">{item.platform} • {item.category} • {item.assetClass}{item.subClass ? ` • ${item.subClass}` : ""}</div>
          <div className="mt-1 text-lg font-semibold text-gray-100 truncate" title={item.name}>
            {item.code} — {item.name}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <Chip label={item.platform} tone={toneForPlatform(item.platform)} />
            <Chip label={item.category} />
            <Chip label={item.assetClass} />
            {item.subClass ? <Chip label={item.subClass} muted /> : null}
            <Chip label={item.type} muted />
            <span
              className={`text-[11px] px-2 py-[2px] rounded border ${
                isUnique
                  ? "bg-emerald-700/30 text-emerald-200 border-emerald-700/60"
                  : "bg-gray-700/30 text-gray-200 border-gray-700/60"
              }`}
              title={isUnique ? "Unique feature" : "Overlaps with others"}
            >
              {isUnique ? "Unique" : "Overlap"}
            </span>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex gap-2">
            <Button
              title="Add to Compare"
              onClick={() => onAddToCompare?.(item.id)}
            >
              + Compare
            </Button>
            <Button title="Copy Code" onClick={() => copyText(item.code)}>
              Copy Code
            </Button>
            <Button
              title="Copy JSON"
              onClick={() => copyText(JSON.stringify(miniJSON, null, 2))}
            >
              Copy JSON
            </Button>
            {item.source ? (
              <a
                href={item.source}
                target="_blank"
                rel="noreferrer"
                className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 border border-[#333]"
                title="Open source"
              >
                Open
              </a>
            ) : null}
            {onClose ? (
              <Button title="Close" onClick={onClose}>Close</Button>
            ) : null}
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">
        {/* Description */}
        <section>
          <h3 className="text-xs text-gray-400 mb-1">Description</h3>
          <p className="text-sm text-gray-200 whitespace-pre-wrap">{item.description || "—"}</p>
        </section>

        {/* Aliases & Tags */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-xs text-gray-400 mb-1">Aliases</h3>
            {(!item.aliases || item.aliases.length === 0) ? (
              <div className="text-xs text-gray-500">—</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {item.aliases!.map((a, i) => (
                  <span key={i} className="text-[11px] px-2 py-[2px] rounded border border-[#333] text-gray-300">
                    {a}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div>
            <h3 className="text-xs text-gray-400 mb-1">Tags</h3>
            {(!item.tags || item.tags.length === 0) ? (
              <div className="text-xs text-gray-500">—</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {item.tags!.map((t, i) => (
                  <span key={i} className="text-[11px] px-2 py-[2px] rounded border border-[#333] text-gray-300">
                    {t}
                  </span>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Overlaps */}
        <section>
          <h3 className="text-xs text-gray-400 mb-1">Overlaps</h3>
          {isUnique ? (
            <div className="text-xs text-emerald-300">None — this feature appears unique.</div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {item.overlaps!.map((id) => (
                <span key={id} className="text-[11px] px-2 py-[2px] rounded border border-[#333] text-gray-300">
                  {id}
                </span>
              ))}
            </div>
          )}
        </section>

        {/* JSON toggle */}
        <section>
          <div className="flex items-center justify-between">
            <h3 className="text-xs text-gray-400">Raw JSON</h3>
            <label className="text-xs text-gray-300 flex items-center gap-2">
              <input
                type="checkbox"
                checked={showJSON}
                onChange={(e) => setShowJSON(e.target.checked)}
              />
              Show
            </label>
          </div>
          {showJSON && (
            <pre className="mt-2 text-xs bg-[#0f0f0f] border border-[#222] rounded p-3 overflow-auto">
{JSON.stringify(item, null, 2)}
            </pre>
          )}
        </section>
      </div>
    </div>
  );
}

/* ===== lil presentational helpers ===== */

function Chip({
  label,
  tone,
  muted,
}: {
  label: string;
  tone?: "purple" | "blue" | "green" | "gray";
  muted?: boolean;
}) {
  const map: Record<string, string> = {
    purple: "bg-purple-700/30 text-purple-200 border border-purple-700/60",
    blue: "bg-blue-700/30 text-blue-200 border border-blue-700/60",
    green: "bg-emerald-700/30 text-emerald-200 border border-emerald-700/60",
    gray: "bg-gray-700/30 text-gray-200 border border-gray-700/60",
  };
  const cls = muted ? "bg-[#1a1a1a] text-gray-300 border border-[#333]" : map[tone || "gray"];
  return <span className={`text-[11px] px-2 py-[2px] rounded ${cls}`}>{label}</span>;
}

function toneForPlatform(p: string): "purple" | "blue" | "green" | "gray" {
  if (p === "Bloomberg") return "purple";
  if (p === "Koyfin") return "blue";
  if (p === "Hammer") return "green";
  return "gray";
}

function Button({
  children,
  onClick,
  title,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 border border-[#333]"
    >
      {children}
    </button>
  );
}