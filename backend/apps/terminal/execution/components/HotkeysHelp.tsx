"use client";

import React, { useMemo, useState } from "react";

/** Platform-aware key symbol mapping */
function usePlatform() {
  const [isMac] = useState(
    typeof navigator !== "undefined" ? /Mac|iPhone|iPad/i.test(navigator.platform) : true
  );
  const mod = isMac ? "⌘" : "Ctrl";
  const alt = isMac ? "⌥" : "Alt";
  const shift = "⇧";
  const ctrl = isMac ? "⌃" : "Ctrl";
  return { isMac, mod, alt, shift, ctrl };
}

/** Hotkey data model */
export type Hotkey = {
  id: string;
  category: string;         // e.g. "Global", "Trading", "Charts", "Terminal", "Navigation"
  action: string;           // e.g. "Place Order"
  /** keys using tokens: mod, alt, shift, ctrl, plus literal keys (A..Z, F1..F12, /, ., ArrowUp, etc.) */
  combo: string[];          // e.g. ["mod","P"] or ["shift","/"]
  when?: string;            // optional condition text
};

export type HotKeysHelpProps = {
  title?: string;
  items: Hotkey[];
  /** initial filter */
  initialCategory?: string | "All";
  /** optionally hide search bar */
  hideSearch?: boolean;
  /** called when user clicks a row (to focus hintable areas, etc.) */
  onPick?: (hk: Hotkey) => void;
};

export default function HotKeysHelp({
  title = "Keyboard Shortcuts",
  items,
  initialCategory = "All",
  hideSearch = false,
  onPick,
}: HotKeysHelpProps) {
  const { isMac, mod, alt, shift, ctrl } = usePlatform();
  const [q, setQ] = useState("");
  const [cat, setCat] = useState<string>(initialCategory);

  const categories = useMemo(
    () => ["All", ...Array.from(new Set(items.map((x) => x.category))).sort()],
    [items]
  );

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    return items.filter((x) => {
      if (cat !== "All" && x.category !== cat) return false;
      if (!s) return true;
      return (
        x.action.toLowerCase().includes(s) ||
        x.category.toLowerCase().includes(s) ||
        x.combo.join(" ").toLowerCase().includes(s)
      );
    });
  }, [items, q, cat]);

  // group by category for display
  const groups = useMemo(() => {
    const by = new Map<string, Hotkey[]>();
    for (const hk of filtered) {
      const k = hk.category;
      if (!by.has(k)) by.set(k, []);
      by.get(k)!.push(hk);
    }
    // stable sort by action
    for (const arr of by.values()) arr.sort((a, b) => a.action.localeCompare(b.action));
    return Array.from(by.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [filtered]);

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-4 py-2 border-b border-[#222] flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-gray-500 hidden sm:inline">
            {isMac ? "macOS" : "Windows/Linux"} layout
          </span>
          <Select value={cat} onChange={setCat} options={categories} />
          {!hideSearch && (
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search shortcuts…"
              className="bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            />
          )}
        </div>
      </div>

      {/* body */}
      {groups.length === 0 ? (
        <div className="p-4 text-xs text-gray-400">No shortcuts found.</div>
      ) : (
        <div className="p-2 divide-y divide-[#141414]">
          {groups.map(([category, arr]) => (
            <div key={category} className="py-2">
              <div className="px-2 pb-1 text-[11px] text-gray-400">{category}</div>
              <ul className="px-1">
                {arr.map((hk) => (
                  <li key={hk.id}>
                    <button
                      onClick={() => {
                        onPick?.(hk);
                        copyCombo(tokensToDisplay(hk.combo, { mod, alt, shift, ctrl }).join(" + "));
                      }}
                      className="w-full text-left px-2 py-1.5 rounded hover:bg-[#101010] focus:outline-none"
                      title={hk.when ? `When: ${hk.when}` : hk.action}
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="text-[12px] text-gray-100 truncate">{hk.action}</div>
                          {hk.when && (
                            <div className="text-[10px] text-gray-500 truncate">{hk.when}</div>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-1 justify-end">
                          {tokensToDisplay(hk.combo, { mod, alt, shift, ctrl }).map((key, i) => (
                            <Keycap key={i} label={key} />
                          ))}
                        </div>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

      {/* footer legend */}
      <div className="px-4 py-2 border-t border-[#222] text-[11px] text-gray-500 flex items-center gap-3 flex-wrap">
        <Legend label="mod" value={mod} />
        <Legend label="alt" value={alt} />
        <Legend label="shift" value={shift} />
        <Legend label="ctrl" value={ctrl} />
        <span className="opacity-60">Click a row to copy the shortcut.</span>
      </div>
    </div>
  );
}

/* ----------------- tiny UI bits ----------------- */

function Keycap({ label }: { label: string }) {
  return (
    <span
      className="px-1.5 py-[2px] border border-[#2a2a2a] bg-[#0f0f0f] rounded text-[11px] text-gray-200 leading-none"
      style={{ fontFeatureSettings: "'ss01' on, 'tnum' on" }}
    >
      {label}
    </span>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function Legend({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="uppercase tracking-wide">{label}</span>
      <Keycap label={value} />
    </span>
  );
}

/* ----------------- helpers ----------------- */

/** Convert tokens to display strings per OS */
function tokensToDisplay(
  tokens: string[],
  ctx: { mod: string; alt: string; shift: string; ctrl: string }
): string[] {
  const map: Record<string, string> = {
    mod: ctx.mod,
    alt: ctx.alt,
    option: ctx.alt,
    shift: ctx.shift,
    ctrl: ctx.ctrl,
    control: ctx.ctrl,
    enter: "↩",
    return: "↩",
    esc: "⎋",
    escape: "⎋",
    tab: "⇥",
    space: "Space",
    up: "↑",
    down: "↓",
    left: "←",
    right: "→",
    pgup: "PgUp",
    pgdn: "PgDn",
    home: "Home",
    end: "End",
    bksp: "⌫",
    del: "Del",
    slash: "/",
    backslash: "\\",
    comma: ",",
    period: ".",
    semicolon: ";",
    quote: "'",
    backquote: "`",
    minus: "-",
    equal: "=",
  };
  return tokens.map((t) => map[t.toLowerCase()] || t.toUpperCase());
}

async function copyCombo(text: string) {
  try {
    await navigator.clipboard?.writeText(text);
  } catch {
    // ignore if not available
  }
}

/* ----------------- example data (remove in prod) -----------------
const SAMPLE: Hotkey[] = [
  { id: "cmdk", category: "Global", action: "Command Palette", combo: ["mod","K"] },
  { id: "search", category: "Global", action: "Quick Search", combo: ["mod","/"] },
  { id: "toggle", category: "Terminal", action: "Toggle Terminal", combo: ["mod","`"] },
  { id: "new-order", category: "Trading", action: "New Order Ticket", combo: ["mod","N"] },
  { id: "basket", category: "Trading", action: "Open Basket Trader", combo: ["mod","B"] },
  { id: "cancel", category: "Trading", action: "Cancel Selected Order", combo: ["del"], when: "Orders panel focused" },
  { id: "chart-1m", category: "Charts", action: "1m Interval", combo: ["1"] },
  { id: "chart-5m", category: "Charts", action: "5m Interval", combo: ["2"] },
  { id: "chart-1h", category: "Charts", action: "1h Interval", combo: ["3"] },
  { id: "next-tab", category: "Navigation", action: "Next Tab", combo: ["ctrl","tab"] },
  { id: "prev-tab", category: "Navigation", action: "Previous Tab", combo: ["ctrl","shift","tab"] },
];
------------------------------------------------------------------- */