"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * MarketToolbar.tsx
 * Minimal, dependency-free toolbar for market pages.
 * - Search box
 * - Timeframe selector
 * - Toggle chips (asset class filters)
 * - Refresh button
 * - Compact mode
 * Tailwind-only. No icon packages (inline SVGs).
 */

type Timeframe = "1D" | "5D" | "1M" | "6M" | "1Y" | "MAX";

export type MarketToolbarProps = {
  /** Called when user types Enter in search or clicks search icon */
  onSearch?: (query: string) => void;
  /** Called when refresh is clicked (or Cmd/Ctrl+R inside toolbar) */
  onRefresh?: () => void;
  /** Controlled timeframe (optional) */
  timeframe?: Timeframe;
  /** Uncontrolled initial timeframe (used if `timeframe` not provided) */
  defaultTimeframe?: Timeframe;
  /** Called when timeframe changes */
  onTimeframeChange?: (tf: Timeframe) => void;
  /** Toggle chips (e.g., asset classes). id should be stable. */
  toggles?: { id: string; label: string; active?: boolean }[];
  /** Called when a toggle chip is clicked */
  onToggleChange?: (id: string, active: boolean) => void;
  /** Placeholder for search input */
  searchPlaceholder?: string;
  /** Start disabled (greys out inputs) */
  disabled?: boolean;
  /** Compact visual density */
  compact?: boolean;
  /** Extra classes for outer container */
  className?: string;
};

const TF_LIST: Timeframe[] = ["1D", "5D", "1M", "6M", "1Y", "MAX"];

export default function MarketToolbar({
  onSearch,
  onRefresh,
  timeframe,
  defaultTimeframe = "1D",
  onTimeframeChange,
  toggles = [
    { id: "equity", label: "Equities", active: true },
    { id: "fx", label: "FX", active: true },
    { id: "crypto", label: "Crypto", active: true },
    { id: "futures", label: "Futures", active: true },
  ],
  onToggleChange,
  searchPlaceholder = "Search symbols, news, ISIN…",
  disabled = false,
  compact = false,
  className = "",
}: MarketToolbarProps) {
  const [q, setQ] = useState("");
  const [internalTF, setInternalTF] = useState<Timeframe>(defaultTimeframe);
  const activeTF = timeframe ?? internalTF;

  const [chips, setChips] = useState(() =>
    toggles.map((t) => ({ ...t, active: t.active ?? true }))
  );

  // keep chips in sync if toggles prop changes
  useEffect(() => {
    setChips(toggles.map((t) => ({ ...t, active: t.active ?? true })));
  }, [JSON.stringify(toggles)]); // shallow-ish

  // keybindings inside toolbar scope
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const handler = (e: KeyboardEvent) => {
      if (disabled) return;
      // Cmd/Ctrl + K => focus search
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        const input = el.querySelector("input[type='search']") as HTMLInputElement | null;
        input?.focus();
        input?.select();
        e.preventDefault();
      }
      // Cmd/Ctrl + R => refresh
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "r") {
        onRefresh?.();
        e.preventDefault();
      }
    };
    el.addEventListener("keydown", handler);
    return () => el.removeEventListener("keydown", handler);
  }, [onRefresh, disabled]);

  const runSearch = () => {
    if (disabled) return;
    onSearch?.(q.trim());
  };

  const setTF = (tf: Timeframe) => {
    if (disabled) return;
    if (!timeframe) setInternalTF(tf);
    onTimeframeChange?.(tf);
  };

  const toggleChip = (id: string) => {
    if (disabled) return;
    setChips((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, active: !c.active } : c
      )
    );
    const newState = !chips.find((c) => c.id === id)?.active;
    onToggleChange?.(id, newState);
  };

  const density = compact
    ? { padX: "px-2", padY: "py-1", text: "text-xs", gap: "gap-1.5", h: "h-8" }
    : { padX: "px-3", padY: "py-2", text: "text-sm", gap: "gap-2", h: "h-10" };

  return (
    <div
      ref={rootRef}
      className={[
        "w-full border-b border-neutral-800 bg-neutral-950/90 backdrop-blur",
        className,
      ].join(" ")}
      tabIndex={0}
    >
      <div className={`mx-auto flex max-w-7xl flex-wrap items-center ${density.gap} px-4 py-3`}>
        {/* Left: Search */}
        <div className="relative flex-1 min-w-[220px]">
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runSearch()}
            disabled={disabled}
            placeholder={searchPlaceholder}
            className={[
              "w-full rounded-md border border-neutral-800 bg-neutral-900",
              "text-neutral-100 placeholder:text-neutral-500",
              "outline-none focus:ring-2 focus:ring-emerald-600",
              density.padX,
              density.padY,
            ].join(" ")}
          />
          <button
            type="button"
            onClick={runSearch}
            disabled={disabled}
            className="absolute right-1 top-1/2 -translate-y-1/2 rounded-md p-1 text-neutral-400 hover:text-neutral-200"
            aria-label="Search"
            title="Search"
          >
            {/* inline magnifier */}
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
              <path d="M20 20L16.65 16.65" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Middle: Timeframes */}
        <div className={`flex items-center ${density.gap}`}>
          {TF_LIST.map((tf) => (
            <button
              key={tf}
              type="button"
              disabled={disabled}
              onClick={() => setTF(tf)}
              className={[
                "rounded-md border",
                activeTF === tf
                  ? "border-emerald-600 bg-emerald-600/20 text-emerald-300"
                  : "border-neutral-800 bg-neutral-900 text-neutral-300 hover:bg-neutral-800/60",
                density.padX,
                density.padY,
                density.text,
              ].join(" ")}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Right: Toggles + Refresh */}
        <div className={`ml-auto flex items-center ${density.gap}`}>
          <div className={`hidden md:flex items-center flex-wrap ${density.gap}`}>
            {chips.map((c) => (
              <button
                key={c.id}
                type="button"
                disabled={disabled}
                onClick={() => toggleChip(c.id)}
                className={[
                  "rounded-full border",
                  c.active
                    ? "border-emerald-600 bg-emerald-700/20 text-emerald-300"
                    : "border-neutral-700 bg-neutral-900 text-neutral-400 hover:text-neutral-200",
                  "px-3 py-1 text-xs",
                ].join(" ")}
                aria-pressed={c.active}
              >
                {c.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            disabled={disabled}
            onClick={() => onRefresh?.()}
            className={[
              "inline-flex items-center justify-center rounded-md border border-neutral-800 bg-neutral-900",
              "text-neutral-300 hover:text-neutral-100 hover:bg-neutral-800",
              density.padX,
              density.padY,
              density.text,
            ].join(" ")}
            title="Refresh (Ctrl/Cmd+R)"
            aria-label="Refresh"
          >
            {/* inline refresh icon */}
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M20 12a8 8 0 1 1-2.34-5.66"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path d="M20 4v6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}