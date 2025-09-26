"use client";

import React, { useEffect, useId, useMemo, useRef, useState } from "react";

export type MarketTab = {
  id: string;
  label: string;
  badge?: number | string;
  disabled?: boolean;
};

type Size = "sm" | "md" | "lg";
type Variant = "underline" | "solid" | "pill";

export type MarketTabsProps = {
  tabs: MarketTab[];
  /** Uncontrolled initial tab id */
  defaultTabId?: string;
  /** Controlled active tab id */
  activeId?: string;
  /** Called on tab change */
  onChange?: (id: string) => void;
  /** Render function for the active panel */
  children?: React.ReactNode | ((activeId: string) => React.ReactNode);
  /** Visual options */
  size?: Size;
  variant?: Variant;
  stretch?: boolean; // make tabs fill width
  className?: string;
};

export default function MarketTabs({
  tabs,
  defaultTabId,
  activeId,
  onChange,
  children,
  size = "md",
  variant = "underline",
  stretch = false,
  className = "",
}: MarketTabsProps) {
  const fallback = useMemo(
    () => tabs.find((t) => !t.disabled)?.id ?? tabs[0]?.id ?? "",
    [tabs]
  );
  const [internal, setInternal] = useState<string>(defaultTabId || activeId || fallback);

  // keep internal in sync with controlled prop
  useEffect(() => {
    if (activeId && activeId !== internal) setInternal(activeId);
  }, [activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ensure selected tab is never disabled/missing
  useEffect(() => {
    const found = tabs.find((t) => t.id === internal && !t.disabled);
    if (!found) setInternal(fallback);
  }, [tabs]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = activeId ?? internal;

  const setActive = (id: string) => {
    if (tabs.find((t) => t.id === id && !t.disabled)) {
      if (!activeId) setInternal(id);
      onChange?.(id);
    }
  };

  /* ---------------------------- accessibility ---------------------------- */

  const listRef = useRef<HTMLDivElement>(null);
  const uid = useId();

  const idx = Math.max(
    0,
    tabs.findIndex((t) => t.id === current)
  );

  const move = (dir: 1 | -1) => {
    if (!tabs.length) return;
    let i = idx;
    for (let step = 0; step < tabs.length; step++) {
      i = (i + dir + tabs.length) % tabs.length;
      if (!tabs[i].disabled) {
        setActive(tabs[i].id);
        focusButton(i);
        break;
      }
    }
  };

  const focusButton = (i: number) => {
    const btns = listRef.current?.querySelectorAll<HTMLButtonElement>("[role='tab']");
    const el = btns?.[i];
    el?.focus();
    el?.scrollIntoView({ inline: "nearest", block: "nearest", behavior: "smooth" });
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        move(1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        move(-1);
        break;
      case "Home":
        e.preventDefault();
        focusButton(0);
        setActive(tabs.find((t) => !t.disabled)?.id ?? tabs[0]?.id ?? "");
        break;
      case "End":
        e.preventDefault();
        // last enabled
        for (let i = tabs.length - 1; i >= 0; i--) {
          if (!tabs[i].disabled) {
            focusButton(i);
            setActive(tabs[i].id);
            break;
          }
        }
        break;
    }
  };

  /* -------------------------------- styles ------------------------------- */

  const sizeCls: Record<Size, string> = {
    sm: "text-xs px-2.5 py-1.5",
    md: "text-sm px-3.5 py-2",
    lg: "text-base px-4.5 py-2.5",
  };

  const baseBtn =
    "relative inline-flex items-center gap-2 rounded-md transition select-none focus:outline-none focus:ring-2 focus:ring-offset-0";
  const disabledCls = "opacity-50 cursor-not-allowed";

  const variantBtn: Record<Variant, { base: string; active: string; inactive: string; underline?: boolean }> = {
    underline: {
      base: "bg-transparent",
      active: "text-neutral-100",
      inactive: "text-neutral-400 hover:text-neutral-200",
      underline: true,
    },
    solid: {
      base: "bg-neutral-800",
      active: "text-neutral-100 bg-neutral-700",
      inactive: "text-neutral-300 hover:bg-neutral-700/80",
    },
    pill: {
      base: "bg-neutral-900 border border-neutral-700",
      active: "text-neutral-100 bg-neutral-800 border-neutral-600",
      inactive: "text-neutral-300 hover:border-neutral-500",
    },
  };

  const barCls =
    "w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-700 scrollbar-track-transparent";

  const buttonLayout = stretch ? "flex-1 justify-center" : "";

  /* -------------------------------- render -------------------------------- */

  return (
    <div className={`w-full ${className}`}>
      {/* Tablist */}
      <div
        ref={listRef}
        role="tablist"
        aria-orientation="horizontal"
        className={`${barCls} border-b border-neutral-800`}
        onKeyDown={onKeyDown}
      >
        <div className={`flex items-end gap-1 p-1`}>
          {tabs.map((t, i) => {
            const isActive = t.id === current;
            const styles = variantBtn[variant];
            return (
              <button
                key={t.id}
                role="tab"
                aria-selected={isActive}
                aria-controls={`${uid}-panel-${t.id}`}
                id={`${uid}-tab-${t.id}`}
                disabled={t.disabled}
                onClick={() => !t.disabled && setActive(t.id)}
                className={[
                  baseBtn,
                  sizeCls[size],
                  buttonLayout,
                  styles.base,
                  t.disabled ? disabledCls : isActive ? styles.active : styles.inactive,
                ].join(" ")}
              >
                <span>{t.label}</span>
                {t.badge != null && (
                  <span className="rounded-md bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-300">
                    {t.badge}
                  </span>
                )}

                {/* underline indicator for 'underline' variant */}
                {variant === "underline" && (
                  <span
                    className={`absolute -bottom-[7px] left-0 right-0 mx-auto h-[2px] w-8 rounded-full transition-all ${
                      isActive ? "bg-emerald-500" : "bg-transparent"
                    }`}
                  />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Panel */}
      <div
        role="tabpanel"
        id={`${uid}-panel-${current}`}
        aria-labelledby={`${uid}-tab-${current}`}
        className="pt-3"
      >
        {typeof children === "function" ? children(current) : children}
      </div>
    </div>
  );
}

/* ------------------------------ Example use ------------------------------
<MarketTabs
  tabs={[
    { id: "energy", label: "Energy", badge: 3 },
    { id: "metals", label: "Metals" },
    { id: "agri", label: "Agri", disabled: false },
  ]}
  defaultTabId="energy"
  variant="underline"
  size="md"
  stretch
>
  {(active) => (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4 text-sm">
      Active: <span className="text-emerald-400 font-medium">{active}</span>
    </div>
  )}
</MarketTabs>
--------------------------------------------------------------------------- */