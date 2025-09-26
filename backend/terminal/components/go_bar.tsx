// go-bar.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";

type GoItemBase = {
  id: string;
  title: string;
  subtitle?: string;
  group?: string; // e.g. "Pages", "Actions"
  icon?: React.ReactNode;
  keywords?: string; // extra search terms
  disabled?: boolean;
};

export type GoLink = GoItemBase & {
  kind: "link";
  href: string;
  target?: "_blank" | "_self";
};

export type GoAction = GoItemBase & {
  kind: "action";
  onAction: () => void | Promise<void>;
};

export type GoItem = GoLink | GoAction;

export interface GoBarProps {
  items: GoItem[];
  placeholder?: string;
  hotkey?: string; // "cmd+k" | "ctrl+k"
  maxResults?: number;
  onClose?: () => void;
  /** If you want to control open state externally; otherwise component manages it */
  open?: boolean;
  onOpenChange?: (o: boolean) => void;
  /** When true, keeps a small “recent” list in localStorage */
  rememberRecent?: boolean;
  storageKey?: string; // localStorage key for recents
}

const cls = (...x: (string | false | null | undefined)[]) => x.filter(Boolean).join(" ");

const useHotkey = (combo: string, handler: (e: KeyboardEvent) => void) => {
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      const isCmd = e.metaKey || (navigator.platform.includes("Mac") && e.ctrlKey);
      const isCtrl = e.ctrlKey;
      const key = e.key.toLowerCase();
      if (combo === "cmd+k" && isCmd && key === "k") {
        e.preventDefault();
        handler(e);
      } else if (combo === "ctrl+k" && isCtrl && key === "k") {
        e.preventDefault();
        handler(e);
      }
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [combo, handler]);
};

const norm = (s: string) =>
  s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");

type Match = { score: number; indices: number[] };

/** Lightweight fuzzy scorer: subsequence match with bonuses for starts/word boundaries */
function fuzzyScore(query: string, text: string): Match {
  if (!query) return { score: 0, indices: [] };
  const q = norm(query);
  const t = norm(text);
  let qi = 0;
  let score = 0;
  const idx: number[] = [];
  for (let i = 0; i < t.length && qi < q.length; i++) {
    if (t[i] === q[qi]) {
      idx.push(i);
      qi++;
      // bonuses
      score += 2;
      if (i === 0 || /\s|[-_/]/.test(t[i - 1])) score += 3;
      if (i > 0 && t[i - 1] === q[qi - 1]) score += 1; // streak
    }
  }
  return qi === q.length ? { score, indices: idx } : { score: -1, indices: [] };
}

function highlight(text: string, indices: number[]) {
  if (!indices.length) return <>{text}</>;
  const parts: React.ReactNode[] = [];
  let last = 0;
  indices.forEach((i, k) => {
    if (i > last) parts.push(<span key={`t${k}-${last}`}>{text.slice(last, i)}</span>);
    parts.push(
      <mark key={`m${k}-${i}`} className="bg-amber-200/60 text-amber-900 rounded px-0.5">
        {text[i]}
      </mark>
    );
    last = i + 1;
  });
  if (last < text.length) parts.push(<span key={`e-${last}`}>{text.slice(last)}</span>);
  return <>{parts}</>;
}

const RECENT_LIMIT = 6;

const GoBar: React.FC<GoBarProps> = ({
  items,
  placeholder = "Go to… (type to search, ↑/↓ to navigate)",
  hotkey = /Mac|iPhone|iPad/.test(navigator.platform) ? "cmd+k" : "ctrl+k",
  maxResults = 50,
  onClose,
  open,
  onOpenChange,
  rememberRecent = true,
  storageKey = "gobar:recents",
}) => {
  const [internalOpen, setInternalOpen] = useState(false);
  const isOpen = open ?? internalOpen;
  const setOpen = (v: boolean) => {
    onOpenChange ? onOpenChange(v) : setInternalOpen(v);
  };

  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // recents
  const [recents, setRecents] = useState<string[]>(() => {
    if (!rememberRecent) return [];
    try {
      const raw = localStorage.getItem(storageKey);
      return raw ? (JSON.parse(raw) as string[]) : [];
    } catch {
      return [];
    }
  });

  const grouped = useMemo(() => {
    const map = new Map<string, GoItem[]>();
    items.forEach((it) => {
      const g = it.group ?? (it.kind === "link" ? "Pages" : "Actions");
      if (!map.has(g)) map.set(g, []);
      map.get(g)!.push(it);
    });
    return Array.from(map.entries()).map(([g, arr]) => [g, arr] as const);
  }, [items]);

  const results = useMemo(() => {
    if (!q.trim()) {
      // show recents first if any
      const byId = new Map(items.map((i) => [i.id, i]));
      const recentItems = recents
        .map((id) => byId.get(id))
        .filter(Boolean) as GoItem[];
      const rest = items.filter((i) => !recents.includes(i.id)).slice(0, Math.max(0, maxResults - recentItems.length));
      return [
        ...(recentItems.length ? [["Recent", recentItems] as const] : []),
        ...grouped.filter(([, arr]) => arr.length > 0),
      ];
    }
    // score all items on title + subtitle + keywords
    type Scored = { item: GoItem; score: number; titleMatch: Match };
    const scored: Scored[] = [];
    for (const it of items) {
      if (it.disabled) continue;
      const base = [it.title, it.subtitle ?? "", it.keywords ?? ""].join(" ");
      const s = fuzzyScore(q, base);
      if (s.score > 0) scored.push({ item: it, score: s.score, titleMatch: fuzzyScore(q, it.title) });
    }
    scored.sort((a, b) => b.score - a.score || a.item.title.localeCompare(b.item.title));
    const top = scored.slice(0, maxResults);
    // regroup by item.group
    const byGroup = new Map<string, Scored[]>();
    top.forEach((x) => {
      const g = x.item.group ?? (x.item.kind === "link" ? "Pages" : "Actions");
      if (!byGroup.has(g)) byGroup.set(g, []);
      byGroup.get(g)!.push(x);
    });
    return Array.from(byGroup.entries()).map(([g, arr]) => [g, arr.map((x) => x.item)] as const);
  }, [q, items, grouped, maxResults, recents]);

  const flatList = useMemo(() => results.flatMap(([, arr]) => arr), [results]);

  const act = (it: GoItem) => {
    if (it.disabled) return;
    if (rememberRecent) {
      const next = [it.id, ...recents.filter((id) => id !== it.id)].slice(0, RECENT_LIMIT);
      setRecents(next);
      try {
        localStorage.setItem(storageKey, JSON.stringify(next));
      } catch {}
    }
    if (it.kind === "link") {
      const a = document.createElement("a");
      a.href = it.href;
      a.target = it.target ?? "_self";
      a.rel = it.target === "_blank" ? "noopener noreferrer" : "";
      a.click();
    } else {
      Promise.resolve(it.onAction()).catch(console.error);
    }
    setOpen(false);
    onClose?.();
  };

  // keyboard & focus
  useHotkey(hotkey, () => {
    setOpen(!isOpen);
    if (!isOpen) setTimeout(() => inputRef.current?.focus(), 0);
  });

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        onClose?.();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setCursor((c) => Math.min(c + 1, flatList.length - 1));
        scrollIntoView(cursor + 1);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setCursor((c) => Math.max(c - 1, 0));
        scrollIntoView(cursor - 1);
      } else if (e.key === "Enter") {
        e.preventDefault();
        const it = flatList[cursor];
        if (it) act(it);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, flatList, cursor]);

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 0);
  }, [isOpen]);

  useEffect(() => {
    // reset cursor when query changes or menu opens
    setCursor(0);
  }, [q, isOpen]);

  const scrollIntoView = (idx: number) => {
    const container = listRef.current;
    if (!container) return;
    const el = container.querySelector<HTMLDivElement>(`[data-idx="${idx}"]`);
    if (!el) return;
    const { top, bottom } = el.getBoundingClientRect();
    const { top: cTop, bottom: cBottom } = container.getBoundingClientRect();
    if (top < cTop) container.scrollTop -= cTop - top + 8;
    else if (bottom > cBottom) container.scrollTop += bottom - cBottom + 8;
  };

  // Render
  return (
    <>
      {/* Trigger (optional) */}
      <button
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-xl bg-zinc-900 text-zinc-100 px-3 py-2 border border-zinc-800 hover:border-zinc-700"
        title={`${hotkey.toUpperCase()} to open`}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" className="opacity-80">
          <path
            fill="currentColor"
            d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79L20 21.5 21.5 20zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14"
          />
        </svg>
        <span className="text-sm">Go</span>
        <span className="ml-2 hidden md:inline text-xs text-zinc-400 border border-zinc-700 rounded px-1">
          {hotkey.toUpperCase()}
        </span>
      </button>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-[999] bg-black/50 backdrop-blur-sm"
          onClick={() => setOpen(false)}
          aria-hidden
        />
      )}

      {/* Dialog */}
      <div
        className={cls(
          "fixed left-1/2 top-20 z-[1000] w-[92vw] max-w-2xl -translate-x-1/2",
          isOpen ? "opacity-100 scale-100" : "opacity-0 scale-95 pointer-events-none",
          "transition-all"
        )}
        role="dialog"
        aria-modal={isOpen}
        aria-label="Go bar"
      >
        <div
          className="rounded-2xl border border-zinc-800 bg-zinc-900 shadow-2xl overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Search input */}
          <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800">
            <svg width="18" height="18" viewBox="0 0 24 24" className="text-zinc-400">
              <path
                fill="currentColor"
                d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79L20 21.5 21.5 20zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14"
              />
            </svg>
            <input
              ref={inputRef}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={placeholder}
              className="flex-1 bg-transparent outline-none text-zinc-100 placeholder:text-zinc-500 py-2"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
            />
            {q && (
              <button
                onClick={() => setQ("")}
                className="text-xs text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded"
                aria-label="Clear"
              >
                Clear
              </button>
            )}
            <button
              onClick={() => {
                setOpen(false);
                onClose?.();
              }}
              className="text-xs text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded"
              aria-label="Close"
            >
              Esc
            </button>
          </div>

          {/* Results */}
          <div ref={listRef} className="max-h-[60vh] overflow-auto">
            {results.length === 0 && (
              <div className="px-4 py-6 text-center text-zinc-400">No results</div>
            )}
            {results.map(([groupName, arr]) => (
              <div key={groupName} className="py-1">
                <div className="px-3 py-1 text-xs uppercase tracking-wide text-zinc-500">
                  {groupName}
                </div>
                {arr.map((it, i) => {
                  const idx = flatList.findIndex((x) => x.id === it.id); // stable global index
                  const active = idx === cursor;
                  const disabled = !!it.disabled;
                  return (
                    <div
                      key={it.id}
                      role="button"
                      data-idx={idx}
                      aria-disabled={disabled}
                      onMouseEnter={() => setCursor(idx)}
                      onClick={() => !disabled && act(it)}
                      className={cls(
                        "mx-1 my-1 flex items-center gap-3 rounded-xl px-3 py-2",
                        active ? "bg-zinc-800 ring-1 ring-zinc-700" : "hover:bg-zinc-800/60",
                        disabled && "opacity-50 cursor-not-allowed"
                      )}
                    >
                      <div className="w-6 h-6 flex items-center justify-center text-zinc-300">
                        {it.icon ?? <span className="text-zinc-500">•</span>}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-zinc-100 text-sm leading-tight">
                          {q ? highlight(it.title, fuzzyScore(q, it.title).indices) : it.title}
                        </div>
                        {it.subtitle && (
                          <div className="text-xs text-zinc-400 truncate">{it.subtitle}</div>
                        )}
                      </div>
                      {it.kind === "link" && (
                        <span className="text-[10px] text-zinc-500">{new URL(it.href, window.location.href).host}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-3 py-2 border-t border-zinc-800 text-[11px] text-zinc-400">
            <div className="flex items-center gap-3">
              <kbd className="px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900">↑</kbd>
              <kbd className="px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900">↓</kbd>
              <span>to navigate</span>
              <kbd className="px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900">Enter</kbd>
              <span>to select</span>
              <kbd className="px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900">Esc</kbd>
              <span>to close</span>
            </div>
            <div>
              <span className="hidden sm:inline">Open with </span>
              <kbd className="px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900">
                {hotkey.toUpperCase()}
              </kbd>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default GoBar;