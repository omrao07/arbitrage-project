"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

export type CatalogSearchProps = {
  /** Called on every debounced change (default 250ms). */
  onChange?: (q: string) => void;
  /** Called when user presses Enter or clicks Search. */
  onSubmit?: (q: string) => void;
  /** Async suggester: return list of suggestion strings for the current query. */
  asyncSuggest?: (q: string) => Promise<string[]>;
  /** Initial query text. */
  initialQuery?: string;
  /** Debounce ms for onChange + suggest. */
  debounceMs?: number;
  /** Placeholder text. */
  placeholder?: string;
  /** If true, shows a compact, borderless style. */
  minimal?: boolean;
};

export default function CatalogSearch({
  onChange,
  onSubmit,
  asyncSuggest,
  initialQuery = "",
  debounceMs = 250,
  placeholder = "Search functions, codes, tags…",
  minimal = false,
}: CatalogSearchProps) {
  const [q, setQ] = useState(initialQuery);
  const [pending, setPending] = useState(false);
  const [open, setOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [active, setActive] = useState<number>(-1);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const tRef = useRef<number | null>(null);

  // Debounced change + suggestions
  useEffect(() => {
    if (tRef.current) window.clearTimeout(tRef.current);
    tRef.current = window.setTimeout(async () => {
      onChange?.(q);
      if (!asyncSuggest) return setSuggestions([]);
      if (!q.trim()) {
        setSuggestions([]);
        setPending(false);
        return;
      }
      setPending(true);
      try {
        const res = await asyncSuggest(q.trim());
        setSuggestions(res.slice(0, 10));
        setOpen(true);
        setActive(-1);
      } catch {
        // swallow
      } finally {
        setPending(false);
      }
    }, debounceMs) as any;

    return () => {
      if (tRef.current) window.clearTimeout(tRef.current);
    };
  }, [q, debounceMs, onChange, asyncSuggest]);

  // Close on outside click
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!boxRef.current) return;
      if (!boxRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  function submit(value?: string) {
    const val = (value ?? q).trim();
    if (!val) return;
    onSubmit?.(val);
    setOpen(false);
    // keep query after submit
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || suggestions.length === 0) {
      if (e.key === "Enter") submit();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => (i - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0) submit(suggestions[active]);
      else submit();
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  const shellCls = minimal
    ? "rounded px-2 py-1"
    : "rounded-lg border border-[#222] px-3 py-2";

  return (
    <div ref={boxRef} className={`relative w-full ${minimal ? "" : "bg-[#0b0b0b]"}`}>
      <div className={`flex items-center gap-2 ${shellCls}`}>
        {/* search icon */}
        <svg width="16" height="16" viewBox="0 0 24 24" className="text-gray-400">
          <path
            fill="currentColor"
            d="M15.5 14h-.79l-.28-.27a6.471 6.471 0 0 0 1.57-4.23C16 6.01 13.99 4 11.5 4S7 6.01 7 9.5 9.01 15 11.5 15c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l4.25 4.25c.41.41 1.08.41 1.49 0s.41-1.08 0-1.49L15.5 14Zm-4 0C9.01 14 7 11.99 7 9.5S9.01 5 11.5 5 16 7.01 16 9.5 13.99 14 11.5 14Z"
          />
        </svg>

        <input
          ref={inputRef}
          className="flex-1 bg-transparent outline-none text-sm text-gray-200 placeholder:text-gray-500"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            if (e.target.value.trim()) setOpen(true);
          }}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          spellCheck={false}
        />

        {/* clear */}
        {q ? (
          <button
            title="Clear (Esc)"
            className="text-gray-400 hover:text-gray-200 text-xs"
            onClick={() => {
              setQ("");
              setSuggestions([]);
              setOpen(false);
              inputRef.current?.focus();
              onChange?.("");
            }}
          >
            ✕
          </button>
        ) : null}

        {/* search button */}
        <button
          onClick={() => submit()}
          className="bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs px-2 py-1 rounded"
          disabled={!q.trim() || pending}
          title="Enter"
        >
          {pending ? "…" : "Search"}
        </button>
      </div>

      {/* suggestions dropdown */}
      {open && suggestions.length > 0 && (
        <div className="absolute left-0 right-0 mt-1 z-20 bg-[#0b0b0b] border border-[#222] rounded-lg shadow-xl overflow-hidden">
          {suggestions.map((s, i) => (
            <button
              key={`${s}-${i}`}
              className={`w-full text-left px-3 py-2 text-sm ${
                i === active ? "bg-[#101010] text-gray-100" : "text-gray-300"
              } hover:bg-[#101010]`}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => submit(s)}
              title={s}
            >
              {highlight(s, q)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** simple highlighter */
function highlight(text: string, query: string) {
  const t = query.trim();
  if (!t) return text;
  const idx = text.toLowerCase().indexOf(t.toLowerCase());
  if (idx < 0) return text;
  const a = text.slice(0, idx);
  const b = text.slice(idx, idx + t.length);
  const c = text.slice(idx + t.length);
  return (
    <>
      {a}
      <mark className="bg-yellow-600/50 text-yellow-100 rounded px-[2px]">{b}</mark>
      {c}
    </>
  );
}