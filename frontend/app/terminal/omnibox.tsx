'use client';
import React, { useEffect, useRef, useState } from 'react';

type Result = {
  id: string;
  label: string;
  type?: string;
};

type Props = {
  endpoint?: string; // API endpoint for search
  placeholder?: string;
  className?: string;
};

export default function Omnibox({
  endpoint = '/api/search',
  placeholder = 'Search or type a command…',
  className = '',
}: Props) {
  const [q, setQ] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [show, setShow] = useState(false);

  // debounce ref
  const qRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!q.trim()) {
      setResults([]);
      return;
    }
    if (qRef.current) clearTimeout(qRef.current);
    qRef.current = setTimeout(() => {
      runSearch();
    }, 300);
    return () => {
      if (qRef.current) clearTimeout(qRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  async function runSearch() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Result[] = await res.json();
      setResults(data);
      setShow(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`relative w-full ${className}`}>
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 px-3 py-2 text-sm bg-white dark:bg-neutral-900"
      />
      {loading && <div className="absolute right-3 top-2 text-xs text-neutral-400">…</div>}
      {error && <div className="absolute right-3 top-2 text-xs text-rose-500">{error}</div>}

      {show && results.length > 0 && (
        <div className="absolute z-10 mt-1 w-full rounded-md border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 shadow-lg max-h-64 overflow-auto">
          {results.map((r) => (
            <div
              key={r.id}
              className="px-3 py-2 text-sm hover:bg-neutral-100 dark:hover:bg-neutral-800 cursor-pointer"
              onClick={() => {
                setQ(r.label);
                setShow(false);
              }}
            >
              <div className="font-medium">{r.label}</div>
              {r.type && <div className="text-xs text-neutral-500">{r.type}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}