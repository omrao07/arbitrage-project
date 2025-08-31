'use client';
import React, { useEffect, useRef, useState } from 'react';

type Stock = {
  symbol: string;
  name: string;
  sector?: string;
  price: number;
  mcap: number;
  vol: number;
};

type Props = {
  endpoint?: string;
  title?: string;
  className?: string;
};

export default function Screener({
  endpoint = '/api/screener',
  title = 'Stock Screener',
  className = '',
}: Props) {
  const [q, setQ] = useState('');
  const [sector, setSector] = useState('');
  const [minMcap, setMinMcap] = useState<number>(0);
  const [results, setResults] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // debounce ref
  const qRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (qRef.current) clearTimeout(qRef.current);
    qRef.current = setTimeout(() => {
      fetchData();
    }, 400);
    return () => {
      if (qRef.current) clearTimeout(qRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, sector, minMcap]);

  async function fetchData() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q, sector, minMcap }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Stock[] = await res.json();
      setResults(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className={`flex flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-950 ${className}`}
    >
      <div className="px-3 py-2 border-b text-sm font-medium">{title}</div>

      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search symbol or name"
          className="rounded border px-2 py-1 text-sm"
        />
        <input
          value={sector}
          onChange={(e) => setSector(e.target.value)}
          placeholder="Sector"
          className="rounded border px-2 py-1 text-sm"
        />
        <input
          type="number"
          value={minMcap}
          onChange={(e) => setMinMcap(Number(e.target.value))}
          placeholder="Min Market Cap"
          className="rounded border px-2 py-1 text-sm"
        />
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="min-w-full text-xs">
          <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800">
            <tr>
              <th className="px-2 py-1 text-left">Symbol</th>
              <th className="px-2 py-1 text-left">Name</th>
              <th className="px-2 py-1 text-left">Sector</th>
              <th className="px-2 py-1 text-right">Price</th>
              <th className="px-2 py-1 text-right">MCap</th>
              <th className="px-2 py-1 text-right">Volume</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr
                key={r.symbol + i}
                className={i % 2 ? 'bg-neutral-50 dark:bg-neutral-950' : ''}
              >
                <td className="px-2 py-1">{r.symbol}</td>
                <td className="px-2 py-1">{r.name}</td>
                <td className="px-2 py-1">{r.sector || '-'}</td>
                <td className="px-2 py-1 text-right">{r.price.toFixed(2)}</td>
                <td className="px-2 py-1 text-right">{r.mcap.toLocaleString()}</td>
                <td className="px-2 py-1 text-right">{r.vol.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {loading && <div className="p-2 text-xs">Loading...</div>}
        {error && <div className="p-2 text-xs text-rose-600">{error}</div>}
      </div>
    </div>
  );
}