"use client";

import React, { useMemo, useState } from "react";

// ------- Types -------
export type Comp = {
  ticker: string;
  name: string;
  sector: string;
  evEbitda: number;
  pe: number;
  ps: number;
  marketCap: number;
};

type Props = {
  data: Comp[];
  title?: string;
};

// ------- Small local UI bits (no external deps) -------
const Card: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`rounded-2xl border border-neutral-200/70 bg-white shadow ${className ?? ""}`}>{children}</div>
);

const CardHeader: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`flex items-center justify-between gap-3 border-b px-4 py-3 ${className ?? ""}`}>{children}</div>
);

const CardTitle: React.FC<React.PropsWithChildren> = ({ children }) => (
  <h2 className="text-lg font-semibold">{children}</h2>
);

const CardContent: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`px-4 py-3 ${className ?? ""}`}>{children}</div>
);

const Badge: React.FC<React.PropsWithChildren> = ({ children }) => (
  <span className="inline-flex items-center rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-800">
    {children}
  </span>
);

// ------- Utils -------
const fmt = (n: number, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : "–");

// ------- Component -------
const CompsPanel: React.FC<Props> = ({ data, title = "Comparable Companies" }) => {
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<keyof Comp>("evEbitda");
  const [sortAsc, setSortAsc] = useState(true);

  const columns: { key: keyof Comp; label: string; num?: boolean; render?: (c: Comp) => React.ReactNode }[] = [
    { key: "ticker", label: "Ticker" },
    { key: "name", label: "Company" },
    { key: "sector", label: "Sector" },
    { key: "evEbitda", label: "EV/EBITDA", num: true },
    { key: "pe", label: "P/E", num: true },
    { key: "ps", label: "P/Sales", num: true },
    {
      key: "marketCap",
      label: "Mkt Cap ($B)",
      num: true,
      render: (c) => fmt(c.marketCap / 1e9, 1),
    },
  ];

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return data;
    return data.filter(
      (d) =>
        d.ticker.toLowerCase().includes(q) ||
        d.name.toLowerCase().includes(q) ||
        d.sector.toLowerCase().includes(q)
    );
  }, [data, filter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      const as = String(av).toLowerCase();
      const bs = String(bv).toLowerCase();
      return sortAsc ? as.localeCompare(bs) : bs.localeCompare(as);
    });
    return arr;
  }, [filtered, sortKey, sortAsc]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by ticker, name, or sector…"
          className="h-9 w-72 rounded-md border border-neutral-300 bg-white px-3 text-sm outline-none focus:border-neutral-500"
        />
      </CardHeader>

      <CardContent className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr className="border-b bg-neutral-50 text-left text-xs font-semibold uppercase tracking-wide text-neutral-600">
              {columns.map((col) => (
                <th
                  key={col.key as string}
                  className="cursor-pointer px-3 py-2"
                  onClick={() => {
                    if (sortKey === col.key) setSortAsc((s) => !s);
                    else {
                      setSortKey(col.key);
                      setSortAsc(true);
                    }
                  }}
                  title="Click to sort"
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {sortKey === col.key && (
                      <span className="text-[10px] text-neutral-500">{sortAsc ? "▲" : "▼"}</span>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>

            <tbody className="text-sm">
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={columns.length} className="px-3 py-6 text-center text-neutral-500">
                    No results found
                  </td>
                </tr>
              ) : (
                sorted.map((c) => (
                  <tr key={c.ticker} className="border-b last:border-0 hover:bg-neutral-50/60">
                    <td className="px-3 py-2">
                      <Badge>{c.ticker}</Badge>
                    </td>
                    <td className="px-3 py-2">{c.name}</td>
                    <td className="px-3 py-2">{c.sector}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{fmt(c.evEbitda, 1)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{fmt(c.pe, 1)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{fmt(c.ps, 1)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {fmt(c.marketCap / 1e9, 1)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
        </table>
      </CardContent>
    </Card>
  );
};

export default CompsPanel;