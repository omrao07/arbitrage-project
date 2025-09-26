"use client";

import React, { useState, useMemo } from "react";

export type Strategy = {
  id: string;
  name: string;
  family: string;
  region: string;
  type: string;
  risk: string;
  pnlYTD?: number;
};

type Props = {
  strategies: Strategy[];
  title?: string;
};

const StrategiesTable: React.FC<Props> = ({ strategies, title = "Strategies" }) => {
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<keyof Strategy>("name");
  const [sortAsc, setSortAsc] = useState(true);

  const filtered = useMemo(() => {
    const q = filter.toLowerCase().trim();
    if (!q) return strategies;
    return strategies.filter(
      (s) =>
        s.name.toLowerCase().includes(q) ||
        s.family.toLowerCase().includes(q) ||
        s.region.toLowerCase().includes(q) ||
        s.type.toLowerCase().includes(q) ||
        s.risk.toLowerCase().includes(q)
    );
  }, [strategies, filter]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return arr;
  }, [filtered, sortKey, sortAsc]);

  const columns: { key: keyof Strategy; label: string; num?: boolean }[] = [
    { key: "name", label: "Name" },
    { key: "family", label: "Family" },
    { key: "region", label: "Region" },
    { key: "type", label: "Type" },
    { key: "risk", label: "Risk" },
    { key: "pnlYTD", label: "PnL YTD", num: true },
  ];

  return (
    <div className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search strategies…"
          className="h-9 w-72 rounded-md border border-neutral-300 px-3 text-sm outline-none focus:border-neutral-500"
        />
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
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
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {sortKey === col.key && (
                      <span className="text-[10px] text-neutral-500">
                        {sortAsc ? "▲" : "▼"}
                      </span>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-3 py-6 text-center text-neutral-500">
                  No strategies found
                </td>
              </tr>
            ) : (
              sorted.map((s) => (
                <tr key={s.id} className="border-b last:border-0 hover:bg-neutral-50/60">
                  <td className="px-3 py-2">{s.name}</td>
                  <td className="px-3 py-2">{s.family}</td>
                  <td className="px-3 py-2">{s.region}</td>
                  <td className="px-3 py-2">{s.type}</td>
                  <td className="px-3 py-2">{s.risk}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {s.pnlYTD !== undefined ? s.pnlYTD.toFixed(2) + "%" : "–"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StrategiesTable;