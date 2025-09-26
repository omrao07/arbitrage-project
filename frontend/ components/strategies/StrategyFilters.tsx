"use client";

import React from "react";

export type Filters = {
  family?: string;
  region?: string;
  type?: string;
  risk?: string;
  query?: string;
};

type Props = {
  filters: Filters;
  families: string[];
  regions: string[];
  types: string[];
  risks: string[];
  onChange: (next: Filters) => void;
};

const StrategyFilters: React.FC<Props> = ({
  filters,
  families,
  regions,
  types,
  risks,
  onChange,
}) => {
  const set = (key: keyof Filters, value: string) =>
    onChange({ ...filters, [key]: value || undefined });

  return (
    <div className="w-full rounded-xl border border-neutral-200 bg-white shadow px-4 py-3 space-y-3">
      <h2 className="text-sm font-semibold text-neutral-700">Filters</h2>

      {/* Search */}
      <input
        type="text"
        placeholder="Search strategies…"
        value={filters.query || ""}
        onChange={(e) => set("query", e.target.value)}
        className="w-full rounded-md border border-neutral-300 px-3 py-1.5 text-sm outline-none focus:border-neutral-500"
      />

      {/* Dropdowns */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <select
          value={filters.family || ""}
          onChange={(e) => set("family", e.target.value)}
          className="rounded-md border border-neutral-300 px-2 py-1.5 text-sm focus:border-neutral-500"
        >
          <option value="">Family</option>
          {families.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>

        <select
          value={filters.region || ""}
          onChange={(e) => set("region", e.target.value)}
          className="rounded-md border border-neutral-300 px-2 py-1.5 text-sm focus:border-neutral-500"
        >
          <option value="">Region</option>
          {regions.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>

        <select
          value={filters.type || ""}
          onChange={(e) => set("type", e.target.value)}
          className="rounded-md border border-neutral-300 px-2 py-1.5 text-sm focus:border-neutral-500"
        >
          <option value="">Type</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>

        <select
          value={filters.risk || ""}
          onChange={(e) => set("risk", e.target.value)}
          className="rounded-md border border-neutral-300 px-2 py-1.5 text-sm focus:border-neutral-500"
        >
          <option value="">Risk</option>
          {risks.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {/* Reset */}
      <button
        onClick={() => onChange({})}
        className="rounded-md border border-neutral-300 bg-neutral-50 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100"
      >
        Reset Filters
      </button>
    </div>
  );
};

export default StrategyFilters;