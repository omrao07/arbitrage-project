"use client";

import React, { useState } from "react";

export type CatalogFiltersState = {
  platform: string;          // "All" | "Bloomberg" | "Koyfin" | "Hammer"
  category: string;          // e.g. "Market", "Analytics", "Execution"
  search: string;            // free text search
  showUniqueOnly: boolean;   // only show unique features vs Bloomberg
};

export default function CatalogFilters({
  onChange,
  initial,
}: {
  onChange: (f: CatalogFiltersState) => void;
  initial?: Partial<CatalogFiltersState>;
}) {
  const [platform, setPlatform] = useState(initial?.platform ?? "All");
  const [category, setCategory] = useState(initial?.category ?? "All");
  const [search, setSearch] = useState(initial?.search ?? "");
  const [showUniqueOnly, setShowUniqueOnly] = useState(initial?.showUniqueOnly ?? false);

  function emit(next: Partial<CatalogFiltersState>) {
    const merged: CatalogFiltersState = {
      platform,
      category,
      search,
      showUniqueOnly,
      ...next,
    };
    onChange(merged);
  }

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg p-3 flex flex-wrap gap-4 items-end">
      {/* Platform filter */}
      <div className="flex flex-col">
        <label className="text-xs text-gray-400 mb-1">Platform</label>
        <select
          className="bg-[#121212] border border-[#333] rounded px-2 py-1"
          value={platform}
          onChange={(e) => {
            setPlatform(e.target.value);
            emit({ platform: e.target.value });
          }}
        >
          <option>All</option>
          <option>Bloomberg</option>
          <option>Koyfin</option>
          <option>Hammer</option>
        </select>
      </div>

      {/* Category filter */}
      <div className="flex flex-col">
        <label className="text-xs text-gray-400 mb-1">Category</label>
        <select
          className="bg-[#121212] border border-[#333] rounded px-2 py-1"
          value={category}
          onChange={(e) => {
            setCategory(e.target.value);
            emit({ category: e.target.value });
          }}
        >
          <option>All</option>
          <option>Market</option>
          <option>Analytics</option>
          <option>Execution</option>
          <option>Portfolio</option>
          <option>Risk</option>
          <option>News</option>
          <option>Other</option>
        </select>
      </div>

      {/* Search box */}
      <div className="flex flex-col flex-1 min-w-[200px]">
        <label className="text-xs text-gray-400 mb-1">Search</label>
        <input
          className="bg-[#121212] border border-[#333] rounded px-2 py-1"
          placeholder="Search functions..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            emit({ search: e.target.value });
          }}
        />
      </div>

      {/* Unique toggle */}
      <label className="flex items-center gap-2 text-xs text-gray-300">
        <input
          type="checkbox"
          checked={showUniqueOnly}
          onChange={(e) => {
            setShowUniqueOnly(e.target.checked);
            emit({ showUniqueOnly: e.target.checked });
          }}
        />
        Unique only
      </label>
    </div>
  );
}