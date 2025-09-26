"use client";

import React, { useState, useEffect } from "react";

/* =====================================================
   TYPES
===================================================== */
type CatalogItem = {
  id: string;
  platform: string;
  category: string;
  assetClass: string;
  subClass?: string;
  type: string;
  code: string;
  name: string;
  description: string;
  aliases?: string[];
  tags?: string[];
  region?: string;
  overlaps?: string[];
  source?: string;
};

/* =====================================================
   CATALOG FILTERS
===================================================== */
function CatalogFilters({
  onChange,
  initial,
}: {
  onChange: (f: any) => void;
  initial?: any;
}) {
  const [platform, setPlatform] = useState(initial?.platform ?? "All");
  const [category, setCategory] = useState(initial?.category ?? "All");
  const [search, setSearch] = useState(initial?.search ?? "");
  const [showUniqueOnly, setShowUniqueOnly] = useState(
    initial?.showUniqueOnly ?? false
  );

  function emit(next: Partial<any>) {
    const merged = { platform, category, search, showUniqueOnly, ...next };
    onChange(merged);
  }

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg p-3 flex flex-wrap gap-4 items-end">
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
        </select>
      </div>

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

/* =====================================================
   CATALOG SEARCH
===================================================== */
function CatalogSearch({
  onChange,
  onSubmit,
  initialQuery = "",
}: {
  onChange?: (q: string) => void;
  onSubmit?: (q: string) => void;
  initialQuery?: string;
}) {
  const [q, setQ] = useState(initialQuery);

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg p-3 flex items-center gap-2">
      <input
        className="flex-1 bg-transparent outline-none text-sm text-gray-200 placeholder:text-gray-500"
        placeholder="Search functions, codes, tags…"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          onChange?.(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSubmit?.(q);
        }}
      />
      <button
        onClick={() => onSubmit?.(q)}
        className="bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs px-3 py-1 rounded"
      >
        Search
      </button>
    </div>
  );
}

/* =====================================================
   CATALOG RESULTS
===================================================== */
function CatalogResults({
  items,
  search,
  onSelect,
}: {
  items: CatalogItem[];
  search: string;
  onSelect: (it: CatalogItem) => void;
}) {
  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {items.length === 0 ? (
        <div className="p-6 text-sm text-gray-400">No results.</div>
      ) : (
        items.map((it) => (
          <button
            key={it.id}
            onClick={() => onSelect(it)}
            className="w-full text-left p-3 hover:bg-[#101010] border-b border-[#222]"
          >
            <div className="text-sm font-semibold text-gray-100">{it.code}</div>
            <div className="text-xs text-gray-400">{it.name}</div>
            <div className="text-xs text-gray-500 line-clamp-2">
              {it.description}
            </div>
          </button>
        ))
      )}
    </div>
  );
}

/* =====================================================
   ITEM DETAILS
===================================================== */
function ItemDetails({
  item,
  onAddToCompare,
  onClose,
}: {
  item: CatalogItem | null;
  onAddToCompare: (id: string) => void;
  onClose: () => void;
}) {
  if (!item) return <div className="text-sm text-gray-400">Select an item</div>;
  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg p-4">
      <div className="flex justify-between items-center mb-2">
        <div>
          <div className="text-sm text-gray-400">
            {item.platform} • {item.category}
          </div>
          <div className="text-lg text-gray-100 font-semibold">
            {item.code} — {item.name}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            className="bg-gray-700 hover:bg-gray-600 text-xs px-2 py-1 rounded"
            onClick={() => onAddToCompare(item.id)}
          >
            + Compare
          </button>
          <button
            className="bg-gray-700 hover:bg-gray-600 text-xs px-2 py-1 rounded"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
      <p className="text-sm text-gray-300">{item.description}</p>
    </div>
  );
}

/* =====================================================
   COMPARE DRAWER
===================================================== */
function CompareDrawer({
  open,
  items,
  onClose,
}: {
  open: boolean;
  items: CatalogItem[];
  onClose: () => void;
}) {
  return (
    <div
      className={`fixed right-0 top-0 h-full w-[400px] bg-[#0b0b0b] border-l border-[#222] transition-transform ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <div className="p-3 flex justify-between border-b border-[#222]">
        <div className="text-sm text-gray-300">Compare ({items.length})</div>
        <button
          onClick={onClose}
          className="bg-gray-700 hover:bg-gray-600 text-xs px-2 py-1 rounded"
        >
          Close
        </button>
      </div>
      <div className="p-3 space-y-2 overflow-y-auto">
        {items.map((it) => (
          <div key={it.id} className="text-xs text-gray-300">
            {it.code} — {it.name}
          </div>
        ))}
      </div>
    </div>
  );
}

/* =====================================================
   PAGE
===================================================== */
export default function CatalogPage() {
  const [filters, setFilters] = useState<any>({
    platform: "All",
    category: "All",
    search: "",
    showUniqueOnly: false,
  });
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<CatalogItem[]>([]);
  const [selected, setSelected] = useState<CatalogItem | null>(null);
  const [compare, setCompare] = useState<CatalogItem[]>([]);
  const [drawer, setDrawer] = useState(false);

  // simulate fetch
  useEffect(() => {
    if (!query) return;
    const mock: CatalogItem[] = [
      {
        id: "bloomberg.EASY",
        platform: "Bloomberg",
        category: "Extra",
        assetClass: "Utilities",
        type: "function",
        code: "EASY",
        name: "Shortcut Panel",
        description: "Keyboard shortcuts and quick tips.",
      },
      {
        id: "koyfin.MACRO",
        platform: "Koyfin",
        category: "Analytics",
        assetClass: "Macro",
        type: "feature",
        code: "MACRO",
        name: "Macro Dashboard",
        description: "Charts & data for macro trends.",
      },
    ];
    setItems(mock.filter((it) => it.code.toLowerCase().includes(query.toLowerCase())));
  }, [query, filters]);

  return (
    <div className="flex flex-col gap-4 p-6">
      <CatalogSearch
        onChange={(q) => setFilters({ ...filters, search: q })}
        onSubmit={(q) => setQuery(q)}
      />
      <CatalogFilters initial={filters} onChange={setFilters} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <CatalogResults
            items={items}
            search={query}
            onSelect={(it) => setSelected(it)}
          />
        </div>
        <div>
          <ItemDetails
            item={selected}
            onAddToCompare={(id) => {
              const found = items.find((i) => i.id === id);
              if (found && !compare.some((c) => c.id === id)) {
                setCompare([...compare, found]);
                setDrawer(true);
              }
            }}
            onClose={() => setSelected(null)}
          />
        </div>
      </div>

      <CompareDrawer open={drawer} items={compare} onClose={() => setDrawer(false)} />
    </div>
  );
}