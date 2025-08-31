// frontend/components/CustomIndex.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  LineChart,
  Line,
} from "recharts";

/* ------------------------------- Types ------------------------------- */

type PricePoint = { t: number; // epoch ms
  [symbol: string]: number | undefined; // e.g. { AAPL: 233.1, MSFT: 425.2 }
};

type Constituent = {
  symbol: string;
  weight: number; // percent, 0..100
};

type Method = "value_weighted" | "equal_weighted" | "price_weighted";

interface Props {
  /** REST endpoint that returns historical prices aligned by time.
   *  Expected response: PricePoint[] where each row has { t, TICKER1, TICKER2, ... } */
  pricesEndpoint?: string; // default: /api/prices?symbols=AAPL,MSFT,GOOG&from=YYYY-MM-DD&to=YYYY-MM-DD&interval=1d
  defaultSymbols?: string[]; // convenience seed
  baseValue?: number; // initial index value (e.g., 100 or 1000)
  interval?: "1d" | "1h" | "5m";
}

export default function CustomIndex({
  pricesEndpoint = "/api/prices",
  defaultSymbols = ["AAPL", "MSFT", "GOOGL"],
  baseValue = 100,
  interval = "1d",
}: Props) {
  /* ------------------------------ State ------------------------------ */
  const [constituents, setConstituents] = useState<Constituent[]>(
    defaultSymbols.map((s) => ({ symbol: s, weight: +(100 / defaultSymbols.length).toFixed(2) }))
  );
  const [method, setMethod] = useState<Method>("value_weighted");
  const [from, setFrom] = useState<string>(isoDaysAgo(365));
  const [to, setTo] = useState<string>(isoDaysAgo(0));
  const [name, setName] = useState<string>("My Custom Index");
  const [prices, setPrices] = useState<PricePoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  /* --------------------------- Derived vals -------------------------- */

  // total weight (for validation)
  const totalWeight = useMemo(
    () => constituents.reduce((a, c) => a + (Number.isFinite(c.weight) ? c.weight : 0), 0),
    [constituents]
  );

  // query string for API
  const symbolList = constituents.map((c) => c.symbol.trim().toUpperCase()).filter(Boolean);
  const query = useMemo(() => {
    const u = new URLSearchParams({
      symbols: symbolList.join(","),
      from,
      to,
      interval,
    });
    return `${pricesEndpoint}?${u.toString()}`;
  }, [symbolList.join(","), from, to, interval, pricesEndpoint]);

  /* ------------------------------ Effects --------------------------- */

  useEffect(() => {
    if (symbolList.length === 0) { setPrices([]); return; }
    let ignore = false;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const res = await fetch(query);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as PricePoint[];
        if (!ignore) setPrices(json);
      } catch (e: any) {
        if (!ignore) { setErr(e?.message || "Failed to fetch prices"); setPrices([]); }
      } finally {
        if (!ignore) setLoading(false);
      }
    })();
    return () => { ignore = true; };
  }, [query]);

  /* -------------------------- Index calculus ------------------------- */

  const indexSeries = useMemo(() => {
    if (!prices || prices.length === 0 || symbolList.length === 0) return [];

    // normalize weights depending on method
    let weights: Record<string, number> = {};
    if (method === "equal_weighted") {
      const w = 1 / symbolList.length;
      symbolList.forEach((s) => (weights[s] = w));
    } else if (method === "price_weighted") {
      // equal shares is price-weighted; normalize by first row price
      const first = prices[0];
      let denom = 0;
      symbolList.forEach((s) => {
        const p = (first[s] as number) ?? 0;
        denom += p;
      });
      symbolList.forEach((s) => {
        const p = (first[s] as number) ?? 0;
        weights[s] = denom ? p / denom : 0;
      });
    } else {
      // value_weighted — use user weights (% → fraction)
      const total = constituents.reduce((a, c) => a + (c.weight || 0), 0) || 1;
      constituents.forEach((c) => (weights[c.symbol.toUpperCase()] = (c.weight || 0) / total));
    }

    // build returns and level
    const out: { t: number; level: number }[] = [];
    let level = baseValue;

    // compute daily (period) return as sum_i (w_i * r_i)
    for (let i = 0; i < prices.length; i++) {
      const row = prices[i];
      if (i === 0) { out.push({ t: row.t, level }); continue; }
      const prev = prices[i - 1];
      let r = 0;
      symbolList.forEach((s) => {
        const p0 = (prev[s] as number) ?? 0;
        const p1 = (row[s] as number) ?? 0;
        if (p0 > 0 && Number.isFinite(p1)) {
          r += (weights[s] || 0) * (p1 / p0 - 1);
        }
      });
      level = level * (1 + r);
      out.push({ t: row.t, level });
    }
    return out;
  }, [prices, method, constituents, baseValue, symbolList]);

  /* ------------------------------- UI -------------------------------- */

  function addRow() {
    setConstituents((rows) => [...rows, { symbol: "", weight: 0 }]);
  }
  function updateRow(i: number, patch: Partial<Constituent>) {
    setConstituents((rows) => rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }
  function removeRow(i: number) {
    setConstituents((rows) => rows.filter((_, idx) => idx !== i));
  }
  function normalizeWeights() {
    const nonEmpty = constituents.filter((c) => c.symbol.trim());
    if (nonEmpty.length === 0) return;
    const w = +(100 / nonEmpty.length).toFixed(4);
    setConstituents(nonEmpty.map((c) => ({ ...c, weight: w })));
  }
  function exportJSON() {
    const payload = { name, method, baseValue, from, to, interval, constituents };
    download(`index_${slug(name)}.json`, JSON.stringify(payload, null, 2), "application/json");
  }
  function exportCSV() {
    const headers = ["t","level"];
    const rows = indexSeries.map((r) => [r.t, r.level]);
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    download(`index_series_${slug(name)}.csv`, csv, "text/csv;charset=utf-8;");
  }

  const weightError =
    method === "value_weighted" && Math.abs(totalWeight - 100) > 0.001
      ? `Weights sum is ${totalWeight.toFixed(2)}% (must be 100%)`
      : null;

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <h2 className="text-lg font-semibold">Custom Index Builder</h2>
          <div className="text-xs opacity-70">
            Build & backtest a synthetic index from your chosen constituents.
          </div>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportJSON}>
            Export Config (JSON)
          </button>
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportCSV} disabled={!indexSeries.length}>
            Export Series (CSV)
          </button>
        </div>
      </header>

      {/* Config row */}
      <div className="grid gap-3 md:grid-cols-3 mb-4">
        <input
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          placeholder="Index name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80">Method</label>
          <select
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
            value={method}
            onChange={(e) => setMethod(e.target.value as Method)}
          >
            <option value="value_weighted">Value-weighted (use weights)</option>
            <option value="equal_weighted">Equal-weighted</option>
            <option value="price_weighted">Price-weighted</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80">Base</label>
          <input
            type="number"
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-28"
            value={baseValue}
            onChange={() => {/* baseValue is prop; adjust in parent if needed */}}
            readOnly
            title="Set via prop baseValue"
          />
        </div>
      </div>

      {/* Date + interval */}
      <div className="grid gap-3 md:grid-cols-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80">From</label>
          <input type="date" className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
            value={from} onChange={(e) => setFrom(e.target.value)} />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80">To</label>
          <input type="date" className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
            value={to} onChange={(e) => setTo(e.target.value)} />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80">Interval</label>
          <select className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
            value={interval} onChange={(e) => location.assign(updateQueryParam(location.href, "interval", e.target.value))}>
            <option value="1d">1d</option>
            <option value="1h">1h</option>
            <option value="5m">5m</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80">Symbols</label>
          <span className="text-xs opacity-70">{symbolList.join(", ") || "—"}</span>
        </div>
      </div>

      {/* Table of constituents */}
      <div className="rounded-xl border dark:border-gray-800 mb-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th className="p-2 text-left">Symbol</th>
              <th className="p-2 text-right">Weight %</th>
              <th className="p-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {constituents.map((c, i) => (
              <tr key={i} className="border-t dark:border-gray-800">
                <td className="p-2">
                  <input
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-40"
                    value={c.symbol}
                    onChange={(e) => updateRow(i, { symbol: e.target.value.toUpperCase() })}
                    placeholder="e.g., AAPL"
                  />
                </td>
                <td className="p-2 text-right">
                  <input
                    type="number"
                    step="0.01"
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-32 text-right"
                    value={c.weight}
                    onChange={(e) => updateRow(i, { weight: Number(e.target.value) })}
                    disabled={method !== "value_weighted"}
                    title={method !== "value_weighted" ? "Weights fixed by method" : ""}
                  />
                </td>
                <td className="p-2 text-right">
                  <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => removeRow(i)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
            {constituents.length === 0 && (
              <tr>
                <td className="p-3 text-center opacity-60" colSpan={3}>Add at least one symbol</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="text-sm opacity-80">
          Total weight: <b>{totalWeight.toFixed(2)}%</b>{" "}
          {weightError && <span className="text-red-600 ml-2">{weightError}</span>}
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={addRow}>Add symbol</button>
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={normalizeWeights}>Equal weights</button>
        </div>
      </div>

      {/* Charts */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-2 rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">{name} — Index Level</div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={indexSeries}>
                <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
                <YAxis />
                <Tooltip
                  labelFormatter={(t) => new Date(t as number).toLocaleString()}
                  formatter={(v) => Number(v as number).toFixed(2)}
                />
                <Area type="monotone" dataKey="level" stroke="#3b82f6" fill="#93c5fd" fillOpacity={0.35} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">Constituent Prices (preview)</div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prices ?? []}>
                <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
                <YAxis />
                <Tooltip labelFormatter={(t) => new Date(t as number).toLocaleString()} />
                <Legend />
                {symbolList.slice(0, 6).map((s, i) => (
                  <Line key={s} type="monotone" dataKey={s} dot={false} stroke={lineColor(i)} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] opacity-70 mt-1">
            Showing up to 6 symbols. Use the left chart for index level.
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 text-xs opacity-70">
        {loading ? "Loading prices…" : err ? `Error: ${err}` : prices?.length ? `Loaded ${prices.length} rows.` : "—"}
      </div>
    </div>
  );
}

/* -------------------------------- Utils ------------------------------- */

function isoDaysAgo(n: number) {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 10);
}
function miniDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}
function slug(s: string) { return s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, ""); }
function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}
function lineColor(i: number) {
  const colors = ["#2563eb","#16a34a","#f59e0b","#ef4444","#a855f7","#06b6d4","#84cc16","#e11d48"];
  return colors[i % colors.length];
}
function updateQueryParam(url: string, key: string, value: string) {
  const u = new URL(url); u.searchParams.set(key, value); return u.toString();
}