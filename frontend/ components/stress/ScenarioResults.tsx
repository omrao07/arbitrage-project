// frontend/components/ScenarioResults.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell,
} from "recharts";

/* ------------------------------- Types ------------------------------- */

export type ScenarioPathRow = { t: number; pnl: number };

export type ScenarioResult = {
  id: string;                   // stable key
  name: string;                 // display name
  path: ScenarioPathRow[];      // PnL over time (currency units)
  meta?: Record<string, any>;
  factors?: {                   // optional contributions: factor -> {t,v}[]
    [factor: string]: { t: number; v: number }[];
  };
};

interface Props {
  /** If provided, fetch is skipped. Expect JSON: ScenarioResult[] or ScenarioResult. */
  results?: ScenarioResult[] | ScenarioResult | null;
  /** GET endpoint returning ScenarioResult[] (or single) if results not provided */
  endpoint?: string;            // default: /api/scenario/results
  /** Which scenarios to preselect & plot (by id). If omitted, selects all. */
  preselectIds?: string[];
  height?: number;              // main chart height
  baseCurrency?: string;        // $ labeling
}

/* ------------------------------ Component ---------------------------- */

export default function ScenarioResults({
  results,
  endpoint = "/api/scenario/results",
  preselectIds,
  height = 320,
  baseCurrency = "USD",
}: Props) {
  const [data, setData] = useState<ScenarioResult[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>(preselectIds ?? []);

  /* load / normalize */
  useEffect(() => {
    if (results) {
      const arr = Array.isArray(results) ? results : [results];
      setData(normalize(arr));
      return;
    }
    (async () => {
      try {
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = await res.json();
        const arr: ScenarioResult[] = Array.isArray(json) ? json : [json];
        setData(normalize(arr));
      } catch (e: any) {
        setErr(e?.message || "Failed to load scenario results");
        setData([]);
      }
    })();
  }, [results, endpoint]);

  /* selection defaults */
  useEffect(() => {
    if (!data.length) return;
    if (!selected.length) setSelected(data.map((d) => d.id));
  }, [data]);

  /* computed rows for selected scenarios */
  const sel = useMemo(
    () => data.filter((d) => selected.includes(d.id)),
    [data, selected]
  );

  const merged = useMemo(() => alignByTime(sel), [sel]);        // { t, name1, name2, ... }
  const kpis = useMemo(() => sel.map((s) => ({ ...statsFromPath(s.path), id: s.id, name: s.name })), [sel]);
  const ddSeries = useMemo(() => buildDrawdownSeries(sel), [sel]);
  const dist = useMemo(() => buildDistribution(sel), [sel]);

  /* factor contributions (optional) */
  const factorKeys = useMemo(() => {
    const set = new Set<string>();
    sel.forEach((s) => Object.keys(s.factors ?? {}).forEach((f) => set.add(f)));
    return Array.from(set).slice(0, 8); // render up to 8
  }, [sel]);

  /* actions */
  function toggle(id: string) {
    setSelected((xs) => (xs.includes(id) ? xs.filter((x) => x !== id) : [...xs, id]));
  }
  function importJSON(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        const arr: ScenarioResult[] = Array.isArray(parsed) ? parsed : [parsed];
        setData(normalize([...data, ...arr]));
      } catch {
        alert("Invalid results JSON");
      }
    };
    reader.readAsText(file);
  }
  function exportCSV() {
    if (!merged.length) return;
    const headers = ["t", ...sel.map((s) => s.name)];
    const rows = merged.map((r) => [r.t, ...sel.map((s) => r[s.name] ?? "")]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    download(`scenario_results_${Date.now()}.csv`, csv, "text/csv;charset=utf-8;");
  }

  /* UI */
  if (err) return <div className="rounded-xl border p-3 text-sm text-red-600">Error: {err}</div>;
  if (!data.length) return <div className="rounded-xl border p-3 text-sm opacity-70">No scenario results yet.</div>;

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Scenario Results</h2>
          <div className="text-xs opacity-70">
            Loaded: {data.length} • Selected: {sel.length}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportCSV}>
            Export CSV
          </button>
          <label className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800 cursor-pointer">
            Import
            <input type="file" accept="application/json" className="hidden" onChange={(e) => {
              const f = e.target.files?.[0]; if (f) importJSON(f);
            }} />
          </label>
        </div>
      </header>

      {/* Selector */}
      <div className="rounded-xl border dark:border-gray-800 mb-3">
        <div className="p-3 text-sm font-medium">Compare</div>
        <div className="p-3 pt-0 flex flex-wrap gap-2">
          {data.map((s, i) => (
            <button
              key={s.id}
              onClick={() => toggle(s.id)}
              className={`px-2 py-1 rounded-md border text-xs ${selected.includes(s.id) ? "bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700" : "border-gray-300 dark:border-gray-700"}`}
              title={s.name}
            >
              <span className="inline-block w-2 h-2 mr-2 rounded-sm" style={{ backgroundColor: lineColor(i) }} />
              {s.name}
            </button>
          ))}
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid gap-3 md:grid-cols-3 mb-4">
        {kpis.map((k, i) => (
          <div key={k.id} className="rounded-xl border dark:border-gray-800 p-3">
            <div className="text-sm font-semibold mb-1" title={k.id}>
              <span className="inline-block w-2 h-2 mr-2 rounded-sm" style={{ backgroundColor: lineColor(i) }} />
              {k.name}
            </div>
            <div className="grid grid-cols-2 text-sm gap-y-1">
              <span className="opacity-70">Total PnL</span><b className={k.total >= 0 ? "text-green-600" : "text-red-600"}>{money(k.total, baseCurrency)}</b>
              <span className="opacity-70">Peak DD</span><b className="text-red-600">{money(k.peakDD, baseCurrency)}</b>
              <span className="opacity-70">Avg Daily</span><b>{money(k.avgDaily, baseCurrency)}</b>
              <span className="opacity-70">Hit Ratio</span><b>{(k.hitRatio * 100).toFixed(1)}%</b>
            </div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid gap-4 md:grid-cols-2 mb-4">
        {/* PnL curve */}
        <div className="rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">PnL Over Time</div>
          <div style={{ height }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={merged}>
                <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
                <YAxis />
                <Tooltip
                  labelFormatter={(t) => new Date(t as number).toLocaleString()}
                  formatter={(v, n) => [money(v as number, baseCurrency), n as string]}
                />
                <Legend />
                {sel.map((s, i) => (
                  <Line key={s.id} type="monotone" dataKey={s.name} dot={false} stroke={lineColor(i)} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Drawdown */}
        <div className="rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">Drawdown</div>
          <div style={{ height }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ddSeries}>
                <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
                <YAxis />
                <Tooltip labelFormatter={(t) => new Date(t as number).toLocaleString()} formatter={(v, n) => [money(v as number, baseCurrency), n as string]} />
                <Legend />
                {sel.map((s, i) => (
                  <Area
                    key={s.id}
                    type="monotone"
                    dataKey={`${s.name}_dd`}
                    stroke={lineColor(i)}
                    fill={lineColor(i)}
                    fillOpacity={0.25}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Distribution */}
      <div className="rounded-xl border dark:border-gray-800 p-3 mb-4">
        <div className="text-sm font-medium mb-2">PnL Distribution (daily)</div>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={dist.bins}>
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip formatter={(v) => Number(v as number).toFixed(0)} />
              {sel.map((s, i) => (
                <Bar key={s.id} dataKey={s.name} stackId="a" fill={lineColor(i)} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[11px] opacity-70 mt-1">Binned by daily PnL; stacked for multiple scenarios.</div>
      </div>

      {/* Factor contributions (optional) */}
      {factorKeys.length > 0 && (
        <div className="rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">Factor Contributions (if provided)</div>
          <div className="grid gap-4 md:grid-cols-2">
            {factorKeys.map((f, idx) => (
              <div key={f} className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mergeFactor(sel, f)}>
                    <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(t) => new Date(t as number).toLocaleString()}
                      formatter={(v, n) => [money(v as number, baseCurrency), n as string]}
                    />
                    <Legend />
                    {sel.map((s, i) => (
                      <Line key={s.id} type="monotone" dataKey={s.name} stroke={lineColor(i)} dot={false} isAnimationActive={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <div className="text-xs mt-1 opacity-80">{f}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* --------------------------------- Utils ------------------------------- */

function normalize(arr: ScenarioResult[]) {
  return arr.map((s, idx) => ({
    id: s.id ?? `scn_${idx}_${Math.random().toString(36).slice(2)}`,
    name: s.name ?? `Scenario ${idx + 1}`,
    path: Array.isArray(s.path) ? s.path : [],
    meta: s.meta ?? {},
    factors: s.factors ?? undefined,
  }));
}

/** Align multiple scenario paths by timestamp to a single array of rows. */
function alignByTime(scenarios: ScenarioResult[]) {
  const tsSet = new Set<number>();
  scenarios.forEach((s) => s.path.forEach((r) => tsSet.add(r.t)));
  const ts = Array.from(tsSet).sort((a, b) => a - b);
  return ts.map((t) => {
    const row: Record<string, any> = { t };
    scenarios.forEach((s) => {
      // last-known value carry-forward (so lines look continuous)
      row[s.name] = carryForwardAt(s.path, t);
    });
    return row;
  });
}

/** Build drawdown series for each scenario; keys like "<name>_dd" (negative values). */
function buildDrawdownSeries(scenarios: ScenarioResult[]) {
  const tsSet = new Set<number>();
  scenarios.forEach((s) => s.path.forEach((r) => tsSet.add(r.t)));
  const ts = Array.from(tsSet).sort((a, b) => a - b);

  // precompute cumulative PnL and running peak → drawdown
  const mapDD: Record<string, { t: number; dd: number }[]> = {};
  for (const s of scenarios) {
    let cum = 0;
    let peak = 0;
    mapDD[s.name] = ts.map((t) => {
      const v = valueAt(s.path, t) ?? 0;
      cum = v;                         // path already PnL-to-date; if it was increments, replace with cum += v;
      peak = Math.max(peak, cum);
      const dd = Math.min(0, cum - peak);
      return { t, dd };
    });
  }
  // merge
  return ts.map((t, i) => {
    const row: Record<string, any> = { t };
    for (const s of scenarios) {
      row[`${s.name}_dd`] = mapDD[s.name][i].dd;
    }
    return row;
  });
}

/** Daily PnL histogram per scenario, stacked. */
function buildDistribution(scenarios: ScenarioResult[], bins = 20) {
  // collect day changes (diffs)
  const diffs: Record<string, number[]> = {};
  for (const s of scenarios) {
    diffs[s.name] = [];
    for (let i = 1; i < s.path.length; i++) {
      diffs[s.name].push(s.path[i].pnl - s.path[i - 1].pnl);
    }
  }
  // shared bin edges
  const all = Object.values(diffs).flat();
  const min = Math.min(...all, 0);
  const max = Math.max(...all, 0);
  const step = (max - min || 1) / bins;
  const edges = Array.from({ length: bins + 1 }, (_, i) => min + i * step);

  // form bin table: { label, name1, name2, ... }
  const binRows: Record<string, any>[] = [];
  for (let b = 0; b < bins; b++) {
    const x1 = edges[b], x2 = edges[b + 1];
    const label = `${money(x1, "")}..${money(x2, "")}`;
    const row: Record<string, any> = { label };
    for (const s of scenarios) {
      const arr = diffs[s.name];
      row[s.name] = arr.filter((v) => v >= x1 && (b === bins - 1 ? v <= x2 : v < x2)).length;
    }
    binRows.push(row);
  }
  return { bins: binRows };
}

/** Factor contribution merge by factor name; returns rows { t, name1, name2, ... } */
function mergeFactor(scenarios: ScenarioResult[], factor: string) {
  const tsSet = new Set<number>();
  scenarios.forEach((s) => {
    const series = s.factors?.[factor] ?? [];
    series.forEach((r) => tsSet.add(r.t));
  });
  const ts = Array.from(tsSet).sort((a, b) => a - b);
  return ts.map((t) => {
    const row: Record<string, any> = { t };
    scenarios.forEach((s) => {
      row[s.name] = carryForwardAt(s.factors?.[factor] ?? [], t);
    });
    return row;
  });
}

function carryForwardAt(series: { t: number; pnl?: number; v?: number }[], t: number) {
  // get value at or last before t
  let val: number | null = null;
  for (let i = 0; i < series.length; i++) {
    const row = series[i] as any;
    if (row.t <= t) {
      val = (row.pnl ?? row.v ?? 0);
    } else break;
  }
  return val ?? 0;
}
function valueAt(series: { t: number; pnl: number }[], t: number) {
  let val: number | null = null;
  for (let i = 0; i < series.length; i++) {
    const row = series[i];
    if (row.t <= t) val = row.pnl; else break;
  }
  return val;
}

function money(x: number, ccy = "USD") {
  try {
    if (!ccy) return (x >= 0 ? "+" : "") + x.toFixed(0);
    return x.toLocaleString(undefined, { style: "currency", currency: ccy });
  } catch { return `${ccy} ${x.toFixed(2)}`; }
}
function miniDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}
function lineColor(i: number) {
  const colors = ["#2563eb","#16a34a","#f59e0b","#ef4444","#a855f7","#06b6d4","#84cc16","#e11d48","#0ea5e9","#22c55e"];
  return colors[i % colors.length];
}
function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}

function statsFromPath(path: ScenarioPathRow[]): any {
    throw new Error("Function not implemented.");
}
