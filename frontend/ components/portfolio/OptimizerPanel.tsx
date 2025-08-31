// frontend/components/OptimizerPanel.tsx
import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
} from "recharts";

/* ------------------------------- Types ------------------------------- */

type AssetRow = {
  symbol: string;
  mu?: number | null;   // expected return (annualized, in %), optional
  sector?: string | null;
  cap?: number | null;  // max weight (%)
};

type Objective = "max_sharpe" | "min_var" | "target_return";

type OptimizeRequest = {
  assets: { symbol: string; mu?: number | null; sector?: string | null; cap?: number | null }[];
  cov?: number[][] | null;  // optional covariance (if not provided, backend estimates)
  rf?: number;              // risk-free (%)
  objective: Objective;
  targetReturn?: number | null; // % for target_return
  longOnly: boolean;
  grossLimit?: number | null;   // e.g. 100 for fully invested, >100 allows leverage
  sectorCaps?: Record<string, number>; // % caps by sector
};

type OptimizeResponse = {
  weights: Record<string, number>;       // fraction (0..1)
  risk: number;                           // stdev (%)
  ret: number;                            // expected (%)
  sharpe: number;
  frontier?: { risk: number; ret: number }[]; // optional efficient frontier
  diag?: Record<string, any>;             // optional solver diagnostics
};

interface Props {
  endpoint?: string; // POST -> OptimizeRequest, returns OptimizeResponse
  defaultAssets?: string[]; // seed symbols
}

/* ------------------------------ Component ---------------------------- */

export default function OptimizerPanel({
  endpoint = "/api/optimize",
  defaultAssets = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
}: Props) {
  const [rows, setRows] = useState<AssetRow[]>(
    defaultAssets.map((s) => ({ symbol: s, mu: null, sector: null, cap: null }))
  );
  const [objective, setObjective] = useState<Objective>("max_sharpe");
  const [targetReturn, setTargetReturn] = useState<number | "">(8);
  const [rf, setRf] = useState<number>(2);             // %
  const [longOnly, setLongOnly] = useState(true);
  const [grossLimit, setGrossLimit] = useState<number>(100); // %
  const [sectorCaps, setSectorCaps] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [resp, setResp] = useState<OptimizeResponse | null>(null);

  /* derived */
  const symbols = useMemo(() => rows.map((r) => r.symbol.trim()).filter(Boolean), [rows]);
  const weightsTable = useMemo(() => {
    if (!resp) return [];
    return Object.entries(resp.weights).map(([s, w]) => ({ symbol: s, weight: w * 100 }));
  }, [resp]);

  /* actions */
  function addRow() {
    setRows((r) => [...r, { symbol: "", mu: null, sector: null, cap: null }]);
  }
  function updateRow(i: number, patch: Partial<AssetRow>) {
    setRows((r) => r.map((row, idx) => (idx === i ? { ...row, ...patch } : row)));
  }
  function removeRow(i: number) {
    setRows((r) => r.filter((_, idx) => idx !== i));
  }
  function normalizeCaps() {
    setRows((r) => r.map((x) => ({ ...x, cap: null })));
    setSectorCaps({});
  }

  async function runOptimize() {
    try {
      setBusy(true); setErr(null);
      const payload: OptimizeRequest = {
        assets: rows.map(({ symbol, mu, sector, cap }) => ({
          symbol: symbol.trim().toUpperCase(),
          mu: isFiniteNum(mu) ? Number(mu) : null,
          sector: sector?.trim() || null,
          cap: isFiniteNum(cap) ? Number(cap) : null,
        })),
        cov: null,
        rf: Number(rf),
        objective,
        targetReturn: objective === "target_return" && targetReturn !== "" ? Number(targetReturn) : null,
        longOnly,
        grossLimit: Number(grossLimit),
        sectorCaps,
      };
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = (await res.json()) as OptimizeResponse;
      setResp(json);
    } catch (e: any) {
      setErr(e?.message || "Optimize failed");
      setResp(null);
    } finally {
      setBusy(false);
    }
  }

  function exportWeightsCSV() {
    if (!resp) return;
    const headers = ["symbol", "weight_pct"];
    const rows = Object.entries(resp.weights).map(([s, w]) => `${s},${(w * 100).toFixed(4)}`);
    download(`weights_${Date.now()}.csv`, [headers.join(","), ...rows].join("\n"), "text/csv;charset=utf-8;");
  }
  function exportConfigJSON() {
    const cfg = {
      objective, rf, longOnly, grossLimit, targetReturn: targetReturn || null, sectorCaps,
      assets: rows,
    };
    download(`optimizer_config_${Date.now()}.json`, JSON.stringify(cfg, null, 2), "application/json");
  }

  /* UI */
  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Portfolio Optimizer</h2>
          <p className="text-xs opacity-70">
            Select assets, choose objective, set constraints, and optimize.
          </p>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportConfigJSON}>
            Export Config
          </button>
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportWeightsCSV} disabled={!resp}>
            Export Weights
          </button>
        </div>
      </header>

      {/* Controls */}
      <div className="grid gap-3 md:grid-cols-3 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80 w-28">Objective</label>
          <select
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-full"
            value={objective}
            onChange={(e) => setObjective(e.target.value as Objective)}
          >
            <option value="max_sharpe">Max Sharpe</option>
            <option value="min_var">Min Variance</option>
            <option value="target_return">Target Return</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80 w-28">Risk-free %</label>
          <input
            type="number" step="0.01"
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-full"
            value={rf} onChange={(e) => setRf(Number(e.target.value))}
          />
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80 w-28">Gross limit %</label>
          <input
            type="number" step="0.1"
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-full"
            value={grossLimit} onChange={(e) => setGrossLimit(Number(e.target.value))}
            title="Sum of absolute weights (100 = fully invested)"
          />
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80 w-28">Long-only</label>
          <input type="checkbox" checked={longOnly} onChange={(e) => setLongOnly(e.target.checked)} />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-80 w-28">Target return %</label>
          <input
            type="number" step="0.01" disabled={objective !== "target_return"}
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-full disabled:opacity-50"
            value={targetReturn} onChange={(e) => setTargetReturn(e.target.value === "" ? "" : Number(e.target.value))}
          />
        </div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={normalizeCaps}>
            Clear Caps
          </button>
        </div>
      </div>

      {/* Assets table */}
      <div className="rounded-xl border dark:border-gray-800 overflow-x-auto mb-4">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th className="p-2 text-left">Symbol</th>
              <th className="p-2 text-right">μ (exp. %)</th>
              <th className="p-2 text-left">Sector</th>
              <th className="p-2 text-right">Cap %</th>
              <th className="p-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t dark:border-gray-800">
                <td className="p-2">
                  <input
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-36"
                    value={r.symbol}
                    onChange={(e) => updateRow(i, { symbol: e.target.value.toUpperCase() })}
                    placeholder="AAPL"
                  />
                </td>
                <td className="p-2 text-right">
                  <input
                    type="number" step="0.01"
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-28 text-right"
                    value={r.mu ?? ""} onChange={(e) => updateRow(i, { mu: e.target.value === "" ? null : Number(e.target.value) })}
                    placeholder="(optional)"
                  />
                </td>
                <td className="p-2">
                  <input
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-40"
                    value={r.sector ?? ""} onChange={(e) => updateRow(i, { sector: e.target.value })}
                    placeholder="Tech"
                  />
                </td>
                <td className="p-2 text-right">
                  <input
                    type="number" step="0.01"
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-28 text-right"
                    value={r.cap ?? ""} onChange={(e) => updateRow(i, { cap: e.target.value === "" ? null : Number(e.target.value) })}
                    placeholder="(opt.)"
                  />
                </td>
                <td className="p-2 text-right">
                  <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => removeRow(i)}>Remove</button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan={5} className="p-3 text-center opacity-60">Add at least one asset</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="text-sm opacity-80">
          Assets: <b>{symbols.length}</b>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={addRow}>Add asset</button>
          <button
            className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800 disabled:opacity-50"
            onClick={runOptimize} disabled={!symbols.length || busy}
          >
            {busy ? "Optimizing…" : "Optimize"}
          </button>
        </div>
      </div>

      {/* Results */}
      {err && <div className="rounded-md border border-red-300 text-red-700 p-2 mb-3 text-sm">Error: {err}</div>}

      {resp && (
        <>
          <div className="grid gap-4 md:grid-cols-3 mb-4">
            <div className="rounded-xl border dark:border-gray-800 p-3">
              <div className="text-sm font-medium mb-2">Weights</div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weightsTable}>
                    <XAxis dataKey="symbol" />
                    <YAxis />
                    <Tooltip formatter={(v) => `${Number(v as number).toFixed(2)}%`} />
                    <Bar dataKey="weight" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border dark:border-gray-800 p-3">
              <div className="text-sm font-medium mb-2">Risk vs Return</div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <XAxis type="number" dataKey="risk" name="Risk" unit="%" />
                    <YAxis type="number" dataKey="ret" name="Return" unit="%" />
                    <ZAxis range={[100, 100]} />
                    <Tooltip formatter={(v: any) => `${Number(v).toFixed(2)}%`} />
                    {/* current solution */}
                    <Scatter name="Solution" data={[{ risk: resp.risk, ret: resp.ret }]} fill="#ef4444" />
                    {/* frontier (if provided) */}
                    {resp.frontier?.length ? (
                      <Scatter name="Frontier" data={resp.frontier} fill="#10b981" />
                    ) : null}
                    <Legend />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div className="text-xs opacity-80 mt-2">
                Sharpe: <b>{resp.sharpe.toFixed(2)}</b> • Risk: <b>{resp.risk.toFixed(2)}%</b> • Return: <b>{resp.ret.toFixed(2)}%</b>
              </div>
            </div>

            <div className="rounded-xl border dark:border-gray-800 p-3">
              <div className="text-sm font-medium mb-2">Efficient Frontier (area)</div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={resp.frontier ?? []}>
                    <XAxis dataKey="risk" />
                    <YAxis />
                    <Tooltip formatter={(v) => `${Number(v as number).toFixed(2)}%`} />
                    <Area type="monotone" dataKey="ret" stroke="#16a34a" fill="#86efac" fillOpacity={0.35} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {resp.diag && (
            <details className="rounded-xl border dark:border-gray-800 p-3 text-xs">
              <summary className="cursor-pointer font-medium">Diagnostics</summary>
              <pre className="whitespace-pre-wrap mt-2">{JSON.stringify(resp.diag, null, 2)}</pre>
            </details>
          )}
        </>
      )}
    </div>
  );
}

/* --------------------------------- Utils ------------------------------- */

function isFiniteNum(x: any): x is number {
  return typeof x === "number" && Number.isFinite(x);
}
function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}