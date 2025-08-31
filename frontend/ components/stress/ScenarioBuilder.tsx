// frontend/components/ScenarioBuilder.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from "recharts";

/* ------------------------------ Types ------------------------------ */

type Decay = "none" | "linear" | "exponential";
type ShockMode = "percent" | "absolute";

export type Shock = {
  id: string;
  label: string;          // e.g., "SPX", "10Y", "Oil", "INRUSD"
  mode: ShockMode;        // percent or absolute
  value: number;          // +5 (% or units)
  start: string;          // ISO date (YYYY-MM-DD)
  end: string;            // ISO date
  decay: Decay;           // how the shock fades
  pinned?: boolean;       // prevents edits (optional)
};

export type Scenario = {
  name: string;
  description?: string;
  resolution: "1d" | "1h";
  from: string;           // ISO date
  to: string;             // ISO date
  shocks: Shock[];
};

type SimRequest = {
  scenario: Scenario;
  portfolio?: Record<string, number>; // optional positions by symbol
};

type SimResponse = {
  path: { t: number; pnl: number }[];     // portfolio PnL over time
  factorPaths?: Record<string, { t: number; v: number }[]>; // optional from server
  meta?: Record<string, any>;
};

interface Props {
  simulateEndpoint?: string;  // POST -> SimRequest => SimResponse
  seed?: Partial<Scenario>;
}

/* ---------------------------- Component ---------------------------- */

export default function ScenarioBuilder({
  simulateEndpoint = "/api/scenario/simulate",
  seed,
}: Props) {
  // --- scenario state
  const [name, setName] = useState(seed?.name ?? "What-if: Fed 50bps surprise");
  const [description, setDescription] = useState(seed?.description ?? "");
  const [resolution, setResolution] = useState<Scenario["resolution"]>(seed?.resolution ?? "1d");
  const [from, setFrom] = useState(seed?.from ?? isoDaysAgo(120));
  const [to, setTo] = useState(seed?.to ?? isoDaysAgo(0));
  const [shocks, setShocks] = useState<Shock[]>(
    seed?.shocks ?? [
      mkShock("SPX", "percent", -5),
      mkShock("10Y", "absolute", +0.25, "linear"),
    ]
  );

  // --- simulation state
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [resp, setResp] = useState<SimResponse | null>(null);

  // derived
  const scenario: Scenario = useMemo(
    () => ({ name, description, resolution, from, to, shocks }),
    [name, description, resolution, from, to, shocks]
  );

  const ticks = useMemo(() => makeTimeGrid(from, to, resolution), [from, to, resolution]);

  // local preview of factor paths (client-side generator)
  const factorPreview = useMemo(() => {
    const out: Record<string, { t: number; v: number }[]> = {};
    for (const s of shocks) {
      out[s.label] = buildShockPath(ticks, s);
    }
    return out;
  }, [shocks, ticks]);

  // quick actions
  function addShock() {
    setShocks((xs) => [...xs, mkShock("")]);
  }
  function updateShock(i: number, patch: Partial<Shock>) {
    setShocks((xs) => xs.map((x, k) => (k === i ? { ...x, ...patch } : x)));
  }
  function removeShock(i: number) {
    setShocks((xs) => xs.filter((_, k) => k !== i));
  }
  function normalizeDates() {
    // align shocks to scenario window if outside
    setShocks((xs) =>
      xs.map((s) => ({
        ...s,
        start: s.start < from ? from : s.start,
        end: s.end > to ? to : s.end,
      }))
    );
  }

  async function simulate() {
    try {
      setBusy(true); setErr(null); setResp(null);
      const body: SimRequest = { scenario };
      const res = await fetch(simulateEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = (await res.json()) as SimResponse;
      setResp(json);
    } catch (e: any) {
      setErr(e?.message || "Simulation failed");
    } finally {
      setBusy(false);
    }
  }

  function exportScenario() {
    download(`${slug(name)}.scenario.json`, JSON.stringify(scenario, null, 2), "application/json");
  }
  function importScenario(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result)) as Scenario;
        setName(parsed.name ?? name);
        setDescription(parsed.description ?? "");
        setResolution(parsed.resolution ?? "1d");
        setFrom(parsed.from ?? from);
        setTo(parsed.to ?? to);
        setShocks(parsed.shocks ?? []);
      } catch (e) {
        alert("Invalid scenario JSON");
      }
    };
    reader.readAsText(file);
  }
  function exportResultsCSV() {
    if (!resp?.path?.length) return;
    const headers = ["t", "pnl"];
    const rows = resp.path.map((r) => [r.t, r.pnl]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    download(`${slug(name)}_pnl.csv`, csv, "text/csv;charset=utf-8;");
  }

  /* ------------------------------- UI -------------------------------- */

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Scenario Builder</h2>
          <p className="text-xs opacity-70">Compose market shocks, preview factor paths, and simulate PnL.</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportScenario}>
            Export Scenario
          </button>
          <label className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800 cursor-pointer">
            Import
            <input type="file" accept="application/json" className="hidden" onChange={(e) => {
              const f = e.target.files?.[0]; if (f) importScenario(f);
            }} />
          </label>
          <button
            className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800 disabled:opacity-50"
            onClick={simulate} disabled={busy}
          >
            {busy ? "Simulating…" : "Run Simulation"}
          </button>
        </div>
      </header>

      {/* Scenario meta */}
      <div className="grid gap-3 md:grid-cols-3 mb-3">
        <input
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          placeholder="Scenario name"
          value={name} onChange={(e) => setName(e.target.value)}
        />
        <select
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          value={resolution} onChange={(e) => setResolution(e.target.value as Scenario["resolution"])}
        >
          <option value="1d">Daily</option>
          <option value="1h">Hourly</option>
        </select>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-70">Window</label>
          <input type="date" value={from} onChange={(e) => setFrom(e.target.value)}
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm" />
          <span className="opacity-60">→</span>
          <input type="date" value={to} onChange={(e) => setTo(e.target.value)}
            className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm" />
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={normalizeDates}>Clamp</button>
        </div>
      </div>

      {/* Shocks table */}
      <div className="rounded-xl border dark:border-gray-800 overflow-x-auto mb-4">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th className="p-2 text-left">Factor / Asset</th>
              <th className="p-2 text-left">Mode</th>
              <th className="p-2 text-right">Value</th>
              <th className="p-2 text-left">Decay</th>
              <th className="p-2 text-left">Start</th>
              <th className="p-2 text-left">End</th>
              <th className="p-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {shocks.map((s, i) => (
              <tr key={s.id} className="border-t dark:border-gray-800">
                <td className="p-2">
                  <input
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-44"
                    value={s.label}
                    onChange={(e) => updateShock(i, { label: e.target.value })}
                    placeholder="e.g., SPX, 10Y, OIL, INRUSD"
                  />
                </td>
                <td className="p-2">
                  <select
                    className="px-2 py-1 rounded-md border dark:border-gray-800"
                    value={s.mode}
                    onChange={(e) => updateShock(i, { mode: e.target.value as ShockMode })}
                  >
                    <option value="percent">Percent</option>
                    <option value="absolute">Absolute</option>
                  </select>
                </td>
                <td className="p-2 text-right">
                  <input
                    type="number" step="0.01"
                    className="px-2 py-1 rounded-md border dark:border-gray-800 w-28 text-right"
                    value={s.value}
                    onChange={(e) => updateShock(i, { value: Number(e.target.value) })}
                  />
                </td>
                <td className="p-2">
                  <select
                    className="px-2 py-1 rounded-md border dark:border-gray-800"
                    value={s.decay}
                    onChange={(e) => updateShock(i, { decay: e.target.value as Decay })}
                  >
                    <option value="none">None</option>
                    <option value="linear">Linear</option>
                    <option value="exponential">Exponential</option>
                  </select>
                </td>
                <td className="p-2">
                  <input type="date" className="px-2 py-1 rounded-md border dark:border-gray-800"
                    value={s.start} onChange={(e) => updateShock(i, { start: e.target.value })} />
                </td>
                <td className="p-2">
                  <input type="date" className="px-2 py-1 rounded-md border dark:border-gray-800"
                    value={s.end} onChange={(e) => updateShock(i, { end: e.target.value })} />
                </td>
                <td className="p-2 text-right">
                  <button className="px-2 py-1 rounded-md border dark:border-gray-800"
                          onClick={() => removeShock(i)}>Remove</button>
                </td>
              </tr>
            ))}
            {shocks.length === 0 && (
              <tr><td className="p-3 text-center opacity-60" colSpan={7}>Add at least one shock</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="text-sm opacity-80">
          Shocks: <b>{shocks.length}</b>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={addShock}>
            Add shock
          </button>
        </div>
      </div>

      {/* Preview factor paths (client-side) */}
      <div className="rounded-xl border dark:border-gray-800 p-3 mb-4">
        <div className="text-sm font-medium mb-2">Factor shock paths (preview)</div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mergePaths(factorPreview, ticks)}>
              <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
              <YAxis />
              <Tooltip
                labelFormatter={(t) => new Date(t as number).toLocaleString()}
                formatter={(v, n) => [Number(v as number).toFixed(3), n as string]}
              />
              <Legend />
              {Object.keys(factorPreview).map((k, idx) => (
                <Line key={k} type="monotone" dataKey={k} dot={false} stroke={lineColor(idx)} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[11px] opacity-70 mt-1">
          Preview shows normalized shock trajectories per factor (0 baseline; +/− indicates applied move).
        </div>
      </div>

      {/* Simulation result (PnL) */}
      <div className="rounded-xl border dark:border-gray-800 p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-medium">Simulated PnL</div>
          <div className="flex items-center gap-2">
            {resp?.path?.length ? (
              <button className="px-2 py-1 rounded-md border dark:border-gray-800 text-sm" onClick={exportResultsCSV}>
                Export CSV
              </button>
            ) : null}
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={resp?.path ?? []}>
              <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} />
              <YAxis />
              <Tooltip labelFormatter={(t) => new Date(t as number).toLocaleString()} />
              <Area type="monotone" dataKey="pnl" stroke="#22c55e" fill="#86efac" fillOpacity={0.35} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        {err && <div className="mt-2 text-sm text-red-600">Error: {err}</div>}
        {!resp?.path?.length && !busy && (
          <div className="text-xs opacity-70 mt-2">Run simulation to see portfolio PnL under this scenario.</div>
        )}
      </div>
    </div>
  );
}

/* ----------------------------- Helpers ----------------------------- */

function mkShock(label: string, mode: ShockMode = "percent", value = 0, decay: Decay = "none"): Shock {
  const today = isoDaysAgo(0);
  return {
    id: Math.random().toString(36).slice(2),
    label,
    mode,
    value,
    start: today,
    end: today,
    decay,
  };
}

function isoDaysAgo(n: number) {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 10);
}

function makeTimeGrid(from: string, to: string, res: "1d" | "1h") {
  const out: number[] = [];
  const start = new Date(from + "T00:00:00Z").getTime();
  const end = new Date(to + "T23:59:59Z").getTime();
  const step = res === "1d" ? 24 * 3600 * 1000 : 3600 * 1000;
  for (let t = start; t <= end; t += step) out.push(t);
  return out;
}

/** Build a simple, unitless path for a single shock over ticks, centered at 0 baseline. */
function buildShockPath(ticks: number[], s: Shock) {
  const start = new Date(s.start + "T00:00:00Z").getTime();
  const end = new Date(s.end + "T23:59:59Z").getTime();
  const val = s.mode === "percent" ? s.value / 100 : s.value; // normalize
  const out: { t: number; v: number }[] = [];
  const activeIdx = ticks
    .map((t, i) => ({ t, i }))
    .filter(({ t }) => t >= start && t <= end)
    .map(({ i }) => i);

  const N = activeIdx.length;
  for (let i = 0; i < ticks.length; i++) {
    if (N === 0 || i < activeIdx[0] || i > activeIdx[N - 1]) {
      out.push({ t: ticks[i], v: 0 });
      continue;
    }
    const rel = (i - activeIdx[0]) / Math.max(1, N - 1); // 0..1 across window
    let amp = val;
    if (s.decay === "linear") amp = val * (1 - rel);
    else if (s.decay === "exponential") amp = val * Math.exp(-3 * rel); // ~5% at end
    out.push({ t: ticks[i], v: amp });
  }
  return out;
}

/** Merge factor paths into a single chart dataset keyed by factor labels. */
function mergePaths(paths: Record<string, { t: number; v: number }[]>, ticks: number[]) {
  return ticks.map((t, idx) => {
    const row: Record<string, any> = { t };
    for (const k of Object.keys(paths)) {
      row[k] = paths[k]?.[idx]?.v ?? 0;
    }
    return row;
  });
}

function miniDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

function lineColor(i: number) {
  const colors = ["#2563eb","#16a34a","#f59e0b","#ef4444","#a855f7","#06b6d4","#84cc16","#e11d48"];
  return colors[i % colors.length];
}

function slug(s: string) { return s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, ""); }
function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}