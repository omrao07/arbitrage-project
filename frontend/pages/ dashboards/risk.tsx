// frontend/components/Risk.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  Legend,
} from "recharts";

type McSummary = {
  alpha: number;
  VaR: number;
  ES: number;
  mean_pnl: number;
  std_pnl: number;
  max_drawdown_est: number;
};

type RiskKpis = {
  nav?: number;
  var?: number;
  es?: number;
  dd?: number;
  vol?: number;
};

type ScenarioRow = {
  name: string;
  pnl: number;
  drivers?: Record<string, number>; // factor -> pnl
};

type TimeseriesPoint = { t: string; var: number; es: number; pnl: number };

export default function Risk() {
  const [kpis, setKpis] = useState<RiskKpis>({});
  const [mc, setMc] = useState<McSummary | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioRow[]>([]);
  const [hist, setHist] = useState<TimeseriesPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const [k, m, s, h] = await Promise.all([
          fetch("/api/risk/kpis").then((r) => r.json()),
          fetch("/api/risk/monte_carlo").then((r) => r.json()),
          fetch("/api/risk/scenarios").then((r) => r.json()),
          fetch("/api/risk/timeseries").then((r) => r.json()),
        ]);
        setKpis(k);
        setMc(m);
        setScenarios(s);
        setHist(h);
      } catch (e: any) {
        setErr(e?.message || "Risk API error");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const scenarioChartData = useMemo(() => {
    return scenarios.map((s) => ({ name: s.name, pnl: s.pnl }));
  }, [scenarios]);

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4">
        <h2 className="text-xl font-semibold">Risk Dashboard</h2>
        <p className="text-sm opacity-70">
          Live VaR/ES, stress scenarios, Monte Carlo summary
        </p>
      </header>

      {loading && <div className="text-sm opacity-70">Loading risk…</div>}
      {err && !loading && (
        <div className="text-sm text-red-500">Error: {err}</div>
      )}

      {/* KPI Row */}
      {!loading && !err && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
          <Kpi label="NAV" value={fmtUsd(kpis.nav)} />
          <Kpi label="VaR" value={fmtUsd(kpis.var)} />
          <Kpi label="ES" value={fmtUsd(kpis.es)} />
          <Kpi label="Drawdown" value={fmtPct(kpis.dd)} />
          <Kpi label="Volatility" value={fmtUsd(kpis.vol)} />
        </div>
      )}

      {/* VaR & ES over time */}
      {!loading && !err && (
        <section className="mb-6">
          <h3 className="text-lg font-medium mb-2">VaR / ES History</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={hist}>
              <XAxis dataKey="t" hide />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="var" name="VaR" stroke="#FF3B30" dot={false} />
              <Line type="monotone" dataKey="es" name="ES" stroke="#0A84FF" dot={false} />
              <Line type="monotone" dataKey="pnl" name="PnL" stroke="#34C759" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {/* Monte Carlo summary */}
      {!loading && !err && mc && (
        <section className="mb-6">
          <h3 className="text-lg font-medium mb-2">Monte Carlo Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
            <Kpi label={`Tail (α)`} value={`${Math.round((mc.alpha || 0) * 100)}%`} />
            <Kpi label="VaR (MC)" value={fmtUsd(mc.VaR)} />
            <Kpi label="ES (MC)" value={fmtUsd(mc.ES)} />
            <Kpi label="Mean PnL" value={fmtUsd(mc.mean_pnl)} />
            <Kpi label="Vol (σ)" value={fmtUsd(mc.std_pnl)} />
            <Kpi label="Max DD (est)" value={fmtUsd(mc.max_drawdown_est)} />
          </div>
        </section>
      )}

      {/* Scenario bar chart */}
      {!loading && !err && scenarios.length > 0 && (
        <section className="mb-6">
          <h3 className="text-lg font-medium mb-2">Stress Scenarios — PnL</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={scenarioChartData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="pnl" fill="#FF9500" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {/* Scenario table with top drivers if present */}
      {!loading && !err && (
        <section>
          <h3 className="text-lg font-medium mb-2">Scenario Details</h3>
          <div className="overflow-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="p-2 text-left">Scenario</th>
                  <th className="p-2 text-right">PnL</th>
                  <th className="p-2 text-left">Top Drivers</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s, i) => {
                  const drivers = topDrivers(s.drivers || {}, 3)
                    .map(([k, v]) => `${k}: ${fmtUsd(v)}`)
                    .join(", ");
                  return (
                    <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
                      <td className="p-2">{s.name}</td>
                      <td className={`p-2 text-right ${s.pnl < 0 ? "text-red-500" : "text-green-500"}`}>
                        {fmtUsd(s.pnl)}
                      </td>
                      <td className="p-2">{drivers || "—"}</td>
                    </tr>
                  );
                })}
                {scenarios.length === 0 && (
                  <tr><td className="p-2 opacity-60" colSpan={3}>No scenarios.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

/* ----------------------------- Small components ---------------------------- */

function Kpi({ label, value }: { label: string; value?: string }) {
  return (
    <div className="rounded-xl border dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800">
      <div className="text-xs opacity-70">{label}</div>
      <div className="text-lg font-semibold">{value ?? "—"}</div>
    </div>
  );
}

/* --------------------------------- Helpers -------------------------------- */

function fmtUsd(x?: number) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  try {
    return (x < 0 ? "-$" : "$") + Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 });
  } catch {
    return String(x);
  }
}
function fmtPct(x?: number) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}
function topDrivers(drivers: Record<string, number>, n: number) {
  return Object.entries(drivers)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, n);
}