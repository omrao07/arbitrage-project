"use client";

/**
 * risk/page.tsx
 * Zero-import, self-contained Risk Dashboard page.
 * Shows portfolio risk metrics and stress scenarios.
 * Tailwind-only. No external imports.
 */

export default function RiskPage() {
  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      {/* Header */}
      <header className="border-b border-neutral-800 px-6 py-4">
        <h1 className="text-xl font-semibold">Risk Dashboard</h1>
        <p className="mt-1 text-sm text-neutral-400">
          Key portfolio risk metrics, exposures, and stress tests.
        </p>
      </header>

      {/* Main content */}
      <main className="mx-auto max-w-7xl space-y-8 p-6">
        {/* Metrics grid */}
        <section>
          <h2 className="mb-3 text-sm font-semibold text-neutral-200">
            Core Metrics
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: "Value-at-Risk (95%)", value: "9.5M USD" },
              { label: "Expected Shortfall (95%)", value: "11.2M USD" },
              { label: "Gross Leverage", value: "3.2×" },
              { label: "Liquidity Coverage", value: "0.95×" },
              { label: "Top 5 Concentration", value: "47%" },
              { label: "Rates DV01", value: "-120k USD" },
              { label: "FX Delta", value: "0.9M USD" },
              { label: "Portfolio Beta", value: "1.08 β" },
            ].map((m) => (
              <MetricCard key={m.label} label={m.label} value={m.value} />
            ))}
          </div>
        </section>

        {/* Stress tests */}
        <section>
          <h2 className="mb-3 text-sm font-semibold text-neutral-200">
            Stress Scenarios
          </h2>
          <div className="overflow-x-auto rounded-xl border border-neutral-800 bg-neutral-900">
            <table className="w-full text-sm">
              <thead className="bg-neutral-950 text-[11px] uppercase text-neutral-500">
                <tr className="border-b border-neutral-800">
                  <th className="px-3 py-2 text-left">Scenario</th>
                  <th className="px-3 py-2 text-left">Shocked Vars</th>
                  <th className="px-3 py-2 text-right">PnL Impact</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800">
                {[
                  {
                    label: "Equities -20%",
                    vars: ["Equities", "Credit"],
                    impact: "-12%",
                  },
                  {
                    label: "Rates +100bps",
                    vars: ["Rates"],
                    impact: "-5%",
                  },
                  {
                    label: "USD +5%",
                    vars: ["FX"],
                    impact: "-2%",
                  },
                  {
                    label: "2008-like Shock",
                    vars: ["Equities", "Credit", "FX", "Rates"],
                    impact: "-25%",
                  },
                ].map((s) => (
                  <tr
                    key={s.label}
                    className="hover:bg-neutral-800/40 transition-colors"
                  >
                    <td className="px-3 py-2">{s.label}</td>
                    <td className="px-3 py-2 text-neutral-300">
                      {s.vars.join(", ")}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-red-400">
                      {s.impact}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

/* ------------------------------ Subcomponents ------------------------------ */

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4 shadow">
      <p className="text-xs text-neutral-400">{label}</p>
      <p className="mt-2 text-lg font-semibold text-neutral-100">{value}</p>
    </div>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;