"use client";

/**
 * fixed income/page.tsx
 * Zero-import, self-contained page container for Fixed Income section.
 * Combines chart panes, tables, and controls with a dark theme.
 * Tailwind-only. No imports/links.
 */

export default function FixedIncomePage() {
  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      {/* Page header */}
      <header className="border-b border-neutral-800 px-6 py-4">
        <h1 className="text-xl font-semibold">Fixed Income Dashboard</h1>
        <p className="mt-1 text-sm text-neutral-400">
          Explore bond markets, yield curves, and credit spreads.
        </p>
      </header>

      {/* Main content grid */}
      <main className="mx-auto max-w-7xl space-y-6 p-6">
        {/* Top row: Yield curve + Maturity ladder */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Panel title="Yield Curves">
            <div className="h-64 w-full rounded-lg bg-neutral-900" />
          </Panel>
          <Panel title="Maturity Ladder">
            <div className="h-64 w-full rounded-lg bg-neutral-900" />
          </Panel>
        </div>

        {/* Middle row: Spread timeseries */}
        <Panel title="Credit Spread Timeseries">
          <div className="h-72 w-full rounded-lg bg-neutral-900" />
        </Panel>

        {/* Bottom row: Issuer tree + Bonds table */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Panel title="Issuer Tree">
            <div className="h-72 w-full rounded-lg bg-neutral-900" />
          </Panel>
          <Panel title="Bonds Table">
            <div className="h-72 w-full rounded-lg bg-neutral-900" />
          </Panel>
        </div>
      </main>
    </div>
  );
}

/* ------------------------------ Reusable Panel ----------------------------- */
function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-950 p-4 shadow">
      <h2 className="mb-3 text-sm font-semibold text-neutral-200">{title}</h2>
      {children}
    </section>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;