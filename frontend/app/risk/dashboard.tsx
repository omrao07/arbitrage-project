'use client';
import React from 'react';
import CounterpartyExposure from './counterparty_exposure';
import AnalystPanel from './AnalystPanel';

// ✅ Corrected relative imports (2 levels up from app/risk to components)


export default function Dashboard() {
  return (
    <div className="flex h-screen w-screen flex-col bg-neutral-50 dark:bg-neutral-950">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <h1 className="text-base font-semibold">Trading & Risk Dashboard</h1>
        <div className="flex items-center gap-3 text-sm">
          <button className="px-3 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800">
            Refresh
          </button>
          <button className="px-3 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800">
            Settings
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 overflow-hidden p-3 grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-3 auto-rows-fr">
        {/* Analyst Panel */}
        <AnalystPanel className="h-full" />

        {/* Counterparty Exposure */}
        <CounterpartyExposure className="h-full" />

        {/* Placeholder for future panels */}
        <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4 flex items-center justify-center text-neutral-500">
          Add another panel here…
        </div>
      </main>
    </div>
  );
}