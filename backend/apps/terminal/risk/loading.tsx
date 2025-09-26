"use client";

/**
 * risk/loading.tsx
 * Zero-import skeleton loader for the Risk section.
 * Tailwind-only. No external imports.
 */

export default function RiskLoading() {
  return (
    <div className="min-h-[60vh] w-full bg-neutral-950 px-6 py-8 text-neutral-200">
      {/* Header skeleton */}
      <div className="mb-6 space-y-2">
        <div className="h-6 w-44 rounded bg-neutral-800 animate-pulse" />
        <div className="h-4 w-72 rounded bg-neutral-900 animate-pulse" />
      </div>

      {/* Metrics grid skeleton */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="rounded-xl border border-neutral-800 bg-neutral-900 p-4"
          >
            <div className="mb-3 h-4 w-28 rounded bg-neutral-800 animate-pulse" />
            <div className="h-6 w-20 rounded bg-neutral-700 animate-pulse" />
          </div>
        ))}
      </div>

      {/* Stress test table skeleton */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900">
        <div className="border-b border-neutral-800 bg-neutral-950 px-4 py-2">
          <div className="h-4 w-36 rounded bg-neutral-800 animate-pulse" />
        </div>
        <div className="divide-y divide-neutral-800">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="grid grid-cols-12 gap-2 px-4 py-3">
              <div className="col-span-3 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-3 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;