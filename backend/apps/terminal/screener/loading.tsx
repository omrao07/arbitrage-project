"use client";

/**
 * screener/loading.tsx
 * Zero-import skeleton loader for Screener pages.
 * Tailwind-only. Shows placeholder filters + results table.
 */

export default function ScreenerLoading() {
  return (
    <div className="min-h-[80vh] w-full bg-neutral-950 px-6 py-8 text-neutral-200">
      {/* Header skeleton */}
      <div className="mb-6 space-y-2">
        <div className="h-6 w-40 rounded bg-neutral-800 animate-pulse" />
        <div className="h-4 w-80 rounded bg-neutral-900 animate-pulse" />
      </div>

      {/* Screener form skeleton */}
      <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="rounded-xl border border-neutral-800 bg-neutral-900 p-4 space-y-3"
          >
            <div className="h-4 w-24 rounded bg-neutral-800 animate-pulse" />
            <div className="h-9 w-full rounded bg-neutral-800 animate-pulse" />
            <div className="h-9 w-2/3 rounded bg-neutral-800 animate-pulse" />
          </div>
        ))}
      </div>

      {/* Results table skeleton */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900">
        <div className="border-b border-neutral-800 bg-neutral-950 px-4 py-2">
          <div className="h-4 w-32 rounded bg-neutral-800 animate-pulse" />
        </div>
        <div className="divide-y divide-neutral-800">
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={i}
              className="grid grid-cols-12 gap-2 px-4 py-3"
            >
              {Array.from({ length: 6 }).map((__, j) => (
                <div
                  key={j}
                  className="h-3 rounded bg-neutral-800 animate-pulse"
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;