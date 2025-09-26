"use client";

/**
 * equities/loading.tsx
 * Zero-import skeleton loader for the Equities section.
 * - Full-screen dark background
 * - Animated shimmer placeholder blocks for header, tiles, chart
 * - Tailwind only
 */

export default function EquitiesLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="w-full max-w-5xl space-y-6 p-6">
        {/* Header skeleton */}
        <div className="flex items-center justify-between">
          <div className="h-6 w-40 animate-pulse rounded-md bg-neutral-800" />
          <div className="h-6 w-24 animate-pulse rounded-md bg-neutral-800" />
        </div>

        {/* Tiles skeleton */}
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-2 rounded-xl border border-neutral-800 bg-neutral-900 p-4">
              <div className="h-3 w-20 animate-pulse rounded bg-neutral-800" />
              <div className="h-5 w-28 animate-pulse rounded bg-neutral-700" />
            </div>
          ))}
        </div>

        {/* Chart skeleton */}
        <div className="h-64 animate-pulse rounded-xl border border-neutral-800 bg-neutral-900" />

        {/* Peers + Headlines skeleton */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {[0, 1].map((col) => (
            <div key={col} className="space-y-2 rounded-xl border border-neutral-800 bg-neutral-900 p-4">
              <div className="h-4 w-32 animate-pulse rounded bg-neutral-800" />
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-3 w-full animate-pulse rounded bg-neutral-800" />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}