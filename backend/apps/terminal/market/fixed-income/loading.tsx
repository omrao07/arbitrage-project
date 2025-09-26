/**
 * fixed income/loading.tsx
 * Zero-import skeleton loader for the Fixed Income section.
 * Tailwind-only. No links/imports.
 */

export default function FixedIncomeLoading() {
  return (
    <div className="min-h-[60vh] w-full bg-neutral-950 px-4 py-6 text-neutral-200">
      {/* Header skeleton */}
      <div className="mb-4 flex items-center justify-between">
        <div className="space-y-2">
          <div className="h-5 w-40 rounded bg-neutral-800 animate-pulse" />
          <div className="h-3 w-64 rounded bg-neutral-900 animate-pulse" />
        </div>
        <div className="flex items-center gap-2">
          <div className="h-8 w-24 rounded-md bg-neutral-900 animate-pulse" />
          <div className="h-8 w-24 rounded-md bg-neutral-900 animate-pulse" />
        </div>
      </div>

      {/* Content grid skeleton */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Left: big chart placeholder */}
        <CardSkeleton lines={6} />

        {/* Right: stacked panes */}
        <div className="space-y-4">
          <CardSkeleton lines={3} />
          <CardSkeleton lines={4} />
        </div>
      </div>

      {/* Table skeleton */}
      <div className="mt-4 overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
        <div className="border-b border-neutral-800 bg-neutral-950 px-3 py-2">
          <div className="h-4 w-28 rounded bg-neutral-800 animate-pulse" />
        </div>
        <div className="divide-y divide-neutral-800">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="grid grid-cols-12 gap-2 px-3 py-3">
              <div className="col-span-3 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-1 h-3 rounded bg-neutral-800 animate-pulse" />
              <div className="col-span-2 h-3 rounded bg-neutral-800 animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Skeleton Card ------------------------------ */

function CardSkeleton({ lines = 4 }: { lines?: number }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-3">
      <div className="mb-3 flex items-center justify-between">
        <div className="h-4 w-32 rounded bg-neutral-800 animate-pulse" />
        <div className="h-7 w-24 rounded bg-neutral-900 animate-pulse" />
      </div>
      <div className="h-48 w-full rounded-lg bg-neutral-950 animate-pulse" />
      <div className="mt-3 space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="h-3 w-full rounded bg-neutral-800 animate-pulse"
            style={{ width: `${80 - i * 6}%` }}
          />
        ))}
      </div>
    </div>
  );
}