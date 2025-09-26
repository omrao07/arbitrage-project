"use client";

/**
 * derivatives/loading.tsx
 * Minimal, zero-import loading screen for the Derivatives section.
 * - Full-screen dark backdrop
 * - Inline SVG spinner
 * - Skeleton tiles + table rows to hint layout
 */

export default function DerivativesLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="w-full max-w-5xl px-4">
        {/* Header */}
        <div className="mb-6 flex items-center gap-3">
          <Spinner />
          <div>
            <div className="h-4 w-40 rounded bg-neutral-800/80" />
            <div className="mt-2 h-3 w-56 rounded bg-neutral-800/60" />
          </div>
        </div>

        {/* Metric tiles */}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <TileSkeleton />
          <TileSkeleton />
          <TileSkeleton />
          <TileSkeleton />
        </div>

        {/* Two pane skeletons */}
        <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <PaneSkeleton titleWidth="w-40" />
          <PaneSkeleton titleWidth="w-32" />
        </div>

        {/* Table skeleton */}
        <div className="mt-6 overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
          <div className="border-b border-neutral-800 px-4 py-3">
            <div className="h-4 w-48 rounded bg-neutral-800/80" />
          </div>
          <div className="divide-y divide-neutral-800">
            {Array.from({ length: 6 }).map((_, i) => (
              <RowSkeleton key={i} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------- UI bits ---------------------------- */

function Spinner() {
  return (
    <svg
      className="h-8 w-8 animate-spin text-emerald-500"
      viewBox="0 0 24 24"
      aria-label="Loading"
      role="status"
    >
      <circle
        className="opacity-20"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
        fill="none"
      />
      <path
        className="opacity-90"
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

function TileSkeleton() {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="h-3 w-1/3 rounded bg-neutral-800/80" />
      <div className="mt-2 h-6 w-2/3 rounded bg-neutral-800/80" />
      <div className="mt-1 h-3 w-1/2 rounded bg-neutral-800/60" />
    </div>
  );
}

function PaneSkeleton({ titleWidth = "w-36" as string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className={`h-4 ${titleWidth} rounded bg-neutral-800/80`} />
      <div className="mt-3 h-40 rounded bg-neutral-800/50" />
      <div className="mt-3 grid grid-cols-3 gap-2">
        <div className="h-3 rounded bg-neutral-800/70" />
        <div className="h-3 rounded bg-neutral-800/70" />
        <div className="h-3 rounded bg-neutral-800/70" />
      </div>
    </div>
  );
}

function RowSkeleton() {
  return (
    <div className="grid grid-cols-6 gap-3 px-4 py-3">
      <div className="h-3 rounded bg-neutral-800/80" />
      <div className="h-3 rounded bg-neutral-800/80" />
      <div className="h-3 rounded bg-neutral-800/80" />
      <div className="h-3 rounded bg-neutral-800/80" />
      <div className="h-3 rounded bg-neutral-800/80" />
      <div className="h-3 rounded bg-neutral-800/80" />
    </div>
  );
}