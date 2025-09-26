"use client";

/**
 * app/crypto/loading.tsx
 * Zero-import loading screen for the Crypto section.
 * - Full-screen dark backdrop
 * - Inline SVG spinner (no icon libs)
 * - Optional skeleton rows
 */

export default function CryptoLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="w-full max-w-md">
        {/* Spinner + message */}
        <div className="mb-5 flex flex-col items-center gap-3">
          <Spinner />
          <p className="text-sm text-neutral-400">Loading crypto data…</p>
        </div>

        {/* Skeletons */}
        <div className="space-y-2">
          <SkeletonLine />
          <SkeletonLine w="w-5/6" />
          <SkeletonLine w="w-3/4" />
          <div className="grid grid-cols-3 gap-2 pt-2">
            <SkeletonTile />
            <SkeletonTile />
            <SkeletonTile />
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
      className="h-9 w-9 animate-spin text-emerald-500"
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

function SkeletonLine({ w = "w-full" as "w-full" | "w-5/6" | "w-3/4" | "w-2/3" }) {
  return <div className={`h-3 ${w} rounded bg-neutral-800/80`} />;
}

function SkeletonTile() {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
      <div className="mb-2 h-3 w-1/2 rounded bg-neutral-800/80" />
      <div className="h-5 w-3/4 rounded bg-neutral-800/80" />
    </div>
  );
}