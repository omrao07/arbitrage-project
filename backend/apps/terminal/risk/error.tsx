"use client";

/**
 * risk/error.tsx
 * Zero-import error UI for the Risk section.
 * Tailwind-only. Pass an optional onRetry() to show a Retry button.
 */

export default function RiskError({
  error,
  onRetry,
}: {
  error?: Error | null;
  onRetry?: () => void;
}) {
  const msg =
    (error && (error.message || String(error))) ||
    "Something went wrong while loading risk data.";

  return (
    <div className="flex min-h-[60vh] items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="mx-auto w-full max-w-md rounded-xl border border-neutral-800 bg-neutral-900 p-6 shadow">
        {/* Icon */}
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-red-900/30 text-red-400">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M21 12A9 9 0 1 1 3 12a9 9 0 0 1 18 0Z" />
          </svg>
        </div>

        {/* Heading */}
        <h2 className="text-center text-lg font-semibold">Risk module failed to load</h2>
        <p className="mt-2 break-words text-center text-sm text-neutral-400">{msg}</p>

        {/* Actions */}
        <div className="mt-4 flex items-center justify-center gap-2">
          {onRetry && (
            <button
              onClick={onRetry}
              className="rounded-md border border-neutral-700 bg-neutral-900 px-4 py-2 text-sm font-medium text-neutral-200 hover:bg-neutral-800"
            >
              Retry
            </button>
          )}
          <button
            onClick={() => location.reload()}
            className="rounded-md border border-neutral-800 bg-neutral-950 px-4 py-2 text-sm text-neutral-300 hover:bg-neutral-800"
          >
            Refresh
          </button>
        </div>

        {/* Footnote */}
        <p className="mt-3 text-center text-[11px] leading-snug text-neutral-500">
          If this keeps happening, check your API keys, network connectivity, and permissions.
        </p>
      </div>
    </div>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;