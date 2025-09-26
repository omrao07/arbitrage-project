"use client";

/**
 * screener/error.tsx
 * Zero-import generic error UI for the Screener views.
 * Tailwind-only. Pass an optional onRetry() to show a Retry button.
 */

export default function ScreenerError({
  error,
  onRetry,
  details,
}: {
  error?: Error | string | null;
  onRetry?: () => void;
  details?: string; // optional extra context (e.g., query JSON)
}) {
  const message =
    (typeof error === "string" && error) ||
    ((error as any)?.message as string) ||
    "Something went wrong while running the screen.";

  return (
    <div className="flex min-h-[60vh] items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="mx-auto w-full max-w-2xl rounded-2xl border border-neutral-800 bg-neutral-900 p-6 shadow">
        {/* Icon + Title */}
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-10 w-10 items-center justify-center rounded-lg bg-rose-900/30 text-rose-400">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 9v4m0 4h.01" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20Z" />
            </svg>
          </div>
          <div className="min-w-0">
            <h2 className="text-lg font-semibold">Screen failed</h2>
            <p className="mt-1 break-words text-sm text-neutral-300">
              {message}
            </p>
          </div>
        </div>

        {/* Hints */}
        <ul className="mt-4 list-disc space-y-1 pl-6 text-xs text-neutral-400">
          <li>Check your filters (ranges, symbols, sectors) and try again.</li>
          <li>If you’re pulling from an API, verify keys, quota, and CORS.</li>
          <li>Large result sets? Lower the limit or add more criteria.</li>
        </ul>

        {/* Details (optional) */}
        {details && (
          <details className="mt-4 rounded-md border border-neutral-800 bg-neutral-950">
            <summary className="cursor-pointer px-3 py-2 text-xs text-neutral-400">Details</summary>
            <pre className="max-h-56 overflow-auto px-3 pb-3 pt-1 text-[11px] text-neutral-300">
{details}
            </pre>
          </details>
        )}

        {/* Actions */}
        <div className="mt-5 flex items-center justify-end gap-2">
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

        {/* Footer tip */}
        <p className="mt-3 text-center text-[11px] leading-snug text-neutral-500">
          Need reproducibility? Capture the query JSON and timestamp when reporting bugs.
        </p>
      </div>
    </div>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;