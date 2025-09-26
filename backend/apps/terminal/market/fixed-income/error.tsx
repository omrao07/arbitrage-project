"use client";

/**
 * fixed income/error.tsx
 * Zero-import error boundary UI for the Fixed Income section.
 * - Dark background
 * - Prominent error icon + message
 * - Retry button (optional callback)
 * - Tailwind only
 */

export default function FixedIncomeError({
  error,
  onRetry,
}: {
  error?: Error | null;
  onRetry?: () => void;
}) {
  return (
    <div className="flex min-h-[60vh] items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="max-w-md space-y-4 text-center">
        {/* Icon */}
        <div className="mx-auto h-12 w-12 rounded-full bg-red-900/30 text-red-400 flex items-center justify-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M21 12A9 9 0 1 1 3 12a9 9 0 0 1 18 0Z" />
          </svg>
        </div>

        {/* Title */}
        <h2 className="text-lg font-semibold">Something went wrong</h2>

        {/* Error details */}
        {error?.message && (
          <p className="text-sm text-neutral-400">{error.message}</p>
        )}

        {/* Retry button */}
        {onRetry && (
          <button
            onClick={onRetry}
            className="mt-2 rounded-md border border-neutral-700 bg-neutral-900 px-4 py-2 text-sm font-medium text-neutral-200 hover:bg-neutral-800"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}