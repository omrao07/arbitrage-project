"use client";

/**
 * terminal/error.tsx
 * Self-contained error boundary UI for the Terminal section.
 * - No external imports, only React ambient + Tailwind.
 * - Shows error icon, message, optional stack, and retry button.
 */

export default function TerminalError({
  error,
  onRetry,
}: {
  error?: Error | null;
  onRetry?: () => void;
}) {
  const msg =
    (typeof error?.message === "string" && error?.message) ||
    "Something went wrong in the terminal.";

  return (
    <div className="flex min-h-[60vh] w-full items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="max-w-lg space-y-4 text-center">
        {/* Icon */}
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-red-900/30 text-red-400">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v4m0 4h.01M21 12A9 9 0 1 1 3 12a9 9 0 0 1 18 0Z"
            />
          </svg>
        </div>

        {/* Message */}
        <h2 className="text-lg font-semibold">Terminal Error</h2>
        <p className="text-sm text-neutral-400">{msg}</p>

        {/* Retry */}
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

/* Ambient React (no imports) */
declare const React: any;