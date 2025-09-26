"use client";

/**
 * terminal/loading.tsx
 * Self-contained skeleton/loading state for the Terminal section.
 * - No external imports, just React ambient + Tailwind classes.
 * - Shows spinner, label, and optional tips.
 */

export default function TerminalLoading({ tip }: { tip?: string }) {
  return (
    <div className="flex min-h-[60vh] w-full items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="flex flex-col items-center gap-3 text-center">
        {/* Spinner */}
        <div className="relative h-12 w-12">
          <div className="absolute inset-0 animate-ping rounded-full bg-sky-600/20" />
          <svg
            className="absolute inset-0 m-auto h-12 w-12 animate-spin text-sky-400"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
            />
            <path
              className="opacity-75"
              d="M4 12a8 8 0 018-8"
              strokeLinecap="round"
            />
          </svg>
        </div>

        {/* Text */}
        <div className="text-sm font-medium">Starting terminal…</div>
        {tip && <div className="text-xs text-neutral-500">{tip}</div>}
      </div>
    </div>
  );
}

/* Ambient React (no imports) */
declare const React: any;