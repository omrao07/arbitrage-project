"use client";

/**
 * rypto/error.tsx  (use in app/crypto/error.tsx or similar)
 * Zero-import error boundary UI:
 * - Shows a friendly error message
 * - Optional "Retry" (calls `reset`) and "Reload" buttons
 * - Inline SVG (no icon libs), Tailwind styles
 */

export default function CryptoError({
  error,
  reset,
}: {
  error?: Error & { digest?: string };
  reset?: () => void;
}) {
  const msg =
    (error && (error.message || (error as any).toString())) ||
    "Something went wrong while loading crypto data.";

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="w-full max-w-md rounded-xl border border-neutral-800 bg-neutral-900 p-6 shadow-lg">
        <div className="mb-3 flex items-center gap-3">
          {/* inline warning icon */}
          <svg width="24" height="24" viewBox="0 0 24 24" className="text-rose-500" aria-hidden>
            <path d="M12 2L1 21h22L12 2Z" fill="currentColor" opacity="0.12" />
            <path d="M12 8v6m0 3h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <h1 className="text-lg font-semibold">Crypto Module Error</h1>
        </div>

        <p className="text-sm text-neutral-400">
          {msg}
          {error?.digest && (
            <>
              <br />
              <span className="text-neutral-500">Ref:</span> {error.digest}
            </>
          )}
        </p>

        <div className="mt-5 flex flex-wrap gap-3">
          {reset && (
            <button
              onClick={reset}
              className="inline-flex items-center justify-center rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              Retry
            </button>
          )}
          <button
            onClick={() => (typeof window !== "undefined" ? window.location.reload() : null)}
            className="inline-flex items-center justify-center rounded-lg border border-neutral-700 px-4 py-2 text-sm font-medium text-neutral-300 hover:bg-neutral-800 focus:outline-none focus:ring-2 focus:ring-neutral-600"
          >
            Reload
          </button>
          <button
            onClick={() => (typeof window !== "undefined" ? (window as any).history?.back?.() : null)}
            className="ml-auto inline-flex items-center justify-center rounded-lg border border-neutral-800 px-3 py-2 text-xs text-neutral-400 hover:text-neutral-200"
            title="Go back"
          >
            ← Back
          </button>
        </div>

        <details className="mt-4 rounded-lg border border-neutral-800 bg-neutral-950/60 p-3 text-xs text-neutral-400">
          <summary className="cursor-pointer select-none text-neutral-300">Technical details</summary>
          <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-neutral-400">
            {safeStringify(error)}
          </pre>
        </details>
      </div>
    </div>
  );
}

/* ------------------------------ helpers ------------------------------ */
function safeStringify(err: unknown): string {
  try {
    if (!err) return "No error object.";
    if (err instanceof Error) {
      const anyErr = err as any;
      const o = {
        name: err.name,
        message: err.message,
        stack: err.stack,
        digest: anyErr?.digest,
      };
      return JSON.stringify(o, null, 2);
    }
    if (typeof err === "object") return JSON.stringify(err, null, 2);
    return String(err);
  } catch {
    return "Unable to serialize error.";
  }
}