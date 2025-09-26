"use client";

/**
 * components/error.tsx
 * Zero-import, self-contained error panel (Tailwind only).
 *
 * Props
 * - title?: string                // heading text
 * - message?: string | Error      // short message (Error -> .message)
 * - details?: string              // optional long text (stack, payload, etc.)
 * - onRetry?: () => void          // show Retry button if provided
 * - onDismiss?: () => void        // optional Dismiss button
 * - errorId?: string              // correlation id to show the user
 * - severity?: "info"|"warn"|"error"|"fatal"  // visual accent
 * - timestamp?: string            // ISO string to display
 *
 * Usage:
 * <ErrorPanel message={err} details={json} onRetry={()=>run()} errorId={rid} />
 */

export default function ErrorPanel({
  title = "Something went wrong",
  message,
  details,
  onRetry,
  onDismiss,
  errorId,
  severity = "error",
  timestamp,
  className = "",
}: {
  title?: string;
  message?: string | Error | null;
  details?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  errorId?: string;
  severity?: "info" | "warn" | "error" | "fatal";
  timestamp?: string;
  className?: string;
}) {
  const msg =
    (typeof message === "string" && message) ||
    ((message as any)?.message as string) ||
    "Unexpected error.";

  const theme = tone(severity);

  return (
    <div
      className={`w-full rounded-xl border ${theme.border} ${theme.bg} ${className}`}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-start gap-3 px-4 py-3">
        <Badge severity={severity} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
            {timestamp && (
              <span className="text-[11px] text-neutral-500">
                {new Date(timestamp).toLocaleString()}
              </span>
            )}
          </div>
          <p className="mt-1 break-words text-sm text-neutral-300">{msg}</p>
          {errorId && (
            <p className="mt-1 text-xs text-neutral-500">
              Error ID: <code className="rounded bg-neutral-900 px-1 py-0.5">{errorId}</code>
            </p>
          )}

          {details && (
            <details className="mt-3 overflow-hidden rounded-md border border-neutral-800">
              <summary className="cursor-pointer bg-neutral-950 px-3 py-2 text-xs text-neutral-400">
                Show details
              </summary>
              <pre className="max-h-64 overflow-auto px-3 pb-3 pt-2 text-[11px] leading-5 text-neutral-300">
{details}
              </pre>
            </details>
          )}

          <div className="mt-3 flex items-center gap-2">
            {onRetry && (
              <button
                onClick={onRetry}
                className="rounded-md border border-emerald-700 bg-emerald-600/20 px-3 py-1.5 text-sm font-medium text-emerald-300 hover:bg-emerald-600/30"
              >
                Retry
              </button>
            )}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-200 hover:bg-neutral-800"
              >
                Dismiss
              </button>
            )}
            <button
              onClick={() => location.reload()}
              className="rounded-md border border-neutral-800 bg-neutral-950 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
            >
              Refresh
            </button>
            <button
              onClick={() => copyContext(msg, details, errorId)}
              className="ml-auto rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-300 hover:bg-neutral-800"
              title="Copy error context"
            >
              Copy context
            </button>
          </div>

          <p className="mt-2 text-[11px] text-neutral-500">
            Tip: include the error ID and steps to reproduce when reporting.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------- Visuals -------------------------------- */

function Badge({ severity }: { severity: "info"|"warn"|"error"|"fatal" }) {
  const c = tone(severity).icon;
  return (
    <div className={`mt-0.5 flex h-10 w-10 items-center justify-center rounded-lg ${c.bg} ${c.text}`}>
      {/* simple alert glyph */}
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 9v4m0 4h.01" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18Z" />
      </svg>
    </div>
  );
}

function tone(sev: "info"|"warn"|"error"|"fatal") {
  switch (sev) {
    case "info":
      return {
        border: "border-sky-800",
        bg: "bg-sky-950/40",
        icon: { bg: "bg-sky-900/30", text: "text-sky-300" },
      };
    case "warn":
      return {
        border: "border-amber-800",
        bg: "bg-amber-950/40",
        icon: { bg: "bg-amber-900/30", text: "text-amber-300" },
      };
    case "fatal":
      return {
        border: "border-rose-900",
        bg: "bg-rose-950/50",
        icon: { bg: "bg-rose-900/40", text: "text-rose-300" },
      };
    case "error":
    default:
      return {
        border: "border-rose-800",
        bg: "bg-rose-950/40",
        icon: { bg: "bg-rose-900/30", text: "text-rose-300" },
      };
  }
}

/* ------------------------------- Helpers -------------------------------- */

function copyContext(msg: string, details?: string, id?: string) {
  const payload = {
    message: msg,
    details: details || "",
    errorId: id || "",
    userAgent: (typeof navigator !== "undefined" && navigator.userAgent) || "",
    ts: new Date().toISOString(),
  };
  try { (navigator as any).clipboard?.writeText(JSON.stringify(payload, null, 2)); } catch {}
}

/* ----------------------- Ambient React (no imports) ---------------------- */
declare const React: any;