"use client";

/**
 * components/loading.tsx
 * Zero-import, self-contained loading/skeleton panel (Tailwind only).
 *
 * Props
 * - label?: string           // small caption under the spinner
 * - tip?: string             // muted helper text
 * - variant?: "spinner"|"bar"|"skeleton"  // visual style
 * - rows?: number            // for skeleton rows
 * - className?: string
 */

export default function Loading({
  label = "Loading…",
  tip,
  variant = "spinner",
  rows = 8,
  className = "",
}: {
  label?: string;
  tip?: string;
  variant?: "spinner" | "bar" | "skeleton";
  rows?: number;
  className?: string;
}) {
  return (
    <div
      className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 p-6 text-neutral-100 ${className}`}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      {variant === "spinner" && <Spinner label={label} tip={tip} />}
      {variant === "bar" && <Bar label={label} tip={tip} />}
      {variant === "skeleton" && <Skeleton label={label} tip={tip} rows={rows} />}
    </div>
  );
}

/* -------------------------------- Variants ------------------------------- */

function Spinner({ label, tip }: { label: string; tip?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className="relative h-10 w-10">
        <div className="absolute inset-0 animate-ping rounded-full bg-sky-600/30" />
        <svg className="absolute inset-0 m-auto h-10 w-10 animate-spin" viewBox="0 0 24 24">
          <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-80"
            d="M22 12a10 10 0 0 1-10 10"
            stroke="currentColor"
            strokeWidth="4"
            strokeLinecap="round"
            fill="none"
          />
        </svg>
      </div>
      <div className="text-sm">{label}</div>
      {tip && <div className="text-xs text-neutral-400">{tip}</div>}
    </div>
  );
}

function Bar({ label, tip }: { label: string; tip?: string }) {
  return (
    <div>
      <div className="mb-2 text-sm">{label}</div>
      <div className="relative h-2 w-full overflow-hidden rounded bg-neutral-800">
        <div className="absolute inset-y-0 left-0 h-full w-1/3 animate-[loading_1.2s_infinite] rounded bg-sky-600/70" />
      </div>
      {tip && <div className="mt-2 text-xs text-neutral-400">{tip}</div>}

      {/* keyframes (inline) */}
      <style>{`@keyframes loading{0%{transform:translateX(-100%)}50%{transform:translateX(80%)}100%{transform:translateX(120%)}}`}</style>
    </div>
  );
}

function Skeleton({ label, tip, rows = 8 }: { label: string; tip?: string; rows?: number }) {
  return (
    <div>
      <div className="mb-3 text-sm">{label}</div>
      <div className="space-y-2">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="grid grid-cols-12 gap-2">
            <div className="col-span-3 h-4 rounded bg-neutral-800 animate-pulse" />
            <div className="col-span-3 h-4 rounded bg-neutral-800 animate-pulse" />
            <div className="col-span-2 h-4 rounded bg-neutral-800 animate-pulse" />
            <div className="col-span-2 h-4 rounded bg-neutral-800 animate-pulse" />
            <div className="col-span-2 h-4 rounded bg-neutral-800 animate-pulse" />
          </div>
        ))}
      </div>
      {tip && <div className="mt-3 text-xs text-neutral-400">{tip}</div>}
    </div>
  );
}

/* ----------------------- Ambient React (no imports) ---------------------- */
declare const React: any;