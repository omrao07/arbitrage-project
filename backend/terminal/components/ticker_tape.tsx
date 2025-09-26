// ticker-tape.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";

export type TapeItem = {
  /** Primary label (e.g., ticker) */
  label: string;
  /** Optional secondary text (e.g., price) */
  value?: string | number;
  /** Optional percentage/absolute change (drives color) */
  change?: number;
  /** Optional click-through link */
  href?: string;
  /** Optional custom node; if set, we render this instead of the default pill */
  node?: React.ReactNode;
};

export interface TickerTapeProps {
  items: TapeItem[];
  /** Height of the tape (px) */
  height?: number;
  /** Gap between items (px) */
  gap?: number;
  /** Scroll speed in px/second (affects duration; higher = faster) */
  speed?: number;
  /** Direction of travel */
  direction?: "left" | "right";
  /** Pause when user hovers */
  pauseOnHover?: boolean;
  /** Soft gradient mask on edges */
  fadeEdges?: boolean;
  /** Start/stop externally */
  playing?: boolean;
  /** ARIA label for the whole region */
  ariaLabel?: string;
}

const cx = (...xs: (string | false | null | undefined)[]) => xs.filter(Boolean).join(" ");

const Arrow = ({ up }: { up: boolean }) => (
  <svg viewBox="0 0 16 16" width="12" height="12" className={cx(up ? "text-emerald-400" : "text-rose-400")}>
    {up ? (
      <path fill="currentColor" d="M8 3l5 6H9v4H7V9H3l5-6z" />
    ) : (
      <path fill="currentColor" d="M8 13l-5-6h4V3h2v4h4l-5 6z" />
    )}
  </svg>
);

/**
 * A silky-smooth, accessible ticker tape (a.k.a. marquee) component.
 * - Canvas-free; pure CSS animation with dynamic duration based on content width.
 * - Seamless loop using a duplicated track (track A + track B).
 * - Hover to pause (optional). Respects prefers-reduced-motion.
 * - Each item can be a link or a custom node; default pill is opinionated and pretty.
 */
const TickerTape: React.FC<TickerTapeProps> = ({
  items,
  height = 40,
  gap = 28,
  speed = 80, // px/sec
  direction = "left",
  pauseOnHover = true,
  fadeEdges = true,
  playing = true,
  ariaLabel = "Live market ticker",
}) => {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [duration, setDuration] = useState<number>(30);

  // Duplicate items to create a seamless loop (ensure at least a few screens worth)
  const trackItems = useMemo(() => (items.length ? [...items, ...items] : []), [items]);

  // Measure track width → compute animation duration
  useEffect(() => {
    const calc = () => {
      if (!wrapRef.current || !trackRef.current) return;
      const wrapW = wrapRef.current.clientWidth;
      const trackW = trackRef.current.scrollWidth / 2; // original (before duplication)
      // distance to travel for one full loop = trackW + wrapW (to fully leave + re-enter)
      const distance = trackW + wrapW;
      // seconds = pixels / (px per sec)
      const sec = Math.max(5, distance / Math.max(10, speed));
      setDuration(sec);
    };
    calc();
    const ro = new ResizeObserver(calc);
    if (wrapRef.current) ro.observe(wrapRef.current);
    if (trackRef.current) ro.observe(trackRef.current);
    window.addEventListener("orientationchange", calc);
    return () => {
      ro.disconnect();
      window.removeEventListener("orientationchange", calc);
    };
  }, [speed, items]);

  const dirClass = direction === "left" ? "tt-scroll-left" : "tt-scroll-right";
  const runState = playing ? "running" : "paused";

  return (
    <div
      className={cx(
        "relative w-full select-none",
        "bg-zinc-950 border border-zinc-800 rounded-2xl"
      )}
      style={{ height }}
      aria-label={ariaLabel}
      role="region"
    >
      {/* Edge fade masks */}
      {fadeEdges && (
        <>
          <div className="pointer-events-none absolute left-0 top-0 h-full w-10 z-10"
               style={{ background: "linear-gradient(90deg, rgba(24,24,27,1), rgba(24,24,27,0))" }} />
          <div className="pointer-events-none absolute right-0 top-0 h-full w-10 z-10"
               style={{ background: "linear-gradient(270deg, rgba(24,24,27,1), rgba(24,24,27,0))" }} />
        </>
      )}

      {/* Track wrapper */}
      <div
        ref={wrapRef}
        className={cx(
          "relative h-full overflow-hidden rounded-2xl",
          pauseOnHover && "hover:[&_.tt-track]:[animation-play-state:paused]"
        )}
      >
        {/* Track (duplicated) */}
        <div
          ref={trackRef}
          className={cx("absolute top-1/2 -translate-y-1/2 flex tt-track", dirClass)}
          style={
            {
              gap: `${gap}px`,
              animationDuration: `${duration}s`,
              animationPlayState: runState,
              // Reduce motion courtesy
              ["@media (prefers-reduced-motion: reduce) as any"]: { animationPlayState: "paused" },
            } as React.CSSProperties
          }
          aria-live="polite"
        >
          {trackItems.map((it, idx) => {
            const key = `${it.label}-${idx}`;
            if (it.node) {
              return (
                <div key={key} className="flex items-center whitespace-nowrap">
                  {it.node}
                </div>
              );
            }
            const ch = typeof it.change === "number" ? it.change : undefined;
            const up = ch !== undefined ? ch >= 0 : undefined;
            const color =
              up === undefined ? "text-zinc-200" : up ? "text-emerald-400" : "text-rose-400";
            const pill = (
              <div
                className={cx(
                  "flex items-center gap-2 rounded-xl border px-3 py-1.5",
                  "border-zinc-800 bg-zinc-900/60 backdrop-blur",
                  "whitespace-nowrap text-sm"
                )}
              >
                <span className="font-semibold text-zinc-100">{it.label}</span>
                {it.value !== undefined && (
                  <span className="tabular-nums text-zinc-300">{it.value}</span>
                )}
                {ch !== undefined && (
                  <span className={cx("flex items-center gap-1 tabular-nums", color)}>
                    <Arrow up={!!up} />
                    {ch > 0 ? "+" : ""}
                    {typeof ch === "number" ? ch.toFixed(2) : ch}
                    %
                  </span>
                )}
              </div>
            );
            return it.href ? (
              <a key={key} href={it.href} className="focus:outline-none focus:ring-2 focus:ring-amber-400/50 rounded-xl">
                {pill}
              </a>
            ) : (
              <div key={key}>{pill}</div>
            );
          })}
        </div>
      </div>

      {/* Keyframes (scoped) */}
      <style>
        {`
        .tt-track { will-change: transform; }
        .tt-scroll-left {
          animation-name: tt-marquee-left;
          animation-timing-function: linear;
          animation-iteration-count: infinite;
        }
        .tt-scroll-right {
          animation-name: tt-marquee-right;
          animation-timing-function: linear;
          animation-iteration-count: infinite;
        }
        @keyframes tt-marquee-left {
          0%   { transform: translateX(0); }
          100% { transform: translateX(-50%); } /* duplicated track = 50% width step */
        }
        @keyframes tt-marquee-right {
          0%   { transform: translateX(-50%); }
          100% { transform: translateX(0); }
        }
        @media (prefers-reduced-motion: reduce) {
          .tt-track { animation-play-state: paused !important; }
        }
      `}
      </style>
    </div>
  );
};

export default TickerTape;