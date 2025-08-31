// frontend/components/RegimePlayer.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
} from "recharts";

/* ------------------------------- Types ------------------------------- */

export type RegimeId = string; // e.g. "bull", "bear", "crisis", "range"

export type RegimeSample = {
  t: number;            // epoch ms
  regime: RegimeId;     // label/id
  score?: number;       // confidence [0..1] or any metric to overlay
  metric?: number;      // optional numeric series to plot (e.g., VIX/return)
};

export type RegimePalette = Record<RegimeId, string>; // fill color by regime

interface Props {
  /** If provided, fetch is skipped. */
  samples?: RegimeSample[];
  /** GET endpoint returning RegimeSample[] when samples not provided */
  endpoint?: string;                 // default: /api/regimes?symbol=SPY&period=1d
  symbol?: string;
  title?: string;
  palette?: RegimePalette;           // custom colors
  metricLabel?: string;              // label for metric axis
  height?: number;
  /** initial playback speed multiplier (1 = 1x) */
  speed?: 0.5 | 1 | 2 | 4;
}

/* ------------------------------ Component ---------------------------- */

export default function RegimePlayer({
  samples,
  endpoint = "/api/regimes",
  symbol = "SPY",
  title = "Regime Player",
  metricLabel = "Metric",
  height = 420,
  speed = 1,
  palette = {
    bull: "#16a34a55",
    bear: "#ef444455",
    range: "#64748b55",
    crisis: "#f59e0b55",
  },
}: Props) {
  const [rows, setRows] = useState<RegimeSample[] | null>(samples ?? null);
  const [err, setErr] = useState<string | null>(null);

  // playback state
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [spd, setSpd] = useState(speed);

  // viewport (window) for focus
  const [windowSize, setWindowSize] = useState<number>(250); // points in view

  const timer = useRef<number | null>(null);

  /* ------------------------------- Data ------------------------------ */

  useEffect(() => {
    if (samples) { setRows(samples); return; }
    let ignore = false;
    (async () => {
      try {
        const q = new URLSearchParams({ symbol }).toString();
        const res = await fetch(`${endpoint}?${q}`);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as RegimeSample[];
        if (!ignore) setRows(json);
      } catch (e: any) {
        if (!ignore) { setErr(e?.message || "Failed to load regimes"); setRows([]); }
      }
    })();
    return () => { ignore = true; };
  }, [samples, endpoint, symbol]);

  const data = rows ?? [];
  const tMin = data[0]?.t ?? 0;
  const tMax = data[data.length - 1]?.t ?? 0;

  // compress consecutive samples with same regime into segments for banding
  const segments = useMemo(() => {
    const segs: { start: number; end: number; regime: RegimeId }[] = [];
    if (!data.length) return segs;
    let cur = { start: data[0].t, end: data[0].t, regime: data[0].regime };
    for (let k = 1; k < data.length; k++) {
      const s = data[k];
      if (s.regime === cur.regime) {
        cur.end = s.t;
      } else {
        segs.push({ ...cur });
        cur = { start: s.t, end: s.t, regime: s.regime };
      }
    }
    segs.push(cur);
    return segs;
  }, [data]);

  // stats per regime
  const stats = useMemo(() => {
    const by: Record<string, { points: number; durationMs: number }> = {};
    for (let k = 0; k < data.length; k++) {
      const r = data[k].regime;
      by[r] ??= { points: 0, durationMs: 0 };
      by[r].points += 1;
      if (k > 0) by[r].durationMs += data[k].t - data[k - 1].t;
    }
    const total = Object.values(by).reduce((a, x) => a + x.durationMs, 0) || 1;
    return Object.entries(by)
      .map(([regime, x]) => ({
        regime,
        points: x.points,
        pctTime: (x.durationMs / total) * 100,
      }))
      .sort((a, b) => b.pctTime - a.pctTime);
  }, [data]);

  // current frame & window
  const cur = data[i] ?? null;
  const winStart = Math.max(0, i - windowSize + 1);
  const view = data.slice(winStart, i + 1);

  // auto-play
  useEffect(() => {
    if (!playing || !data.length) return;
    const frameMs = 33 / spd; // ~30fps at 1x
    timer.current = window.setTimeout(() => {
      setI((prev) => {
        const next = prev + 1;
        if (next < data.length) return next;
        return loop ? 0 : prev;
      });
    }, frameMs) as unknown as number;
    return () => { if (timer.current) window.clearTimeout(timer.current); };
  }, [playing, data.length, i, spd, loop]);

  /* ------------------------------- UI -------------------------------- */

  if (err) return <div className="rounded-xl border p-3 text-sm text-red-600">Error: {err}</div>;
  if (!data.length) return <div className="rounded-xl border p-3 text-sm opacity-70">Loading regimes…</div>;

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{title} — {symbol}</h2>
          <div className="text-xs opacity-70">
            {new Date(tMin).toLocaleDateString()} → {new Date(tMax).toLocaleDateString()} •
            {" "}frames: {data.length} • window: {windowSize}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800"
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800"
            onClick={() => setI((k) => Math.max(0, k - 1))}
          >
            ‹ Prev
          </button>
          <button
            className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800"
            onClick={() => setI((k) => Math.min(data.length - 1, k + 1))}
          >
            Next ›
          </button>
          <select
            className="px-2 py-1.5 text-sm rounded-md border dark:border-gray-800"
            value={spd}
            onChange={(e) => setSpd(Number(e.target.value) as any)}
            title="Playback speed"
          >
            {[0.5, 1, 2, 4].map((x) => <option key={x} value={x}>{x}×</option>)}
          </select>
          <label className="text-sm flex items-center gap-1">
            <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} />
            loop
          </label>
        </div>
      </header>

      {/* Current frame */}
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm">
          <span className="opacity-70">t:</span> <b>{cur ? new Date(cur.t).toLocaleString() : "—"}</b>{" "}
          <span className="opacity-70">regime:</span>{" "}
          <b style={{ color: solidColor(cur?.regime, palette) }}>
            {cur?.regime ?? "—"}
          </b>{" "}
          {cur?.score !== undefined && (
            <>
              <span className="opacity-70">score:</span> <b>{cur.score?.toFixed(2)}</b>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm opacity-70">Window</label>
          <input
            type="range" min={50} max={data.length} value={windowSize}
            onChange={(e) => setWindowSize(Number(e.target.value))}
          />
        </div>
      </div>

      {/* Timeline chart with colored bands + metric line */}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={view}>
            <XAxis
              dataKey="t"
              tickFormatter={(t) => miniDate(t as number)}
              domain={["auto", "auto"]}
              type="number"
            />
            <YAxis yAxisId="y" width={50} />
            <Tooltip
              labelFormatter={(t) => new Date(t as number).toLocaleString()}
              formatter={(v, name) => {
                if (name === "metric") return [Number(v as number).toFixed(2), metricLabel];
                return [String(v), name];
              }}
            />
            {/* Regime bands (draw behind) */}
            {bandAreas(segments, view[0]?.t ?? tMin, view[view.length - 1]?.t ?? tMax, palette).map((b, idx) => (
              <ReferenceArea
                key={idx}
                x1={b.x1}
                x2={b.x2}
                y1={Number.NEGATIVE_INFINITY}
                y2={Number.POSITIVE_INFINITY}
                ifOverflow="extendDomain"
                fill={b.fill}
                fillOpacity={1}
                strokeOpacity={0}
              />
            ))}
            {/* metric line */}
            <Line
              yAxisId="y"
              type="monotone"
              dataKey="metric"
              stroke="#2563eb"
              dot={false}
              isAnimationActive={false}
              name={metricLabel}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend + stats */}
      <div className="mt-3 grid gap-3 md:grid-cols-3">
        <div className="rounded-xl border dark:border-gray-800 p-3 text-sm">
          <div className="font-medium mb-2">Regimes</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(collectRegimes(segments)).map(([r, count]) => (
              <span
                key={r}
                className="text-xs px-2 py-1 rounded-md"
                style={{ backgroundColor: palette[r] ?? "#e5e7eb", color: textOn(palette[r] ?? "#e5e7eb") }}
                title={`${count} segments`}
              >
                {r}
              </span>
            ))}
          </div>
        </div>

        <div className="md:col-span-2 rounded-xl border dark:border-gray-800 p-3 text-sm">
          <div className="font-medium mb-2">Time in Regime</div>
          <div className="grid gap-1" style={{ gridTemplateColumns: "160px 1fr 60px" }}>
            {stats.map((s) => (
              <React.Fragment key={s.regime}>
                <div className="opacity-80">{s.regime}</div>
                <div className="h-2 my-2 rounded-full overflow-hidden bg-gray-200 dark:bg-gray-800">
                  <div
                    className="h-full"
                    style={{
                      width: `${s.pctTime.toFixed(2)}%`,
                      backgroundColor: solidColor(s.regime, palette),
                    }}
                  />
                </div>
                <div className="text-right">{s.pctTime.toFixed(1)}%</div>
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      {/* Transport bar */}
      <div className="mt-3 flex items-center gap-2">
        <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setI(0)}>⏮</button>
        <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setI((k) => Math.max(0, k - 25))}>⏪</button>
        <input
          type="range"
          min={0}
          max={data.length - 1}
          value={i}
          className="flex-1"
          onChange={(e) => setI(Number(e.target.value))}
        />
        <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setI((k) => Math.min(data.length - 1, k + 25))}>⏩</button>
        <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setI(data.length - 1)}>⏭</button>
      </div>
    </div>
  );
}

/* ------------------------------- Helpers ------------------------------ */

function miniDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

/** produce colored band ranges intersecting the current viewport */
function bandAreas(
  segs: { start: number; end: number; regime: RegimeId }[],
  xMin: number,
  xMax: number,
  palette: RegimePalette
) {
  const out: { x1: number; x2: number; fill: string }[] = [];
  for (const s of segs) {
    const x1 = Math.max(s.start, xMin);
    const x2 = Math.min(s.end, xMax);
    if (x2 <= xMin || x1 >= xMax) continue;
    out.push({ x1, x2, fill: palette[s.regime] ?? "#e5e7eb" });
  }
  return out;
}

function solidColor(regime: RegimeId | undefined, palette: RegimePalette) {
  const c = (regime && palette[regime]) || "#9ca3af55";
  // ensure solid fill (strip alpha if provided)
  return c.length === 9 && c.endsWith("55") ? c.slice(0, 7) : c.replace(/[\da-f]{2}$/i, "");
}

function textOn(bg: string) {
  // simple luminance heuristic for contrast
  const hex = bg.replace("#", "").slice(0, 6);
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b);
  return lum > 160 ? "#111827" : "#f9fafb";
}

function collectRegimes(segs: { regime: RegimeId }[]) {
  const m: Record<string, number> = {};
  segs.forEach((s) => (m[s.regime] = (m[s.regime] ?? 0) + 1));
  return m;
}