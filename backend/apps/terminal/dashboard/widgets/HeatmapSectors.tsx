"use client";

type SectorLeaf = {
  id: string;                // unique (e.g., "AAPL")
  label: string;             // display name
  pct: number;               // % change for the day (e.g., -1.23)
  weight?: number;           // market-cap or turnover weight (0..1 or raw)
};

export type SectorNode = {
  id: string;                // e.g., "Technology"
  label: string;             // display name
  pct: number;               // aggregated % change (cap-weighted or avg)
  weight?: number;           // sector weight (0..1 or raw)
  count?: number;            // number of members
  children?: SectorLeaf[];   // optional constituents for tooltip
};

export type HeatmapSectorsProps = {
  data: SectorNode[];
  title?: string;
  /** -10..+10 default domain (in %) */
  domain?: { min: number; max: number };
  /** tile size policy */
  sizing?: "byWeight" | "uniform";
  /** sort order */
  sortBy?: "weight" | "pct" | "label";
  /** click handler */
  onSelect?: (id: string) => void;
  /** show legend */
  legend?: boolean;
  /** max rows before scroll */
  maxHeight?: number; // px
};

/**
 * HeatmapSectors.tsx
 * - No chart lib; uses CSS grid + inline HSL backgrounds
 * - Diverging color scale (red↔gray↔green)
 * - Tile size can reflect weight bucket
 */
export default function HeatmapSectors({
  data,
  title = "Sector Heatmap",
  domain = { min: -10, max: 10 },
  sizing = "byWeight",
  sortBy = "weight",
  onSelect,
  legend = true,
  maxHeight = 520,
}: HeatmapSectorsProps) {
  const items = [...(data || [])]
    .map((s) => normalizeSector(s))
    .sort((a, b) => {
      if (sortBy === "pct") return Math.abs(b.pct) - Math.abs(a.pct);
      if (sortBy === "label") return a.label.localeCompare(b.label);
      return (b.weight ?? 0) - (a.weight ?? 0);
    });

  // normalize weights into buckets to size tiles
  const weights = items.map((x) => x.weight ?? 0);
  const maxW = Math.max(1, ...weights);
  const bucket = (w?: number) => {
    const v = (w ?? 0) / maxW;
    if (sizing === "uniform") return "w-48"; // ~192px
    if (v >= 0.66) return "w-56";            // large
    if (v >= 0.33) return "w-48";            // medium
    return "w-40";                           // small
  };

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-4 py-2 border-b border-[#222] flex items-center justify-between">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">
          {domain.min}% … {domain.max}%
        </div>
      </div>

      {/* grid */}
      <div
        className="p-3 overflow-auto"
        style={{ maxHeight }}
      >
        <div className="flex flex-wrap gap-3">
          {items.map((s) => {
            const bg = cellColor(s.pct, domain);
            const txt = pickTextColor(s.pct);
            const sz = bucket(s.weight);
            const tip = buildTooltip(s);

            return (
              <button
                key={s.id}
                title={tip}
                onClick={() => onSelect?.(s.id)}
                className={`rounded-lg border border-[#1f1f1f] ${sz} min-w-[10rem] h-28 p-3 text-left hover:brightness-110 focus:outline-none`}
                style={{ background: bg, color: txt }}
              >
                <div className="flex items-start justify-between">
                  <div className="text-sm font-semibold truncate">{s.label}</div>
                  <span className="text-[10px] px-1.5 py-[1px] rounded border border-white/20">
                    {formatPct(s.pct)}
                  </span>
                </div>

                <div className="mt-2 text-[11px] opacity-80">
                  Weight: {formatWeight(s.weight)}
                  {s.count != null ? ` • ${s.count} members` : ""}
                </div>

                {/* mini bar to show absolute magnitude */}
                <div className="mt-3 h-1.5 bg-black/20 rounded overflow-hidden">
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.min(100, (Math.abs(s.pct) / (domain.max - domain.min) * 200))}%`,
                      background: "rgba(255,255,255,0.35)",
                    }}
                  />
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* legend */}
      {legend && (
        <div className="px-4 py-2 border-t border-[#222]">
          <Legend domain={domain} />
        </div>
      )}
    </div>
  );
}

/* ================= helpers ================= */

function normalizeSector(s: SectorNode): SectorNode {
  return {
    id: String(s.id),
    label: s.label ?? s.id,
    pct: clampNum(Number(s.pct), -100, 100),
    weight: s.weight != null ? Number(s.weight) : undefined,
    count: s.count,
    children: s.children?.map((c) => ({
      id: String(c.id),
      label: c.label ?? c.id,
      pct: clampNum(Number(c.pct), -100, 100),
      weight: c.weight != null ? Number(c.weight) : undefined,
    })),
  };
}

function clampNum(n: number, a: number, b: number) {
  if (!Number.isFinite(n)) return 0;
  return Math.max(a, Math.min(b, n));
}

function formatPct(n: number) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function formatWeight(w?: number) {
  if (w == null) return "—";
  if (w > 1_000_000_000) return (w / 1_000_000_000).toFixed(1) + "B";
  if (w > 1_000_000) return (w / 1_000_000).toFixed(1) + "M";
  if (w > 1_000) return (w / 1_000).toFixed(1) + "K";
  return w.toFixed(2);
}

/** Diverging red→gray→green using HSL (no lib) */
function cellColor(pct: number, domain: { min: number; max: number }) {
  const { min, max } = domain;
  const mid = 0;
  const clamped = Math.max(min, Math.min(max, pct));
  // map [-|+ ] to hue [0=red .. 120=green]
  const ratio = clamped >= mid
    ? (clamped - mid) / (max - mid || 1)
    : -1 * (clamped - mid) / (min - mid || 1);
  const hue = clamped >= mid ? 120 : 0;  // green side vs red side
  const light = 16 + Math.min(70, 70 * ratio); // lighten with magnitude
  const sat = 60 + Math.min(35, 35 * ratio);
  return `hsl(${hue} ${sat}% ${light}%)`;
}

function pickTextColor(pct: number) {
  // brighter bg -> darker text, else light text
  return Math.abs(pct) > 5 ? "#0b0b0b" : "#f3f4f6";
}

function buildTooltip(s: SectorNode) {
  const header = `${s.label} • ${formatPct(s.pct)} • Weight ${formatWeight(s.weight)}`;
  if (!s.children || s.children.length === 0) return header;
  const lines = s.children
    .slice(0, 6)
    .map((c) => ` - ${c.label}: ${formatPct(c.pct)}`)
    .join("\n");
  const more = s.children.length > 6 ? `\n(+${s.children.length - 6} more)` : "";
  return `${header}\n${lines}${more}`;
}

/* ============ legend component ============ */

function Legend({ domain }: { domain: { min: number; max: number } }) {
  const stops = [-10, -5, -2, 0, 2, 5, 10];
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="text-[11px] text-gray-400">Performance</div>
      <div className="flex-1 h-3 bg-[#111] rounded overflow-hidden mx-3 flex">
        {stops.map((v, i) => (
          <div
            key={i}
            className="flex-1"
            style={{ background: cellColor(v, domain) }}
            title={`${v}%`}
          />
        ))}
      </div>
      <div className="text-[11px] text-gray-400">{domain.min}% / {domain.max}%</div>
    </div>
  );
}