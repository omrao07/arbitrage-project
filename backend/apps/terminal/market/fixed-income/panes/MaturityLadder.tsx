"use client";

/**
 * maturityladder.tsx
 * Zero-import, self-contained bond maturity ladder.
 *
 * What it shows
 * - Stacked bars by period bucket (Month/Quarter/Year)
 * - Height = face/market value (user-provided "amount")
 * - Color by sector (or rating), legend, hover tooltips
 * - Totals row + weighted avg YTM per bucket
 *
 * Controls
 * - Bucket: Month / Quarter / Year
 * - Color by: Sector / Rating
 * - Range: From .. To (years)
 * - Search filter (symbol/issuer)
 * - CSV export (current view)
 *
 * Tailwind + inline SVG only. No external imports.
 */

export type BondLot = {
  id?: string;
  symbol: string;           // e.g. "AAPL 3.25% 2029"
  issuer?: string;          // optional
  sector?: string;          // e.g. "Tech"
  rating?: string;          // e.g. "A", "BBB+"
  coupon?: number;          // decimal (0.0325)
  ytm?: number;             // decimal
  amount: number;           // face or MV
  maturity: string;         // ISO date e.g. "2029-05-15"
};

type Bucketed = {
  key: string;              // formatted label (e.g. "2029", "2029 Q2", "2029-05")
  startISO: string;         // bucket start date
  endISO: string;           // bucket end date (exclusive)
  total: number;
  wAvgYtm: number | null;   // weighted avg ytm by amount
  stacks: { label: string; value: number; items: BondLot[] }[]; // grouped by legend key
};

export default function MaturityLadder({
  lots = [],
  title = "Maturity Ladder",
  className = "",
  defaultBucket = "Year",
  defaultColorBy = "Sector",
}: {
  lots: BondLot[];
  title?: string;
  className?: string;
  defaultBucket?: "Month" | "Quarter" | "Year";
  defaultColorBy?: "Sector" | "Rating";
}) {
  const [bucket, setBucket] = useState<"Month" | "Quarter" | "Year">(defaultBucket);
  const [colorBy, setColorBy] = useState<"Sector" | "Rating">(defaultColorBy);
  const [fromY, setFromY] = useState<number>(() => new Date().getFullYear());
  const [toY, setToY] = useState<number>(() => new Date().getFullYear() + 10);
  const [q, setQ] = useState("");

  // Filtered data
  const filtered = useMemo(() => {
    const qlc = q.trim().toLowerCase();
    return (lots || []).filter((l) => {
      // range
      const y = new Date(l.maturity).getFullYear();
      if (!(y >= fromY && y <= toY)) return false;
      // query
      if (!qlc) return true;
      const hay = (l.symbol + " " + (l.issuer || "") + " " + (l.sector || "") + " " + (l.rating || "")).toLowerCase();
      return hay.includes(qlc);
    });
  }, [lots, q, fromY, toY]);

  // Domain for legend
  const legendKeys = useMemo(() => {
    const keyOf = (l: BondLot) => (colorBy === "Sector" ? (l.sector || "Other") : (l.rating || "Unrated"));
    return uniq(filtered.map(keyOf)).sort(orderLegend(colorBy));
  }, [filtered, colorBy]);

  // Bucketing
  const buckets = useMemo<Bucketed[]>(() => {
    const spec = bucketSpec(bucket);
    const map = new Map<string, Bucketed>();
    const keyOf = (l: BondLot) => (colorBy === "Sector" ? (l.sector || "Other") : (l.rating || "Unrated"));

    for (const lot of filtered) {
      const d = new Date(lot.maturity);
      if (isNaN(+d)) continue;
      const { key, startISO, endISO } = spec.key(d);
      let b = map.get(key);
      if (!b) {
        b = { key, startISO, endISO, total: 0, wAvgYtm: null, stacks: [] };
        for (const k of legendKeys) b.stacks.push({ label: k, value: 0, items: [] });
        map.set(key, b);
      }
      const idx = b.stacks.findIndex((s) => s.label === keyOf(lot));
      const pick = idx >= 0 ? b.stacks[idx] : (b.stacks[b.stacks.push({ label: "Other", value: 0, items: [] }) - 1]);
      const amt = Math.max(0, +lot.amount || 0);
      pick.value += amt;
      pick.items.push(lot);
      b.total += amt;

      // accum for wAvg
      const ytm = Number.isFinite(lot.ytm as number) ? (lot.ytm as number) : NaN;
      if (!isNaN(ytm) && amt > 0) {
        const prevAmt = (b.wAvgYtm == null ? 0 : b.total - amt); // previous total before adding
        const prevW = b.wAvgYtm == null ? 0 : (b.wAvgYtm as number) * prevAmt;
        b.wAvgYtm = (prevW + ytm * amt) / (prevAmt + amt || 1);
      }
    }

    const out = Array.from(map.values()).sort((a, b) => +new Date(a.startISO) - +new Date(b.startISO));
    return out;
  }, [filtered, bucket, colorBy, legendKeys]);

  // Layout for chart
  const w = 980;
  const h = 320;
  const pad = { l: 64, r: 16, t: 18, b: 54 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);

  const maxVal = Math.max(1, ...buckets.map((b) => b.total));
  const X = (i: number) => pad.l + (buckets.length <= 1 ? innerW / 2 : (i / (buckets.length - 1)) * innerW);
  const Y = (v: number) => pad.t + innerH - (v / maxVal) * innerH;

  // Tooltip state
  const [tip, setTip] = useState<{ x: number; y: number; html: string } | null>(null);

  // CSV export (current bucket view)
  const onCopyCSV = () => {
    const cols = ["bucket","start","end","total","wAvgYTM", ...legendKeys];
    const lines = [cols.join(",")];
    for (const b of buckets) {
      const row = [
        b.key, b.startISO, b.endISO,
        b.total, b.wAvgYtm == null ? "" : b.wAvgYtm,
        ...legendKeys.map((k) => (b.stacks.find((s) => s.label === k)?.value ?? 0))
      ];
      lines.push(row.map(csv).join(","));
    }
    const csvStr = lines.join("\n");
    try { (navigator as any).clipboard?.writeText(csvStr); } catch {}
  };

  // Colors
  const palette = (label: string) => {
    // deterministic pastel-ish palette
    const c = hue(label);
    return `hsl(${c}, 65%, 55%)`;
  };

  // Axis ticks
  const yTicks = niceTicks(0, maxVal, 4);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header / Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {filtered.length} lots · {fromY}–{toY} · {bucket}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="relative">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search symbol / issuer"
              className="w-56 rounded-md border border-neutral-700 bg-neutral-950 pl-7 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
            />
            <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
              <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </label>
          <Select label="Bucket" value={bucket} onChange={(v) => setBucket(v as any)} options={["Month","Quarter","Year"]} />
          <Select label="Color" value={colorBy} onChange={(v) => setColorBy(v as any)} options={["Sector","Rating"]} />
          <div className="flex items-center gap-1">
            <span className="text-neutral-400">From</span>
            <input
              type="number"
              value={String(fromY)}
              onChange={(e) => setFromY(Math.min(+e.target.value || fromY, toY))}
              className="w-20 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
            />
            <span className="text-neutral-400">To</span>
            <input
              type="number"
              value={String(toY)}
              onChange={(e) => setToY(Math.max(+e.target.value || toY, fromY))}
              className="w-20 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
            />
          </div>
          <button
            onClick={onCopyCSV}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 hover:bg-neutral-800"
          >
            Copy CSV
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="p-3">
        <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
          {/* Y grid + ticks */}
          {yTicks.map((v, i) => (
            <g key={`gy-${i}`}>
              <line x1={pad.l} y1={Y(v)} x2={w - pad.r} y2={Y(v)} stroke="#27272a" strokeDasharray="3 3" />
              <text x={4} y={Y(v) + 4} fill="#9ca3af" fontSize="10">${fmtInt(v)}</text>
            </g>
          ))}

          {/* Bars */}
          {buckets.map((b, i) => {
            const x = X(i);
            const bw = Math.min(40, (innerW / (buckets.length || 1)) * 0.65);
            let y = Y(0);
            return (
              <g key={b.key} transform={`translate(${x - bw / 2},0)`}>
                {/* Stacks */}
                {b.stacks.map((s, k) => {
                  if (s.value <= 0) return null;
                  const y1 = y - (s.value / maxVal) * innerH;
                  const color = palette(s.label);
                  const rect = (
                    <rect
                      key={k}
                      x={0}
                      y={y1}
                      width={bw}
                      height={y - y1}
                      fill={color}
                      opacity="0.9"
                      onMouseMove={(e) => setTip({
                        x: (e as any).clientX,
                        y: (e as any).clientY,
                        html: tipHTML(b, s),
                      })}
                      onMouseLeave={() => setTip(null)}
                    />
                  );
                  y = y1;
                  return rect;
                })}
                {/* X label */}
                <text x={bw / 2} y={h - pad.b + 14} textAnchor="middle" fontSize="10" fill="#cbd5e1">{b.key}</text>
                {/* Totals */}
                <text x={bw / 2} y={Y(b.total) - 4} textAnchor="middle" fontSize="10" fill="#cbd5e1">
                  {b.total > 0 ? `$${formatShort(b.total)}` : ""}
                </text>
              </g>
            );
          })}

          {/* X axis line */}
          <line x1={pad.l} y1={pad.t + innerH} x2={w - pad.r} y2={pad.t + innerH} stroke="#3f3f46" />
        </svg>
      </div>

      {/* Legend + Stats */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-neutral-800 px-3 py-2 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          {legendKeys.map((k) => (
            <div key={k} className="inline-flex items-center gap-1">
              <span className="h-3 w-3 rounded-sm" style={{ background: palette(k) }} />
              <span className="text-neutral-300">{k}</span>
            </div>
          ))}
        </div>
        <div className="text-neutral-400">
          Total ${fmtInt(buckets.reduce((s, b) => s + b.total, 0))}
          {" · "}
          Avg YTM {fmtPct(wAvg(buckets))}
        </div>
      </div>

      {/* Tooltip */}
      {tip && (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 shadow"
          style={{ left: tip.x + 12, top: tip.y + 12 }}
          dangerouslySetInnerHTML={{ __html: tip.html }}
        />
      )}
    </div>
  );
}

/* --------------------------------- Pieces --------------------------------- */

function Select({
  label, value, onChange, options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="text-neutral-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-200"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );
}

/* -------------------------------- Bucketing -------------------------------- */

function bucketSpec(kind: "Month" | "Quarter" | "Year") {
  if (kind === "Month") {
    return {
      key(d: Date) {
        const y = d.getFullYear(), m = d.getMonth();
        const start = new Date(Date.UTC(y, m, 1));
        const end = new Date(Date.UTC(y, m + 1, 1));
        return { key: `${y}-${String(m + 1).padStart(2, "0")}`, startISO: iso(start), endISO: iso(end) };
      },
    };
  }
  if (kind === "Quarter") {
    return {
      key(d: Date) {
        const y = d.getFullYear(), q = Math.floor(d.getMonth() / 3) + 1;
        const start = new Date(Date.UTC(y, (q - 1) * 3, 1));
        const end = new Date(Date.UTC(y, q * 3, 1));
        return { key: `${y} Q${q}`, startISO: iso(start), endISO: iso(end) };
      },
    };
  }
  return {
    key(d: Date) {
      const y = d.getFullYear();
      const start = new Date(Date.UTC(y, 0, 1));
      const end = new Date(Date.UTC(y + 1, 0, 1));
      return { key: `${y}`, startISO: iso(start), endISO: iso(end) };
    },
  };
}

/* --------------------------------- Tooltip -------------------------------- */

function tipHTML(b: Bucketed, s: { label: string; value: number; items: BondLot[] }) {
  const lines = [
    `<div class="text-neutral-300 font-medium">${b.key}</div>`,
    `<div class="text-neutral-400">${s.label}: $${fmtInt(s.value)}</div>`,
  ];
  if (b.wAvgYtm != null) lines.push(`<div class="text-neutral-500">Bucket w.avg YTM ${fmtPct(b.wAvgYtm)}</div>`);
  // top 3 items
  const top = s.items.slice(0, 3);
  if (top.length) {
    lines.push(`<div class="mt-1 text-neutral-400">Top:</div>`);
    for (const it of top) {
      const y = new Date(it.maturity).getFullYear();
      lines.push(
        `<div class="text-neutral-300">• ${escapeHtml(it.symbol)} <span class="text-neutral-500">(${y})</span> — $${fmtInt(it.amount)}${it.ytm!=null?` · ${fmtPct(it.ytm)}`:""}</div>`
      );
    }
    if (s.items.length > 3) lines.push(`<div class="text-neutral-500">… and ${s.items.length - 3} more</div>`);
  }
  return lines.join("");
}

/* --------------------------------- Utils --------------------------------- */

function uniq<T>(a: T[]) { return Array.from(new Set(a)); }
function orderLegend(colorBy: "Sector" | "Rating") {
  if (colorBy === "Rating") {
    const ord = ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-","B+","B","B-","CCC","CC","C","D","Unrated"];
    const pos = (x: string) => { const i = ord.indexOf(x.toUpperCase()); return i < 0 ? ord.length + x.charCodeAt(0) : i; };
    return (a: string, b: string) => pos(a) - pos(b);
  }
  return (a: string, b: string) => a.localeCompare(b);
}

function iso(d: Date) { return new Date(d).toISOString().slice(0, 10); }

function niceTicks(min: number, max: number, n = 4) {
  const span = Math.max(1, max - min);
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step0;
  const mult = err >= 7.5 ? 10 : err >= 3 ? 5 : err >= 1.5 ? 2 : 1;
  const step = mult * step0;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

function fmtInt(n: number) { return Math.round(n).toLocaleString("en-US"); }
function fmtPct(x: number) { return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%"; }
function formatShort(n: number) {
  const abs = Math.abs(n);
  if (abs >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return String(Math.round(n));
}
function wAvg(buckets: Bucketed[]) {
  let sum = 0, w = 0;
  for (const b of buckets) {
    if (b.wAvgYtm == null) continue;
    sum += b.wAvgYtm * b.total;
    w += b.total;
  }
  return w > 0 ? sum / w : 0;
}

function csv(x: any) {
  if (x == null) return "";
  if (typeof x === "number") return String(x);
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function hue(label: string) {
  // deterministic hue 0..360
  const h = hash(label);
  return h % 360;
}
function hash(s: string) { let h = 2166136261 >>> 0; for (let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0; }
function escapeHtml(s: string) { return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]!)); }

/* ------------------- Ambient React (keep zero imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;