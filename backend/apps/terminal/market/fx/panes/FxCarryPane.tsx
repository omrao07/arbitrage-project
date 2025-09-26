"use client";

/**
 * fx carrypane.tsx
 * Zero-import, self-contained FX Carry pane.
 *
 * What it shows
 * - For each pair (e.g., EURUSD), computes interest-rate differential (carry) and theoretical forwards
 *   using Covered Interest Parity:  F = S * (1 + r_quote*T) / (1 + r_base*T)
 * - Displays annualized carry (r_base − r_quote), forward premium/discount by tenor,
 *   and forward points in pips.
 * - Sortable table, search, compact bar chart of annualized carry, CSV export.
 *
 * Assumptions
 * - Rates are simple annualized decimals (e.g., 0.035 = 3.5%) on ACT/365 approximation.
 * - If you already have forward points, you can pass them via `overrides` (per tenor).
 *
 * Tailwind + inline SVG only. No imports.
 */

export type FxCarryItem = {
  pair: string;              // "EURUSD"
  spot: number;              // e.g., 1.0832  (quote units per base)
  baseRate: number;          // r_base  (e.g., EUR O/N->1Y proxy as decimal)
  quoteRate: number;         // r_quote (e.g., USD …)
  tenorsDays?: number[];     // optional custom tenors (days). Default: [30, 90, 180, 365]
  overrides?: {              // optional overrides for forward points (quote units)
    // map tenorDays -> forward points (F - S), if you want to plug your own
    [tenorDays: number]: number;
  };
};

export default function FxCarryPane({
  title = "FX Carry",
  items = [],
  className = "",
  defaultTenors = [30, 90, 180, 365],
}: {
  title?: string;
  items: FxCarryItem[];
  className?: string;
  defaultTenors?: number[];
}) {
  // ---- Controls ----
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "carry", dir: "desc" });

  // ---- Normalize / compute ----
  const rows = useMemo(() => {
    const out: Row[] = [];
    for (const it of items) {
      const pair = normPair(it.pair);
      if (!pair) continue;
      const base = pair.slice(0, 3), quote = pair.slice(3);
      const spot = +it.spot;
      if (!isFinite(spot) || spot <= 0) continue;

      const rB = +it.baseRate || 0;
      const rQ = +it.quoteRate || 0;
      const carry = rB - rQ; // simple annualized rate differential

      const tenors = (it.tenorsDays && it.tenorsDays.length ? sortNums(it.tenorsDays) : defaultTenors).filter((d) => d > 0);
      const row: Row = {
        pair, base, quote, spot, rBase: rB, rQuote: rQ, carry,
        tenors,
        forwards: [],
      };

      for (const T of tenors) {
        const y = T / 365;
        const cipFwd = spot * ((1 + rQ * y) / (1 + rB * y)); // Covered Interest Parity
        const overridePts = it.overrides?.[T];
        const F = isFinite(overridePts as number) ? spot + (overridePts as number) : cipFwd;

        const points = F - spot;                                  // quote units
        const pip = pipSize(pair);
        const fwdPips = points / pip;                             // in pips
        const prem = (F / spot) - 1;                              // forward premium (total)
        const premAnn = prem / y;                                 // annualized
        row.forwards.push({ tenorDays: T, F, points, fwdPips, prem, premAnn });
      }

      out.push(row);
    }
    return out;
  }, [items, defaultTenors]);

  const filtered = useMemo(() => {
    const term = q.trim().toLowerCase();
    const arr = term
      ? rows.filter(r => (r.pair + " " + r.base + " " + r.quote).toLowerCase().includes(term))
      : rows.slice();
    arr.sort((a, b) => {
      const dir = sort.dir === "asc" ? 1 : -1;
      switch (sort.key) {
        case "pair": return a.pair.localeCompare(b.pair) * dir;
        case "spot": return (a.spot - b.spot) * dir;
        case "carry": return (a.carry - b.carry) * dir;
        case "rBase": return (a.rBase - b.rBase) * dir;
        case "rQuote": return (a.rQuote - b.rQuote) * dir;
        default: return 0;
      }
    });
    return arr;
  }, [rows, q, sort]);

  // ---- Chart domain ----
  const carryVals = filtered.map(r => r.carry);
  const yMin = Math.min(0, ...(carryVals.length ? carryVals : [0]));
  const yMax = Math.max(0, ...(carryVals.length ? carryVals : [0.01]));
  const pad = (yMax - yMin) * 0.1 || 0.01;
  const Y0 = yMin - pad;
  const Y1 = yMax + pad;

  // ---- Clipboard ----
  const copy = () => copyCSV(filtered);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {filtered.length} pairs · annualized carry shown as r<sub>base</sub> − r<sub>quote</sub>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <label className="relative">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search pair (e.g., usd, jpy)…"
              className="w-56 rounded-md border border-neutral-700 bg-neutral-950 pl-7 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
            />
            <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
              <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </label>
          <button onClick={copy} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 hover:bg-neutral-800">
            Copy CSV
          </button>
        </div>
      </div>

      {/* Carry bar chart */}
      <BarChart rows={filtered} y0={Y0} y1={Y1} />

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-neutral-950 text-[11px] uppercase text-neutral-500">
            <tr className="border-b border-neutral-800">
              <Th label="Pair"     k="pair"   sort={sort} onSort={setSort} />
              <Th label="Spot"     k="spot"   sort={sort} onSort={setSort} right />
              <Th label="r Base"   k="rBase"  sort={sort} onSort={setSort} right />
              <Th label="r Quote"  k="rQuote" sort={sort} onSort={setSort} right />
              <Th label="Carry (ann.)" k="carry" sort={sort} onSort={setSort} right />
              <th className="px-3 py-2 text-right">Forwards / Points (by tenor)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {filtered.length === 0 && (
              <tr><td colSpan={6} className="px-3 py-6 text-center text-xs text-neutral-500">No matches.</td></tr>
            )}
            {filtered.map((r) => (
              <tr key={r.pair} className="hover:bg-neutral-800/40">
                <td className="px-3 py-2 font-medium text-neutral-100">{prettyPair(r.pair)}</td>
                <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.spot, dp(r.pair))}</td>
                <td className="px-3 py-2 text-right font-mono tabular-nums">{fmtPct(r.rBase)}</td>
                <td className="px-3 py-2 text-right font-mono tabular-nums">{fmtPct(r.rQuote)}</td>
                <td className={`px-3 py-2 text-right font-mono tabular-nums ${r.carry>=0?"text-emerald-400":"text-rose-400"}`}>{fmtPct(r.carry)}</td>
                <td className="px-3 py-2">
                  <div className="flex flex-wrap items-center gap-3 justify-end">
                    {r.forwards.map(f => (
                      <div key={`${r.pair}-${f.tenorDays}`} className="text-xs text-neutral-300">
                        <span className="text-neutral-500">{tenorLabel(f.tenorDays)}:</span>{" "}
                        <span className="font-mono">{fmt(f.F, dp(r.pair))}</span>
                        <span className="text-neutral-500"> · </span>
                        <span className={`${f.points>=0?"text-emerald-400":"text-rose-400"} font-mono`}>{fmtPips(f.fwdPips)} pts</span>
                        <span className="text-neutral-500"> · </span>
                        <span className="font-mono">{fmtPct(f.premAnn)}</span>
                      </div>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* --------------------------------- Bar Chart -------------------------------- */

function BarChart({ rows, y0, y1 }: { rows: Row[]; y0: number; y1: number }) {
  const w = 980, h = 220;
  const pad = { l: 56, r: 16, t: 16, b: 40 };
  const innerW = Math.max(1, w - pad.l - pad.r);
  const innerH = Math.max(1, h - pad.t - pad.b);

  const N = Math.max(1, rows.length);
  const bw = Math.min(50, innerW / N * 0.7);
  const X = (i: number) => pad.l + (i + 0.5) * (innerW / N);
  const Y = (v: number) => pad.t + (1 - (v - y0) / (y1 - y0 || 1)) * innerH;

  const ticks = niceTicks(y0, y1, 5);

  const [tip, setTip] = useState<{ x: number; y: number; html: string } | null>(null);

  return (
    <div className="p-3 relative">
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
        {/* Grid */}
        {ticks.map((v, i) => (
          <g key={i}>
            <line x1={pad.l} y1={Y(v)} x2={w - pad.r} y2={Y(v)} stroke="#27272a" strokeDasharray="3 3" />
            <text x={4} y={Y(v) + 4} fontSize="10" fill="#9ca3af">{fmtPct(v)}</text>
          </g>
        ))}

        {/* Bars */}
        {rows.map((r, i) => {
          const y = r.carry;
          const yZero = Y(0), yVal = Y(y);
          const x = X(i) - bw / 2;
          const barH = Math.abs(yVal - yZero);
          const yTop = Math.min(yVal, yZero);
          const color = y >= 0 ? "#34d399" : "#fb7185";
          return (
            <g key={r.pair} transform={`translate(${x},0)`}>
              <rect
                x={0} y={yTop} width={bw} height={barH}
                fill={color} opacity="0.9"
                onMouseMove={(e) => setTip({
                  x: (e as any).clientX, y: (e as any).clientY,
                  html: `<div class="text-neutral-200 font-medium">${prettyPair(r.pair)}</div>
                         <div class="text-neutral-400">Spot <span class="font-mono text-neutral-200">${fmt(r.spot, dp(r.pair))}</span></div>
                         <div class="text-neutral-400">Carry <span class="font-mono text-neutral-200">${fmtPct(r.carry)}</span></div>`
                })}
                onMouseLeave={() => setTip(null)}
              />
              <text x={bw/2} y={h - 10} textAnchor="middle" fontSize="10" fill="#cbd5e1">{r.pair}</text>
            </g>
          );
        })}

        {/* Axis line */}
        <line x1={pad.l} y1={Y(0)} x2={w - pad.r} y2={Y(0)} stroke="#3f3f46" />
      </svg>

      {tip && (
        <div
          className="pointer-events-none absolute z-50 max-w-xs rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 shadow"
          style={{ left: Math.min(w - 180, Math.max(8, tip.x + 12)), top: Math.max(8, tip.y + 12) }}
          dangerouslySetInnerHTML={{ __html: tip.html }}
        />
      )}
    </div>
  );
}

/* ---------------------------------- Bits ---------------------------------- */

type SortKey = "pair" | "spot" | "carry" | "rBase" | "rQuote";

type Row = {
  pair: string; base: string; quote: string;
  spot: number; rBase: number; rQuote: number; carry: number;
  tenors: number[];
  forwards: { tenorDays: number; F: number; points: number; fwdPips: number; prem: number; premAnn: number }[];
};

function Th({
  label, k, sort, onSort, right = false,
}: {
  label: string;
  k: SortKey;
  sort: { key: SortKey; dir: "asc" | "desc" };
  onSort: (s: { key: SortKey; dir: "asc" | "desc" }) => void;
  right?: boolean;
}) {
  const active = sort.key === k;
  const dir = active ? sort.dir : undefined;
  return (
    <th
      className={`cursor-pointer select-none px-3 py-2 ${right ? "text-right" : "text-left"}`}
      onClick={() => onSort({ key: k, dir: active && dir === "desc" ? "asc" : "desc" })}
      title="Sort"
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <svg width="10" height="10" viewBox="0 0 24 24" className={`opacity-70 ${active ? "" : "invisible"}`}>
          {dir === "asc"
            ? <path d="M7 14l5-5 5 5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />
            : <path d="M7 10l5 5 5-5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />
          }
        </svg>
      </span>
    </th>
  );
}

/* ---------------------------------- Utils ---------------------------------- */

function normPair(p: string) {
  const s = (p || "").toUpperCase().replace(/[^A-Z]/g, "");
  return s.length === 6 ? s : "";
}
function prettyPair(p: string) { return `${p.slice(0,3)}/${p.slice(3)}`; }
function pipSize(pair: string) { return pair.slice(3).toUpperCase() === "JPY" ? 0.01 : 0.0001; }
function dp(pair: string) { return pair.slice(3).toUpperCase() === "JPY" ? 3 : 5; }
function sortNums(a: number[]) { return [...a].sort((x, y) => x - y); }
function tenorLabel(d: number) {
  if (d % 365 === 0) return `${Math.round(d/365)}Y`;
  if (d % 30 === 0) return `${Math.round(d/30)}M`;
  return `${d}d`;
}

function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtPct(x: number) { return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%"; }
function fmtPips(x: number) {
  const sign = x >= 0 ? "" : "-";
  const abs = Math.abs(x);
  return sign + abs.toLocaleString("en-US", { maximumFractionDigits: 1 });
}
function niceTicks(min: number, max: number, n = 5) {
  if (!(max > min)) return [min];
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / Math.max(1, n))));
  const err = (span / n) / step0;
  const mult = err >= 7.5 ? 10 : err >= 3 ? 5 : err >= 1.5 ? 2 : 1;
  const step = mult * step0;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + 1e-9; v += step) out.push(v);
  if (!out.includes(0)) out.push(0);
  return out.sort((a,b)=>a-b);
}

function copyCSV(rows: Row[]) {
  // columns: pair, spot, rBase, rQuote, carry, for each tenor: F, points(pips), premAnn
  const tenors = Array.from(new Set(rows.flatMap(r => r.tenors))).sort((a,b)=>a-b);
  const head = [
    "pair","spot","r_base","r_quote","carry_ann",
    ...tenors.flatMap(t => [`F_${t}d`,`points_${t}d_pips`,`premAnn_${t}d`]),
  ];
  const lines = [head.join(",")];

  for (const r of rows) {
    const map: Record<number, Row["forwards"][number]> = {};
    for (const f of r.forwards) map[f.tenorDays] = f;
    const base = [r.pair, r.spot, r.rBase, r.rQuote, r.carry].map(csv);
    const tail: string[] = [];
    for (const t of tenors) {
      const f = map[t];
      if (f) {
        tail.push(csv(f.F), csv(Math.round(f.fwdPips * 10) / 10), csv(f.premAnn));
      } else {
        tail.push("","","");
      }
    }
    lines.push([...base, ...tail].join(","));
  }

  const csvStr = lines.join("\n");
  try { (navigator as any).clipboard?.writeText(csvStr); } catch {}
}

function csv(x: any) {
  if (x == null) return "";
  if (typeof x === "number") return String(x);
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;

/* ---------------------------------- Example --------------------------------
const items: FxCarryItem[] = [
  { pair: "EURUSD", spot: 1.0832, baseRate: 0.035, quoteRate: 0.045 },
  { pair: "USDJPY", spot: 155.12, baseRate: 0.055, quoteRate: 0.010 },
  { pair: "GBPUSD", spot: 1.2705, baseRate: 0.045, quoteRate: 0.045 },
];
<FxCarryPane items={items} />
---------------------------------------------------------------------------- */