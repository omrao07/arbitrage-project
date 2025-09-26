"use client";

/**
 * bondscreenerpane.tsx
 * Zero-import, self-contained bond screener pane (gov/credit).
 *
 * Features:
 * - Accepts pre-fetched quotes and optional gov curve (for G-spread)
 * - Text search (symbol/ISIN), quick filters (tenor / YTM / Duration ranges)
 * - Sortable table, sticky header, pagination
 * - CSV export (Copy) & selection count
 * - Tailwind + inline SVG only (no imports)
 */

export type CurvePoint = { tenor: string; years: number; yield: number }; // decimal
export type BondQuote = {
  symbol: string; isin?: string; tenorYears: number;
  coupon: number; maturity: string; cleanPrice: number; ytm: number; // decimals
  duration: number; convexity: number;
  bid: number; ask: number; bidSize: number; askSize: number;
};

type SortKey =
  | "symbol" | "isin" | "tenorYears" | "coupon" | "maturity"
  | "cleanPrice" | "ytm" | "duration" | "convexity"
  | "bid" | "ask" | "bidSize" | "askSize" | "gspread";

export default function BondScreenerPane({
  title = "Bond Screener",
  quotes = [],
  curve,                 // optional gov curve for G-spread computation
  className = "",
  pageSize = 12,
}: {
  title?: string;
  quotes: BondQuote[];
  curve?: CurvePoint[];
  className?: string;
  pageSize?: number;
}) {
  // Controls
  const [q, setQ] = useState("");
  const [tenorMin, setTenorMin] = useState(0);
  const [tenorMax, setTenorMax] = useState(30);
  const [ytmMin, setYtmMin] = useState(0);
  const [ytmMax, setYtmMax] = useState(0.20);
  const [durMin, setDurMin] = useState(0);
  const [durMax, setDurMax] = useState(20);
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "tenorYears", dir: "asc" });
  const [page, setPage] = useState(1);
  const [sel, setSel] = useState<Record<number, boolean>>({});

  // Range auto from data
  const stats = useMemo(() => {
    const ts = quotes.map((x) => x.tenorYears);
    const ys = quotes.map((x) => x.ytm);
    const ds = quotes.map((x) => x.duration);
    return {
      tMin: Math.min(...(ts.length ? ts : [0])),
      tMax: Math.max(...(ts.length ? ts : [30])),
      yMin: clamp(Math.min(...(ys.length ? ys : [0])), -0.01, 1),
      yMax: clamp(Math.max(...(ys.length ? ys : [0.2])), -0.01, 1),
      dMin: Math.max(0, Math.min(...(ds.length ? ds : [0]))),
      dMax: Math.max(1, Math.max(...(ds.length ? ds : [10]))),
    };
  }, [quotes]);

  // Initialize ranges once when data mounts
  useEffect(() => {
    setTenorMin(Math.floor(stats.tMin));
    setTenorMax(Math.ceil(stats.tMax));
    setYtmMin(Math.max(0, round(stats.yMin, 3)));
    setYtmMax(round(stats.yMax, 3));
    setDurMin(Math.floor(stats.dMin));
    setDurMax(Math.ceil(stats.dMax));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stats.tMin, stats.tMax, stats.yMin, stats.yMax, stats.dMin, stats.dMax]);

  // G-spread compute (decimal). Interpolate curve by tenor.
  const curveFn = useMemo(() => {
    if (!curve || !curve.length) return undefined;
    const pts = curve.slice().sort((a, b) => a.years - b.years);
    const xs = pts.map((p) => p.years), ys = pts.map((p) => p.yield);
    return (T: number) => interp(xs, ys, T);
  }, [curve]);

  // Derived rows with g-spread
  const rows = useMemo(() => {
    return quotes.map((r, i) => {
      const base = curveFn ? curveFn(r.tenorYears) : NaN;
      const g = isFinite(base) ? r.ytm - (base as number) : NaN;
      return { ...r, _idx: i, gspread: g };
    });
  }, [quotes, curveFn]);

  const qlc = q.trim().toLowerCase();
  const filtered = useMemo(() => {
    let arr = rows.filter((r) => (
      r.tenorYears >= tenorMin && r.tenorYears <= tenorMax &&
      r.ytm >= ytmMin && r.ytm <= ytmMax &&
      r.duration >= durMin && r.duration <= durMax &&
      (qlc
        ? (
            (r.symbol + " " + (r.isin || "") + " " + r.maturity)
            .toLowerCase()
            .includes(qlc)
          )
        : true)
    ));
    arr.sort((a, b) => {
      const dir = sort.dir === "asc" ? 1 : -1;
      const ka = (a as any)[sort.key], kb = (b as any)[sort.key];
      if (typeof ka === "string" && typeof kb === "string") return ka.localeCompare(kb) * dir;
      const av = (isNaN(ka) ? -Infinity : ka) as number;
      const bv = (isNaN(kb) ? -Infinity : kb) as number;
      return (av - bv) * dir;
    });
    return arr;
  }, [rows, qlc, tenorMin, tenorMax, ytmMin, ytmMax, durMin, durMax, sort]);

  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  useEffect(() => { if (page > pages) setPage(pages); }, [pages, page]);

  // Keyboard shortcuts
  const searchRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (!inField && e.key === "/") { e.preventDefault(); searchRef.current?.focus(); }
      if (inField && e.key === "Escape") {
        (e.target as HTMLInputElement).blur(); setQ("");
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // Page slice
  const view = filtered.slice((page - 1) * pageSize, page * pageSize);

  // Selection helpers
  const allChecked = view.length > 0 && view.every((r) => sel[r._idx]);
  const anyChecked = view.some((r) => sel[r._idx]);
  const toggleAll = () => {
    const next = { ...sel };
    if (allChecked) {
      for (const r of view) delete next[r._idx];
    } else {
      for (const r of view) next[r._idx] = true;
    }
    setSel(next);
  };

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {quotes.length} total · {filtered.length} match
            {curveFn ? " · G-spread enabled" : " · No curve (G-spread —)"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label className="relative">
            <input
              ref={searchRef}
              value={q}
              onChange={(e) => { setQ(e.target.value); setPage(1); }}
              placeholder="Search symbol / ISIN  (press /)"
              className="w-56 rounded-md border border-neutral-700 bg-neutral-950 pl-7 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
            />
            <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
              <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </label>

          <Range label="Tenor" value={[tenorMin, tenorMax]} onChange={(a, b) => { setTenorMin(a); setTenorMax(b); setPage(1); }} fmt={(v)=>`${v}y`} step={1} min={Math.floor(stats.tMin)} max={Math.ceil(stats.tMax)} />
          <Range label="YTM" value={[ytmMin, ytmMax]} onChange={(a, b) => { setYtmMin(a); setYtmMax(b); setPage(1); }} fmt={(v)=>fmtPct(v)} step={0.001} min={round(stats.yMin,3)} max={round(stats.yMax,3)} />
          <Range label="Dur" value={[durMin, durMax]} onChange={(a, b) => { setDurMin(a); setDurMax(b); setPage(1); }} fmt={(v)=>v.toFixed(1)} step={0.5} min={Math.floor(stats.dMin)} max={Math.ceil(stats.dMax)} />

          <button
            onClick={() => copyCSV(view, curveFn != null)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 hover:bg-neutral-800"
            title="Copy current page as CSV"
          >
            Copy CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-neutral-950 text-[11px] uppercase text-neutral-500">
            <tr className="border-b border-neutral-800">
              <th className="px-3 py-2">
                <input type="checkbox" checked={allChecked} ref={(el) => { if (el) el.indeterminate = !allChecked && anyChecked; }} onChange={toggleAll} />
              </th>
              <Th label="Symbol" k="symbol" sort={sort} onSort={setSort} />
              <Th label="ISIN" k="isin" sort={sort} onSort={setSort} />
              <Th label="Tenor" k="tenorYears" sort={sort} onSort={setSort} right />
              <Th label="Coupon" k="coupon" sort={sort} onSort={setSort} right />
              <Th label="Maturity" k="maturity" sort={sort} onSort={setSort} />
              <Th label="Price" k="cleanPrice" sort={sort} onSort={setSort} right />
              <Th label="YTM" k="ytm" sort={sort} onSort={setSort} right />
              <Th label="Dur" k="duration" sort={sort} onSort={setSort} right />
              <Th label="Conv" k="convexity" sort={sort} onSort={setSort} right />
              <Th label="Bid" k="bid" sort={sort} onSort={setSort} right />
              <Th label="Ask" k="ask" sort={sort} onSort={setSort} right />
              <Th label="G-Spread" k="gspread" sort={sort} onSort={setSort} right />
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {view.length === 0 && (
              <tr>
                <td colSpan={13} className="px-3 py-6 text-center text-xs text-neutral-500">No bonds match your filters.</td>
              </tr>
            )}
            {view.map((r) => {
              const on = !!sel[r._idx];
              const pos = (r.ytm - (curveFn ? curveFn(r.tenorYears) : r.ytm)) >= 0;
              return (
                <tr key={r._idx} className={`hover:bg-neutral-800/40 ${on ? "bg-neutral-800/30" : ""}`}>
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={(e) => setSel({ ...sel, [r._idx]: e.target.checked })}
                    />
                  </td>
                  <td className="px-3 py-2 font-medium text-neutral-100">{r.symbol}</td>
                  <td className="px-3 py-2 text-neutral-300">{r.isin ?? "—"}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.tenorYears, 2)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmtPct(r.coupon)}</td>
                  <td className="px-3 py-2 text-neutral-300">{r.maturity}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.cleanPrice, 3)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmtPct(r.ytm)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.duration, 2)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.convexity, 2)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.bid, 3)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(r.ask, 3)}</td>
                  <td className={`px-3 py-2 text-right font-mono tabular-nums ${isFinite(r.gspread) ? (pos ? "text-emerald-400" : "text-rose-400") : "text-neutral-500"}`}>
                    {isFinite(r.gspread) ? fmtBps(r.gspread) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-neutral-800 px-3 py-2 text-xs text-neutral-400">
        <div>
          Page {page} / {pages} · Selected {Object.values(sel).filter(Boolean).length}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800"
            disabled={page <= 1}
          >
            ← Prev
          </button>
          <button
            onClick={() => setPage((p) => Math.min(pages, p + 1))}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800"
            disabled={page >= pages}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------- Components -------------------------------- */

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

function Range({
  label, value, onChange, fmt, min, max, step,
}: {
  label: string;
  value: [number, number];
  onChange: (a: number, b: number) => void;
  fmt: (v: number) => string;
  min: number;
  max: number;
  step: number;
}) {
  const [a, b] = value;
  return (
    <div className="flex items-center gap-2">
      <span className="text-neutral-400">{label}</span>
      <input
        type="number"
        step={step}
        value={String(a)}
        onChange={(e) => onChange(clamp(+e.target.value || a, min, b), b)}
        className="w-20 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
      />
      <span>–</span>
      <input
        type="number"
        step={step}
        value={String(b)}
        onChange={(e) => onChange(a, clamp(+e.target.value || b, a, max))}
        className="w-20 rounded border border-neutral-700 bg-neutral-950 px-2 py-1 text-right text-neutral-200"
      />
      <span className="text-neutral-500">{fmt(a)} – {fmt(b)}</span>
    </div>
  );
}

/* ---------------------------------- Utils ---------------------------------- */

function interp(xs: number[], ys: number[], x: number) {
  if (!xs.length) return NaN;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (i < xs.length && xs[i] < x) i++;
  const x0 = xs[i - 1], x1 = xs[i], y0 = ys[i - 1], y1 = ys[i];
  const t = (x - x0) / (x1 - x0 || 1);
  return y0 + (y1 - y0) * t;
}

function copyCSV(rows: (BondQuote & { _idx: number; gspread: number })[], hasG: boolean) {
  const cols = ["symbol","isin","tenorYears","coupon","maturity","cleanPrice","ytm","duration","convexity","bid","ask","bidSize","askSize"];
  if (hasG) cols.push("gspread_bps");
  const head = cols.join(",");
  const lines = rows.map(r => {
    const base = [
      r.symbol, r.isin ?? "",
      r.tenorYears, r.coupon, r.maturity, r.cleanPrice,
      r.ytm, r.duration, r.convexity, r.bid, r.ask, r.bidSize, r.askSize
    ].map(safeCSV);
    if (hasG) base.push(String(Math.round((r.gspread || 0) * 10000)));
    return base.join(",");
  });
  const csv = [head, ...lines].join("\n");
  try {
    (navigator as any).clipboard?.writeText(csv);
  } catch {
    // fallback: textarea trick
    const ta = document.createElement("textarea");
    ta.value = csv; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); document.body.removeChild(ta);
  }
}

function safeCSV(x: any) {
  if (x == null) return "";
  if (typeof x === "number") return String(x);
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtPct(x: number) { return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%"; }
function fmtBps(x: number) { return (x * 10000).toLocaleString("en-US", { maximumFractionDigits: 0 }) + " bps"; }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function round(n: number, d = 3) { const p = 10 ** d; return Math.round(n * p) / p; }

/* -------------------------- Ambient React (no imports) -------------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useRef<T>(v: T | null): { current: T | null };

/* ---------------------------------- Example ----------------------------------
import { fetchBondSnapshot } from "@/server/fetchbondsserver";
const snap = await fetchBondSnapshot({ country: "US" });
<BondScreenerPane quotes={snap.quotes} curve={snap.govCurve} />
-------------------------------------------------------------------------------- */