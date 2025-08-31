// frontend/components/TradeLogPanel.tsx
import React, { useEffect, useMemo, useState } from "react";

/* ------------------------------- Types ------------------------------- */

export type Side = "BUY" | "SELL";
export type Status = "NEW" | "PARTIAL" | "FILLED" | "CANCELLED" | "REJECTED";

export type Fill = {
  id: string;
  ts: number;       // epoch ms
  qty: number;
  price: number;
  venue?: string;
  fee?: number;
};

export type TradeRow = {
  id: string;
  tsOpen: number;
  tsClose?: number | null;
  symbol: string;
  side: Side;
  qty: number;
  avgPrice?: number | null;
  status: Status;
  strategy?: string | null;
  venue?: string | null;
  tag?: string | null;
  pnl?: number | null;
  notional?: number | null;
  notes?: string | null;
  fills?: Fill[];
};

interface Props {
  trades?: TradeRow[];          // if provided, no fetch
  endpoint?: string;            // GET -> TradeRow[] (default /api/trades)
  pollMs?: number;              // default 10s
  pageSize?: number;            // default 20
  title?: string;               // UI title
  currency?: string;            // money format, default USD
}

/* ------------------------------ Component ---------------------------- */

export default function TradeLogPanel({
  trades,
  endpoint = "/api/trades",
  pollMs = 10_000,
  pageSize = 20,
  title = "Trade Log",
  currency = "USD",
}: Props) {
  const [rows, setRows] = useState<TradeRow[]>(trades ?? []);
  const [err, setErr] = useState<string | null>(null);

  // filters
  const [q, setQ] = useState("");
  const [fSide, setFSide] = useState<Side | "ALL">("ALL");
  const [fStatus, setFStatus] = useState<Status | "ALL">("ALL");
  const [fStrategy, setFStrategy] = useState<string>("ALL");
  const [dateFrom, setDateFrom] = useState<string>("");
  const [dateTo, setDateTo] = useState<string>("");

  // sort / paging
  const [sortKey, setSortKey] = useState<keyof TradeRow>("tsOpen");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);
  const [page, setPage] = useState(1);

  // load/poll if endpoint is provided and no explicit trades prop
  useEffect(() => {
    if (trades) { setRows(trades); return; }
    let ignore = false;
    async function load() {
      try {
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as TradeRow[];
        if (!ignore) { setRows(Array.isArray(json) ? json : []); setErr(null); }
      } catch (e: any) {
        if (!ignore) setErr(e?.message || "Failed to load trades");
      }
    }
    load();
    const t = setInterval(load, pollMs);
    return () => { ignore = true; clearInterval(t); };
  }, [trades, endpoint, pollMs]);

  // strategy list for filter
  const strategies = useMemo(() => {
    const s = new Set<string>();
    rows.forEach(r => r.strategy && s.add(r.strategy));
    return ["ALL", ...Array.from(s).sort()];
  }, [rows]);

  // filtered dataset
  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const t0 = dateFrom ? new Date(dateFrom + "T00:00:00Z").getTime() : -Infinity;
    const t1 = dateTo ? new Date(dateTo + "T23:59:59Z").getTime() : Infinity;

    return rows.filter(r => {
      if (r.tsOpen < t0 || r.tsOpen > t1) return false;
      if (fSide !== "ALL" && r.side !== fSide) return false;
      if (fStatus !== "ALL" && r.status !== fStatus) return false;
      if (fStrategy !== "ALL" && r.strategy !== fStrategy) return false;
      if (!ql) return true;
      const hay = `${r.symbol} ${r.id} ${r.strategy ?? ""} ${r.venue ?? ""} ${r.tag ?? ""} ${r.status}`.toLowerCase();
      return hay.includes(ql);
    });
  }, [rows, q, fSide, fStatus, fStrategy, dateFrom, dateTo]);

  // sorted + paged
  const sorted = useMemo(() => {
    const xs = [...filtered];
    xs.sort((a, b) => {
      const av = (a[sortKey] ?? 0) as any;
      const bv = (b[sortKey] ?? 0) as any;
      if (av < bv) return -1 * sortDir;
      if (av > bv) return 1 * sortDir;
      return 0;
    });
    return xs;
  }, [filtered, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageRows = useMemo(
    () => sorted.slice((page - 1) * pageSize, page * pageSize),
    [sorted, page, pageSize]
  );

  // KPIs
  const kpi = useMemo(() => {
    if (filtered.length === 0) return { count: 0, filled: 0, gross: 0, hit: 0, holdMs: 0 };
    let filled = 0, gross = 0, wins = 0, holdMs = 0, held = 0;
    for (const r of filtered) {
      if (r.status === "FILLED" || r.status === "PARTIAL") {
        filled += Math.abs(r.notional ?? (r.avgPrice ?? 0) * Math.abs(r.qty));
      }
      if (typeof r.pnl === "number") {
        gross += r.pnl;
        if (r.pnl > 0) wins += 1;
      }
      if (r.tsClose && r.tsClose > r.tsOpen) {
        holdMs += (r.tsClose - r.tsOpen);
        held += 1;
      }
    }
    return {
      count: filtered.length,
      filled,
      gross,
      hit: filtered.length ? wins / filtered.length : 0,
      holdMs: held ? holdMs / held : 0,
    };
  }, [filtered]);

  function toggleSort(key: keyof TradeRow) {
    if (sortKey === key) setSortDir(d => (d === 1 ? -1 : 1));
    else { setSortKey(key); setSortDir(key === "tsOpen" ? -1 : 1); }
  }

  function exportCSV() {
    const headers = ["id","time","symbol","side","qty","avgPrice","status","strategy","venue","tag","pnl","notional","holdSec"];
    const lines = sorted.map(r => {
      const hold = r.tsClose ? Math.max(0, r.tsClose - r.tsOpen) / 1000 : "";
      return [
        r.id,
        new Date(r.tsOpen).toISOString(),
        r.symbol,
        r.side,
        r.qty,
        safeNum(r.avgPrice),
        r.status,
        r.strategy ?? "",
        r.venue ?? "",
        r.tag ?? "",
        safeNum(r.pnl),
        safeNum(r.notional),
        hold,
      ].join(",");
    });
    const csv = [headers.join(","), ...lines].join("\n");
    download(`trades_${Date.now()}.csv`, csv, "text/csv;charset=utf-8;");
  }

  /* -------------------------------- Render ------------------------------- */

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <div className="text-xs opacity-70">{rows.length} total • {filtered.length} filtered</div>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-2 py-1 rounded-md border text-sm dark:border-gray-800" onClick={exportCSV}>
            Export CSV
          </button>
        </div>
      </header>

      {/* Filters */}
      <div className="grid gap-2 md:grid-cols-5 mb-3">
        <input
          placeholder="Search (symbol / id / strategy / venue / tag)"
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm md:col-span-2"
          value={q}
          onChange={(e) => { setQ(e.target.value); setPage(1); }}
        />
        <select className="px-2 py-1.5 rounded-md border text-sm dark:border-gray-800" value={fSide}
                onChange={(e) => { setFSide(e.target.value as any); setPage(1); }}>
          <option value="ALL">All Sides</option>
          <option value="BUY">Buy</option>
          <option value="SELL">Sell</option>
        </select>
        <select className="px-2 py-1.5 rounded-md border text-sm dark:border-gray-800" value={fStatus}
                onChange={(e) => { setFStatus(e.target.value as any); setPage(1); }}>
          <option value="ALL">All Status</option>
          {["NEW","PARTIAL","FILLED","CANCELLED","REJECTED"].map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select className="px-2 py-1.5 rounded-md border text-sm dark:border-gray-800" value={fStrategy}
                onChange={(e) => { setFStrategy(e.target.value); setPage(1); }}>
          {strategies.map(s => <option key={s} value={s}>{s === "ALL" ? "All Strategies" : s}</option>)}
        </select>
        <div className="flex items-center gap-2 md:col-span-2">
          <input type="date" className="px-2 py-1.5 rounded-md border text-sm dark:border-gray-800 flex-1"
                 value={dateFrom} onChange={(e) => { setDateFrom(e.target.value); setPage(1); }} />
          <span className="opacity-60 text-xs">→</span>
          <input type="date" className="px-2 py-1.5 rounded-md border text-sm dark:border-gray-800 flex-1"
                 value={dateTo} onChange={(e) => { setDateTo(e.target.value); setPage(1); }} />
        </div>
      </div>

      {/* KPI Strip (inline, no separate component) */}
      <div className="grid gap-2 md:grid-cols-5 mb-3">
        {/* Trades */}
        <div className="p-3 rounded-lg border dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
          <div className="text-xs opacity-70 mb-1">Trades</div>
          <div className="text-base font-semibold">{String(kpi.count)}</div>
        </div>
        {/* Filled Notional */}
        <div className="p-3 rounded-lg border dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
          <div className="text-xs opacity-70 mb-1">Filled Notional</div>
          <div className="text-base font-semibold">{formatMoney(kpi.filled, currency)}</div>
        </div>
        {/* Gross PnL with color */}
        <div className="p-3 rounded-lg border dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
          <div className="text-xs opacity-70 mb-1">Gross PnL</div>
          <div className={`text-base font-semibold ${
            kpi.gross > 0 ? "text-green-600" : kpi.gross < 0 ? "text-red-600" : "opacity-80"
          }`}>
            {formatMoney(kpi.gross, currency)}
          </div>
        </div>
        {/* Hit Rate */}
        <div className="p-3 rounded-lg border dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
          <div className="text-xs opacity-70 mb-1">Hit Rate</div>
          <div className="text-base font-semibold">{`${(kpi.hit * 100).toFixed(1)}%`}</div>
        </div>
        {/* Avg Hold */}
        <div className="p-3 rounded-lg border dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
          <div className="text-xs opacity-70 mb-1">Avg Hold</div>
          <div className="text-base font-semibold">{formatDuration(kpi.holdMs)}</div>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-xl border dark:border-gray-800 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
            <tr>
              <Th label="Time" onClick={() => toggleSort("tsOpen")} active={sortKey === "tsOpen"} dir={sortDir} />
              <Th label="Symbol" onClick={() => toggleSort("symbol")} active={sortKey === "symbol"} dir={sortDir} />
              <Th label="Side" onClick={() => toggleSort("side")} active={sortKey === "side"} dir={sortDir} />
              <Th label="Qty" onClick={() => toggleSort("qty")} active={sortKey === "qty"} dir={sortDir} right />
              <Th label="Avg Px" onClick={() => toggleSort("avgPrice")} active={sortKey === "avgPrice"} dir={sortDir} right />
              <Th label="Status" onClick={() => toggleSort("status")} active={sortKey === "status"} dir={sortDir} />
              <Th label="Strategy" onClick={() => toggleSort("strategy")} active={sortKey === "strategy"} dir={sortDir} />
              <Th label="Venue" onClick={() => toggleSort("venue")} active={sortKey === "venue"} dir={sortDir} />
              <Th label="Tag" onClick={() => toggleSort("tag")} active={sortKey === "tag"} dir={sortDir} />
              <Th label="PnL" onClick={() => toggleSort("pnl")} active={sortKey === "pnl"} dir={sortDir} right />
              <Th label="" />
            </tr>
          </thead>
          <tbody>
            {pageRows.map((r) => (
              <Row key={r.id} r={r} currency={currency} />
            ))}
            {pageRows.length === 0 && (
              <tr><td colSpan={11} className="p-4 text-center opacity-60">No trades match your filters.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="mt-3 flex items-center justify-between text-sm">
        <div className="opacity-70">
          Page <b>{page}</b> / {totalPages} • Rows {Math.min((page - 1) * pageSize + 1, sorted.length)}–
          {Math.min(page * pageSize, sorted.length)} of {sorted.length}
        </div>
        <div className="flex gap-2">
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(1)} disabled={page === 1}>⏮</button>
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>‹ Prev</button>
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next ›</button>
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(totalPages)} disabled={page === totalPages}>⏭</button>
        </div>
      </div>

      {err && <div className="mt-2 text-sm text-red-600">Error: {err}</div>}
    </div>
  );
}

/* --------------------------- Subcomponents --------------------------- */

function Th({ label, onClick, active, dir, right }: { label: string; onClick?: () => void; active?: boolean; dir?: 1 | -1; right?: boolean }) {
  return (
    <th
      className={`p-2 text-left ${right ? "text-right" : ""} cursor-pointer select-none`}
      onClick={onClick}
      title="Sort"
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {active && <span className="opacity-60 text-[10px]">{dir === -1 ? "▼" : "▲"}</span>}
      </span>
    </th>
  );
}

function Row({ r, currency }: { r: TradeRow; currency: string }) {
  const [open, setOpen] = useState(false);
  const tone = (r.pnl ?? 0) > 0 ? "text-green-600" : (r.pnl ?? 0) < 0 ? "text-red-600" : "opacity-80";
  return (
    <>
      <tr className="border-t dark:border-gray-800">
        <td className="p-2 whitespace-nowrap">{new Date(r.tsOpen).toLocaleString()}</td>
        <td className="p-2 font-medium">{r.symbol}</td>
        <td className="p-2">
          <span className={`text-[11px] px-2 py-0.5 rounded-full ${r.side === "BUY" ? "bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300" : "bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300"}`}>
            {r.side}
          </span>
        </td>
        <td className="p-2 text-right">{formatNumber(r.qty)}</td>
        <td className="p-2 text-right">{formatNumber(r.avgPrice)}</td>
        <td className="p-2">
          <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800">{r.status}</span>
        </td>
        <td className="p-2">{r.strategy ?? "—"}</td>
        <td className="p-2">{r.venue ?? "—"}</td>
        <td className="p-2">{r.tag ?? "—"}</td>
        <td className={`p-2 text-right ${tone}`}>{formatMoney(r.pnl, currency)}</td>
        <td className="p-2 text-right">
          {r.fills?.length ? (
            <button className="px-2 py-1 rounded-md border text-xs dark:border-gray-800" onClick={() => setOpen(o => !o)}>
              {open ? "Hide" : "Fills"}
            </button>
          ) : null}
        </td>
      </tr>
      {open && r.fills && (
        <tr className="bg-gray-50 dark:bg-gray-800/40">
          <td className="p-2" colSpan={11}>
            <div className="text-xs font-medium mb-1">Fills ({r.fills.length})</div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="opacity-70">
                    <th className="p-1 text-left">ID</th>
                    <th className="p-1 text-left">Time</th>
                    <th className="p-1 text-right">Qty</th>
                    <th className="p-1 text-right">Price</th>
                    <th className="p-1 text-left">Venue</th>
                    <th className="p-1 text-right">Fee</th>
                  </tr>
                </thead>
                <tbody>
                  {r.fills.map(f => (
                    <tr key={f.id} className="border-t dark:border-gray-800">
                      <td className="p-1">{f.id}</td>
                      <td className="p-1">{new Date(f.ts).toLocaleString()}</td>
                      <td className="p-1 text-right">{formatNumber(f.qty)}</td>
                      <td className="p-1 text-right">{formatNumber(f.price)}</td>
                      <td className="p-1">{f.venue ?? "—"}</td>
                      <td className="p-1 text-right">{formatMoney(f.fee, currency)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

/* -------------------------------- Helpers ------------------------------- */

function formatNumber(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return "—";
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 8 }); } catch { return String(x); }
}
function formatMoney(x: number | null | undefined, ccy = "USD"): string {
  if (x == null || !Number.isFinite(x)) return "—";
  try { return x.toLocaleString(undefined, { style: "currency", currency: ccy }); }
  catch { return `${ccy} ${Number(x).toFixed(2)}`; }
}
function formatDuration(ms: number | null | undefined): string {
  if (!ms || !Number.isFinite(ms)) return "—";
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${ss}s`;
  return `${ss}s`;
}
function safeNum(v: any) { return Number.isFinite(v) ? v : ""; }

function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}