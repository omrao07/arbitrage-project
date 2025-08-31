// frontend/components/LedgerView.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
} from "recharts";

/* -------------------------------- Types -------------------------------- */

export type LedgerType =
  | "trade"
  | "deposit"
  | "withdrawal"
  | "fee"
  | "interest"
  | "dividend"
  | "transfer";

export type Side = "buy" | "sell" | null;

export interface LedgerEntry {
  id: string;
  ts: number;              // epoch ms
  type: LedgerType;
  symbol?: string | null;
  side?: Side;             // for trade rows
  qty?: number | null;
  price?: number | null;
  fee?: number | null;     // negative for charges
  cashDelta: number;       // +credit, -debit
  pnl?: number | null;     // realized PnL for closing trades (can be 0)
  venue?: string | null;
  account?: string | null;
  orderId?: string | null;
  note?: string | null;
}

interface Props {
  entries?: LedgerEntry[];        // if provided, no fetch
  endpoint?: string;              // GET -> LedgerEntry[]
  baseCurrency?: string;          // label for money, e.g., "USD"
  pageSize?: number;
}

/* ------------------------------- Component ------------------------------ */

export default function LedgerView({
  entries,
  endpoint = "/api/ledger",
  baseCurrency = "USD",
  pageSize = 25,
}: Props) {
  const [raw, setRaw] = useState<LedgerEntry[] | null>(entries ?? null);
  const [err, setErr] = useState<string | null>(null);

  // filters
  const [q, setQ] = useState("");
  const [type, setType] = useState<LedgerType | "all">("all");
  const [side, setSide] = useState<Side | "all">("all");
  const [from, setFrom] = useState<string>(""); // yyyy-mm-dd
  const [to, setTo] = useState<string>("");

  // sorting
  const [sortKey, setSortKey] = useState<keyof LedgerEntry>("ts");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // pagination
  const [page, setPage] = useState(1);

  useEffect(() => {
    if (entries) { setRaw(entries); return; }
    (async () => {
      try {
        const res = await fetch(endpoint);
        const json = (await res.json()) as LedgerEntry[];
        setRaw(Array.isArray(json) ? json : []);
      } catch (e: any) {
        setErr(e?.message || "Failed to load ledger");
        setRaw([]);
      }
    })();
  }, [entries, endpoint]);

  const filtered = useMemo(() => {
    if (!raw) return [];
    let rows = [...raw];

    // date filter
    if (from) {
      const t = new Date(from + "T00:00:00Z").getTime();
      rows = rows.filter((r) => r.ts >= t);
    }
    if (to) {
      const t = new Date(to + "T23:59:59Z").getTime();
      rows = rows.filter((r) => r.ts <= t);
    }

    if (type !== "all") rows = rows.filter((r) => r.type === type);
    if (side !== "all") rows = rows.filter((r) => r.side === side);

    if (q.trim()) {
      const needle = q.trim().toLowerCase();
      rows = rows.filter((r) =>
        [
          r.id,
          r.symbol,
          r.venue,
          r.account,
          r.orderId,
          r.note,
          r.type,
          r.side,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase()
          .includes(needle)
      );
    }

    // sort
    rows.sort((a, b) => {
      const va = (a[sortKey] ?? "") as any;
      const vb = (b[sortKey] ?? "") as any;
      if (va < vb) return sortDir === "asc" ? -1 : 1;
      if (va > vb) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return rows;
  }, [raw, q, type, side, from, to, sortKey, sortDir]);

  // running balance & equity curve
  const withBalance = useMemo(() => {
    let bal = 0;
    return filtered.map((r) => {
      bal += r.cashDelta;
      return { ...r, balance: bal };
    });
  }, [filtered]);

  // summaries
  const stats = useMemo(() => {
    const totalFees = sum(filtered.map((r) => r.fee ?? 0));
    const realized = sum(filtered.map((r) => r.pnl ?? 0));
    const deposits = sum(filtered.filter((r) => r.type === "deposit").map((r) => r.cashDelta));
    const withdrawals = sum(filtered.filter((r) => r.type === "withdrawal").map((r) => r.cashDelta));
    const trades = filtered.filter((r) => r.type === "trade").length;
    const endBal = withBalance.length ? withBalance[withBalance.length - 1].balance : 0;
    return { trades, totalFees, realized, deposits, withdrawals, endBal };
  }, [filtered, withBalance]);

  // chart data
  const equityCurve = withBalance.map((r) => ({ t: r.ts, balance: r.balance }));
  const pnlBars = filtered
    .filter((r) => (r.pnl ?? 0) !== 0)
    .map((r) => ({ t: r.ts, pnl: r.pnl ?? 0 }));

  // pagination slices
  const pages = Math.max(1, Math.ceil(withBalance.length / pageSize));
  const startIdx = (page - 1) * pageSize;
  const pageRows = withBalance.slice(startIdx, startIdx + pageSize);

  function toggleSort(key: keyof LedgerEntry) {
    if (key === sortKey) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
  }

  function exportCSV() {
    const headers = [
      "id","ts","date","type","symbol","side","qty","price","fee","cashDelta","pnl","venue","account","orderId","note","balance",
    ];
    const rows = withBalance.map((r) => [
      r.id,
      r.ts,
      fmtDate(r.ts),
      r.type,
      r.symbol ?? "",
      r.side ?? "",
      safeNum(r.qty),
      safeNum(r.price),
      safeNum(r.fee),
      safeNum(r.cashDelta),
      safeNum(r.pnl),
      r.venue ?? "",
      r.account ?? "",
      r.orderId ?? "",
      (r.note ?? "").replace(/\n/g, " "),
      safeNum((r as any).balance),
    ]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ledger_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (err) {
    return <div className="rounded-xl border p-3 text-sm text-red-600">Error: {err}</div>;
  }

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      {/* Header + KPIs */}
      <header className="flex items-center justify-between gap-4 mb-4">
        <div>
          <h2 className="text-lg font-semibold">Ledger</h2>
          <p className="text-xs opacity-70">
            Entries: {filtered.length} • Trades: {stats.trades} • Realized PnL:{" "}
            <b className={stats.realized >= 0 ? "text-green-600" : "text-red-600"}>
              {money(stats.realized, baseCurrency)}
            </b>{" "}
            • Fees: {money(stats.totalFees, baseCurrency)} • End Balance:{" "}
            <b>{money(stats.endBal, baseCurrency)}</b>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800" onClick={exportCSV}>
            Export CSV
          </button>
        </div>
      </header>

      {/* Filters */}
      <div className="grid gap-2 md:grid-cols-6 mb-4">
        <input
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          placeholder="Search (symbol, venue, account, note…) "
          value={q}
          onChange={(e) => { setQ(e.target.value); setPage(1); }}
        />
        <select
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          value={type}
          onChange={(e) => { setType(e.target.value as any); setPage(1); }}
        >
          <option value="all">All types</option>
          {["trade","deposit","withdrawal","fee","interest","dividend","transfer"].map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          onChange={(e) => { setSide(e.target.value as any); setPage(1); }}
        >
          <option value="all">Any side</option>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
        <input
          type="date"
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          value={from}
          onChange={(e) => { setFrom(e.target.value); setPage(1); }}
        />
        <input
          type="date"
          className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
          value={to}
          onChange={(e) => { setTo(e.target.value); setPage(1); }}
        />
        <button
          className="px-3 py-1.5 text-sm rounded-md border dark:border-gray-800"
          onClick={() => { setQ(""); setType("all"); setSide("all"); setFrom(""); setTo(""); setPage(1); }}
        >
          Reset
        </button>
      </div>

      {/* Charts */}
      <div className="grid gap-4 md:grid-cols-3 mb-4">
        <div className="rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">Equity Curve</div>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityCurve}>
                <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} hide />
                <YAxis hide />
                <Tooltip labelFormatter={(t) => fmtDate(t as number)} formatter={(v) => money(v as number, baseCurrency)} />
                <Area type="monotone" dataKey="balance" stroke="#3b82f6" fill="#93c5fd" fillOpacity={0.35} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="md:col-span-2 rounded-xl border dark:border-gray-800 p-3">
          <div className="text-sm font-medium mb-2">Realized PnL (by event)</div>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pnlBars}>
                <XAxis dataKey="t" tickFormatter={(t) => miniDate(t as number)} hide />
                <YAxis hide />
                <Tooltip labelFormatter={(t) => fmtDate(t as number)} formatter={(v) => money(v as number, baseCurrency)} />
                <Bar dataKey="pnl" fill="#16a34a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto">
        <table className="w-full text-sm border-separate border-spacing-0">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-800 sticky top-0">
              {th("Date", "ts")}
              {th("Type", "type")}
              <th className="p-2 text-left">Symbol</th>
              <th className="p-2 text-left">Side</th>
              {th("Qty", "qty")}
              {th("Price", "price")}
              {th(`Cash (${baseCurrency})`, "cashDelta")}
              {th(`Fee (${baseCurrency})`, "fee")}
              {th(`PnL (${baseCurrency})`, "pnl")}
              <th className="p-2 text-left">Venue</th>
              <th className="p-2 text-left">Account</th>
              <th className="p-2 text-left">Order</th>
              {th(`Balance (${baseCurrency})`, "balance" as any)}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((r) => (
              <tr key={r.id} className="border-t dark:border-gray-800">
                <td className="p-2 whitespace-nowrap">{fmtDate(r.ts)}</td>
                <td className="p-2">{r.type}</td>
                <td className="p-2">{r.symbol ?? "—"}</td>
                <td className="p-2">{r.side ?? "—"}</td>
                <td className="p-2 text-right">{num(r.qty)}</td>
                <td className="p-2 text-right">{num(r.price)}</td>
                <td className={`p-2 text-right ${r.cashDelta >= 0 ? "text-green-600" : "text-red-600"}`}>
                  {money(r.cashDelta, baseCurrency)}
                </td>
                <td className="p-2 text-right">{money(r.fee ?? 0, baseCurrency)}</td>
                <td className={`p-2 text-right ${safeNum(r.pnl) >= 0 ? "text-green-600" : "text-red-600"}`}>
                  {money(r.pnl ?? 0, baseCurrency)}
                </td>
                <td className="p-2">{r.venue ?? "—"}</td>
                <td className="p-2">{r.account ?? "—"}</td>
                <td className="p-2">{r.orderId ?? "—"}</td>
                <td className="p-2 text-right font-medium">{money((r as any).balance ?? 0, baseCurrency)}</td>
              </tr>
            ))}
            {pageRows.length === 0 && (
              <tr><td className="p-3 text-center opacity-60" colSpan={13}>No results</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="mt-3 flex items-center justify-between text-sm">
        <div>Page {page} / {pages}</div>
        <div className="flex gap-2">
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(1)} disabled={page === 1}>⟪</button>
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>‹</button>
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages}>›</button>
          <button className="px-2 py-1 rounded-md border dark:border-gray-800" onClick={() => setPage(pages)} disabled={page === pages}>⟫</button>
        </div>
      </div>
    </div>
  );

  /* table header cell with sort control */
  function th(label: string, key: keyof LedgerEntry) {
    const active = sortKey === key;
    return (
      <th
        className="p-2 text-left cursor-pointer select-none"
        onClick={() => toggleSort(key)}
        title="Sort"
      >
        <span className="inline-flex items-center gap-1">
          {label}
          <span className={`text-[10px] opacity-70 ${active ? "" : "invisible"}`}>
            {sortDir === "asc" ? "▲" : "▼"}
          </span>
        </span>
      </th>
    );
  }
}

/* --------------------------------- Utils -------------------------------- */

function sum(arr: number[]) { return arr.reduce((a, b) => a + b, 0); }
function num(x?: number | null) {
  if (x === null || x === undefined) return "—";
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 6 }); } catch { return String(x); }
}
function money(x: number, ccy: string) {
  try { return x.toLocaleString(undefined, { style: "currency", currency: ccy }); }
  catch { return `${ccy} ${x.toFixed(2)}`; }
}
function fmtDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleString(undefined, { year: "2-digit", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}
function miniDate(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}
function safeNum(x?: number | null) { return x == null ? 0 : +x; }