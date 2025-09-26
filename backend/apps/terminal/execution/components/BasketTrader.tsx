"use client";

import React, { useMemo, useState } from "react";

/* ---------- shared types (aligned with submitBasket.actions.ts) ---------- */

export type OrderSide = "BUY" | "SELL";
export type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
export type TIF = "DAY" | "GTC";

export type BasketLeg = {
  symbol: string;
  qty: number;
  side: OrderSide;
  type: OrderType;
  limitPrice?: number;
  stopPrice?: number;
  tif?: TIF;
  tag?: string;
};

export type SubmitBasketInput = {
  accountId: string;
  legs: BasketLeg[];
  basketTag?: string;
  dryRun?: boolean;
};

export type BasketLegResult = {
  index: number;
  symbol: string;
  side: OrderSide;
  qty: number;
  type: OrderType;
  status: "accepted" | "rejected" | "placed";
  message?: string;
  orderId?: string;
  error?: string;
};

export type SubmitBasketResult = {
  success: boolean;
  basketId: string;
  accountId: string;
  dryRun: boolean;
  legsAccepted: number;
  legsRejected: number;
  legsPlaced?: number;
  grossNotional: number;
  netNotional: number;
  buyNotional: number;
  sellNotional: number;
  results: BasketLegResult[];
  warnings?: string[];
  error?: string;
};

/* ---------- component props ---------- */

type Props = {
  /** default account for orders */
  defaultAccountId?: string;
  /** optional default universe (for placeholder / UX only) */
  placeholderUniverse?: string;
  /** if provided, use this handler instead of POSTing to endpoint */
  onSubmit?: (input: SubmitBasketInput) => Promise<SubmitBasketResult>;
  /** API endpoint to call when onSubmit not supplied */
  endpoint?: string; // default "/api/trade/submit-basket"
  /** initial legs (optional) */
  initialLegs?: BasketLeg[];
  /** called on successful placement/preview */
  onResult?: (res: SubmitBasketResult) => void;
  /** panel title */
  title?: string;
};

export default function BasketTrader({
  defaultAccountId = "",
  placeholderUniverse = "US Equities",
  onSubmit,
  endpoint = "/api/trade/submit-basket",
  initialLegs,
  onResult,
  title = "Basket Trader",
}: Props) {
  /* -------------------- state -------------------- */
  const [accountId, setAccountId] = useState(defaultAccountId);
  const [basketTag, setBasketTag] = useState<string>("");
  const [dryRun, setDryRun] = useState<boolean>(true);

  const [rows, setRows] = useState<BasketLeg[]>(
    initialLegs && initialLegs.length
      ? initialLegs
      : [
          { symbol: "", qty: 0, side: "BUY", type: "MARKET", tif: "DAY" },
        ]
  );

  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [last, setLast] = useState<SubmitBasketResult | null>(null);

  /* -------------------- derived -------------------- */

  const issues = useMemo(() => rows.map(validateRow), [rows]);

  const canSubmit = useMemo(() => {
    if (!accountId) return false;
    if (!rows.length) return false;
    // must have at least one valid row
    return rows.some((_, i) => issues[i].length === 0);
  }, [accountId, rows, issues]);

  const totals = useMemo(() => {
    // naive preview notionals using limit/stop if provided, else 1
    let buy = 0, sell = 0;
    for (const r of rows) {
      const px = firstNum(r.limitPrice, r.stopPrice, 1);
      const notional = Math.abs((r.qty || 0) * px);
      if (r.side === "BUY") buy += notional; else sell += notional;
    }
    return {
      gross: buy + sell,
      net: buy - sell,
      buy,
      sell,
    };
  }, [rows]);

  /* -------------------- handlers -------------------- */

  function updateRow(i: number, patch: Partial<BasketLeg>) {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }

  function addRow(afterIdx?: number) {
    setRows((prev) => {
      const base: BasketLeg = { symbol: "", qty: 0, side: "BUY", type: "MARKET", tif: "DAY" };
      if (afterIdx == null || afterIdx < 0 || afterIdx >= prev.length) return [...prev, base];
      const next = [...prev];
      next.splice(afterIdx + 1, 0, base);
      return next;
    });
  }

  function removeRow(i: number) {
    setRows((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function runSubmit() {
    setErr(null);
    setSubmitting(true);
    setLast(null);
    try {
      const payload: SubmitBasketInput = {
        accountId,
        legs: rows,
        basketTag: basketTag || undefined,
        dryRun,
      };

      const res = onSubmit
        ? await onSubmit(payload)
        : await postJSON<SubmitBasketResult>(endpoint, payload);

      setLast(res);
      onResult?.(res);
      if (!res.success && res.error) setErr(res.error);
    } catch (e: any) {
      setErr(e?.message || "Submit failed");
    } finally {
      setSubmitting(false);
    }
  }

  /* -------------------- render -------------------- */

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-4 py-2 border-b border-[#222] flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">{placeholderUniverse}</div>
      </div>

      {/* controls */}
      <div className="p-3 flex flex-wrap items-center gap-3">
        <Field label="Account">
          <input
            value={accountId}
            onChange={(e) => setAccountId(e.target.value)}
            placeholder="ACC-123"
            className="w-40 bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
          />
        </Field>
        <Field label="Basket Tag">
          <input
            value={basketTag}
            onChange={(e) => setBasketTag(e.target.value)}
            placeholder="Rebalance_2025-09-24"
            className="w-56 bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
          />
        </Field>
        <label className="flex items-center gap-2 text-[12px] text-gray-300 ml-auto">
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
          />
          Dry run (preview)
        </label>
      </div>

      {/* legs table */}
      <div className="px-3 pb-3 overflow-x-auto">
        <table className="min-w-full text-[12px]">
          <thead className="bg-[#0f0f0f] border border-[#1f1f1f] text-gray-400">
            <tr>
              <Th>#</Th>
              <Th>Symbol</Th>
              <Th>Qty</Th>
              <Th>Side</Th>
              <Th>Type</Th>
              <Th>Limit</Th>
              <Th>Stop</Th>
              <Th>TIF</Th>
              <Th>Tag</Th>
              <Th children={undefined}></Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#101010]">
            {rows.map((r, i) => {
              const errs = issues[i];
              const bad = (key: keyof BasketLeg) => errs.some((e) => e.field === key);

              return (
                <tr key={i} className="hover:bg-[#101010]">
                  <Td className="text-gray-500">{i + 1}</Td>
                  <Td>
                    <input
                      value={r.symbol}
                      onChange={(e) => updateRow(i, { symbol: e.target.value.toUpperCase() })}
                      placeholder="AAPL"
                      className={`w-28 bg-transparent border rounded px-2 py-1 outline-none ${
                        bad("symbol") ? "border-red-600/60" : "border-[#1f1f1f]"
                      }`}
                    />
                  </Td>
                  <Td>
                    <input
                      type="number"
                      value={r.qty || 0}
                      onChange={(e) => updateRow(i, { qty: toPosNum(e.target.value) })}
                      placeholder="100"
                      className={`w-24 bg-transparent border rounded px-2 py-1 outline-none ${
                        bad("qty") ? "border-red-600/60" : "border-[#1f1f1f]"
                      }`}
                    />
                  </Td>
                  <Td>
                    <Select
                      value={r.side}
                      onChange={(v) => updateRow(i, { side: v as OrderSide })}
                      options={["BUY", "SELL"]}
                    />
                  </Td>
                  <Td>
                    <Select
                      value={r.type}
                      onChange={(v) => updateRow(i, { type: v as OrderType })}
                      options={["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]}
                    />
                  </Td>
                  <Td>
                    <input
                      type="number"
                      value={r.limitPrice ?? ""}
                      onChange={(e) =>
                        updateRow(i, { limitPrice: toOptNum(e.target.value) })
                      }
                      placeholder="—"
                      className={`w-24 bg-transparent border rounded px-2 py-1 outline-none ${
                        bad("limitPrice") ? "border-red-600/60" : "border-[#1f1f1f]"
                      }`}
                    />
                  </Td>
                  <Td>
                    <input
                      type="number"
                      value={r.stopPrice ?? ""}
                      onChange={(e) =>
                        updateRow(i, { stopPrice: toOptNum(e.target.value) })
                      }
                      placeholder="—"
                      className={`w-24 bg-transparent border rounded px-2 py-1 outline-none ${
                        bad("stopPrice") ? "border-red-600/60" : "border-[#1f1f1f]"
                      }`}
                    />
                  </Td>
                  <Td>
                    <Select
                      value={r.tif ?? "DAY"}
                      onChange={(v) => updateRow(i, { tif: v as TIF })}
                      options={["DAY", "GTC"]}
                    />
                  </Td>
                  <Td>
                    <input
                      value={r.tag ?? ""}
                      onChange={(e) => updateRow(i, { tag: e.target.value })}
                      placeholder="leg note"
                      className="w-28 bg-transparent border border-[#1f1f1f] rounded px-2 py-1 outline-none"
                    />
                  </Td>
                  <Td>
                    <div className="flex gap-2">
                      <button
                        onClick={() => addRow(i)}
                        title="Add row below"
                        className="text-emerald-300 hover:text-emerald-200"
                      >
                        + Add
                      </button>
                      <button
                        onClick={() => removeRow(i)}
                        title="Remove row"
                        className="text-red-400 hover:text-red-300"
                        disabled={rows.length <= 1}
                      >
                        Remove
                      </button>
                    </div>
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {/* row-level errors */}
        <div className="mt-2 space-y-1">
          {issues.flatMap((errs, i) =>
            errs.map((e, k) => (
              <div key={`${i}-${k}`} className="text-[11px] text-red-400">
                Row {i + 1}: {e.msg}
              </div>
            ))
          )}
        </div>
      </div>

      {/* totals + actions */}
      <div className="px-3 py-3 border-t border-[#1f1f1f] flex flex-wrap items-center gap-3">
        <Chip label="Buy Notional" value={fmtMoney(totals.buy)} tone="pos" />
        <Chip label="Sell Notional" value={fmtMoney(totals.sell)} tone="neg" />
        <Chip label="Gross" value={fmtMoney(totals.gross)} />
        <Chip label="Net" value={fmtMoney(totals.net)} tone={totals.net >= 0 ? "pos" : "neg"} />

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={runSubmit}
            disabled={!canSubmit || submitting}
            className={`px-3 py-2 rounded text-sm ${
              canSubmit && !submitting
                ? "bg-emerald-600 text-white hover:bg-emerald-500"
                : "bg-[#141414] text-gray-500"
            }`}
          >
            {submitting ? "Submitting…" : dryRun ? "Preview Basket" : "Submit Basket"}
          </button>
          {!dryRun && (
            <button
              onClick={() => setDryRun(true)}
              className="px-3 py-2 rounded text-sm bg-[#141414] text-gray-300 hover:bg-[#181818]"
            >
              Switch to Preview
            </button>
          )}
        </div>
      </div>

      {/* result panel */}
      {err ? (
        <div className="px-3 pb-3 text-xs text-red-400">{err}</div>
      ) : null}

      {last && (
        <div className="px-3 pb-4">
          <div className="mt-2 bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg overflow-hidden">
            <div className="px-3 py-2 border-b border-[#1f1f1f] flex items-center justify-between">
              <div className="text-sm text-gray-200 font-semibold">
                {last.dryRun ? "Preview" : "Submission"} · Basket {last.basketId}
              </div>
              <div className="text-[11px] text-gray-500">
                {last.legsAccepted} accepted / {last.legsRejected} rejected
              </div>
            </div>
            <div className="p-3 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              <Chip label="Gross Notional" value={fmtMoney(last.grossNotional)} />
              <Chip label="Net Notional" value={fmtMoney(last.netNotional)} tone={last.netNotional >= 0 ? "pos" : "neg"} />
              <Chip label="Buy Notional" value={fmtMoney(last.buyNotional)} tone="pos" />
              <Chip label="Sell Notional" value={fmtMoney(last.sellNotional)} tone="neg" />
            </div>

            {last.warnings && last.warnings.length > 0 && (
              <div className="px-3 pb-3">
                {last.warnings.map((w, i) => (
                  <div key={i} className="text-[11px] text-amber-300">⚠ {w}</div>
                ))}
              </div>
            )}

            <div className="px-3 pb-3 overflow-x-auto">
              <table className="min-w-full text-[12px]">
                <thead className="bg-[#0f0f0f] border border-[#1f1f1f] text-gray-400">
                  <tr>
                    <Th>#</Th>
                    <Th>Symbol</Th>
                    <Th>Side</Th>
                    <Th>Qty</Th>
                    <Th>Type</Th>
                    <Th>Status</Th>
                    <Th>Message</Th>
                    <Th>Order ID</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#101010]">
                  {last.results.map((r, i) => {
                    const col =
                      r.status === "placed"
                        ? "text-emerald-300"
                        : r.status === "accepted"
                        ? "text-amber-300"
                        : "text-red-300";
                    return (
                      <tr key={i}>
                        <Td className="text-gray-500">{r.index + 1}</Td>
                        <Td className="text-gray-100">{r.symbol}</Td>
                        <Td className="text-gray-300">{r.side}</Td>
                        <Td className="text-gray-300">{fmtNum(r.qty)}</Td>
                        <Td className="text-gray-300">{r.type}</Td>
                        <Td className={`font-semibold ${col}`}>{r.status.toUpperCase()}</Td>
                        <Td className="text-gray-400">{r.error ?? r.message ?? "—"}</Td>
                        <Td className="text-gray-400">{r.orderId ?? "—"}</Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {!last.dryRun && (
              <div className="px-3 pb-3 text-[11px] text-gray-500">
                Success: {String(last.success)}
                {last.error ? ` • Error: ${last.error}` : ""}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------- tiny UI helpers -------------------- */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-2 text-[12px] text-gray-300">
      <span className="text-gray-400">{label}</span>
      {children}
    </label>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-3 py-2 text-left">{children}</th>;
}

function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-3 py-2 ${className}`}>{children}</td>;
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-transparent border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function Chip({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "pos" | "neg";
}) {
  const color =
    tone === "pos" ? "text-emerald-300" : tone === "neg" ? "text-red-300" : "text-gray-100";
  const chip =
    tone === "pos"
      ? "bg-emerald-700/30 text-emerald-200 border-emerald-700/60"
      : tone === "neg"
      ? "bg-red-700/30 text-red-200 border-red-700/60"
      : "bg-gray-700/30 text-gray-200 border-gray-700/60";
  return (
    <div className="bg-[#0e0e0e] border border-[#1f1f1f] rounded px-2 py-1">
      <div className="text-[10px] text-gray-400">{label}</div>
      <div className={`text-[12px] font-semibold ${color}`}>
        <span className={`px-1.5 py-[1px] rounded border ${chip}`}>{value}</span>
      </div>
    </div>
  );
}

/* -------------------- validation & utils -------------------- */

function validateRow(r: BasketLeg): Array<{ field: keyof BasketLeg; msg: string }> {
  const out: Array<{ field: keyof BasketLeg; msg: string }> = [];
  if (!r.symbol) out.push({ field: "symbol", msg: "symbol is required" });
  if (!Number.isFinite(r.qty) || r.qty <= 0) out.push({ field: "qty", msg: "qty must be > 0" });
  if ((r.type === "LIMIT" || r.type === "STOP_LIMIT") && !isFiniteNum(r.limitPrice))
    out.push({ field: "limitPrice", msg: "limitPrice required for LIMIT/STOP_LIMIT" });
  if ((r.type === "STOP" || r.type === "STOP_LIMIT") && !isFiniteNum(r.stopPrice))
    out.push({ field: "stopPrice", msg: "stopPrice required for STOP/STOP_LIMIT" });
  return out;
}

function isFiniteNum(v: any): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function toPosNum(v: string): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

function toOptNum(v: string): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function firstNum(...vals: Array<number | undefined>): number {
  for (const v of vals) if (typeof v === "number" && Number.isFinite(v)) return v as number;
  return 1;
}

function fmtMoney(n: number) {
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (a >= 1_000_000_000) return `${sign}$${(a / 1_000_000_000).toFixed(2)}B`;
  if (a >= 1_000_000) return `${sign}$${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 1_000) return `${sign}$${(a / 1_000).toFixed(2)}K`;
  return `${sign}$${a.toFixed(2)}`;
}

function fmtNum(n?: number) {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/* -------------------- fallback client submit -------------------- */

async function postJSON<T>(endpoint: string, body: any): Promise<T> {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} ${txt}`.trim());
  }
  return (await res.json()) as T;
}