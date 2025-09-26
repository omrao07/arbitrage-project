"use client";

/**
 * TradeTicket.tsx
 * - Controlled, stateless order ticket (no imports)
 * - MARKET / LIMIT / STOP / STOP_LIMIT (+ TP/SL brackets)
 * - Account, route, TIF, advanced flags (post-only, reduce-only, iceberg)
 * - Live quote panel (bid/ask/last), est. notional/fees preview
 */

type Side = "BUY" | "SELL";
type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
type TIF = "DAY" | "GTC";

export type TicketValue = {
  accountId?: string;
  symbol: string;
  side: Side;
  type: OrderType;
  qty: number;
  limitPrice?: number;
  stopPrice?: number;
  tif: TIF;
  route?: string;

  // optional brackets
  takeProfit?: number;
  stopLoss?: number;

  // advanced
  postOnly?: boolean;
  reduceOnly?: boolean;
  icebergQty?: number; // if set, will slice qty in lots of icebergQty
  leverage?: number;   // for margin/crypto (e.g., 1..125)

  // quotes (display-only; parent feeds these)
  bestBid?: number;
  bestAsk?: number;
  last?: number;
  mark?: number;

  // fees/commission (bps or flat) for preview, optional
  feeBps?: number;     // e.g., 1.5 = 1.5 bps
  feeFlat?: number;    // currency units
};

export type TradeTicketProps = {
  value: TicketValue;
  onChange: (patch: Partial<TicketValue>) => void;
  onSubmit: () => void;
  onCancel?: () => void;

  // chart helpers (optional)
  onUsePriceFromChart?: (target: "limitPrice" | "stopPrice" | "takeProfit" | "stopLoss") => void;

  // UX options
  title?: string;
  minQty?: number;
  pxTick?: number;           // price tick
  qtyStep?: number;          // qty nudge
  currency?: string;         // for preview labels
  disabled?: boolean;
};

export default function TradeTicket({
  value: v,
  onChange,
  onSubmit,
  onCancel,
  onUsePriceFromChart,
  title = "Order Ticket",
  minQty = 1,
  pxTick = 0.01,
  qtyStep = 100,
  currency = "USD",
  disabled,
}: TradeTicketProps) {
  const needsLimit = v.type === "LIMIT" || v.type === "STOP_LIMIT";
  const needsStop  = v.type === "STOP"  || v.type === "STOP_LIMIT";

  const validQty   = Number.isFinite(v.qty) && v.qty >= minQty;
  const validLimit = !needsLimit || isNum(v.limitPrice);
  const validStop  = !needsStop  || isNum(v.stopPrice);

  const refPx = refPrice(v); // preview ref price based on type/quotes
  const notional = (v.qty || 0) * (refPx || 0);
  const fees = estFees(notional, v.feeBps, v.feeFlat);
  const canSubmit = !disabled && v.symbol && validQty && validLimit && validStop;

  const tone = v.side === "BUY" ? "text-emerald-300" : "text-red-300";
  const pxTone =
    v.type === "MARKET" ? "text-gray-300"
      : v.side === "BUY" ? "text-emerald-300"
      : "text-red-300";

    function clampInt(arg0: number, minQty: number): number | undefined {
        throw new Error("Function not implemented.");
    }

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-3 py-2 border-b border-[#222] flex items-center justify-between gap-2">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <QuoteStrip bid={v.bestBid} ask={v.bestAsk} last={v.last} />
      </div>

      {/* basics */}
      <div className="p-3 grid grid-cols-2 gap-3">
        <Field label="Account">
          <input
            value={v.accountId ?? ""}
            onChange={(e) => onChange({ accountId: e.target.value })}
            placeholder="ACC-123"
            className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            disabled={disabled}
          />
        </Field>

        <Field label="Symbol">
          <input
            value={v.symbol}
            onChange={(e) => onChange({ symbol: e.target.value.toUpperCase() })}
            placeholder="AAPL / ESZ5 / BTC-USD"
            className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            disabled={disabled}
          />
        </Field>

        <Field label="Side">
          <Select
            value={v.side}
            onChange={(s) => onChange({ side: s as Side })}
            options={["BUY","SELL"]}
            disabled={disabled}
          />
        </Field>

        <Field label="Type">
          <Select
            value={v.type}
            onChange={(t) => onChange({ type: t as OrderType })}
            options={["MARKET","LIMIT","STOP","STOP_LIMIT"]}
            disabled={disabled}
          />
        </Field>

        <Field label="Qty">
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={minQty}
              step={1}
              value={safeInt(v.qty)}
              onChange={(e) => onChange({ qty: toPosInt(e.target.value, minQty) })}
              className={`w-full bg-[#0f0f0f] border rounded px-2 py-1 text-[12px] text-gray-200 outline-none ${
                validQty ? "border-[#1f1f1f]" : "border-red-600/60"
              }`}
              disabled={disabled}
            />
            <Btn small label={`+${qtyStep}`} onClick={() => onChange({ qty: clampInt((v.qty || 0) + qtyStep, minQty) })} />
            <Btn small label={`-${qtyStep}`} onClick={() => onChange({ qty: clampInt((v.qty || 0) - qtyStep, minQty) })} />
          </div>
        </Field>

        <Field label="TIF">
          <Select
            value={v.tif}
            onChange={(t) => onChange({ tif: t as TIF })}
            options={["DAY","GTC"]}
            disabled={disabled}
          />
        </Field>

        {/* price editors */}
        {needsLimit ? (
          <Field label="Limit Price">
            <PriceEditor
              value={v.limitPrice}
              last={v.last ?? v.mark}
              tick={pxTick}
              bad={!validLimit}
              tone={pxTone}
              onChange={(px) => onChange({ limitPrice: px })}
              onUseChart={() => onUsePriceFromChart?.("limitPrice")}
              disabled={disabled}
            />
          </Field>
        ) : null}

        {needsStop ? (
          <Field label="Stop Price">
            <PriceEditor
              value={v.stopPrice}
              last={v.last ?? v.mark}
              tick={pxTick}
              bad={!validStop}
              tone="text-amber-300"
              onChange={(px) => onChange({ stopPrice: px })}
              onUseChart={() => onUsePriceFromChart?.("stopPrice")}
              disabled={disabled}
            />
          </Field>
        ) : null}

        {/* route + leverage */}
        <Field label="Route (optional)">
          <input
            value={v.route ?? ""}
            onChange={(e) => onChange({ route: e.target.value })}
            placeholder="SMART / NYSE / DARK"
            className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            disabled={disabled}
          />
        </Field>

        <Field label="Leverage (opt)">
          <input
            type="number"
            step={0.1}
            min={1}
            value={v.leverage ?? ""}
            onChange={(e) => onChange({ leverage: toOptNum(e.target.value) })}
            placeholder="1"
            className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            disabled={disabled}
          />
        </Field>

        {/* brackets */}
        <Field label="Take Profit (opt)">
          <PriceEditor
            value={v.takeProfit}
            last={v.last ?? v.mark}
            tick={pxTick}
            onChange={(px) => onChange({ takeProfit: px })}
            onUseChart={() => onUsePriceFromChart?.("takeProfit")}
            disabled={disabled}
            tone="text-gray-300"
          />
        </Field>

        <Field label="Stop Loss (opt)">
          <PriceEditor
            value={v.stopLoss}
            last={v.last ?? v.mark}
            tick={pxTick}
            onChange={(px) => onChange({ stopLoss: px })}
            onUseChart={() => onUsePriceFromChart?.("stopLoss")}
            disabled={disabled}
            tone="text-red-300"
          />
        </Field>

        {/* advanced */}
        <Field label="Advanced">
          <div className="flex flex-wrap items-center gap-3">
            <Chk
              label="Post Only"
              checked={!!v.postOnly}
              onChange={(x) => onChange({ postOnly: x })}
              disabled={disabled}
            />
            <Chk
              label="Reduce Only"
              checked={!!v.reduceOnly}
              onChange={(x) => onChange({ reduceOnly: x })}
              disabled={disabled}
            />
            <span className="inline-flex items-center gap-2">
              <span className="text-[11px] text-gray-400">Iceberg</span>
              <input
                type="number"
                min={0}
                step={1}
                value={v.icebergQty ?? ""}
                onChange={(e) => onChange({ icebergQty: toOptInt(e.target.value) })}
                placeholder="0"
                className="w-24 bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
                disabled={disabled}
              />
            </span>
          </div>
        </Field>
      </div>

      {/* preview */}
      <div className="px-3 py-2 border-t border-[#1f1f1f] text-[12px] text-gray-300 flex flex-wrap items-center gap-3">
        <span className={tone}>{v.side}</span>
        <span>· {v.type}</span>
        <span>· Qty {fmtInt(v.qty)}</span>
        <span>· Ref {fmtPx(refPx)}</span>
        <span>· Notional {fmtMoney(notional, currency)}</span>
        {fees ? <span>· Est Fees {fmtMoney(fees, currency)}</span> : null}
        {v.takeProfit ? <span>· TP {fmtPx(v.takeProfit)}</span> : null}
        {v.stopLoss ? <span>· SL {fmtPx(v.stopLoss)}</span> : null}
      </div>

      {/* actions */}
      <div className="px-3 py-3 border-t border-[#1f1f1f] flex items-center gap-2">
        <button
          onClick={onSubmit}
          disabled={!canSubmit}
          className={`px-3 py-2 rounded text-sm ${
            canSubmit
              ? (v.side === "BUY" ? "bg-emerald-600 text-white hover:bg-emerald-500" : "bg-red-600 text-white hover:bg-red-500")
              : "bg-[#141414] text-gray-500"
          }`}
          title={canSubmit ? `${v.side} ${v.qty} ${v.symbol}` : "Fill required fields"}
        >
          {v.side === "BUY" ? "Buy" : "Sell"} {fmtInt(v.qty)} {v.symbol}
          {v.type !== "MARKET" && (needsLimit || needsStop) ? (
            <> @ {fmtPx(needsLimit ? v.limitPrice! : v.stopPrice!)}</>
          ) : null}
        </button>

        <button
          onClick={onCancel ?? (() => onChange({}))}
          className="px-3 py-2 rounded text-sm bg-[#141414] text-gray-300 hover:bg-[#181818]"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

/* ---------------- tiny UI bits ---------------- */

function Field({ label, children }: { label: string; children: any }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] text-gray-400">{label}</span>
      {children}
    </label>
  );
}

function Select({
  value, onChange, options, disabled,
}: { value: string; onChange: (v: string) => void; options: string[]; disabled?: boolean }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
    >
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function Btn({ label, onClick, small }: { label: string; onClick: () => void; small?: boolean }) {
  return (
    <button
      onClick={onClick}
      className={`rounded border border-[#1f1f1f] bg-[#101010] text-gray-300 hover:bg-[#141414] ${
        small ? "text-[11px] px-2 py-1" : "text-sm px-3 py-1.5"
      }`}
    >
      {label}
    </button>
  );
}

function Chk({ label, checked, onChange, disabled }: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="inline-flex items-center gap-2 text-[12px] text-gray-300">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} disabled={disabled} />
      {label}
    </label>
  );
}

function PriceEditor({
  value, last, tick, bad, tone, onChange, onUseChart, disabled,
}: {
  value?: number;
  last?: number;
  tick: number;
  bad?: boolean;
  tone?: string; // tailwind text class
  onChange: (v: number | undefined) => void;
  onUseChart?: () => void;
  disabled?: boolean;
}) {
  const cls = `w-32 bg-[#0f0f0f] border rounded px-2 py-1 text-[12px] outline-none ${
    bad ? "border-red-600/60" : "border-[#1f1f1f]"
  } ${tone ?? "text-gray-200"}`;

    function fmtTick(tick: number) {
        throw new Error("Function not implemented.");
    }

  return (
    <div className="flex items-center gap-2">
      <input
        type="number"
        inputMode="decimal"
        step={tick}
        value={value ?? ""}
        onChange={(e) => onChange(toOptNum(e.target.value))}
        placeholder="—"
        className={cls}
        disabled={disabled}
      />
      <Btn small label={`+${fmtTick(tick)}`} onClick={() => onChange(roundTick((value ?? last ?? 0) + tick, tick))} />
      <Btn small label={`-${fmtTick(tick)}`} onClick={() => onChange(roundTick((value ?? last ?? 0) - tick, tick))} />
      {isNum(last) ? <Btn small label="= Last" onClick={() => onChange(roundTick(last!, tick))} /> : null}
      {onUseChart ? <Btn small label="Use Chart" onClick={onUseChart} /> : null}
    </div>
  );
}

function QuoteStrip({ bid, ask, last }: { bid?: number; ask?: number; last?: number }) {
  const spread = isNum(bid) && isNum(ask) ? (ask! - bid!) : undefined;
  const mid = isNum(bid) && isNum(ask) ? (ask! + bid!) / 2 : undefined;
  return (
    <div className="text-[11px] text-gray-500">
      {isNum(bid) ? <>Bid <span className="text-emerald-300">{fmtPx(bid!)}</span> · </> : null}
      {isNum(ask) ? <>Ask <span className="text-red-300">{fmtPx(ask!)}</span> · </> : null}
      {isNum(spread) ? <>Spr <span className="text-gray-300">{fmtPx(spread!)}</span> · </> : null}
      {isNum(mid) ? <>Mid <span className="text-gray-300">{fmtPx(mid!)}</span> · </> : null}
      {isNum(last) ? <>Last <span className="text-gray-300">{fmtPx(last!)}</span></> : null}
    </div>
  );
}

/* ---------------- helpers ---------------- */

function isNum(v: any): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function safeInt(n: any) {
  const x = Math.floor(Number(n));
  return Number.isFinite(x) ? x : 0;
}

function toPosInt(v: string, min: number) {
  const n = Math.floor(Number(v));
  return Number.isFinite(n) && n >= min ? n : min;
}

function toOptInt(v: string): number | undefined {
  const n = Math.floor(Number(v));
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function toOptNum(v: string): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function roundTick(x: number, tick: number) {
  if (!Number.isFinite(x) || !Number.isFinite(tick) || tick <= 0) return x;
  return Math.round(x / tick) * tick;
}

function refPrice(v: TicketValue): number {
  // choose a sensible preview price
  if (v.type === "MARKET") {
    // prefer mid, else last, else mark, else best side
    const mid = isNum(v.bestBid) && isNum(v.bestAsk) ? (v.bestBid! + v.bestAsk!) / 2 : undefined;
    return pickNum(mid, v.last, v.mark, v.side === "BUY" ? v.bestAsk : v.bestBid, 0);
  }
  if (v.type === "LIMIT") return v.limitPrice ?? v.last ?? v.mark ?? 0;
  if (v.type === "STOP") return v.stopPrice ?? v.last ?? v.mark ?? 0;
  // STOP_LIMIT
  return pickNum(v.limitPrice, v.stopPrice, v.last, v.mark, 0);
}

function pickNum(...vals: Array<number | undefined>): number {
  for (const x of vals) if (isNum(x)) return x!;
  return 0;
}

function estFees(notional: number, bps?: number, flat?: number) {
  const a = Math.abs(notional);
  const fee = (isNum(bps) ? (a * (bps! / 10000)) : 0) + (isNum(flat) ? flat! : 0);
  return fee && fee > 0 ? fee : 0;
}

function fmtPx(n?: number) {
  if (!isNum(n)) return "—";
  const a = Math.abs(n);
  const dp =
    a >= 1000 ? 2 :
    a >= 100 ? 3 :
    a >= 10 ? 4 :
    a >= 1 ? 5 :
    6;
  return n.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function fmtInt(n: number) {
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function fmtMoney(n: number, ccy: string) {
  if (!Number.isFinite(n)) return `— ${ccy}`;
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  const txt =
    a >= 1_000_000_000 ? `${(a / 1_000_000_000).toFixed(2)}B` :
    a >= 1_000_000     ? `${(a / 1_000_000).toFixed(2)}M` :
    a >= 1_000         ? `${(a / 1_000).toFixed(2)}K` :
                         a.toFixed(2);
  return `${sign}${txt} ${ccy}`;
}