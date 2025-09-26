"use client";

/**
 * TradeFromChart.tsx
 * - Controlled, stateless "trade from chart" ticket
 * - No imports, no hooks (parent owns all state)
 * - Works for MARKET / LIMIT / STOP / STOP_LIMIT
 * - Quick price nudges (% / ticks), set-from-last, and chart attach hooks
 */

type Side = "BUY" | "SELL";
type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
type TIF = "DAY" | "GTC";

export type TradeFromChartState = {
  symbol: string;
  last?: number;          // last traded / mid
  side: Side;
  type: OrderType;
  qty: number;
  limitPrice?: number;
  stopPrice?: number;
  tif: TIF;
  route?: string;
  takeProfit?: number;    // optional TP price (absolute)
  stopLoss?: number;      // optional SL price (absolute)
};

export type TradeFromChartProps = {
  value: TradeFromChartState;
  onChange: (patch: Partial<TradeFromChartState>) => void;
  onSubmit: () => void;

  /** called when user asks to "use chart price" (e.g., crosshair price) */
  onUseChartPrice?: (field: "limitPrice" | "stopPrice" | "takeProfit" | "stopLoss") => void;
  /** called when user toggles/requests a draggable order line on the chart */
  onToggleOrderLine?: (opts: { field: "limitPrice" | "stopPrice" | "takeProfit" | "stopLoss"; enable: boolean }) => void;
  /** optional cancel handler */
  onCancel?: () => void;

  /** display tweaks */
  title?: string;
  minQty?: number;          // default 1
  priceTick?: number;       // default 0.01
  pctNudges?: number[];     // default [0.1, 0.5, 1]
  disabled?: boolean;
};

export default function TradeFromChart({
  value,
  onChange,
  onSubmit,
  onUseChartPrice,
  onToggleOrderLine,
  onCancel,
  title = "Trade From Chart",
  minQty = 1,
  priceTick = 0.01,
  pctNudges = [0.1, 0.5, 1],
  disabled,
}: TradeFromChartProps) {
  const v = value;
  const needsLimit = v.type === "LIMIT" || v.type === "STOP_LIMIT";
  const needsStop  = v.type === "STOP"  || v.type === "STOP_LIMIT";

  const validQty   = Number.isFinite(v.qty) && v.qty >= minQty;
  const validLimit = !needsLimit || isFiniteNum(v.limitPrice);
  const validStop  = !needsStop  || isFiniteNum(v.stopPrice);

  const canSubmit = !disabled && v.symbol && validQty && validLimit && validStop;

  const tone = v.side === "BUY" ? "text-emerald-300" : "text-red-300";

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-3 py-2 border-b border-[#222] flex items-center justify-between gap-2">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">
          {v.symbol ? v.symbol : "—"}{v.last != null ? <> · Last <span className="text-gray-300">{fmtPx(v.last)}</span></> : null}
        </div>
      </div>

      {/* core fields */}
      <div className="p-3 grid grid-cols-2 gap-3">
        <Field label="Symbol">
          <input
            value={v.symbol}
            onChange={(e) => onChange({ symbol: e.target.value.toUpperCase() })}
            placeholder="AAPL"
            className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            disabled={disabled}
          />
        </Field>

        <Field label="Qty">
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={minQty}
              step={1}
              value={safeNum(v.qty)}
              onChange={(e) => onChange({ qty: toPosInt(e.target.value, minQty) })}
              className={`w-full bg-[#0f0f0f] border rounded px-2 py-1 text-[12px] text-gray-200 outline-none ${
                validQty ? "border-[#1f1f1f]" : "border-red-600/60"
              }`}
              disabled={disabled}
            />
            <NudgeBtn label="+100" onClick={() => onChange({ qty: clampInt((v.qty || 0) + 100, minQty) })} disabled={disabled} />
            <NudgeBtn label="-100" onClick={() => onChange({ qty: clampInt((v.qty || 0) - 100, minQty) })} disabled={disabled} />
          </div>
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

        {/* Limit */}
        {needsLimit ? (
          <Field label="Limit">
            <PriceEditor
              value={v.limitPrice}
              last={v.last}
              tick={priceTick}
              pctNudges={pctNudges}
              bad={!validLimit}
              onChange={(px) => onChange({ limitPrice: px })}
              onUseChart={() => onUseChartPrice?.("limitPrice")}
              onToggleLine={(en) => onToggleOrderLine?.({ field: "limitPrice", enable: en })}
              disabled={disabled}
              tone={v.side === "BUY" ? "pos" : "neg"}
            />
          </Field>
        ) : null}

        {/* Stop */}
        {needsStop ? (
          <Field label="Stop">
            <PriceEditor
              value={v.stopPrice}
              last={v.last}
              tick={priceTick}
              pctNudges={pctNudges}
              bad={!validStop}
              onChange={(px) => onChange({ stopPrice: px })}
              onUseChart={() => onUseChartPrice?.("stopPrice")}
              onToggleLine={(en) => onToggleOrderLine?.({ field: "stopPrice", enable: en })}
              disabled={disabled}
              tone="warn"
            />
          </Field>
        ) : null}

        {/* Brackets (optional) */}
        <Field label="Take Profit (optional)">
          <PriceEditor
            value={v.takeProfit}
            last={v.last}
            tick={priceTick}
            pctNudges={pctNudges}
            onChange={(px) => onChange({ takeProfit: px })}
            onUseChart={() => onUseChartPrice?.("takeProfit")}
            onToggleLine={(en) => onToggleOrderLine?.({ field: "takeProfit", enable: en })}
            disabled={disabled}
            tone="neutral"
          />
        </Field>

        <Field label="Stop Loss (optional)">
          <PriceEditor
            value={v.stopLoss}
            last={v.last}
            tick={priceTick}
            pctNudges={pctNudges}
            onChange={(px) => onChange({ stopLoss: px })}
            onUseChart={() => onUseChartPrice?.("stopLoss")}
            onToggleLine={(en) => onToggleOrderLine?.({ field: "stopLoss", enable: en })}
            disabled={disabled}
            tone="neg"
          />
        </Field>

        <Field label="TIF">
          <Select
            value={v.tif}
            onChange={(t) => onChange({ tif: t as TIF })}
            options={["DAY","GTC"]}
            disabled={disabled}
          />
        </Field>

        <Field label="Route (optional)">
          <input
            value={v.route ?? ""}
            onChange={(e) => onChange({ route: e.target.value })}
            placeholder="SMART / NYSE / DARK"
            className="w-full bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
            disabled={disabled}
          />
        </Field>
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
          {v.side === "BUY" ? "Buy" : "Sell"} {fmtQty(v.qty)} {v.symbol}
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

        <div className="ml-auto text-[11px] text-gray-500">
          <span className={tone}>{v.side}</span> · {v.type} · {v.tif}
        </div>
      </div>
    </div>
  );
}

/* ----------------- tiny UI widgets ----------------- */

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

function NudgeBtn({ label, onClick, disabled }: { label: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="text-[11px] px-2 py-1 rounded border border-[#1f1f1f] bg-[#101010] text-gray-300 hover:bg-[#141414] disabled:opacity-50"
    >
      {label}
    </button>
  );
}

function PriceEditor({
  value,
  last,
  tick,
  pctNudges,
  bad,
  onChange,
  onUseChart,
  onToggleLine,
  disabled,
  tone = "neutral",
}: {
  value?: number;
  last?: number;
  tick: number;
  pctNudges: number[];
  bad?: boolean;
  onChange: (v: number | undefined) => void;
  onUseChart?: () => void;
  onToggleLine?: (enable: boolean) => void;
  disabled?: boolean;
  tone?: "pos" | "neg" | "warn" | "neutral";
}) {
  const col =
    tone === "pos" ? "text-emerald-300" :
    tone === "neg" ? "text-red-300" :
    tone === "warn" ? "text-amber-300" : "text-gray-200";

  return (
    <div className="flex items-center gap-2">
      <input
        type="number"
        inputMode="decimal"
        step={tick}
        value={value ?? ""}
        onChange={(e) => onChange(toOptNum(e.target.value))}
        placeholder="—"
        className={`w-28 bg-[#0f0f0f] border rounded px-2 py-1 text-[12px] outline-none ${col} ${
          bad ? "border-red-600/60" : "border-[#1f1f1f]"
        }`}
        disabled={disabled}
      />
      <NudgeBtn label={`+${fmtTick(tick)}`} onClick={() => onChange(roundTick((value ?? last ?? 0) + tick, tick))} disabled={disabled} />
      <NudgeBtn label={`-${fmtTick(tick)}`} onClick={() => onChange(roundTick((value ?? last ?? 0) - tick, tick))} disabled={disabled} />
      {pctNudges.map((p) => (
        <NudgeBtn
          key={p}
          label={`${p}%`}
          onClick={() => onChange(roundTick((value ?? last ?? 0) * (1 + p/100), tick))}
          disabled={disabled}
        />
      ))}
      {last != null ? (
        <NudgeBtn label="= Last" onClick={() => onChange(roundTick(last, tick))} disabled={disabled} />
      ) : null}
      {onUseChart ? (
        <NudgeBtn label="Use Chart" onClick={onUseChart} disabled={disabled} />
      ) : null}
      {onToggleLine ? (
        <>
          <button
            onClick={() => onToggleLine(true)}
            className="text-[11px] px-2 py-1 rounded border border-[#1f1f1f] bg-[#101010] text-gray-300 hover:bg-[#141414]"
            disabled={disabled}
          >
            Show Line
          </button>
          <button
            onClick={() => onToggleLine(false)}
            className="text-[11px] px-2 py-1 rounded border border-[#1f1f1f] bg-[#101010] text-gray-300 hover:bg-[#141414]"
            disabled={disabled}
          >
            Hide
          </button>
        </>
      ) : null}
    </div>
  );
}

/* ----------------- helpers ----------------- */

function isFiniteNum(v: any): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function toPosInt(v: string, min: number) {
  const n = Math.floor(Number(v));
  return Number.isFinite(n) && n >= min ? n : min;
}

function toOptNum(v: string): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function clampInt(n: number, min: number) {
  return n >= min ? n : min;
}

function roundTick(x: number, tick: number) {
  if (!Number.isFinite(x) || !Number.isFinite(tick) || tick <= 0) return x;
  return Math.round(x / tick) * tick;
}

function fmtPx(n?: number) {
  if (n == null || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  const dp =
    a >= 1000 ? 2 :
    a >= 100 ? 3 :
    a >= 10 ? 4 :
    a >= 1 ? 5 :
    6;
  return n.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function fmtQty(q: number) {
  return Number(q).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function fmtTick(t: number) {
  if (!Number.isFinite(t)) return String(t);
  const s = t.toString();
  if (s.includes("e") || s.includes("E")) return t.toFixed(6);
  const dp = (s.split(".")[1]?.length ?? 0) || 2;
  return t.toFixed(Math.min(6, Math.max(0, dp)));
}

function safeNum(n: any) {
  const x = Number(n);
  return Number.isFinite(x) ? x : 0;
}