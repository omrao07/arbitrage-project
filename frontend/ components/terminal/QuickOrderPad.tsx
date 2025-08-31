// frontend/components/QuickOrderPad.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------- Types ------------------------------- */

type Side = "BUY" | "SELL";
type OrdType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
type TIF = "DAY" | "IOC" | "FOK" | "GTC";

export interface EstimateResponse {
  notional: number;       // qty * px (or last)
  fees: number;           // estimated fees/commissions
  slippage: number;       // absolute slippage cost
  total: number;          // notional + fees + slippage (signed)
}

export interface SubmitResponse {
  ok: boolean;
  id?: string;
  error?: string;
}

interface Props {
  title?: string;
  defaultSymbol?: string;
  defaultVenue?: string;
  venues?: string[];                // dropdown; if omitted, free text
  maxNotional?: number;             // risk rail (ccy)
  maxLeverage?: number;             // risk rail
  priceBandPct?: number;            // e.g., 10 => limit must be within ±10% of mark (if mark provided)
  markPrice?: number | null;        // optional live mark to anchor bands/estimates
  /**
   * If provided, submit will POST here with JSON:
   * { symbol, venue, side, qty, type, limitPrice?, stopPrice?, tif, leverage?, clientTag? }
   * Expected response shape SubmitResponse
   */
  submitEndpoint?: string;
  /** Optional estimate endpoint (GET): /api/estimate?symbol=...&side=...&qty=...&px=... */
  estimateEndpoint?: string;
  /** Called before submit; return string to block submit with an error. */
  onValidate?: (payload: any) => string | null | undefined;
  /** Called after submit success */
  onSubmitted?: (resp: SubmitResponse, payload: any) => void;
  /** If set, show a compact layout */
  dense?: boolean;
}

/* -------------------------------- Component ------------------------------- */

export default function QuickOrderPad({
  title = "Quick Order",
  defaultSymbol = "",
  defaultVenue,
  venues,
  maxNotional = 1_000_000,
  maxLeverage = 10,
  priceBandPct = 10,
  markPrice = null,
  submitEndpoint = "/api/orders",
  estimateEndpoint = "/api/estimate",
  onValidate,
  onSubmitted,
  dense = false,
}: Props) {
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [venue, setVenue] = useState(defaultVenue ?? (venues?.[0] ?? ""));
  const [side, setSide] = useState<Side>("BUY");
  const [qty, setQty] = useState<number>(0);
  const [ordType, setOrdType] = useState<OrdType>("MARKET");
  const [limitPx, setLimitPx] = useState<number | "">("");
  const [stopPx, setStopPx] = useState<number | "">("");
  const [tif, setTif] = useState<TIF>("DAY");
  const [lev, setLev] = useState<number | "">("");
  const [clientTag, setClientTag] = useState<string>("ui.quick");
  const [estim, setEstim] = useState<EstimateResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  /* ------------------------------- Derived ------------------------------- */

  const pxForCalc = useMemo(() => {
    if (ordType === "MARKET") return markPrice ?? toNum(limitPx) ?? 0;
    if (ordType === "LIMIT")  return toNum(limitPx) ?? 0;
    if (ordType === "STOP")   return markPrice ?? 0;
    // STOP_LIMIT
    return toNum(limitPx) ?? markPrice ?? 0;
  }, [ordType, limitPx, markPrice]);

  const notional = useMemo(() => (qty > 0 && pxForCalc > 0 ? qty * pxForCalc : 0), [qty, pxForCalc]);

  const bandError = useMemo(() => {
    if (!priceBandPct || !markPrice) return null;
    const px = ordType === "LIMIT" || ordType === "STOP_LIMIT" ? toNum(limitPx) : null;
    if (!px) return null;
    const low = markPrice * (1 - priceBandPct / 100);
    const high = markPrice * (1 + priceBandPct / 100);
    return px < low || px > high
      ? `Limit ${fmt(px)} outside ±${priceBandPct}% of mark (${fmt(low)}–${fmt(high)})`
      : null;
  }, [ordType, limitPx, markPrice, priceBandPct]);

  const leverageError = useMemo(() => {
    if (!lev || !maxLeverage) return null;
    return Number(lev) > maxLeverage ? `Leverage ${lev}× exceeds max ${maxLeverage}×` : null;
  }, [lev, maxLeverage]);

  const notionalError = useMemo(() => {
    if (!maxNotional) return null;
    return notional > maxNotional ? `Notional ${money(notional)} exceeds limit ${money(maxNotional)}` : null;
  }, [notional, maxNotional]);

  const needsLimit = ordType === "LIMIT" || ordType === "STOP_LIMIT";
  const needsStop = ordType === "STOP" || ordType === "STOP_LIMIT";

  /* -------------------------------- Effects -------------------------------- */

  // Hotkeys inside the pad
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent) => {
      if (!el.contains(document.activeElement)) return;
      if (e.key === "b" || e.key === "B") setSide("BUY");
      if (e.key === "s" || e.key === "S") setSide("SELL");
      if (e.key === "m" || e.key === "M") setOrdType("MARKET");
      if (e.key === "l" || e.key === "L") setOrdType("LIMIT");
      if (e.key === "ArrowUp" && (needsLimit || ordType === "MARKET") ) {
        e.preventDefault();
        bumpPx(+1);
      }
      if (e.key === "ArrowDown" && (needsLimit || ordType === "MARKET") ) {
        e.preventDefault();
        bumpPx(-1);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        submit();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [needsLimit, ordType, limitPx, markPrice, qty, side]);

  // Re-estimate on key inputs
  useEffect(() => {
    let ignore = false;
    (async () => {
      try {
        if (!estimateEndpoint || !symbol || qty <= 0 || pxForCalc <= 0) { setEstim(null); return; }
        const u = new URLSearchParams({
          symbol, side, qty: String(qty), px: String(pxForCalc),
          venue: venue || "", type: ordType
        }).toString();
        const res = await fetch(`${estimateEndpoint}?${u}`);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as EstimateResponse;
        if (!ignore) setEstim(json);
      } catch {
        if (!ignore) setEstim(null); // silent fail → still allow submit
      }
    })();
    return () => { ignore = true; };
  }, [estimateEndpoint, symbol, side, qty, pxForCalc, ordType, venue]);

  /* -------------------------------- Actions -------------------------------- */

  function bumpPx(dir: 1 | -1) {
    const step = tickGuess(pxForCalc);
    if (needsLimit) setLimitPx(round((toNum(limitPx) || pxForCalc) + dir * step));
    else if (ordType === "MARKET" && markPrice) setLimitPx(round(markPrice + dir * step)); // allows quick switch
  }

  function clear() {
    setQty(0);
    setLimitPx("");
    setStopPx("");
    setErr(null);
  }

  async function submit() {
    try {
      setBusy(true); setErr(null);
      // basic validations
      if (!symbol.trim()) throw new Error("Symbol is required");
      if (qty <= 0 || !Number.isFinite(qty)) throw new Error("Quantity must be > 0");
      if (needsLimit && !toNum(limitPx)) throw new Error("Limit price required");
      if (needsStop && !toNum(stopPx)) throw new Error("Stop price required");
      if (bandError) throw new Error(bandError);
      if (leverageError) throw new Error(leverageError);
      if (notionalError) throw new Error(notionalError);

      const payload = {
        symbol: symbol.trim().toUpperCase(),
        venue: venue?.trim() || undefined,
        side,
        qty: Number(qty),
        type: ordType,
        tif,
        limitPrice: needsLimit ? Number(limitPx) : undefined,
        stopPrice: needsStop ? Number(stopPx) : undefined,
        leverage: lev === "" ? undefined : Number(lev),
        clientTag,
        meta: { from: "quick_order_pad" }
      };

      // extra user validation hook
      const vmsg = onValidate?.(payload);
      if (vmsg) throw new Error(vmsg);

      let resp: SubmitResponse = { ok: true, id: `tmp_${Date.now()}` };
      if (submitEndpoint) {
        const res = await fetch(submitEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `${res.status} ${res.statusText}`);
        }
        resp = await res.json();
      }

      onSubmitted?.(resp, payload);
      if (!resp.ok) throw new Error(resp.error || "Order rejected");

      clear();
    } catch (e: any) {
      setErr(e?.message || "Failed to submit order");
    } finally {
      setBusy(false);
    }
  }

  /* ---------------------------------- UI ---------------------------------- */

  return (
    <div ref={rootRef} className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <div className="text-xs opacity-70">
            Hotkeys: <kbd>b</kbd>/<kbd>s</kbd> side • <kbd>m</kbd>/<kbd>l</kbd> type • <kbd>↑</kbd>/<kbd>↓</kbd> price • <kbd>Ctrl/⌘+↩</kbd> submit
          </div>
        </div>
        <div className="text-xs opacity-70">
          {markPrice ? <>Mark: <b>{fmt(markPrice)}</b></> : <span>—</span>}
        </div>
      </header>

      {/* Form */}
      <div className={`grid gap-3 ${dense ? "md:grid-cols-4" : "md:grid-cols-6"}`}>
        <div className="flex items-center gap-2">
          <label className="text-sm w-16 opacity-70">Symbol</label>
          <input className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm flex-1"
                 placeholder="AAPL / NIFTY / BTC-USD"
                 value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())}/>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm w-12 opacity-70">Side</label>
          <div className="flex gap-1">
            <button
              className={`px-3 py-1.5 rounded-md border text-sm ${side === "BUY" ? "bg-green-100 dark:bg-green-900/20 border-green-400 dark:border-green-700" : "dark:border-gray-800"}`}
              onClick={() => setSide("BUY")}
            >Buy</button>
            <button
              className={`px-3 py-1.5 rounded-md border text-sm ${side === "SELL" ? "bg-red-100 dark:bg-red-900/20 border-red-400 dark:border-red-700" : "dark:border-gray-800"}`}
              onClick={() => setSide("SELL")}
            >Sell</button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm w-12 opacity-70">Qty</label>
          <input type="number" step="any" className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-28 text-right"
                 value={qty || ""} onChange={(e) => setQty(numOrZero(e.target.value))}/>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm w-14 opacity-70">Type</label>
          <select className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
                  value={ordType} onChange={(e) => setOrdType(e.target.value as OrdType)}>
            <option value="MARKET">Market</option>
            <option value="LIMIT">Limit</option>
            <option value="STOP">Stop</option>
            <option value="STOP_LIMIT">Stop-Limit</option>
          </select>
        </div>

        {(needsLimit || needsStop) && (
          <>
            {needsLimit && (
              <div className="flex items-center gap-2">
                <label className="text-sm w-14 opacity-70">Limit</label>
                <input type="number" step="any" className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-28 text-right"
                       value={limitPx === "" ? "" : limitPx}
                       onChange={(e) => setLimitPx(numOrBlank(e.target.value))}/>
              </div>
            )}
            {needsStop && (
              <div className="flex items-center gap-2">
                <label className="text-sm w-12 opacity-70">Stop</label>
                <input type="number" step="any" className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-28 text-right"
                       value={stopPx === "" ? "" : stopPx}
                       onChange={(e) => setStopPx(numOrBlank(e.target.value))}/>
              </div>
            )}
          </>
        )}

        <div className="flex items-center gap-2">
          <label className="text-sm w-10 opacity-70">TIF</label>
          <select className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
                  value={tif} onChange={(e) => setTif(e.target.value as TIF)}>
            <option value="DAY">DAY</option>
            <option value="IOC">IOC</option>
            <option value="FOK">FOK</option>
            <option value="GTC">GTC</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm w-14 opacity-70">Venue</label>
          {venues?.length ? (
            <select className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm"
                    value={venue} onChange={(e) => setVenue(e.target.value)}>
              {venues.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          ) : (
            <input className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-28"
                   placeholder="(opt.)" value={venue} onChange={(e) => setVenue(e.target.value)} />
          )}
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm w-16 opacity-70">Leverage</label>
          <input type="number" step="0.1" className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm w-24 text-right"
                 value={lev === "" ? "" : lev} onChange={(e) => setLev(numOrBlank(e.target.value))}/>
        </div>

        <div className="flex items-center gap-2 md:col-span-2">
          <label className="text-sm w-16 opacity-70">Tag</label>
          <input className="px-2 py-1.5 rounded-md border dark:border-gray-800 text-sm flex-1"
                 value={clientTag} onChange={(e) => setClientTag(e.target.value)} />
        </div>
      </div>

      {/* Errors & rails */}
      {err && <div className="mt-2 text-sm text-red-600">{err}</div>}
      <div className="mt-2 text-xs grid gap-1 md:grid-cols-3">
        <Rail label="Notional cap" value={maxNotional ? money(maxNotional) : "—"} warn={!!notionalError} />
        <Rail label="Leverage cap" value={maxLeverage ? `${maxLeverage}×` : "—"} warn={!!leverageError} />
        <Rail label="Price band" value={markPrice ? `±${priceBandPct}% vs mark` : "—"} warn={!!bandError} />
      </div>

      {/* Estimate row */}
      <div className="mt-3 rounded-xl border dark:border-gray-800 p-3 text-sm grid gap-2 md:grid-cols-4">
        <KV k="Notional" v={money(notional)} />
        <KV k="Fees" v={estim ? money(estim.fees) : "—"} />
        <KV k="Slippage" v={estim ? money(estim.slippage) : "—"} />
        <KV k="Total" v={estim ? money(estim.total) : "—"} strong />
      </div>

      {/* Submit */}
      <div className="mt-3 flex items-center justify-between">
        <div className="text-xs opacity-70">
          {side === "BUY" ? "You will **pay** notional + costs." : "You will **receive** notional – costs."}
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 rounded-md border dark:border-gray-800 text-sm" onClick={clear}>Clear</button>
          <button
            className={`px-3 py-1.5 rounded-md text-sm text-white ${side === "BUY" ? "bg-green-600 hover:bg-green-700" : "bg-red-600 hover:bg-red-700"}`}
            onClick={submit}
            disabled={busy}
            title="Ctrl/⌘+Enter"
          >
            {busy ? "Submitting…" : `${side} ${qty || ""} ${symbol || ""}`}
          </button>
        </div>
      </div>
    </div>
  );
}

/* --------------------------------- Bits --------------------------------- */

function Rail({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className={`px-2 py-1 rounded-md ${warn ? "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300" : "bg-gray-50 dark:bg-gray-800 opacity-90"}`}>
      <span className="opacity-70 mr-2">{label}</span>
      <b>{value}</b>
    </div>
  );
}
function KV({ k, v, strong }: { k: string; v: string; strong?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="opacity-70">{k}</span>
      <span className={strong ? "font-semibold" : ""}>{v}</span>
    </div>
  );
}

/* -------------------------------- Utils -------------------------------- */

function toNum(x: number | string | "" | null | undefined): number | null {
  if (typeof x === "number") return Number.isFinite(x) ? x : null;
  if (x === "" || x == null) return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}
function numOrZero(s: string) { const n = Number(s); return Number.isFinite(n) ? n : 0; }
function numOrBlank(s: string) { const n = Number(s); return Number.isFinite(n) ? n : ""; }
function round(x: number) { return Number(x.toFixed(8)); }

function tickGuess(px?: number | null) {
  if (!px || !Number.isFinite(px)) return 0.01;
  if (px >= 1000) return 1;
  if (px >= 100) return 0.1;
  if (px >= 1) return 0.01;
  if (px >= 0.1) return 0.001;
  return 0.0001;
}
function fmt(x: number) {
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 8 }); } catch { return String(x); }
}
function money(x: number) {
  try { return x.toLocaleString(undefined, { style: "currency", currency: "USD" }); } catch { return `$${x.toFixed(2)}`; }
}