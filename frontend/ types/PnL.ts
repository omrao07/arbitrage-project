// frontend/types/PnL.ts
// Strong types for P&L snapshots, time-series, and attribution,
// plus small utilities for aggregation, sharpe, and drawdown.

export type ISODate = string;       // "2025-09-11"
export type ISODateTime = string;   // "2025-09-11T15:30:00Z"

export interface PnLSnapshot {
  asOf: ISODateTime;
  navUSD: number;            // current equity
  grossUSD: number;          // |long| + |short|
  netUSD: number;            // |long - short|
  pnlDayUSD: number;         // realized + unrealized today
  pnlMtdUSD?: number;
  pnlYtdUSD?: number;
  varUSD?: number;           // 1-day VaR
  maxDrawdownUSD?: number;
}

export interface PnLBar {
  date: ISODate;
  pnlUSD: number;            // daily total
  feesUSD?: number;
  slippageUSD?: number;
  tradePnLUSD?: number;      // execution alpha
  pricePnLUSD?: number;      // mark-to-market
  fxPnLUSD?: number;
  carryUSD?: number;
  navEndUSD?: number;        // optional end-of-day NAV for plotting
}

export interface PnLAttributionRow {
  date: ISODate;
  asset: string;
  strategyId?: string;
  pricePnLUSD: number;
  fxPnLUSD: number;
  carryUSD: number;
  tradePnLUSD: number;
  slippageUSD: number;
  feesUSD: number;
  totalUSD: number;
}

export interface PnLSeries {
  bars: PnLBar[];
  asOf: ISODateTime;
}

export interface AttributionSeries {
  byAsset: PnLAttributionRow[];
  byStrategy?: PnLAttributionRow[];
  asOf: ISODateTime;
}

/* ------------------------------------------------------------------------------------------------
 * Aggregation helpers
 * ------------------------------------------------------------------------------------------------ */

export function sumPnL(bars: PnLBar[], from?: ISODate, to?: ISODate): number {
  const f = from ? new Date(from).getTime() : Number.NEGATIVE_INFINITY;
  const t = to ? new Date(to).getTime() : Number.POSITIVE_INFINITY;
  return bars.reduce((acc, b) => {
    const ts = new Date(b.date).getTime();
    return acc + (ts >= f && ts <= t ? (b.pnlUSD || 0) : 0);
  }, 0);
}

export function rollupAttribution(
  rows: PnLAttributionRow[],
  by: "asset" | "strategyId" = "asset"
): Record<string, { totalUSD: number; pricePnLUSD: number; fxPnLUSD: number; carryUSD: number; tradePnLUSD: number; slippageUSD: number; feesUSD: number }> {
  const out: Record<string, any> = {};
  for (const r of rows) {
    const k = String(by === "asset" ? r.asset : r.strategyId ?? "unknown");
    (out[k] ||= { totalUSD: 0, pricePnLUSD: 0, fxPnLUSD: 0, carryUSD: 0, tradePnLUSD: 0, slippageUSD: 0, feesUSD: 0 });
    out[k].totalUSD     += r.totalUSD || 0;
    out[k].pricePnLUSD  += r.pricePnLUSD || 0;
    out[k].fxPnLUSD     += r.fxPnLUSD || 0;
    out[k].carryUSD     += r.carryUSD || 0;
    out[k].tradePnLUSD  += r.tradePnLUSD || 0;
    out[k].slippageUSD  += r.slippageUSD || 0;
    out[k].feesUSD      += r.feesUSD || 0;
  }
  return out;
}

/* ------------------------------------------------------------------------------------------------
 * Risk/quality metrics (daily bars → returns)
 * ------------------------------------------------------------------------------------------------ */

/** Annualized Sharpe from daily PnL and NAV (assumes ~252 trading days). */
export function sharpeAnnualized(bars: PnLBar[], navBaselineUSD: number): number {
  if (!bars.length || navBaselineUSD <= 0) return 0;
  const rets = bars.map(b => (b.pnlUSD || 0) / navBaselineUSD);
  const mu = mean(rets);
  const sd = stdev(rets);
  if (sd <= 0) return 0;
  return (mu / sd) * Math.sqrt(252);
}

/** Max drawdown on cumulative PnL series. Returns absolute USD loss (negative). */
export function maxDrawdownUSD(bars: PnLBar[]): number {
  let peak = 0, cum = 0, mdd = 0;
  for (const b of bars) {
    cum += b.pnlUSD || 0;
    peak = Math.max(peak, cum);
    mdd = Math.min(mdd, cum - peak);
  }
  return mdd; // ≤ 0
}

/* ------------------------------------------------------------------------------------------------
 * Transform utilities
 * ------------------------------------------------------------------------------------------------ */

/** Merge multiple daily series by date, summing PnL & components. */
export function mergeSeries(...series: PnLSeries[]): PnLSeries {
  const map = new Map<string, PnLBar>();
  for (const s of series) {
    for (const b of s.bars) {
      const prev = map.get(b.date) || { date: b.date, pnlUSD: 0, feesUSD: 0, slippageUSD: 0, tradePnLUSD: 0, pricePnLUSD: 0, fxPnLUSD: 0, carryUSD: 0 };
      map.set(b.date, {
        date: b.date,
        pnlUSD: (prev.pnlUSD || 0) + (b.pnlUSD || 0),
        feesUSD: (prev.feesUSD || 0) + (b.feesUSD || 0),
        slippageUSD: (prev.slippageUSD || 0) + (b.slippageUSD || 0),
        tradePnLUSD: (prev.tradePnLUSD || 0) + (b.tradePnLUSD || 0),
        pricePnLUSD: (prev.pricePnLUSD || 0) + (b.pricePnLUSD || 0),
        fxPnLUSD: (prev.fxPnLUSD || 0) + (b.fxPnLUSD || 0),
        carryUSD: (prev.carryUSD || 0) + (b.carryUSD || 0),
        navEndUSD: b.navEndUSD ?? prev.navEndUSD,
      });
    }
  }
  const bars = Array.from(map.values()).sort((a, b) => (a.date < b.date ? -1 : 1));
  return { bars, asOf: new Date().toISOString() };
}

/** Convert attribution rows → daily bars by summing totals per day. */
export function attributionToBars(rows: PnLAttributionRow[]): PnLBar[] {
  const map = new Map<string, PnLBar>();
  for (const r of rows) {
    const prev = map.get(r.date) || { date: r.date, pnlUSD: 0, feesUSD: 0, slippageUSD: 0, tradePnLUSD: 0, pricePnLUSD: 0, fxPnLUSD: 0, carryUSD: 0 };
    map.set(r.date, {
      date: r.date,
      pnlUSD: (prev.pnlUSD || 0) + (r.totalUSD || 0),
      feesUSD: (prev.feesUSD || 0) + (r.feesUSD || 0),
      slippageUSD: (prev.slippageUSD || 0) + (r.slippageUSD || 0),
      tradePnLUSD: (prev.tradePnLUSD || 0) + (r.tradePnLUSD || 0),
      pricePnLUSD: (prev.pricePnLUSD || 0) + (r.pricePnLUSD || 0),
      fxPnLUSD: (prev.fxPnLUSD || 0) + (r.fxPnLUSD || 0),
      carryUSD: (prev.carryUSD || 0) + (r.carryUSD || 0),
    });
  }
  return Array.from(map.values()).sort((a, b) => (a.date < b.date ? -1 : 1));
}

/* ------------------------------------------------------------------------------------------------
 * Formatting helpers for React tables/charts
 * ------------------------------------------------------------------------------------------------ */

export const fmtUSD = (v: number, digits = 0) =>
  (v ?? 0).toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: digits });

export const fmtSignedUSD = (v: number, digits = 0) => {
  const s = fmtUSD(Math.abs(v ?? 0), digits);
  return (v ?? 0) >= 0 ? `+${s}` : `-${s}`;
};

export const pct = (v: number, digits = 2) => `${(100 * (v ?? 0)).toFixed(digits)}%`;

/* ------------------------------------------------------------------------------------------------
 * Small stats
 * ------------------------------------------------------------------------------------------------ */

export function mean(xs: number[]): number {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + (b || 0), 0) / xs.length;
}

export function stdev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const v = xs.reduce((acc, x) => acc + Math.pow((x || 0) - m, 2), 0) / (xs.length - 1);
  return Math.sqrt(v);
}

/* ------------------------------------------------------------------------------------------------
 * Quick builders for UI mocks/tests
 * ------------------------------------------------------------------------------------------------ */

export function makePnLBar(date: ISODate, pnlUSD: number): PnLBar {
  return { date, pnlUSD };
}

export function makeSeries(dates: ISODate[], vals: number[]): PnLSeries {
  return {
    bars: dates.map((d, i) => makePnLBar(d, vals[i] ?? 0)),
    asOf: new Date().toISOString(),
  };
}