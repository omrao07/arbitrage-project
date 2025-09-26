// macro/interestratearb.ts
// Import-free utilities for interest-rate & cost-of-carry arbitrage.
//
// This module focuses on *pricing parity* across spot/forward/futures and
// the carry inputs behind them (funding rates, income/dividends, storage,
// convenience yield). It intentionally uses simple compounding by default,
// with an option for continuous compounding where noted.
//
// What you get (pure functions):
// - yearFrac({start,end,dayCount})
// - fairForward({spot, r, q, T, comp})                         // generic parity: F = S * e^{(r - q)T} or simple
// - forwardMispricing({spot, fwd, r, q, T, comp})              // signed mispricing & annualized basis
// - impliedRateFromForward({spot, fwd, q, T, comp})            // back out funding r
// - equityIndexFairForward({spot, r, divs?, q?, T, comp})      // supports discrete dividends
// - equityIndexImpliedRate({spot, fwd, divs?, T, comp})
// - commodityFairForward({spot, rf, storage?, convYield?, T, comp})
// - impliedRepoFromFutures({spot, fwd, carryCF, T, comp})      // generic IRR from cash-and-carry (carryCF = net carry cashflows)
// - cashAndCarryPnL({spot, fwd, rBorrow, rLend, q, fees, T, comp})
// - calendarSpreadTheo({F1, T1, F2, T2})                       // drift implied by (r-q)
// - breakevenFunding({spot, fwd, q, T, comp})                  // r that makes no-arb
//
// Conventions
// - r, q, rf, storage, convYield are annualized decimals (0.05 = 5%).
// - T is in *years*. Use yearFrac() to compute it.
// - comp: "simple" (default) or "cont" (continuous).
//
// NOTE: These are pedagogical/desk-ready utilities, not a full valuation lib.

export type Comp = "simple" | "cont";

export function yearFrac(args: {
  start: number | Date;
  end: number | Date;
  dayCount?: "ACT/365" | "ACT/360" | "30/360";
}): number {
  const dc = args.dayCount || "ACT/365";
  const t0 = toDate(args.start), t1 = toDate(args.end);
  if (dc === "30/360") {
    const d0 = d30360(t0), d1 = d30360(t1);
    const days = (d1.y - d0.y) * 360 + (d1.m - d0.m) * 30 + (d1.d - d0.d);
    return days / 360;
  }
  const ms = t1.getTime() - t0.getTime();
  const days = ms / 86400000;
  return dc === "ACT/360" ? days / 360 : days / 365;
}

/* ---------------------------- Core Parities ---------------------------- */

export function fairForward(args: {
  spot: number;     // S
  r: number;        // funding (carry cost)
  q?: number;       // income yield (dividends, convenience negative cost)
  T: number;        // in years
  comp?: Comp;
}): number {
  const S = num(args.spot);
  const r = num(args.r);
  const q = num(args.q, 0);
  const T = Math.max(0, num(args.T));
  const comp: Comp = args.comp || "simple";
  if (S <= 0 || T < 0) return NaN;
  return comp === "cont" ? S * Math.exp((r - q) * T) : S * (1 + (r - q) * T);
}

export function forwardMispricing(args: {
  spot: number; fwd: number; r: number; q?: number; T: number; comp?: Comp;
}): { mispricing: number; mispricingPct: number; basisAnn: number } {
  const theo = fairForward(args);
  if (!isFinite(theo)) return { mispricing: NaN, mispricingPct: NaN, basisAnn: NaN };
  const mis = num(args.fwd) - theo;
  const pct = theo !== 0 ? mis / theo : NaN;
  // Annualized “basis” b that reconciles F: treat as extra carry on quote side → here as delta(r-q)
  const b = basisFromF_S(args.spot, args.fwd, args.r, args.q || 0, args.T, args.comp || "simple");
  return { mispricing: round6(mis), mispricingPct: round6(pct), basisAnn: round6(b) };
}

export function impliedRateFromForward(args: {
  spot: number; fwd: number; q?: number; T: number; comp?: Comp;
}): number {
  const S = num(args.spot), F = num(args.fwd), q = num(args.q, 0), T = num(args.T);
  const comp: Comp = args.comp || "simple";
  if (S <= 0 || F <= 0 || T <= 0) return NaN;
  if (comp === "cont") return Math.log(F / S) / T + q;
  return (F / S - 1) / T + q;
}

/* --------------------------- Equity Index Fwds -------------------------- */

export function equityIndexFairForward(args: {
  spot: number;        // index level
  r: number;           // funding
  T: number;           // years
  divs?: Array<{ ts: number | Date; amount: number }>; // discrete cash dividends (per index “unit”)
  q?: number;          // continuous dividend yield alternative (if provided, divs ignored)
  comp?: Comp;
}): number {
  const S = num(args.spot);
  const r = num(args.r);
  const T = num(args.T);
  const comp: Comp = args.comp || "simple";
  if (S <= 0 || T < 0) return NaN;

  if (isFinite(args.q as number)) {
    return fairForward({ spot: S, r, q: num(args.q!, 0), T, comp });
  }

  // Discrete dividends: F = (S - PV(divs)) * growth(r)
  const pvDivs = sumPVDivs(args.divs || [], r, comp, T);
  const carry = comp === "cont" ? Math.exp(r * T) : (1 + r * T);
  return (S - pvDivs) * carry;
}

export function equityIndexImpliedRate(args: {
  spot: number; fwd: number; T: number;
  divs?: Array<{ ts: number | Date; amount: number }>;
  q?: number;
  comp?: Comp;
}): number {
  const S = num(args.spot), F = num(args.fwd), T = num(args.T);
  const comp: Comp = args.comp || "simple";
  if (S <= 0 || F <= 0 || T <= 0) return NaN;

  if (isFinite(args.q as number)) {
    // r = (ln(F/S)+qT)/T for cont; simple: r = ((F/S)-1)/T + q
    return impliedRateFromForward({ spot: S, fwd: F, q: num(args.q!, 0), T, comp });
  }

  // With discrete dividends: solve r from F = (S - PV(divs)) * growth(r)
  // growth(r) = e^{rT} or (1+rT). Rearranged:
  const pv = sumPVDivs(args.divs || [], 0, comp, T); // start with r=0 for PV; effect on PV from r is tiny for short T
  if (comp === "cont") return Math.log(F / Math.max(1e-12, (S - pv))) / T;
  return (F / Math.max(1e-12, (S - pv)) - 1) / T;
}

/* ---------------------------- Commodities Fwds -------------------------- */

export function commodityFairForward(args: {
  spot: number; rf: number; T: number;
  storage?: number;        // + cost
  convYield?: number;      // − benefit
  comp?: Comp;
}): number {
  const S = num(args.spot), rf = num(args.rf), T = num(args.T);
  const storage = num(args.storage, 0), y = num(args.convYield, 0);
  const comp: Comp = args.comp || "simple";
  if (S <= 0 || T < 0) return NaN;
  const carry = (rf + storage - y);
  return comp === "cont" ? S * Math.exp(carry * T) : S * (1 + carry * T);
}

/* ----------------------------- Implied “Repo” ---------------------------- */

export function impliedRepoFromFutures(args: {
  spot: number; fwd: number; carryCF?: number; T: number; comp?: Comp;
}): number {
  // carryCF: net cashflows during life (income positive reduces carry need)
  // Solve r in: F = (S - PV_income + PV_cost) * growth(r)
  const S = num(args.spot), F = num(args.fwd), T = num(args.T);
  const comp: Comp = args.comp || "simple";
  const cf = num(args.carryCF, 0); // use sign convention: income negative CF? We'll treat positive CF as income reducing base: S - CF
  if (S <= 0 || F <= 0 || T <= 0) return NaN;

  // Effective adjusted spot:
  const Se = S - cf; // CF approximated at present; for accuracy discount each CF separately
  if (comp === "cont") return Math.log(F / Math.max(1e-12, Se)) / T;
  return (F / Math.max(1e-12, Se) - 1) / T;
}

/* ---------------------------- Cash-and-carry PnL ------------------------- */

export function cashAndCarryPnL(args: {
  spot: number; fwd: number; T: number;
  rBorrow: number;         // funding paid to buy spot
  rLend: number;           // invest proceeds from selling forward (or shorting spot)
  q?: number;              // income yield on the asset (e.g., dividend yield)
  fees?: number;           // total fees/slippage per unit notional (both legs)
  comp?: Comp;
}): { direction: "cash-and-carry" | "reverse-cash-and-carry" | "none"; pnl: number; pnlAnn: number } {
  // Classic: If F > S * carry_up → buy spot, finance at rBorrow, sell forward (cash-and-carry).
  // If F < S * carry_down → short spot, invest proceeds at rLend, buy forward (reverse).
  const S = num(args.spot), F = num(args.fwd), T = num(args.T);
  const rB = num(args.rBorrow), rL = num(args.rLend), q = num(args.q, 0);
  const fees = num(args.fees, 0);
  const comp: Comp = args.comp || "simple";
  if (S <= 0 || F <= 0 || T <= 0) return { direction: "none", pnl: NaN, pnlAnn: NaN };

  const carryUp = priceWithCarry(S, rB - q, T, comp); // financing minus income
  const carryDn = priceWithCarry(S, rL - q, T, comp);

  if (F > carryUp) {
    // cash-and-carry (long spot, short forward)
    const pnl = F - carryUp - fees;
    return { direction: "cash-and-carry", pnl: round6(pnl), pnlAnn: round6(pnl / T) };
  } else if (F < carryDn) {
    // reverse (short spot, long forward)
    const pnl = carryDn - F - fees;
    return { direction: "reverse-cash-and-carry", pnl: round6(pnl), pnlAnn: round6(pnl / T) };
  }
  return { direction: "none", pnl: 0, pnlAnn: 0 };
}

/* --------------------------- Calendar Spread Theo ----------------------- */

export function calendarSpreadTheo(args: {
  F1: number; T1: number; F2: number; T2: number;
}): { impliedCarry: number; fairDiffPerYear: number } {
  // Solve r - q from two forwards: F = S * (1 + (r - q)T)  (simple)
  // Eliminating S: (F2 - F1) / (T2 - T1) ≈ S*(r - q)
  // As a practical desk proxy, we use: impliedCarry ≈ (F2/F1 - 1) / (T2 - T1) (scale-free)
  const F1 = num(args.F1), F2 = num(args.F2), T1 = num(args.T1), T2 = num(args.T2);
  if (F1 <= 0 || F2 <= 0 || T2 <= T1) return { impliedCarry: NaN, fairDiffPerYear: NaN };
  const impliedCarry = (F2 / F1 - 1) / (T2 - T1);
  // Fair difference per year in outright terms (simple) using F1 as proxy for S carry base:
  const fairDiffPerYear = (F2 - F1) / (T2 - T1);
  return { impliedCarry, fairDiffPerYear };
}

/* ------------------------------ Breakeven r ----------------------------- */

export function breakevenFunding(args: {
  spot: number; fwd: number; q?: number; T: number; comp?: Comp;
}): number {
  // r that makes F = fairForward(...)
  return impliedRateFromForward({ spot: args.spot, fwd: args.fwd, q: num(args.q, 0), T: args.T, comp: args.comp || "simple" });
}

/* -------------------------------- Helpers ------------------------------- */

function basisFromF_S(S: number, F: number, r: number, q: number, T: number, comp: Comp) {
  if (S <= 0 || F <= 0 || T <= 0) return NaN;
  // Return annualized delta to (r - q) that reconciles observed F with parity
  if (comp === "cont") return Math.log(F / S) / T - (r - q);
  return (F / S - 1) / T - (r - q);
}

function priceWithCarry(S: number, carry: number, T: number, comp: Comp) {
  return comp === "cont" ? S * Math.exp(carry * T) : S * (1 + carry * T);
}

function sumPVDivs(divs: Array<{ ts: number | Date; amount: number }>, r: number, comp: Comp, T: number) {
  if (!divs || divs.length === 0) return 0;
  let pv = 0;
  const now = Date.now();
  for (let i = 0; i < divs.length; i++) {
    const amt = num(divs[i].amount, 0);
    const t = Math.max(0, yearFrac({ start: now, end: divs[i].ts, dayCount: "ACT/365" }));
    if (t > T + 1 / 365) continue; // ignore beyond maturity
    pv += comp === "cont" ? amt * Math.exp(-r * t) : amt / (1 + r * t);
  }
  return pv;
}

function toDate(x: number | Date) { return x instanceof Date ? x : new Date(x); }
function d30360(d: Date) {
  let Y = d.getUTCFullYear(), M = d.getUTCMonth() + 1, D = d.getUTCDate();
  if (D === 31) D = 30;
  return { y: Y, m: M, d: D };
}
function num(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function round6(n: number) { return Math.round(n * 1e6) / 1e6; }

/* --------------------------------- Mini Demo ---------------------------- */
// const T = yearFrac({ start: Date.now(), end: Date.now()+90*86400000 });
// const F = fairForward({ spot: 100, r: 0.05, q: 0.02, T });
// const mis = forwardMispricing({ spot: 100, fwd: F*1.002, r: 0.05, q: 0.02, T });
// console.log({ T, F, mis });