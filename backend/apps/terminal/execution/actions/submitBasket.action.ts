 "use server";

/** Sides, types & TIFs kept consistent with your other actions */
export type OrderSide = "BUY" | "SELL";
export type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
export type TIF = "DAY" | "GTC";

/** One leg in the basket */
export interface BasketLeg {
  symbol: string;
  qty: number;               // absolute size (positive)
  side: OrderSide;
  type: OrderType;
  limitPrice?: number;
  stopPrice?: number;
  tif?: TIF;
  tag?: string;              // client tag per-leg
}

/** Request to submit a basket */
export interface SubmitBasketInput {
  accountId: string;
  legs: BasketLeg[];
  basketTag?: string;        // optional basket-level client tag
  dryRun?: boolean;          // if true, validate & summarize, but do not place
}

/** Per-leg outcome */
export interface BasketLegResult {
  index: number;             // original index in input.legs
  symbol: string;
  side: OrderSide;
  qty: number;
  type: OrderType;
  status: "accepted" | "rejected" | "placed";
  message?: string;
  orderId?: string;
  error?: string;
}

/** Basket totals & response */
export interface SubmitBasketResult {
  success: boolean;
  basketId: string;          // synthetic id for audit/UI corr
  accountId: string;
  dryRun: boolean;

  // Exposure & cost summaries (best-effort estimates)
  legsAccepted: number;
  legsRejected: number;
  legsPlaced?: number;       // when dryRun=false
  grossNotional: number;     // sum(|qty * refPrice|)
  netNotional: number;       // sum(sign * qty * refPrice)
  buyNotional: number;
  sellNotional: number;

  // individual leg results
  results: BasketLegResult[];

  // optional aggregate warnings
  warnings?: string[];
  error?: string;
}

/**
 * Submit a basket of orders.
 * - Validates each leg
 * - Computes notional/exposure
 * - If dryRun=true → returns preview only
 * - Else "places" orders (mock here) and returns ids
 *
 * Replace the `mockPlace()` with your execution adapter (IBKR/Alpaca/etc).
 */
export async function submitBasket(
  input: SubmitBasketInput
): Promise<SubmitBasketResult> {
  const basketId = `basket_${Date.now()}`;

  try {
    if (!input.accountId) throw new Error("accountId is required");
    if (!Array.isArray(input.legs) || input.legs.length === 0) {
      throw new Error("at least one leg is required");
    }

    const dryRun = !!input.dryRun;

    // Validate legs + compute notionals using best available price proxy
    const results: BasketLegResult[] = [];
    let gross = 0, net = 0, buyNotional = 0, sellNotional = 0;
    let accepted = 0, rejected = 0;
    const warnings: string[] = [];

    input.legs.forEach((leg, i) => {
      const r: BasketLegResult = {
        index: i,
        symbol: String(leg.symbol || ""),
        side: leg.side,
        qty: Number(leg.qty),
        type: leg.type,
        status: "accepted",
        message: "validated",
      };

      // Basic validations
      if (!r.symbol) {
        r.status = "rejected";
        r.error = "symbol is required";
      } else if (!Number.isFinite(r.qty) || r.qty <= 0) {
        r.status = "rejected";
        r.error = "qty must be a positive number";
      } else if (
        (leg.type === "LIMIT" || leg.type === "STOP_LIMIT") &&
        !isFiniteNum(leg.limitPrice)
      ) {
        r.status = "rejected";
        r.error = "limitPrice required for LIMIT/STOP_LIMIT";
      } else if (
        (leg.type === "STOP" || leg.type === "STOP_LIMIT") &&
        !isFiniteNum(leg.stopPrice)
      ) {
        r.status = "rejected";
        r.error = "stopPrice required for STOP/STOP_LIMIT";
      }

      // Estimate a reference price for notional
      const ref =
        isFiniteNum(leg.limitPrice) ? (leg.limitPrice as number) :
        isFiniteNum(leg.stopPrice)  ? (leg.stopPrice  as number) :
        1; // fallback unit price if none supplied

      const notional = (r.qty || 0) * Math.abs(ref);
      if (r.status === "accepted") {
        gross += Math.abs(notional);
        const s = leg.side === "SELL" ? -1 : 1;
        net += s * notional;
        if (s > 0) buyNotional += notional; else sellNotional += Math.abs(notional);
        accepted++;
      } else {
        rejected++;
      }

      results.push(r);
    });

    if (rejected === input.legs.length) {
      return {
        success: false,
        basketId,
        accountId: input.accountId,
        dryRun,
        legsAccepted: 0,
        legsRejected: rejected,
        grossNotional: 0,
        netNotional: 0,
        buyNotional: 0,
        sellNotional: 0,
        results,
        error: "all legs rejected",
      };
    }

    // Add any basket-level warnings here (limits, exposure, etc.)
    if (gross > 100_000_000) warnings.push("gross notional exceeds $100m preview threshold");

    // Dry-run mode: return preview only
    if (dryRun) {
      return {
        success: true,
        basketId,
        accountId: input.accountId,
        dryRun: true,
        legsAccepted: accepted,
        legsRejected: rejected,
        grossNotional: round2(gross),
        netNotional: round2(net),
        buyNotional: round2(buyNotional),
        sellNotional: round2(sellNotional),
        results,
        warnings: warnings.length ? warnings : undefined,
      };
    }

    // Otherwise, place accepted legs sequentially (or in parallel if your OMS supports it)
    let placed = 0;
    for (const r of results) {
      if (r.status !== "accepted") continue;

      try {
        const leg = input.legs[r.index];
        // Replace this with your broker adapter:
        // const placedId = await adapters.execution.placeOrder({ ... });
        const placedId = await mockPlace(input.accountId, leg);
        r.orderId = placedId;
        r.status = "placed";
        r.message = "order placed";
        placed++;
      } catch (e: any) {
        r.status = "rejected";
        r.error = e?.message || "failed to place leg";
      }
    }

    const legsRejectedAfterPlacement = results.filter((x) => x.status !== "placed").length - (rejected);
    const ok = placed > 0;

    return {
      success: ok,
      basketId,
      accountId: input.accountId,
      dryRun: false,
      legsAccepted: accepted,
      legsRejected: rejected + Math.max(0, legsRejectedAfterPlacement),
      legsPlaced: placed,
      grossNotional: round2(gross),
      netNotional: round2(net),
      buyNotional: round2(buyNotional),
      sellNotional: round2(sellNotional),
      results,
      warnings: warnings.length ? warnings : undefined,
      error: ok ? undefined : "no legs placed",
    };
  } catch (err: any) {
    return {
      success: false,
      basketId,
      accountId: input.accountId,
      dryRun: !!input.dryRun,
      legsAccepted: 0,
      legsRejected: input.legs?.length ?? 0,
      grossNotional: 0,
      netNotional: 0,
      buyNotional: 0,
      sellNotional: 0,
      results: [],
      error: err?.message || "submitBasket failed",
    };
  }
}

/* ---------------- helpers / mock adapter ---------------- */

function isFiniteNum(v: any): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

// Mock placement — replace with real broker/OMS call
async function mockPlace(accountId: string, leg: BasketLeg): Promise<string> {
  await new Promise((r) => setTimeout(r, 120)); // simulate latency
  // Randomly fail a tiny fraction to exercise UI paths
  if (Math.random() < 0.01) throw new Error(`mock reject for ${leg.symbol}`);
  return `ord_${accountId}_${leg.symbol}_${Date.now()}_${Math.floor(Math.random() * 1e4)}`; }
