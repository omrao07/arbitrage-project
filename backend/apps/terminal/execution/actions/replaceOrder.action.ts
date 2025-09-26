"use server";

export type OrderSide = "BUY" | "SELL";
export type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
export type TIF = "DAY" | "GTC";

export interface ReplaceOrderInput {
  /** The existing broker/OMS order id */
  orderId: string;
  /** Account that owns the order (some brokers require it) */
  accountId?: string;

  // --- fields you may replace (all optional) ---
  qty?: number;              // new absolute quantity
  type?: OrderType;          // e.g., LIMIT -> STOP_LIMIT
  limitPrice?: number;       // new limit
  stopPrice?: number;        // new stop
  tif?: TIF;                 // DAY/GTC
  tag?: string;              // new client tag / memo

  // immutable fields (normally NOT allowed to change, included for completeness)
  symbol?: string;           // not used by most replace endpoints
  side?: OrderSide;          // not used by most replace endpoints
  /** Optional reason for audit trail */
  reason?: string;
}

export interface ReplaceOrderResult {
  success: boolean;
  orderId: string;
  replacedFields?: Array<keyof ReplaceOrderInput>;
  message?: string;
  error?: string;
}

/**
 * Replace (amend) an existing working order.
 * NOTE: This is a mock adapter — plug your broker/OMS call here.
 *  - IBKR: modifyOrder
 *  - Alpaca: PATCH /v2/orders/{order_id}
 *  - Zerodha: modify order via kiteconnect
 *  - Binance: order modify endpoint where supported
 */
export async function replaceOrder(
  input: ReplaceOrderInput
): Promise<ReplaceOrderResult> {
  try {
    const {
      orderId,
      accountId,
      qty,
      type,
      limitPrice,
      stopPrice,
      tif,
      tag,
    } = input;

    if (!orderId) throw new Error("orderId is required");
    if (
      type === "LIMIT" && limitPrice == null ||
      type === "STOP" && stopPrice == null ||
      type === "STOP_LIMIT" && (limitPrice == null || stopPrice == null)
    ) {
      throw new Error("missing price fields for the selected order type");
    }
    if (qty != null && qty <= 0) throw new Error("qty must be > 0 when provided");

    // Mock latency (replace with SDK/HTTP call)
    await sleep(250);

    // Determine which fields are actually being changed (for UI/audit)
    const replaced: Array<keyof ReplaceOrderInput> = [];
    (["qty","type","limitPrice","stopPrice","tif","tag"] as const).forEach((k) => {
      if (input[k] != null) replaced.push(k);
    });

    // Example: call your broker adapter
    // await broker.replaceOrder({ orderId, accountId, qty, type, limitPrice, stopPrice, tif, tag });

    console.log(
      `[replaceOrder] id=${orderId} acct=${accountId ?? "-"} fields=${replaced.join(",") || "none"}`
    );

    return {
      success: true,
      orderId,
      replacedFields: replaced,
      message: `Order ${orderId} amended (${replaced.join(", ") || "no changes"})`,
    };
  } catch (err: any) {
    console.error("replaceOrder failed:", err);
    return {
      success: false,
      orderId: input.orderId,
      error: err?.message || "Unknown error",
    };
  }
}

/* ---------- utils ---------- */

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}