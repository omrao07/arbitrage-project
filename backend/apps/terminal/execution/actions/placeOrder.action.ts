"use server";

export type OrderSide = "BUY" | "SELL";
export type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";

export interface PlaceOrderInput {
  accountId: string;
  symbol: string;       // e.g. "AAPL", "BTC-USD"
  qty: number;          // positive size
  side: OrderSide;      // BUY or SELL
  type: OrderType;      // MARKET, LIMIT, STOP...
  limitPrice?: number;  // required if type=LIMIT or STOP_LIMIT
  stopPrice?: number;   // required if type=STOP or STOP_LIMIT
  tif?: "DAY" | "GTC";  // time in force
  tag?: string;         // optional client order ID / tag
}

export interface PlaceOrderResult {
  success: boolean;
  orderId?: string;
  symbol?: string;
  qty?: number;
  side?: OrderSide;
  type?: OrderType;
  message?: string;
  error?: string;
}

/**
 * Places an order.
 * Mock implementation — swap with broker/execution API adapter.
 */
export async function placeOrder(
  input: PlaceOrderInput
): Promise<PlaceOrderResult> {
  try {
    const {
      accountId,
      symbol,
      qty,
      side,
      type,
      limitPrice,
      stopPrice,
      tif,
      tag,
    } = input;

    // Basic validation
    if (!accountId) throw new Error("accountId is required");
    if (!symbol) throw new Error("symbol is required");
    if (qty <= 0) throw new Error("qty must be > 0");

    if ((type === "LIMIT" || type === "STOP_LIMIT") && !limitPrice) {
      throw new Error("limitPrice required for LIMIT / STOP_LIMIT orders");
    }
    if ((type === "STOP" || type === "STOP_LIMIT") && !stopPrice) {
      throw new Error("stopPrice required for STOP / STOP_LIMIT orders");
    }

    // Mock latency
    await new Promise((r) => setTimeout(r, 300));

    // Mock order ID
    const orderId = `ord_${Date.now()}`;

    console.log(
      `[placeOrder] account=${accountId} ${side} ${qty} ${symbol} type=${type} limit=${limitPrice} stop=${stopPrice} tif=${tif} tag=${tag}`
    );

    return {
      success: true,
      orderId,
      symbol,
      qty,
      side,
      type,
      message: `Order placed successfully (ID=${orderId})`,
    };
  } catch (err: any) {
    console.error("placeOrder failed:", err);
    return {
      success: false,
      error: err.message || "Unknown error",
    };
  }
}