"use server";

/**
 * fetchOrders.server.ts
 * - Server action to fetch open/working orders for an account
 * - Replace the mockOrders() with adapter calls (IBKR, Alpaca, Binance, etc.)
 */

export type OrderSide = "BUY" | "SELL";
export type OrderType = "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT";
export type OrderStatus =
  | "NEW"
  | "PARTIALLY_FILLED"
  | "FILLED"
  | "CANCELED"
  | "REPLACED"
  | "REJECTED";

export interface Order {
  id: string;             // unique id from broker
  accountId: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  qty: number;
  filledQty: number;
  limitPrice?: number;
  stopPrice?: number;
  tif: "DAY" | "GTC";
  status: OrderStatus;
  avgFillPrice?: number;
  createdAt: number;      // epoch ms
  updatedAt: number;      // epoch ms
  route?: string;
  tag?: string;           // client tag
}

export interface FetchOrdersInput {
  accountId: string;
  symbols?: string[];     // optional filter
  status?: OrderStatus[]; // optional filter
}

export interface FetchOrdersResult {
  ts: number;
  orders: Order[];
}

export async function fetchOrders(input: FetchOrdersInput): Promise<FetchOrdersResult> {
  const { accountId, symbols, status } = input;
  if (!accountId) throw new Error("accountId is required");

  // TODO: swap with your adapter, e.g. await broker.getOpenOrders(accountId)
  const all = mockOrders(accountId);

  // optional filters
  const filtered = all.filter((o) => {
    if (symbols && symbols.length && !symbols.includes(o.symbol)) return false;
    if (status && status.length && !status.includes(o.status)) return false;
    return true;
  });

  return {
    ts: Date.now(),
    orders: filtered,
  };
}

/* ---------------- helpers ---------------- */

function mockOrders(accountId: string): Order[] {
  const now = Date.now();
  return [
    {
      id: "ord_1",
      accountId,
      symbol: "AAPL",
      side: "BUY",
      type: "LIMIT",
      qty: 100,
      filledQty: 0,
      limitPrice: 192.5,
      tif: "DAY",
      status: "NEW",
      createdAt: now - 30000,
      updatedAt: now - 30000,
      route: "SMART",
      tag: "rebalance",
    },
    {
      id: "ord_2",
      accountId,
      symbol: "MSFT",
      side: "SELL",
      type: "STOP_LIMIT",
      qty: 50,
      filledQty: 0,
      limitPrice: 410,
      stopPrice: 415,
      tif: "GTC",
      status: "NEW",
      createdAt: now - 120000,
      updatedAt: now - 120000,
    },
    {
      id: "ord_3",
      accountId,
      symbol: "BTC-USD",
      side: "BUY",
      type: "MARKET",
      qty: 0.25,
      filledQty: 0.25,
      tif: "DAY",
      status: "FILLED",
      avgFillPrice: 64000,
      createdAt: now - 3600000,
      updatedAt: now - 3550000,
      route: "Binance",
    },
  ];
}