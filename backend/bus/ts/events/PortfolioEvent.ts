// events/portfolio_event.ts

/** ========= Shared primitives ========= */
export type Side = "BUY" | "SELL";
export type OrderType = "MKT" | "LMT" | "STP" | "STP_LMT";
export type TimeInForce = "DAY" | "GTC" | "IOC" | "FOK";
export type Currency = "USD" | "EUR" | "GBP" | "JPY" | string;

export interface BaseEvent {
  ts: string;          // ISO 8601 timestamp
  account: string;     // account or portfolio id
  source?: string;     // producer system/node
  seq?: number;        // monotonic sequence if available
}

/** ========= Orders ========= */
export interface OrderEvent extends BaseEvent {
  type: "order";
  orderId: string;
  symbol: string;
  side: Side;
  qty: number;                     // desired quantity (+ for buy, - for sell is allowed but side is authoritative)
  filledQty?: number;              // cumulative filled
  price?: number;                  // limit or stop price as applicable
  orderType: OrderType;
  tif?: TimeInForce;
  status:
    | "NEW"
    | "ACCEPTED"
    | "PARTIALLY_FILLED"
    | "FILLED"
    | "REPLACED"
    | "CANCELLED"
    | "REJECTED";
  venue?: string;                  // exchange/broker
  notes?: string;
}

/** ========= Positions ========= */
export interface PositionEvent extends BaseEvent {
  type: "position";
  symbol: string;
  qty: number;                     // current position
  avgPx?: number;                  // average cost
  currency?: Currency;
  gross?: number;                  // $ gross exposure
  net?: number;                    // $ net exposure (by sign)
  leverage?: number;               // portfolio leverage if provided
  tags?: string[];                 // e.g. book/desk/strategy labels
}

/** ========= PnL ========= */
export interface PnLEvent extends BaseEvent {
  type: "pnl";
  period: "INTRADAY" | "DAILY" | "MTD" | "YTD" | "TOTAL";
  currency?: Currency;
  /** point-in-time values (USD unless currency set) */
  gross?: number;
  net?: number;
  fees?: number;
  slippage?: number;
  /** attribution buckets */
  byStrategy?: Record<string, number>;
  bySymbol?: Record<string, number>;
}

/** Union of portfolio events */
export type PortfolioEvent = OrderEvent | PositionEvent | PnLEvent;

/** ========= Type guards ========= */
export const isOrderEvent    = (e: PortfolioEvent): e is OrderEvent    => e.type === "order";
export const isPositionEvent = (e: PortfolioEvent): e is PositionEvent => e.type === "position";
export const isPnLEvent      = (e: PortfolioEvent): e is PnLEvent      => e.type === "pnl";

/** ========= Validation (minimal; no deps) ========= */
function isIso(s: unknown): s is string {
  return typeof s === "string" && !Number.isNaN(Date.parse(s));
}
function isFiniteNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}
function req(cond: boolean, msg: string, acc: string[]) { if (!cond) acc.push(msg); }

export function validatePortfolioEvent(e: unknown): string[] {
  const errs: string[] = [];
  const x = e as Partial<PortfolioEvent>;
  req(!!x, "event required", errs);
  if (!x) return ["event null/undefined"];
  req(typeof x.type === "string", "type must be string", errs);
  req(isIso(x.ts), "ts must be ISO8601", errs);
  req(typeof x.account === "string" && x.account.length > 0, "account required", errs);

  switch (x.type) {
    case "order": {
      const o = x as OrderEvent;
      req(typeof o.orderId === "string" && o.orderId.length > 0, "order.orderId required", errs);
      req(typeof o.symbol === "string" && o.symbol.length > 0, "order.symbol required", errs);
      req(o.side === "BUY" || o.side === "SELL", "order.side invalid", errs);
      req(isFiniteNum(o.qty), "order.qty must be number", errs);
      req(["MKT","LMT","STP","STP_LMT"].includes(o.orderType), "order.orderType invalid", errs);
      req(
        ["NEW","ACCEPTED","PARTIALLY_FILLED","FILLED","REPLACED","CANCELLED","REJECTED"].includes(o.status),
        "order.status invalid", errs
      );
      break;
    }
    case "position": {
      const p = x as PositionEvent;
      req(typeof p.symbol === "string" && p.symbol.length > 0, "position.symbol required", errs);
      req(isFiniteNum(p.qty), "position.qty must be number", errs);
      break;
    }
    case "pnl": {
      const p = x as PnLEvent;
      req(["INTRADAY","DAILY","MTD","YTD","TOTAL"].includes(p.period), "pnl.period invalid", errs);
      break;
    }
    default:
      errs.push("unsupported type");
  }
  return errs;
}

/** ========= Constructors ========= */
export const nowIso = () => new Date().toISOString();

export function makeOrder(p: Omit<OrderEvent, "type" | "ts"> & { ts?: string }): OrderEvent {
  return { type: "order", ts: p.ts ?? nowIso(), ...p };
}

export function makePosition(p: Omit<PositionEvent, "type" | "ts"> & { ts?: string }): PositionEvent {
  return { type: "position", ts: p.ts ?? nowIso(), ...p };
}

export function makePnL(p: Omit<PnLEvent, "type" | "ts"> & { ts?: string }): PnLEvent {
  return { type: "pnl", ts: p.ts ?? nowIso(), ...p };
}

/** ========= Serialization helpers ========= */
export function toJSON(e: PortfolioEvent): string {
  return JSON.stringify(e);
}
export function fromJSON(json: string): PortfolioEvent {
  const obj = JSON.parse(json);
  const errs = validatePortfolioEvent(obj);
  if (errs.length) throw new Error("Invalid PortfolioEvent: " + errs.join("; "));
  return obj as PortfolioEvent;
}

/** ========= Default headers for bus ========= */
export function defaultHeaders() {
  return {
    "content-type": "application/json",
    "x-event-type": "portfolio",
  };
}