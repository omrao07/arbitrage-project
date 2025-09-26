// events/market_event.ts

/** ---------- Shared primitives ---------- */
export type Venue = "NYSE" | "NASDAQ" | "ARCA" | "IEX" | "CME" | "ICE" | "FX" | "OTC" | string;

export interface BaseEvent {
  /** ISO8601 (ms precision) */
  ts: string;
  /** e.g., "AAPL", "ESZ5", "EURUSD" */
  symbol: string;
  /** data source or venue */
  venue?: Venue;
  /** producer id (service/node) */
  source?: string;
  /** monotonic sequence if provided by feed */
  seq?: number;
}

/** ---------- Concrete event shapes ---------- */
export interface TickEvent extends BaseEvent {
  type: "tick";
  price: number;
  size?: number;
  bid?: number;
  ask?: number;
}

export interface TradeEvent extends BaseEvent {
  type: "trade";
  price: number;
  size: number;
  side?: "BUY" | "SELL";
  tradeId?: string;
}

export interface QuoteEvent extends BaseEvent {
  type: "quote";
  bid: number;
  bidSize?: number;
  ask: number;
  askSize?: number;
}

export interface OrderBookLevel {
  price: number;
  size: number;
}

export interface OrderBookEvent extends BaseEvent {
  type: "orderbook";
  /** Snapshot or delta */
  mode: "SNAPSHOT" | "DELTA";
  /** top N bids, sorted desc by price */
  bids: OrderBookLevel[];
  /** top N asks, sorted asc by price */
  asks: OrderBookLevel[];
}

/** Union of all market events we support here */
export type MarketEvent = TickEvent | TradeEvent | QuoteEvent | OrderBookEvent;

/** ---------- Type guards ---------- */
export const isTick = (e: MarketEvent): e is TickEvent => e.type === "tick";
export const isTrade = (e: MarketEvent): e is TradeEvent => e.type === "trade";
export const isQuote = (e: MarketEvent): e is QuoteEvent => e.type === "quote";
export const isOrderBook = (e: MarketEvent): e is OrderBookEvent => e.type === "orderbook";

/** ---------- Small validation helpers (no deps) ---------- */
function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}
function isIsoDate(s: unknown): s is string {
  return typeof s === "string" && !Number.isNaN(Date.parse(s));
}
function req<T>(cond: boolean, msg: string, acc: string[]) {
  if (!cond) acc.push(msg);
}

/** Validate a MarketEvent (returns error list; empty => valid) */
export function validateMarketEvent(e: unknown): string[] {
  const errs: string[] = [];
  const m = e as Partial<MarketEvent>;
  req(!!m, "event is required", errs);
  req(typeof m === "object", "event must be an object", errs);
  if (!m) return ["event is null/undefined"];

  req(typeof m.type === "string", "type must be string", errs);
  req(typeof m.symbol === "string" && m.symbol.length > 0, "symbol required", errs);
  req(isIsoDate(m.ts), "ts must be ISO8601 string", errs);

  switch (m.type) {
    case "tick":
      req(isFiniteNumber((m as TickEvent).price), "tick.price must be number", errs);
      break;
    case "trade":
      req(isFiniteNumber((m as TradeEvent).price), "trade.price must be number", errs);
      req(isFiniteNumber((m as TradeEvent).size), "trade.size must be number", errs);
      break;
    case "quote":
      req(isFiniteNumber((m as QuoteEvent).bid), "quote.bid must be number", errs);
      req(isFiniteNumber((m as QuoteEvent).ask), "quote.ask must be number", errs);
      break;
    case "orderbook":
      const ob = m as OrderBookEvent;
      req(ob.mode === "SNAPSHOT" || ob.mode === "DELTA", "orderbook.mode invalid", errs);
      req(Array.isArray(ob.bids), "orderbook.bids must be array", errs);
      req(Array.isArray(ob.asks), "orderbook.asks must be array", errs);
      break;
    default:
      errs.push("unsupported type");
  }
  return errs;
}

/** ---------- Constructors (with sensible defaults) ---------- */
export function nowIso(): string {
  return new Date().toISOString();
}

export function makeTick(p: {
  symbol: string;
  price: number;
  size?: number;
  bid?: number;
  ask?: number;
  venue?: Venue;
  source?: string;
  ts?: string;
  seq?: number;
}): TickEvent {
  return {
    type: "tick",
    ts: p.ts ?? nowIso(),
    symbol: p.symbol,
    price: p.price,
    size: p.size,
    bid: p.bid,
    ask: p.ask,
    venue: p.venue,
    source: p.source,
    seq: p.seq,
  };
}

export function makeTrade(p: {
  symbol: string;
  price: number;
  size: number;
  side?: "BUY" | "SELL";
  tradeId?: string;
  venue?: Venue;
  source?: string;
  ts?: string;
  seq?: number;
}): TradeEvent {
  return {
    type: "trade",
    ts: p.ts ?? nowIso(),
    symbol: p.symbol,
    price: p.price,
    size: p.size,
    side: p.side,
    tradeId: p.tradeId,
    venue: p.venue,
    source: p.source,
    seq: p.seq,
  };
}

export function makeQuote(p: {
  symbol: string;
  bid: number;
  ask: number;
  bidSize?: number;
  askSize?: number;
  venue?: Venue;
  source?: string;
  ts?: string;
  seq?: number;
}): QuoteEvent {
  return {
    type: "quote",
    ts: p.ts ?? nowIso(),
    symbol: p.symbol,
    bid: p.bid,
    ask: p.ask,
    bidSize: p.bidSize,
    askSize: p.askSize,
    venue: p.venue,
    source: p.source,
    seq: p.seq,
  };
}

export function makeOrderBook(p: {
  symbol: string;
  mode: "SNAPSHOT" | "DELTA";
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  venue?: Venue;
  source?: string;
  ts?: string;
  seq?: number;
}): OrderBookEvent {
  return {
    type: "orderbook",
    ts: p.ts ?? nowIso(),
    symbol: p.symbol,
    mode: p.mode,
    bids: p.bids,
    asks: p.asks,
    venue: p.venue,
    source: p.source,
    seq: p.seq,
  };
}

/** ---------- Serialization helpers ---------- */
export function toJSON(e: MarketEvent): string {
  return JSON.stringify(e);
}
export function fromJSON(json: string): MarketEvent {
  const obj = JSON.parse(json);
  const errs = validateMarketEvent(obj);
  if (errs.length) {
    throw new Error("Invalid MarketEvent: " + errs.join("; "));
  }
  return obj as MarketEvent;
}

/** ---------- Header helper (for bus) ---------- */
export function defaultHeaders() {
  return {
    "content-type": "application/json",
    "x-event-type": "market",
  };
}