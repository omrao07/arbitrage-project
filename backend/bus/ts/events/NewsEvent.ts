// events/news_event.ts

/** ---------- Shared primitives ---------- */
export type NewsSource = "Reuters" | "Bloomberg" | "DowJones" | "Twitter" | "Custom" | string;

export interface BaseEvent {
  ts: string;             // ISO8601 timestamp
  source?: string;        // producer system/node
  seq?: number;           // monotonic sequence if available
}

/** ---------- Concrete event shapes ---------- */
export interface HeadlineEvent extends BaseEvent {
  type: "headline";
  id: string;             // unique ID for dedup
  tickerSymbols?: string[];
  headline: string;
  sourceName: NewsSource;
  sentiment?: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
}

export interface ArticleEvent extends BaseEvent {
  type: "article";
  id: string;
  tickerSymbols?: string[];
  headline: string;
  body: string;
  sourceName: NewsSource;
  sentiment?: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
  url?: string;
}

export interface SignalEvent extends BaseEvent {
  type: "signal";
  id: string;
  symbol: string;
  score: number;           // numeric signal strength (-1 to +1 or 0–100)
  model?: string;          // NLP model/tagger name
  tags?: string[];         // e.g. ["earnings", "inflation"]
}

/** Union of all news events */
export type NewsEvent = HeadlineEvent | ArticleEvent | SignalEvent;

/** ---------- Type guards ---------- */
export const isHeadline = (e: NewsEvent): e is HeadlineEvent => e.type === "headline";
export const isArticle  = (e: NewsEvent): e is ArticleEvent  => e.type === "article";
export const isSignal   = (e: NewsEvent): e is SignalEvent   => e.type === "signal";

/** ---------- Validation helpers ---------- */
function isIsoDate(s: unknown): s is string {
  return typeof s === "string" && !Number.isNaN(Date.parse(s));
}
function isNonEmptyString(s: unknown): s is string {
  return typeof s === "string" && s.trim().length > 0;
}
function req(cond: boolean, msg: string, acc: string[]) {
  if (!cond) acc.push(msg);
}

export function validateNewsEvent(e: unknown): string[] {
  const errs: string[] = [];
  const n = e as Partial<NewsEvent>;
  req(!!n, "event required", errs);
  if (!n) return ["event null/undefined"];
  req(typeof n.type === "string", "type must be string", errs);
  req(isIsoDate(n.ts), "ts must be ISO8601 string", errs);

  switch (n.type) {
    case "headline":
      req(isNonEmptyString((n as HeadlineEvent).headline), "headline required", errs);
      break;
    case "article":
      req(isNonEmptyString((n as ArticleEvent).body), "article body required", errs);
      break;
    case "signal":
      req(isNonEmptyString((n as SignalEvent).symbol), "signal.symbol required", errs);
      break;
    default:
      errs.push("unsupported type");
  }
  return errs;
}

/** ---------- Constructors ---------- */
export function nowIso(): string {
  return new Date().toISOString();
}

export function makeHeadline(p: {
  id: string;
  headline: string;
  tickerSymbols?: string[];
  sourceName: NewsSource;
  sentiment?: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
  ts?: string;
  seq?: number;
}): HeadlineEvent {
  return {
    type: "headline",
    ts: p.ts ?? nowIso(),
    id: p.id,
    headline: p.headline,
    tickerSymbols: p.tickerSymbols,
    sourceName: p.sourceName,
    sentiment: p.sentiment,
    seq: p.seq,
  };
}

export function makeArticle(p: {
  id: string;
  headline: string;
  body: string;
  tickerSymbols?: string[];
  sourceName: NewsSource;
  sentiment?: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
  url?: string;
  ts?: string;
  seq?: number;
}): ArticleEvent {
  return {
    type: "article",
    ts: p.ts ?? nowIso(),
    id: p.id,
    headline: p.headline,
    body: p.body,
    tickerSymbols: p.tickerSymbols,
    sourceName: p.sourceName,
    sentiment: p.sentiment,
    url: p.url,
    seq: p.seq,
  };
}

export function makeSignal(p: {
  id: string;
  symbol: string;
  score: number;
  model?: string;
  tags?: string[];
  ts?: string;
  seq?: number;
}): SignalEvent {
  return {
    type: "signal",
    ts: p.ts ?? nowIso(),
    id: p.id,
    symbol: p.symbol,
    score: p.score,
    model: p.model,
    tags: p.tags,
    seq: p.seq,
  };
}

/** ---------- Serialization ---------- */
export function toJSON(e: NewsEvent): string {
  return JSON.stringify(e);
}
export function fromJSON(json: string): NewsEvent {
  const obj = JSON.parse(json);
  const errs = validateNewsEvent(obj);
  if (errs.length) throw new Error("Invalid NewsEvent: " + errs.join("; "));
  return obj as NewsEvent;
}

/** ---------- Headers ---------- */
export function defaultHeaders() {
  return {
    "content-type": "application/json",
    "x-event-type": "news",
  };
}