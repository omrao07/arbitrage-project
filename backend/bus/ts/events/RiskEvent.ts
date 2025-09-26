// events/riskevent.ts

/** ========= Shared primitives ========= */
export type Currency = "USD" | "EUR" | "GBP" | "JPY" | string;

export interface BaseEvent {
  ts: string;          // ISO8601 timestamp
  account?: string;    // portfolio / book / account id (optional for some)
  source?: string;     // producer system/node
  seq?: number;        // monotonic sequence if available
}

/** ========= VaR ========= */
export interface VaREvent extends BaseEvent {
  type: "var";
  horizonDays: number;                 // e.g., 1, 10
  confidence: number;                  // e.g., 0.95, 0.99
  currency?: Currency;
  value: number;                       // VaR amount in currency (positive number represents loss magnitude)
  method?: "HIST" | "MC" | "DELTA_GAMMA";
  windowDays?: number;                 // lookback window for HIST
  details?: {
    perSymbol?: Record<string, number>;
    perBucket?: Record<string, number>; // e.g., sector/strategy
  };
}

/** ========= Stress (predefined shocks) ========= */
export interface StressEvent extends BaseEvent {
  type: "stress";
  name: string;                        // e.g., "GFC-2008", "Rates+200bp", "Oil+20%"
  currency?: Currency;
  pnl: number;                         // portfolio P&L under stress (negative=loss)
  assumptions?: string[];              // human-readable shock notes
  bySymbol?: Record<string, number>;
  byBucket?: Record<string, number>;
}

/** ========= Scenario (custom path/point shocks) ========= */
export interface ScenarioEvent extends BaseEvent {
  type: "scenario";
  scenarioId: string;                  // unique id
  label?: string;                      // display name
  currency?: Currency;
  pnl: number;                         // portfolio P&L under scenario
  drivers?: Record<string, number>;    // factor shocks (e.g., "UST10Y:+0.50", "SPX:-5%")
  notes?: string;
}

/** ========= Risk limit events ========= */
export interface LimitEvent extends BaseEvent {
  type: "limit";
  limitId: string;                     // ID of the limit policy (maps to policy store)
  name: string;                        // e.g., "GrossExposure", "SingleNameWeight", "VaRCap"
  scope: "ACCOUNT" | "BOOK" | "STRATEGY" | "GLOBAL";
  metric: string;                      // e.g., "gross_exposure_usd", "var_99_1d"
  threshold: number;                   // numeric threshold (same units as value)
  comparator: "<" | "<=" | "==" | ">=" | ">";
  currency?: Currency;                 // if metric is monetary
}

export interface BreachEvent extends BaseEvent {
  type: "breach";
  limitId: string;
  name: string;
  scope: "ACCOUNT" | "BOOK" | "STRATEGY" | "GLOBAL";
  metric: string;
  value: number;                       // observed value that caused breach
  threshold: number;
  comparator: "<" | "<=" | "==" | ">=" | ">";
  severity: "INFO" | "WARN" | "CRITICAL";
  currency?: Currency;
  acknowledgedBy?: string;             // user id who acked
  acknowledgedAt?: string;             // ISO8601
  notes?: string;
}

/** ========= Union ========= */
export type RiskEvent = VaREvent | StressEvent | ScenarioEvent | LimitEvent | BreachEvent;

/** ========= Type guards ========= */
export const isVaR      = (e: RiskEvent): e is VaREvent      => e.type === "var";
export const isStress   = (e: RiskEvent): e is StressEvent   => e.type === "stress";
export const isScenario = (e: RiskEvent): e is ScenarioEvent => e.type === "scenario";
export const isLimit    = (e: RiskEvent): e is LimitEvent    => e.type === "limit";
export const isBreach   = (e: RiskEvent): e is BreachEvent   => e.type === "breach";

/** ========= Validation (minimal; no deps) ========= */
function isIso(s: unknown): s is string {
  return typeof s === "string" && !Number.isNaN(Date.parse(s));
}
function isFiniteNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}
function req(cond: boolean, msg: string, acc: string[]) { if (!cond) acc.push(msg); }

export function validateRiskEvent(e: unknown): string[] {
  const errs: string[] = [];
  const x = e as Partial<RiskEvent>;
  req(!!x, "event required", errs);
  if (!x) return ["event null/undefined"];
  req(typeof x.type === "string", "type must be string", errs);
  req(isIso(x.ts), "ts must be ISO8601", errs);

  switch (x.type) {
    case "var": {
      const v = x as VaREvent;
      req(isFiniteNum(v.horizonDays) && v.horizonDays > 0, "var.horizonDays > 0", errs);
      req(isFiniteNum(v.confidence) && v.confidence > 0 && v.confidence < 1, "var.confidence (0,1)", errs);
      req(isFiniteNum(v.value), "var.value must be number", errs);
      break;
    }
    case "stress": {
      const s = x as StressEvent;
      req(typeof s.name === "string" && s.name.length > 0, "stress.name required", errs);
      req(isFiniteNum(s.pnl), "stress.pnl must be number", errs);
      break;
    }
    case "scenario": {
      const s = x as ScenarioEvent;
      req(typeof s.scenarioId === "string" && s.scenarioId.length > 0, "scenario.scenarioId required", errs);
      req(isFiniteNum(s.pnl), "scenario.pnl must be number", errs);
      break;
    }
    case "limit": {
      const l = x as LimitEvent;
      req(typeof l.limitId === "string" && l.limitId.length > 0, "limit.limitId required", errs);
      req(typeof l.name === "string" && l.name.length > 0, "limit.name required", errs);
      req(typeof l.metric === "string" && l.metric.length > 0, "limit.metric required", errs);
      req(isFiniteNum(l.threshold), "limit.threshold must be number", errs);
      req(["<","<=","==",">=",">"].includes(l.comparator), "limit.comparator invalid", errs);
      break;
    }
    case "breach": {
      const b = x as BreachEvent;
      req(typeof b.limitId === "string" && b.limitId.length > 0, "breach.limitId required", errs);
      req(typeof b.metric === "string" && b.metric.length > 0, "breach.metric required", errs);
      req(isFiniteNum(b.value), "breach.value must be number", errs);
      req(isFiniteNum(b.threshold), "breach.threshold must be number", errs);
      req(["<","<=","==",">=",">"].includes(b.comparator), "breach.comparator invalid", errs);
      req(["INFO","WARN","CRITICAL"].includes(b.severity), "breach.severity invalid", errs);
      if (b.acknowledgedAt) req(isIso(b.acknowledgedAt), "breach.acknowledgedAt must be ISO8601", errs);
      break;
    }
    default:
      errs.push("unsupported type");
  }
  return errs;
}

/** ========= Constructors ========= */
export const nowIso = () => new Date().toISOString();

export function makeVaR(p: Omit<VaREvent, "type" | "ts"> & { ts?: string }): VaREvent {
  return { type: "var", ts: p.ts ?? nowIso(), ...p };
}
export function makeStress(p: Omit<StressEvent, "type" | "ts"> & { ts?: string }): StressEvent {
  return { type: "stress", ts: p.ts ?? nowIso(), ...p };
}
export function makeScenario(p: Omit<ScenarioEvent, "type" | "ts"> & { ts?: string }): ScenarioEvent {
  return { type: "scenario", ts: p.ts ?? nowIso(), ...p };
}
export function makeLimit(p: Omit<LimitEvent, "type" | "ts"> & { ts?: string }): LimitEvent {
  return { type: "limit", ts: p.ts ?? nowIso(), ...p };
}
export function makeBreach(p: Omit<BreachEvent, "type" | "ts"> & { ts?: string }): BreachEvent {
  return { type: "breach", ts: p.ts ?? nowIso(), ...p };
}

/** ========= Serialization ========= */
export function toJSON(e: RiskEvent): string {
  return JSON.stringify(e);
}
export function fromJSON(json: string): RiskEvent {
  const obj = JSON.parse(json);
  const errs = validateRiskEvent(obj);
  if (errs.length) throw new Error("Invalid RiskEvent: " + errs.join("; "));
  return obj as RiskEvent;
}

/** ========= Default headers for bus ========= */
export function defaultHeaders() {
  return {
    "content-type": "application/json",
    "x-event-type": "risk",
  };
}