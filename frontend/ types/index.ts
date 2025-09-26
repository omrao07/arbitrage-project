// frontend/types/Strategy.ts
// Core domain types for strategies, risk, signals, orders, attribution, and registry rows.

export type ID = string;
export type ISODate = string; // e.g., "2025-09-11"
export type ISODateTime = string; // e.g., "2025-09-11T14:32:00Z"

/** Top-level catalog grouping (used for filters/menus). */
export enum StrategyGenre {
  EquityLS = "equity_ls",
  StatArb = "stat_arb",
  FuturesMacro = "futures_macro",
  OptionsVol = "options_vol",
  CreditCDS = "credit_cds",
  MultiAsset = "multi_asset",
  Other = "other",
}

/** Execution mode & human-control modes (mirrors orchestrator/modes.py). */
export enum RunMode { BACKTEST = "BACKTEST", PAPER = "PAPER", LIVE = "LIVE" }
export enum ControlMode { MANUAL = "MANUAL", SEMI_AUTO = "SEMI_AUTO", AUTO = "AUTO" }

/** Lightweight risk constraints for UI (aligns with risk/risk_limits.py LimitsConfig). */
export interface RiskCaps {
  maxGrossUSD: number;
  maxNetUSD: number;
  maxPerNameUSD: number;
  maxDailyTurnoverUSD: number;
  maxTicketUSD: number;
  minTicketUSD: number;
  allowShort: boolean;
  maxWeightPerName?: number; // 0..1
}

/** Registry row (one per concrete micro strategy). */
export interface StrategyRegistryRow {
  id: ID;                        // e.g., "CIT-0421"
  name: string;                  // "US Stat-Arb: Overnight Reversal"
  family: StrategyGenre | string;
  engine: string;                // import path "pkg.mod:Adapter"
  yaml: string;                  // "CIT/CIT-0421.yaml"
  firm?: "Bridgewater" | "Citadel" | "Point72" | string;
  tags?: string[];               // ["us","equity","meanrev","overnight"]
  status?: "live" | "paused" | "draft";
  createdAt?: ISODate;
  updatedAt?: ISODate;
}

/** Live runtime descriptor (combines registry + runtime state). */
export interface StrategyDescriptor {
  id: ID;
  name: string;
  family: StrategyGenre | string;
  tags: string[];
  runMode: RunMode;
  controlMode: ControlMode;
  risk: RiskCaps;
  navUSD: number;                // current NAV allocated
  pnlDayUSD: number;
  pnlYtdUSD?: number;
  sharpeYtd?: number;
  turnoverDayUSD?: number;
  health: "INIT" | "WARMED" | "RUNNING" | "ERROR" | "STOPPED";
  breaches?: string[];           // from risk engine (e.g., ["MAX_GROSS","VAR"])
  lastTickTs?: number;           // epoch sec
}

/** Signal payload (adapter output). */
export interface Signal {
  asOf: ISODateTime;
  asset: string;                 // e.g., "AAPL", "IG_A_5Y"
  signalName: string;            // "momentum_20_100"
  strength: number;              // -1..+1 normalized
  score?: number;                // raw score
  features?: Record<string, number | string | boolean>;
}

/** Proposed order (pre-risk). Mirrors orchestrator expectations. */
export interface TradeOrder {
  ticker: string;
  side: "BUY" | "SELL" | "BUY_PROTECTION" | "SELL_PROTECTION";
  tradeNotional: number;         // USD
  clientOrderId?: string;
  pxHintBps?: number;
  advUSD?: number;               // for liquidity checks
  meta?: Record<string, unknown>;
}

/** Execution fill (post-routing). */
export interface Fill {
  ticker: string;
  side: TradeOrder["side"];
  filledUSD: number;
  avgPriceBps?: number;
  status: "FILLED" | "PARTIAL" | "REJECTED";
  ts: number; // epoch sec
}

/** Allocation row from selector (asset-strategy weight). */
export interface Allocation {
  asset: string;
  strategyId: ID;
  weight: number;                // 0..1 within the asset budget
  score?: number;
  reason?: string;               // "ensemble" | "bandit" | "rules"
  meta?: Record<string, unknown>;
}

/** Daily PnL attribution components for charts / tables. */
export interface PnLAttribution {
  date: ISODate;
  asset: string;
  strategyId?: ID;
  pricePnLUSD: number;
  fxPnLUSD: number;
  carryUSD: number;
  tradePnLUSD: number;
  slippageUSD: number;
  feesUSD: number;
  totalUSD: number;
}

/** Linked news / macro context shown in StrategyDetails. */
export interface NewsLink {
  asOf: ISODateTime;
  source: string;                // "Bloomberg", "WSJ", "Reuters", ...
  headline: string;
  url?: string;
  sentiment?: number;            // -1..+1
  topics?: string[];             // ["CPI","Energy"]
}

/** UI table row (flattened view for 3,000+ records). */
export interface StrategiesTableRow {
  id: ID;
  name: string;
  family: string;
  tags: string;
  status?: StrategyRegistryRow["status"];
  navUSD: number;
  pnlDayUSD: number;
  sharpeYtd?: number;
  breaches?: string[];
}

/** Response shapes from backend (optional but handy). */
export interface RegistryResponse { rows: StrategyRegistryRow[]; updatedAt: ISODateTime; }
export interface StrategyRuntimeResponse { descriptors: StrategyDescriptor[]; ts: number; }
export interface SignalsResponse { strategyId: ID; signals: Signal[]; asOf: ISODateTime; }
export interface PnLSeriesResponse { byAsset: PnLAttribution[]; byStrategy?: PnLAttribution[]; }

/* ---------------------------------------------------------------------------------- */
/* Utility helpers                                                                    */
/* ---------------------------------------------------------------------------------- */

export const isLive = (s: StrategyDescriptor) => s.runMode === RunMode.LIVE;
export const isPaused = (r: StrategyRegistryRow | StrategyDescriptor) =>
  (r as StrategyRegistryRow).status ? (r as StrategyRegistryRow).status === "paused" : false;

export function toTableRow(desc: StrategyDescriptor): StrategiesTableRow {
  return {
    id: desc.id,
    name: desc.name,
    family: String(desc.family),
    tags: desc.tags?.join(", ") ?? "",
    navUSD: desc.navUSD,
    pnlDayUSD: desc.pnlDayUSD,
    sharpeYtd: desc.sharpeYtd,
    breaches: desc.breaches,
  };
}

/** Safe default caps for UI forms. Keep in sync with backend defaults. */
export const DEFAULT_RISK_CAPS: RiskCaps = {
  maxGrossUSD: 25_000_000,
  maxNetUSD: 10_000_000,
  maxPerNameUSD: 5_000_000,
  maxDailyTurnoverUSD: 10_000_000,
  maxTicketUSD: 5_000_000,
  minTicketUSD: 5_000,
  allowShort: true,
  maxWeightPerName: 0.25,
};

/** Quick factory for a new descriptor from a registry row. */
export function makeDescriptor(r: StrategyRegistryRow): StrategyDescriptor {
  return {
    id: r.id,
    name: r.name,
    family: (r.family as StrategyGenre) || StrategyGenre.Other,
    tags: r.tags ?? [],
    runMode: RunMode.PAPER,
    controlMode: ControlMode.SEMI_AUTO,
    risk: DEFAULT_RISK_CAPS,
    navUSD: 0,
    pnlDayUSD: 0,
    health: "INIT",
  };
}

/** Narrow type guard for arrays of registry rows. */
export function isRegistryRows(x: unknown): x is StrategyRegistryRow[] {
  return Array.isArray(x) && x.every(y => typeof y?.id === "string" && typeof y?.name === "string");
}

/** Currency/number formatting helpers (UI). */
export const fmtUSD = (v: number) =>
  (v ?? 0).toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const pct = (v: number, digits: number = 1) =>
  `${(100 * (v ?? 0)).toFixed(digits)}%`;

/* Barrel exports for convenience */
export default {};