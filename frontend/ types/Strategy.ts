// frontend/types/strategy.ts
// Minimal, dependency-free alias with safe type guards.
// It re-exports everything from the canonical `Strategy.ts` so both imports work:
//
//   import { StrategyDescriptor } from "@/types/strategy";
//   // or
//   import { StrategyDescriptor } from "@/types/Strategy";
//
// No runtime libraries, no side effects.

import { RiskCaps, StrategyDescriptor, StrategyRegistryRow } from ".";

export * from "./Strategy";

/* ------------------------------------------------------------------------------------
 * Narrow runtime guards (no external libs)
 * ---------------------------------------------------------------------------------- */



/** Quick primitive checks */
const isObj = (x: unknown): x is Record<string, unknown> =>
  typeof x === "object" && x !== null;

const isStr = (v: unknown): v is string => typeof v === "string";
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isBool = (v: unknown): v is boolean => typeof v === "boolean";

/** Safe check for arrays of strings */
const isStrArray = (v: unknown): v is string[] =>
  Array.isArray(v) && v.every(isStr);

/** Partial check for RiskCaps to avoid false positives */
export function isRiskCaps(x: unknown): x is RiskCaps {
  if (!isObj(x)) return false;
  return (
    isNum(x["maxGrossUSD"]) &&
    isNum(x["maxNetUSD"]) &&
    isNum(x["maxPerNameUSD"]) &&
    isNum(x["maxDailyTurnoverUSD"]) &&
    isNum(x["maxTicketUSD"]) &&
    isNum(x["minTicketUSD"]) &&
    isBool(x["allowShort"])
  );
}

/** Guard for StrategyRegistryRow */
export function isStrategyRegistryRow(x: unknown): x is StrategyRegistryRow {
  if (!isObj(x)) return false;
  return (
    isStr(x["id"]) &&
    isStr(x["name"]) &&
    (isStr(x["family"]) || isStr(x["family"])) && // enum or string
    isStr(x["engine"]) &&
    isStr(x["yaml"]) &&
    (x["tags"] === undefined || isStrArray(x["tags"])) &&
    (x["status"] === undefined ||
      x["status"] === "live" ||
      x["status"] === "paused" ||
      x["status"] === "draft")
  );
}

/** Guard for StrategyDescriptor */
export function isStrategyDescriptor(x: unknown): x is StrategyDescriptor {
  if (!isObj(x)) return false;
  // required core fields
  if (!isStr(x["id"]) || !isStr(x["name"]) || !isStr(x["runMode"]) || !isStr(x["controlMode"])) {
    return false;
  }
  // optional fields sanity
  if (x["tags"] !== undefined && !isStrArray(x["tags"])) return false;
  if (x["risk"] !== undefined && !isRiskCaps(x["risk"])) return false;
  if (x["navUSD"] !== undefined && !isNum(x["navUSD"])) return false;
  if (x["pnlDayUSD"] !== undefined && !isNum(x["pnlDayUSD"])) return false;
  if (x["breaches"] !== undefined && !Array.isArray(x["breaches"])) return false;
  return true;
}

/* ------------------------------------------------------------------------------------
 * Lightweight coercers (best-effort, won’t throw)
 * ---------------------------------------------------------------------------------- */

/** Coerce plain JSON into a StrategyRegistryRow (best-effort). */
export function coerceRegistryRow(x: unknown): StrategyRegistryRow {
  const o = (isObj(x) ? x : {}) as Record<string, unknown>;
  return {
    id: String(o.id ?? ""),
    name: String(o.name ?? ""),
    family: String(o.family ?? "other"),
    engine: String(o.engine ?? ""),
    yaml: String(o.yaml ?? ""),
    firm: o.firm !== undefined ? String(o.firm) : undefined,
    tags: Array.isArray(o.tags) ? (o.tags as unknown[]).filter(isStr) as string[] : undefined,
    status: ((): "live" | "paused" | "draft" | undefined => {
      const s = o.status;
      return s === "live" || s === "paused" || s === "draft" ? s : undefined;
    })(),
    createdAt: o.createdAt ? String(o.createdAt) : undefined,
    updatedAt: o.updatedAt ? String(o.updatedAt) : undefined,
  };
}

/** Coerce plain JSON into a StrategyDescriptor (best-effort). */
export function coerceDescriptor(x: unknown): StrategyDescriptor {
  const o = (isObj(x) ? x : {}) as Record<string, unknown>;
  const tags = Array.isArray(o.tags) ? (o.tags as unknown[]).filter(isStr) as string[] : [];
  const risk = isRiskCaps(o.risk)
    ? (o.risk as RiskCaps)
    : {
        maxGrossUSD: 25_000_000,
        maxNetUSD: 10_000_000,
        maxPerNameUSD: 5_000_000,
        maxDailyTurnoverUSD: 10_000_000,
        maxTicketUSD: 5_000_000,
        minTicketUSD: 5_000,
        allowShort: true,
        maxWeightPerName: 0.25,
      } as RiskCaps;

  return {
    id: String(o.id ?? ""),
    name: String(o.name ?? ""),
    family: String(o.family ?? "other"),
    tags,
    runMode: String(o.runMode ?? "PAPER") as any,
    controlMode: String(o.controlMode ?? "SEMI_AUTO") as any,
    risk,
    navUSD: isNum(o.navUSD) ? Number(o.navUSD) : 0,
    pnlDayUSD: isNum(o.pnlDayUSD) ? Number(o.pnlDayUSD) : 0,
    pnlYtdUSD: isNum(o.pnlYtdUSD) ? Number(o.pnlYtdUSD) : undefined,
    sharpeYtd: isNum(o.sharpeYtd) ? Number(o.sharpeYtd) : undefined,
    turnoverDayUSD: isNum(o.turnoverDayUSD) ? Number(o.turnoverDayUSD) : undefined,
    health: String(o.health ?? "INIT") as any,
    breaches: Array.isArray(o.breaches) ? (o.breaches as unknown[]).map(String) : undefined,
    lastTickTs: isNum(o.lastTickTs) ? Number(o.lastTickTs) : undefined,
  };
}