// lib/updatelimits.action.ts
// No imports. Server Action to normalize & validate portfolio/risk limits.
// - Works with FormData (for Next.js Server Actions) or a typed object
// - Percent fields use PERCENT UNITS (e.g., 5 == 5%, 0.25 == 0.25%)
// - Returns a normalized payload plus validation errors/warnings and an optional diff
//
// FormData keys (all optional unless noted):
//   scope                : "account" | "household" | "strategy" | "sleeve" | "symbol"
//   scopeId              : string
//   effectiveAt          : ISO date string
//   notes                : string
//   limits               : JSON blob for the whole limits object (see NormalizedLimits)
//   prev                 : JSON blob of previous saved limits (to compute diff)
//   -- Or provide flattened fields instead of `limits` JSON --
//   maxPositionWeightPct : number (percent)
//   maxSectorWeightPct   : number (percent)
//   grossExposurePct     : number (percent)
//   netExposurePct       : number (percent)
//   leverageMax          : number (>=0)
//   dailyLossLimitPct    : number (percent)
//   trailingDrawdownPct  : number (percent)
//   varPct               : number (percent)
//   turnoverCapPct       : number (percent)
//   minTradeValue        : number
//   minCashPct           : number (percent)
//   maxCashPct           : number (percent)
//   perSymbolCaps        : JSON Record<string, { maxQty?: number; maxNotional?: number; maxWeightPct?: number }>
//
// Example (typed):
//   const res = await updateLimits({
//     scope: "account",
//     scopeId: "ACC-123",
//     limits: { weights: { maxPositionWeightPct: 5 }, risk: { dailyLossLimitPct: 2 } },
//     prev: { ...previousLimits },
//     notes: "Tighten single-name cap",
//   });

"use server";

/* ============================== Types ============================== */

export type Scope = "account" | "household" | "strategy" | "sleeve" | "symbol";

export type NormalizedLimits = {
  weights?: {
    maxPositionWeightPct?: number; // %
    maxSectorWeightPct?: number;   // %
  };
  exposures?: {
    grossExposurePct?: number;     // %
    netExposurePct?: number;       // %
    leverageMax?: number;          // x
  };
  risk?: {
    dailyLossLimitPct?: number;    // %
    trailingDrawdownPct?: number;  // %
    varPct?: number;               // %
  };
  trading?: {
    turnoverCapPct?: number;       // %
    minTradeValue?: number;        // absolute
  };
  cash?: {
    minCashPct?: number;           // %
    maxCashPct?: number;           // %
  };
  perSymbolCaps?: Record<string, {
    maxQty?: number;
    maxNotional?: number;
    maxWeightPct?: number;         // %
  }>;
};

export type UpdateLimitsInput = {
  scope?: Scope;
  scopeId?: string;
  effectiveAt?: string;
  notes?: string;
  limits?: NormalizedLimits;
  prev?: NormalizedLimits; // optional, to compute diff
};

export type UpdateLimitsResult = {
  ok: boolean;
  updatedAt: string; // ISO
  scope?: Scope;
  scopeId?: string;
  effectiveAt?: string;
  notes?: string;
  saved: NormalizedLimits; // normalized + validated
  errors: string[];
  warnings: string[];
  diff?: Array<{ path: string; from: any; to: any }>;
};

/* ============================== Action ============================== */

export default async function updateLimits(
  input: UpdateLimitsInput | FormData,
): Promise<UpdateLimitsResult> {
  const now = new Date().toISOString();

  const parsed = isFormData(input) ? parseForm(input) : sanitizeObjectInput(input);
  const { scope, scopeId, effectiveAt, notes, limits, prev, errors: parseErrors } = parsed;

  // Validate + normalize constraints
  const { normalized, errors: valErrors, warnings } = validateLimits(limits || {});
  const errors = [...parseErrors, ...valErrors];

  // Optional diff against previous
  const diff = prev ? computeDiff(prev, normalized) : undefined;

  return {
    ok: errors.length === 0,
    updatedAt: now,
    scope,
    scopeId,
    effectiveAt,
    notes,
    saved: normalized,
    errors,
    warnings,
    diff,
  };
}

/* ============================== Parsing ============================== */

function isFormData(x: any): x is FormData {
  return typeof x === "object" && x?.constructor?.name === "FormData";
}

function parseForm(fd: FormData): {
  scope?: Scope;
  scopeId?: string;
  effectiveAt?: string;
  notes?: string;
  limits?: NormalizedLimits;
  prev?: NormalizedLimits;
  errors: string[];
} {
  const errors: string[] = [];
  const scope = str(fd.get("scope")) as Scope | undefined;
  const scopeId = str(fd.get("scopeId"));
  const effectiveAt = isoOrUndefined(str(fd.get("effectiveAt")));
  const notes = str(fd.get("notes"));

  // Prefer `limits` JSON if present; otherwise assemble from flat fields
  let limits = readJson<NormalizedLimits>(fd.get("limits")) || undefined;
  if (!limits) {
    limits = {};
    // weights
    const maxPositionWeightPct = num(fd.get("maxPositionWeightPct"));
    const maxSectorWeightPct = num(fd.get("maxSectorWeightPct"));
    if (isNum(maxPositionWeightPct) || isNum(maxSectorWeightPct)) {
      limits.weights = {};
      if (isNum(maxPositionWeightPct)) limits.weights.maxPositionWeightPct = maxPositionWeightPct!;
      if (isNum(maxSectorWeightPct)) limits.weights.maxSectorWeightPct = maxSectorWeightPct!;
    }
    // exposures
    const grossExposurePct = num(fd.get("grossExposurePct"));
    const netExposurePct = num(fd.get("netExposurePct"));
    const leverageMax = num(fd.get("leverageMax"));
    if (isNum(grossExposurePct) || isNum(netExposurePct) || isNum(leverageMax)) {
      limits.exposures = {};
      if (isNum(grossExposurePct)) limits.exposures.grossExposurePct = grossExposurePct!;
      if (isNum(netExposurePct)) limits.exposures.netExposurePct = netExposurePct!;
      if (isNum(leverageMax)) limits.exposures.leverageMax = leverageMax!;
    }
    // risk
    const dailyLossLimitPct = num(fd.get("dailyLossLimitPct"));
    const trailingDrawdownPct = num(fd.get("trailingDrawdownPct"));
    const varPct = num(fd.get("varPct"));
    if (isNum(dailyLossLimitPct) || isNum(trailingDrawdownPct) || isNum(varPct)) {
      limits.risk = {};
      if (isNum(dailyLossLimitPct)) limits.risk.dailyLossLimitPct = dailyLossLimitPct!;
      if (isNum(trailingDrawdownPct)) limits.risk.trailingDrawdownPct = trailingDrawdownPct!;
      if (isNum(varPct)) limits.risk.varPct = varPct!;
    }
    // trading
    const turnoverCapPct = num(fd.get("turnoverCapPct"));
    const minTradeValue = num(fd.get("minTradeValue"));
    if (isNum(turnoverCapPct) || isNum(minTradeValue)) {
      limits.trading = {};
      if (isNum(turnoverCapPct)) limits.trading.turnoverCapPct = turnoverCapPct!;
      if (isNum(minTradeValue)) limits.trading.minTradeValue = minTradeValue!;
    }
    // cash
    const minCashPct = num(fd.get("minCashPct"));
    const maxCashPct = num(fd.get("maxCashPct"));
    if (isNum(minCashPct) || isNum(maxCashPct)) {
      limits.cash = {};
      if (isNum(minCashPct)) limits.cash.minCashPct = minCashPct!;
      if (isNum(maxCashPct)) limits.cash.maxCashPct = maxCashPct!;
    }
    // per symbol
    const psc = readJson<Record<string, any>>(fd.get("perSymbolCaps"));
    if (psc && typeof psc === "object") {
      limits.perSymbolCaps = psc;
    }
  }

  const prev = readJson<NormalizedLimits>(fd.get("prev")) || undefined;

  return { scope, scopeId, effectiveAt, notes, limits, prev, errors };
}

function sanitizeObjectInput(x: UpdateLimitsInput | undefined): {
  scope?: Scope;
  scopeId?: string;
  effectiveAt?: string;
  notes?: string;
  limits?: NormalizedLimits;
  prev?: NormalizedLimits;
  errors: string[];
} {
  const errors: string[] = [];
  if (!x || typeof x !== "object") return { errors: ["no payload"] };

  return {
    scope: x.scope,
    scopeId: str(x.scopeId),
    effectiveAt: isoOrUndefined(str(x.effectiveAt)),
    notes: str(x.notes),
    limits: coerceLimits(x.limits || {}),
    prev: x.prev ? coerceLimits(x.prev) : undefined,
    errors,
  };
}

/* ============================== Validation ============================== */

function validateLimits(lim: NormalizedLimits) {
  const errors: string[] = [];
  const warnings: string[] = [];
  const L = cloneLimits(lim);

  // Clamp/round percents to sensible ranges
  function pctClamp(path: string, v?: number, lo = 0, hi = 1000) {
    if (!isNum(v)) return undefined;
    const n = round(clamp(v!, lo, hi), 6);
    if (n !== v) warnings.push(`${path} adjusted to ${n}`);
    return n;
  }
  function nonNeg(path: string, v?: number) {
    if (!isNum(v)) return undefined;
    const n = round(Math.max(0, v!), 6);
    if (n !== v) warnings.push(`${path} adjusted to ${n}`);
    return n;
  }

  // weights
  if (L.weights) {
    L.weights.maxPositionWeightPct = pctClamp("weights.maxPositionWeightPct", L.weights.maxPositionWeightPct, 0, 100);
    L.weights.maxSectorWeightPct = pctClamp("weights.maxSectorWeightPct", L.weights.maxSectorWeightPct, 0, 100);
  }

  // exposures
  if (L.exposures) {
    L.exposures.grossExposurePct = pctClamp("exposures.grossExposurePct", L.exposures.grossExposurePct, 0, 500);
    L.exposures.netExposurePct = pctClamp("exposures.netExposurePct", L.exposures.netExposurePct, 0, 500);
    L.exposures.leverageMax = nonNeg("exposures.leverageMax", L.exposures.leverageMax);
  }

  // risk
  if (L.risk) {
    L.risk.dailyLossLimitPct = pctClamp("risk.dailyLossLimitPct", L.risk.dailyLossLimitPct, 0, 100);
    L.risk.trailingDrawdownPct = pctClamp("risk.trailingDrawdownPct", L.risk.trailingDrawdownPct, 0, 100);
    L.risk.varPct = pctClamp("risk.varPct", L.risk.varPct, 0, 100);
  }

  // trading
  if (L.trading) {
    L.trading.turnoverCapPct = pctClamp("trading.turnoverCapPct", L.trading.turnoverCapPct, 0, 1000);
    L.trading.minTradeValue = nonNeg("trading.minTradeValue", L.trading.minTradeValue);
  }

  // cash
  if (L.cash) {
    L.cash.minCashPct = pctClamp("cash.minCashPct", L.cash.minCashPct, 0, 100);
    L.cash.maxCashPct = pctClamp("cash.maxCashPct", L.cash.maxCashPct, 0, 100);
    if (isNum(L.cash.minCashPct) && isNum(L.cash.maxCashPct) && (L.cash.minCashPct! > L.cash.maxCashPct!)) {
      errors.push("cash.minCashPct cannot be greater than cash.maxCashPct");
    }
  }

  // perSymbolCaps
  if (L.perSymbolCaps && typeof L.perSymbolCaps === "object") {
    const out: Record<string, any> = {};
    for (const raw of Object.keys(L.perSymbolCaps)) {
      const sym = String(raw || "").trim().toUpperCase();
      if (!sym) continue;
      const row = L.perSymbolCaps[raw] || {};
      const maxQty = isNum(row.maxQty) ? nonNeg(`perSymbolCaps.${sym}.maxQty`, Number(row.maxQty)) : undefined;
      const maxNotional = isNum(row.maxNotional) ? nonNeg(`perSymbolCaps.${sym}.maxNotional`, Number(row.maxNotional)) : undefined;
      const maxWeightPct = isNum(row.maxWeightPct) ? pctClamp(`perSymbolCaps.${sym}.maxWeightPct`, Number(row.maxWeightPct), 0, 100) : undefined;
      if (!isNum(maxQty) && !isNum(maxNotional) && !isNum(maxWeightPct)) continue;
      out[sym] = {};
      if (isNum(maxQty)) out[sym].maxQty = maxQty!;
      if (isNum(maxNotional)) out[sym].maxNotional = maxNotional!;
      if (isNum(maxWeightPct)) out[sym].maxWeightPct = maxWeightPct!;
    }
    L.perSymbolCaps = Object.keys(out).length ? out : undefined;
  }

  return { normalized: L, errors, warnings };
}

/* ============================== Utils ============================== */

function coerceLimits(x: NormalizedLimits): NormalizedLimits {
  const L: NormalizedLimits = {};
  if (x.weights) {
    L.weights = {};
    if (isNum(x.weights.maxPositionWeightPct)) L.weights.maxPositionWeightPct = Number(x.weights.maxPositionWeightPct);
    if (isNum(x.weights.maxSectorWeightPct)) L.weights.maxSectorWeightPct = Number(x.weights.maxSectorWeightPct);
  }
  if (x.exposures) {
    L.exposures = {};
    if (isNum(x.exposures.grossExposurePct)) L.exposures.grossExposurePct = Number(x.exposures.grossExposurePct);
    if (isNum(x.exposures.netExposurePct)) L.exposures.netExposurePct = Number(x.exposures.netExposurePct);
    if (isNum(x.exposures.leverageMax)) L.exposures.leverageMax = Number(x.exposures.leverageMax);
  }
  if (x.risk) {
    L.risk = {};
    if (isNum(x.risk.dailyLossLimitPct)) L.risk.dailyLossLimitPct = Number(x.risk.dailyLossLimitPct);
    if (isNum(x.risk.trailingDrawdownPct)) L.risk.trailingDrawdownPct = Number(x.risk.trailingDrawdownPct);
    if (isNum(x.risk.varPct)) L.risk.varPct = Number(x.risk.varPct);
  }
  if (x.trading) {
    L.trading = {};
    if (isNum(x.trading.turnoverCapPct)) L.trading.turnoverCapPct = Number(x.trading.turnoverCapPct);
    if (isNum(x.trading.minTradeValue)) L.trading.minTradeValue = Number(x.trading.minTradeValue);
  }
  if (x.cash) {
    L.cash = {};
    if (isNum(x.cash.minCashPct)) L.cash.minCashPct = Number(x.cash.minCashPct);
    if (isNum(x.cash.maxCashPct)) L.cash.maxCashPct = Number(x.cash.maxCashPct);
  }
  if (x.perSymbolCaps && typeof x.perSymbolCaps === "object") {
    const out: Record<string, any> = {};
    for (const k of Object.keys(x.perSymbolCaps)) {
      const sym = String(k || "").trim().toUpperCase();
      const row = (x.perSymbolCaps as any)[k] || {};
      const ent: any = {};
      if (isNum(row.maxQty)) ent.maxQty = Number(row.maxQty);
      if (isNum(row.maxNotional)) ent.maxNotional = Number(row.maxNotional);
      if (isNum(row.maxWeightPct)) ent.maxWeightPct = Number(row.maxWeightPct);
      if (Object.keys(ent).length) out[sym] = ent;
    }
    if (Object.keys(out).length) L.perSymbolCaps = out;
  }
  return L;
}

function computeDiff(prev: NormalizedLimits, next: NormalizedLimits) {
  const rows: Array<{ path: string; from: any; to: any }> = [];
  const keys = new Set<string>();
  const walker = (obj: any, base: string) => {
    for (const k of Object.keys(obj || {})) {
      keys.add(base ? `${base}.${k}` : k);
      if (obj[k] && typeof obj[k] === "object" && !Array.isArray(obj[k])) {
        walker(obj[k], base ? `${base}.${k}` : k);
      }
    }
  };
  walker(prev || {}, "");
  walker(next || {}, "");
  for (const path of keys) {
    const a = getPath(prev, path);
    const b = getPath(next, path);
    if (!deepEq(a, b)) rows.push({ path, from: a, to: b });
  }
  return rows.sort((x, y) => x.path.localeCompare(y.path));
}

function getPath(obj: any, path: string): any {
  const parts = path.split(".");
  let cur = obj;
  for (const p of parts) {
    if (!cur) return undefined;
    cur = cur[p];
  }
  return cur;
}

function deepEq(a: any, b: any): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a && b && typeof a === "object") {
    const ak = Object.keys(a);
    const bk = Object.keys(b);
    if (ak.length !== bk.length) return false;
    for (const k of ak) if (!deepEq(a[k], b[k])) return false;
    return true;
  }
  return false;
}

function cloneLimits(x: NormalizedLimits): NormalizedLimits {
  return JSON.parse(JSON.stringify(x || {}));
}

function readJson<T>(v: any): T | undefined {
  const s = str(v);
  if (!s) return undefined;
  try { return JSON.parse(s) as T; } catch { return undefined; }
}

function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}

function num(v: any): number | undefined {
  if (v == null || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function isNum(n: any): n is number {
  return typeof n === "number" && Number.isFinite(n);
}

function isoOrUndefined(s?: string) {
  if (!s) return undefined;
  const t = Date.parse(s);
  return Number.isFinite(t) ? new Date(t).toISOString() : undefined;
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function round(n: number, dp = 6) {
  const m = Math.pow(10, dp);
  return Math.round(n * m) / m;
}
