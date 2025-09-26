// core/fields.ts
// Catalog field specs, provider-to-catalog mapping, and value normalization.
// Pure TypeScript, zero imports. Safe to drop anywhere in the repo.

/* ───────────────────────── Types ───────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type ValueType = "number" | "string" | "date" | "boolean";

export type Unit =
  | "number"   // unitless
  | "shares"
  | "pct"      // fraction (0.0123 = 1.23%)
  | "bps"      // basis points (100 bps = 1% -> 0.01)
  | "USD" | "EUR" | "JPY" | string; // currency 3-letter (normalize via FX)

export type FieldSpec = {
  /** Canonical catalog field name (unique key) */
  name: string;
  /** Expected primitive type (default inferred) */
  type?: ValueType;
  /** Unit semantics (used for normalization) */
  unit?: Unit;
  /** Required in normalized output */
  required?: boolean;
  /** Default value when missing/null (applied post-normalization) */
  default?: any;
  /** Optional description */
  description?: string;
  /** Aliases this field may appear as in provider payloads (fallback order) */
  aliases?: string[];
};

export type NormalizeCtx = {
  /** Normalize monetary values to this currency; default "USD" */
  baseCcy?: string;
  /** FX map: e.g., { "EURUSD": 1.075, "USDJPY": 155.2 } */
  fx?: Record<string, number>;
  /** ISO timestamp to stamp as reference time if missing */
  asOf?: string;
  /** Desired timezone label for date outputs (we output ISO UTC strings) */
  tz?: string;
  /** If true, treat percentage-like strings like "12.3%" as 0.123 */
  acceptPercentStrings?: boolean;
};

export type Mapping = Record<string, string>; // catalogName -> providerKey

/* ───────────────────────── Defaults ───────────────────────── */

export const DefaultSpecs: FieldSpec[] = [
  { name: "asOf",     type: "date",   description: "Record timestamp (ISO)" },
  { name: "ticker",   type: "string", description: "Symbol / TICKER" },
  { name: "name",     type: "string" },
  { name: "currency", type: "string" },
  { name: "mic",      type: "string" },
  { name: "figi",     type: "string" },
  { name: "isin",     type: "string" },
  { name: "bbgid",    type: "string" },
  { name: "open",     type: "number", unit: "USD" },
  { name: "high",     type: "number", unit: "USD" },
  { name: "low",      type: "number", unit: "USD" },
  { name: "close",    type: "number", unit: "USD" },
  { name: "volume",   type: "number", unit: "shares" },
];

/* ───────────────────────── API ───────────────────────── */

/** Build a name→spec index; validates uniqueness. */
export function indexSpecs(specs: FieldSpec[]): Record<string, FieldSpec> {
  const ix: Record<string, FieldSpec> = Object.create(null);
  for (const s of specs) {
    if (!s?.name) continue;
    if (ix[s.name]) throw new Error(`duplicate_field_spec:${s.name}`);
    ix[s.name] = s;
  }
  return ix;
}

/**
 * Compile a mapping from catalog to provider keys.
 * Order of resolution: explicitMapping > spec.aliases > identity (same name).
 */
export function compileMapping(
  specs: FieldSpec[] = DefaultSpecs,
  explicitMapping: Mapping = {}
): Mapping {
  const ix = indexSpecs(specs);
  const out: Mapping = {};
  for (const k in ix) {
    if (explicitMapping[k]) { out[k] = explicitMapping[k]; continue; }
    const s = ix[k];
    const alias = firstExisting(s.aliases || [], []);
    out[k] = alias || k; // identity fallback
  }
  // include any explicit keys that point to ad-hoc fields not in specs
  for (const k in explicitMapping) if (!out[k]) out[k] = explicitMapping[k];
  return out;

  function firstExisting(names: string[], _existing: string[]): string {
    // We cannot inspect provider keys here; return first alias if present.
    return names.length ? names[0] : "";
  }
}

/**
 * Normalize a single provider row into catalog row using mapping + specs.
 * - Currency values converted into ctx.baseCcy (default USD).
 * - pct kept as fraction; bps to fraction (/10,000).
 * - Dates returned as ISO strings (UTC).
 */
export function normalizeRow(
  src: Dict,
  mapping: Mapping,
  specs: Record<string, FieldSpec> | FieldSpec[] = DefaultSpecs,
  ctx: NormalizeCtx = {}
): Dict {
  const specIx = Array.isArray(specs) ? indexSpecs(specs) : specs;
  const out: Dict = {};
  for (const catName in mapping) {
    const provKey = mapping[catName];
    const spec = specIx[catName] || { name: catName } as FieldSpec;
    const raw = pick(src, provKey, spec.aliases);
    const val = normalizeValue(raw, spec, src, ctx);
    out[catName] = val == null ? (spec.required ? spec.default ?? null : (spec.default ?? null)) : val;
  }
  // Attempt to stamp asOf if absent
  if (out.asOf == null) {
    const guess = src["asOf"] || src["date"] || src["timestamp"] || ctx.asOf;
    if (guess) out.asOf = toISO(guess);
  }
  return out;
}

/** Normalize an array of provider rows. */
export function normalizeBatch(
  rows: Dict[],
  mapping: Mapping,
  specs: Record<string, FieldSpec> | FieldSpec[] = DefaultSpecs,
  ctx: NormalizeCtx = {}
): Dict[] {
  return (rows || []).map(r => normalizeRow(r, mapping, specs, ctx));
}

/** Build a mapping from simple alias table: { catalogName: ["PX_LAST","close_px"] } */
export function mappingFromAliases(aliases: Record<string, string[]>): Mapping {
  const out: Mapping = {};
  for (const k in aliases) out[k] = (aliases[k] && aliases[k][0]) || k;
  return out;
}

/* ───────────────────────── Normalizers ───────────────────────── */

export function normalizeValue(raw: any, spec: FieldSpec, row: Dict, ctx: NormalizeCtx): any {
  if (raw == null) return raw;

  // Unit-driven normalization first
  if (spec.unit) {
    if (isCurrency(spec.unit)) {
      const srcCcy = (row.currency || row.ccy || "USD") as string;
      return toBaseCcy(asNumber(raw, ctx), srcCcy, ctx.baseCcy || "USD", ctx.fx || {});
    }
    if (spec.unit === "pct") return asPercentFraction(raw, ctx);
    if (spec.unit === "bps") return asNumber(raw, ctx) / 10_000;
    if (spec.unit === "shares" || spec.unit === "number") return asNumber(raw, ctx);
    // unknown unit: just coerce to number
    if (!isKnownUnit(spec.unit)) return asNumber(raw, ctx);
  }

  // Type-driven normalization next
  if (spec.type === "number") return asNumber(raw, ctx);
  if (spec.type === "boolean") return asBoolean(raw);
  if (spec.type === "date") return toISO(raw);
  return asString(raw);
}

/* ───────────────────────── Helpers ───────────────────────── */

function asNumber(v: any, ctx: NormalizeCtx): number {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const s = v.trim();
    if (ctx.acceptPercentStrings && s.endsWith("%")) {
      const n = Number(s.slice(0, -1).replace(/,/g, ""));
      if (isFinite(n)) return n / 100;
    }
    const n = Number(s.replace(/,/g, ""));
    if (isFinite(n)) return n;
  }
  if (typeof v === "boolean") return v ? 1 : 0;
  const n = Number(v);
  return isFinite(n) ? n : NaN;
}

function asPercentFraction(v: any, ctx: NormalizeCtx): number {
  if (typeof v === "string" && v.trim().endsWith("%")) {
    const n = Number(v.trim().slice(0, -1).replace(/,/g, ""));
    return isFinite(n) ? n / 100 : NaN;
  }
  return asNumber(v, ctx);
}

function asBoolean(v: any): boolean {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "true" || s === "yes" || s === "y" || s === "1") return true;
    if (s === "false" || s === "no" || s === "n" || s === "0") return false;
  }
  return Boolean(v);
}

function asString(v: any): string {
  if (v == null) return "";
  return String(v);
}

function toISO(d: any): string {
  try {
    // If it's already an ISO string, Date will keep semantics; we output UTC ISO.
    return new Date(d).toISOString();
  } catch {
    return String(d);
  }
}

function isKnownUnit(u: string): boolean {
  return u === "number" || u === "shares" || u === "pct" || u === "bps" || isCurrency(u);
}

function isCurrency(u: string): boolean {
  // naive: 3+ uppercase letters means "currency" to be converted if FX present
  if (u === "USD" || u === "EUR" || u === "JPY") return true;
  return /^[A-Z]{3,}$/.test(u);
}

function toBaseCcy(amount: number, from: string, base: string, fx: Record<string, number>): number {
  if (!isFinite(amount)) return amount;
  if (!from || !base || from === base) return amount;
  const p1 = (from + base).toUpperCase();
  const p2 = (base + from).toUpperCase();
  if (fx[p1]) return amount * fx[p1];
  if (fx[p2]) return amount / fx[p2];
  return amount; // fallback if no rate
}

function pick(row: Dict, key: string, aliases?: string[]): any {
  if (key in row) return row[key];
  if (aliases && aliases.length) {
    for (const a of aliases) if (a in row) return row[a];
  }
  // try loose case-insensitive match
  const kLow = key.toLowerCase();
  for (const k in row) if (k.toLowerCase() === kLow) return row[k];
  return undefined;
}

/* ───────────────────────── Convenience ───────────────────────── */

/** Build a quick mapping for OHLCV using common vendor keys. */
export function ohlcvMapping(opts?: {
  open?: string; high?: string; low?: string; close?: string; volume?: string; currency?: string; ticker?: string; asOf?: string;
}): Mapping {
  return {
    open:   opts?.open   || "PX_OPEN",
    high:   opts?.high   || "PX_HIGH",
    low:    opts?.low    || "PX_LOW",
    close:  opts?.close  || "PX_LAST",
    volume: opts?.volume || "PX_VOLUME",
    currency: opts?.currency || "CRNCY",
    ticker:   opts?.ticker   || "TICKER",
    asOf:     opts?.asOf     || "asOf",
  };
}

/** Merge two mappings; right-hand wins on conflicts. */
export function mergeMapping(a: Mapping, b: Mapping): Mapping {
  const out: Mapping = {};
  for (const k in a) out[k] = a[k];
  for (const k in b) out[k] = b[k];
  return out;
}

/** Validate that required fields exist after normalization; throws on missing. */
export function assertRequired(row: Dict, specs: FieldSpec[]): void {
  for (const s of specs) {
    if (s.required && (row[s.name] == null || row[s.name] === "")) {
      throw new Error(`required_missing:${s.name}`);
    }
  }
}

/** Quick spec builder. */
export function spec(name: string, type: ValueType, unit?: Unit, required = false, aliases?: string[]): FieldSpec {
  return { name, type, unit, required, aliases };
}

/** Shallow clone + only keep fields present in specs index. */
export function project(row: Dict, specs: FieldSpec[] | Record<string, FieldSpec>): Dict {
  const ix = Array.isArray(specs) ? indexSpecs(specs) : specs;
  const out: Dict = {};
  for (const k in ix) if (k in row) out[k] = row[k];
  return out;
}

/* ───────────────────────── Example: mini self-test (optional) ───────────────────────── */

export function __selftest__(): string {
  const specs = DefaultSpecs;
  const map = ohlcvMapping();
  const ctx: NormalizeCtx = { baseCcy: "USD", fx: { "EURUSD": 1.1 }, acceptPercentStrings: true };
  const src = { PX_OPEN: "100", PX_HIGH: 110, PX_LOW: 95, PX_LAST: 105, PX_VOLUME: "1,234,567", CRNCY: "EUR", TICKER: "SIM", asOf: "2024-01-02" };
  const row = normalizeRow(src, map, specs, ctx);
  if (Math.abs(row.open - 110) > 1e-6) return "fail_fx";
  if (row.ticker !== "SIM") return "fail_map";
  if (typeof row.volume !== "number") return "fail_num";
  return "ok";
}