// risk/src/limits.ts
// Pure, zero-import helpers for defining, evaluating, and managing risk limits.
// This file is self-contained (re-declares minimal types) and safe to drop in.

// ────────────────────────────────────────────────────────────────────────────
// Types (kept minimal to avoid coupling)
// ────────────────────────────────────────────────────────────────────────────

export type Dict<T = any> = { [k: string]: T };

export type RiskMetrics = {
  gross: number;
  net: number;
  long: number;
  short: number;
  volDaily?: number;
  volAnnual?: number;
  var_95?: number;
  var_99?: number;
  es_95?: number;
  es_99?: number;
  // Allow custom numeric metrics on the object
  [k: string]: any;
};

export type LimitOp = "<" | "<=" | ">" | ">=" | "abs<=" | "abs<" | "between" | "outside";

export type Limit = {
  id: string;
  metric: keyof RiskMetrics | string; // supports nested via dotted path (e.g., "custom.sub.metric")
  op: LimitOp;
  threshold: number | [number, number]; // for between/outside: [lo, hi]
  severity?: "info" | "warn" | "critical";
  note?: string;               // freeform note shown in messages
  units?: string;              // e.g., "$", "bps", "%"
  hysteresisPct?: number;      // optional dead-band (e.g., 0.02 = 2%) to reduce flapping
  useAbs?: boolean;            // convenience: treat metric as absolute before comparison
};

export type Breach = {
  id: string;
  limitId: string;
  metric: string;
  value: number;
  threshold: number | [number, number];
  op: LimitOp;
  severity: NonNullable<Limit["severity"]>;
  message: string;
  at: string; // ISO timestamp
};

export type Ack = {
  // Suppress a limit temporarily (e.g., acknowledged by an operator)
  limitId: string;
  reason?: string;
  until?: string; // ISO time; if omitted, suppress once (by breachId) via `onceForId`
  onceForId?: string; // suppress one specific breach id
};

// ────────────────────────────────────────────────────────────────────────────
// Core evaluation
// ────────────────────────────────────────────────────────────────────────────

/**
 * Evaluate a single limit against provided metrics.
 * Returns a Breach when violated; otherwise null.
 */
export function evalLimit(metrics: RiskMetrics, limit: Limit, now?: Date): Breach | null {
  const sev = limit.severity || "warn";
  const raw = getMetricValue(metrics, String(limit.metric));
  const v = limit.useAbs ? Math.abs(raw) : raw;

  if (!isFiniteNumber(v)) return null;

  const res = compareWithHysteresis(v, limit);
  if (res.ok) return null;

  return {
    id: breachId(limit.id, v, now),
    limitId: limit.id,
    metric: String(limit.metric),
    value: round2(v),
    threshold: limit.threshold,
    op: limit.op,
    severity: sev,
    message: formatMessage(limit, v),
    at: (now || new Date()).toISOString()
  };
}

/**
 * Evaluate a list of limits. Returns all active breaches.
 */
export function evalLimits(metrics: RiskMetrics, limits: Limit[], now?: Date): Breach[] {
  const out: Breach[] = [];
  for (let i = 0; i < limits.length; i++) {
    const b = evalLimit(metrics, limits[i], now);
    if (b) out.push(b);
  }
  // Stable order: critical → warn → info, then by message
  out.sort((a, b) => severityRank(a.severity) - severityRank(b.severity) || (a.message < b.message ? -1 : 1));
  return out;
}

/**
 * Quick boolean: should we pull the killswitch based on current breaches?
 */
export function killswitchSuggested(breaches: Breach[], level: "critical" | "warn" = "critical"): boolean {
  for (let i = 0; i < breaches.length; i++) if (breaches[i].severity === level) return true;
  return false;
}

// ────────────────────────────────────────────────────────────────────────────
/** Acknowledgement/suppression helpers */
// ────────────────────────────────────────────────────────────────────────────

/**
 * Filter out acknowledged breaches.
 * - If ack.until is in the future, suppress all breaches for that limit.
 * - If ack.onceForId matches breach.id, suppress just that breach once.
 */
export function filterAcknowledged(breaches: Breach[], acks: Ack[], now?: Date): Breach[] {
  if (!acks || !acks.length) return breaches.slice();
  const t = (now || new Date()).getTime();

  const activeByLimit = new Map<string, { until?: number }>();
  const oneShot = new Set<string>();

  for (let i = 0; i < acks.length; i++) {
    const a = acks[i];
    if (a.onceForId) {
      oneShot.add(a.onceForId);
    } else {
      const until = a.until ? Date.parse(a.until) : NaN;
      if (!isNaN(until) && until > t) {
        activeByLimit.set(a.limitId, { until });
      }
    }
  }

  const out: Breach[] = [];
  for (let i = 0; i < breaches.length; i++) {
    const b = breaches[i];
    if (oneShot.has(b.id)) continue;
    const lim = activeByLimit.get(b.limitId);
    if (lim && lim.until && lim.until > t) continue;
    out.push(b);
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
/** Rolling counters & escalation (optional) */
// ────────────────────────────────────────────────────────────────────────────

export type BreachEvent = { id: string; limitId: string; at: string; severity: Breach["severity"] };

export type BreachHistory = {
  // limitId → list of timestamps (ms since epoch), kept recent first
  [limitId: string]: number[];
};

export function updateBreachHistory(history: BreachHistory, breaches: Breach[], windowMs: number, now?: Date): BreachHistory {
  const t = (now || new Date()).getTime();
  const lo = t - Math.max(0, windowMs);

  // clone shallow
  const h: BreachHistory = {};
  for (const k in history) h[k] = history[k].slice().filter(x => x >= lo);

  for (let i = 0; i < breaches.length; i++) {
    const b = breaches[i];
    const arr = h[b.limitId] || (h[b.limitId] = []);
    arr.push(t);
  }
  return h;
}

export function breachCounts(history: BreachHistory, windowMs: number, now?: Date): Record<string, number> {
  const t = (now || new Date()).getTime();
  const lo = t - Math.max(0, windowMs);
  const out: Record<string, number> = {};
  for (const k in history) out[k] = history[k].filter(x => x >= lo).length;
  return out;
}

/**
 * Simple escalation rule: if a limit breaches >= count times within window, escalate severity.
 */
export function escalate(breaches: Breach[], history: BreachHistory, windowMs: number, thresholdCount: number): Breach[] {
  if (!breaches.length) return breaches;
  const counts = breachCounts(history, windowMs);
  const out: Breach[] = new Array(breaches.length);
  for (let i = 0; i < breaches.length; i++) {
    const b = { ...breaches[i] };
    const n = counts[b.limitId] || 0;
    if (n >= thresholdCount && b.severity !== "critical") b.severity = "critical";
    out[i] = b;
  }
  out.sort((a, b) => severityRank(a.severity) - severityRank(b.severity));
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Pretty summaries
// ────────────────────────────────────────────────────────────────────────────

export function summarize(breaches: Breach[]): {
  total: number;
  bySeverity: { info: number; warn: number; critical: number };
  byLimit: Record<string, number>;
  messages: string[];
} {
  let info = 0, warn = 0, critical = 0;
  const byLimit: Record<string, number> = {};
  const msgs: string[] = [];
  for (let i = 0; i < breaches.length; i++) {
    const b = breaches[i];
    if (b.severity === "critical") critical++; else if (b.severity === "warn") warn++; else info++;
    byLimit[b.limitId] = (byLimit[b.limitId] || 0) + 1;
    msgs.push(b.message);
  }
  return {
    total: breaches.length,
    bySeverity: { info, warn, critical },
    byLimit,
    messages: msgs
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Internals
// ────────────────────────────────────────────────────────────────────────────

function compareWithHysteresis(v: number, L: Limit): { ok: boolean } {
  const op = L.op;
  const h = Math.max(0, Number(L.hysteresisPct || 0));

  if (op === "<") return { ok: v < n(L.threshold) * (1 - h) };
  if (op === "<=") return { ok: v <= n(L.threshold) * (1 - h) };
  if (op === ">") return { ok: v > n(L.threshold) * (1 + h) };
  if (op === ">=") return { ok: v >= n(L.threshold) * (1 + h) };
  if (op === "abs<=") return { ok: Math.abs(v) <= n(L.threshold) * (1 - h) };
  if (op === "abs<") return { ok: Math.abs(v) < n(L.threshold) * (1 - h) };

  if (op === "between" || op === "outside") {
    const [lo, hi] = pair(L.threshold);
    const loAdj = lo * (1 - h);
    const hiAdj = hi * (1 + h);
    const inside = v >= Math.min(loAdj, hiAdj) && v <= Math.max(loAdj, hiAdj);
    return { ok: op === "between" ? inside : !inside };
  }

  // Fallback: treat as <=
  return { ok: v <= n(L.threshold) * (1 - h) };
}

function getMetricValue(metrics: RiskMetrics, path: string): number {
  // support dotted path, e.g. "custom.metrics.var"
  const parts = String(path).split(".").filter(Boolean);
  let cur: any = metrics;
  for (let i = 0; i < parts.length; i++) {
    if (cur == null) return NaN;
    cur = cur[parts[i]];
  }
  if (typeof cur === "number") return cur;
  // Allow { value: number } shape
  if (cur && typeof cur.value === "number") return cur.value;
  return NaN;
}

function formatMessage(L: Limit, v: number): string {
  const u = L.units || "";
  const sev = (L.severity || "warn").toUpperCase();
  if (L.op === "between") {
    const [lo, hi] = pair(L.threshold);
    return `[${sev}] ${L.id}: ${L.metric}=${fmt(v, u)} must be between ${fmt(lo, u)} and ${fmt(hi, u)}${note(L)}`;
  }
  if (L.op === "outside") {
    const [lo, hi] = pair(L.threshold);
    return `[${sev}] ${L.id}: ${L.metric}=${fmt(v, u)} must be outside ${fmt(lo, u)}…${fmt(hi, u)}${note(L)}`;
  }
  const thr = n(L.threshold);
  const opTxt = L.op.startsWith("abs") ? `${L.op} (${fmt(thr, u)})` : `${L.op} ${fmt(thr, u)}`;
  return `[${sev}] ${L.id}: ${L.metric}=${fmt(v, u)} ${opTxt}${note(L)}`;
}

function note(L: Limit): string {
  return L.note ? ` — ${L.note}` : "";
}

function breachId(limitId: string, v: number, now?: Date): string {
  const ts = (now || new Date()).toISOString();
  return `breach:${limitId}:${round4(v)}:${ts}`;
}

function severityRank(s: "info" | "warn" | "critical"): number {
  if (s === "critical") return 0;
  if (s === "warn") return 1;
  return 2;
}

function n(x: number | [number, number]): number {
  return Array.isArray(x) ? x[0] : Number(x);
}
function pair(x: number | [number, number]): [number, number] {
  return Array.isArray(x) ? [Number(x[0]), Number(x[1])] : [Number(x), Number(x)];
}

function isFiniteNumber(x: any): x is number {
  return typeof x === "number" && isFinite(x);
}

function round2(x: number): number { return Math.round(x * 100) / 100; }
function round4(x: number): number { return Math.round(x * 10000) / 10000; }

function fmt(v: number, u: string): string {
  if (u === "%") return `${round2(v * 100)}%`;
  if (u.toLowerCase() === "bps") return `${round2(v * 10000)} bps`;
  if (u) return `${u}${round2(v)}`;
  return String(round2(v));
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience: one-liners for common patterns
// ────────────────────────────────────────────────────────────────────────────

/** Build a ≤ limit */
export function atMost(id: string, metric: string, thr: number, severity: Limit["severity"] = "warn", note?: string, units?: string): Limit {
  return { id, metric, op: "<=", threshold: thr, severity, note, units };
}

/** Build an absolute ≤ limit */
export function absAtMost(id: string, metric: string, thr: number, severity: Limit["severity"] = "warn", note?: string, units?: string): Limit {
  return { id, metric, op: "abs<=", threshold: thr, severity, note, units, useAbs: true };
}

/** Build a range (inclusive) requirement */
export function between(id: string, metric: string, lo: number, hi: number, severity: Limit["severity"] = "warn", note?: string, units?: string): Limit {
  return { id, metric, op: "between", threshold: [lo, hi], severity, note, units };
}

/** Build outside-range requirement */
export function outside(id: string, metric: string, lo: number, hi: number, severity: Limit["severity"] = "warn", note?: string, units?: string): Limit {
  return { id, metric, op: "outside", threshold: [lo, hi], severity, note, units };
}