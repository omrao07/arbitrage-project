// adapters/src/index.ts
// Zero-dependency adapter registry + pipeline runner.
// Keep it pure (no imports) so this compiles anywhere.

/* ────────────────────────────────────────────────────────────────────────── *
 * Types
 * ────────────────────────────────────────────────────────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type AdapterContext = {
  /** UTC ISO timestamp when the run started */
  startedAt?: string;
  /** Arbitrary environment bag (keys like env, tz, region, dryRun, etc.) */
  env?: Dict;
  /** Version/revision string for tracing */
  version?: string;
  /** Optional logger hook; defaults to console.* */
  log?: {
    info?: (...args: any[]) => void;
    warn?: (...args: any[]) => void;
    error?: (...args: any[]) => void;
    debug?: (...args: any[]) => void;
  };
};

export type AdapterResult<O = any> = {
  ok: boolean;
  /** Primary payload; for row transforms prefer `{rows: any[]}`. */
  data?: O;
  /** Free-form metadata (timings, counts, provenance, etc.) */
  meta?: Dict;
  /** Error message if ok=false */
  error?: string;
};

export type Adapter<I = any, O = any> =
  (input: I, ctx: AdapterContext) => AdapterResult<O> | Promise<AdapterResult<O>>;

/* ────────────────────────────────────────────────────────────────────────── *
 * Registry
 * ────────────────────────────────────────────────────────────────────────── */

const REGISTRY: Dict<Adapter> = Object.create(null);

export function registerAdapter(name: string, fn: Adapter): void {
  if (!name || typeof name !== "string") throw new Error("registerAdapter: name must be a non-empty string");
  if (typeof fn !== "function") throw new Error("registerAdapter: fn must be a function");
  REGISTRY[name] = fn;
}

export function unregisterAdapter(name: string): void {
  delete REGISTRY[name];
}

export function getAdapter(name: string): Adapter | undefined {
  return REGISTRY[name];
}

export function hasAdapter(name: string): boolean {
  return typeof REGISTRY[name] === "function";
}

export function listAdapters(): string[] {
  return Object.keys(REGISTRY).sort();
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Runner
 * ────────────────────────────────────────────────────────────────────────── */

export async function runAdapter<I = any, O = any>(
  name: string,
  input: I,
  ctx?: AdapterContext
): Promise<AdapterResult<O>> {
  const fn = REGISTRY[name];
  const _ctx = withDefaults(ctx);
  if (!fn) {
    const err = `Adapter not found: ${name}`;
    _ctx.log?.error?.(err);
    return { ok: false, error: err };
  }
  try {
    const res = await fn(input, _ctx);
    if (!res || typeof res.ok !== "boolean") {
      const err = `Adapter "${name}" returned invalid result`;
      _ctx.log?.error?.(err);
      return { ok: false, error: err };
    }
    return res as AdapterResult<O>;
  } catch (e: any) {
    const err = `Adapter "${name}" threw: ${stringifyError(e)}`;
    _ctx.log?.error?.(err);
    return { ok: false, error: err };
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Pipelines
 * ────────────────────────────────────────────────────────────────────────── */

export type PipelineStep = string;

export type PipelineResult<T = any> = AdapterResult<T> & {
  /** Per-step results (ok/error/meta) for debugging */
  steps?: Array<{ name: string; ok: boolean; meta?: Dict; error?: string }>;
};

export async function runPipeline<T = any>(
  steps: PipelineStep[],
  initialData: T,
  ctx?: AdapterContext
): Promise<PipelineResult<T>> {
  const _ctx = withDefaults(ctx);
  const outSteps: Array<{ name: string; ok: boolean; meta?: Dict; error?: string }> = [];
  let current: any = initialData;

  for (let i = 0; i < steps.length; i++) {
    const name = steps[i];
    const res = await runAdapter<any, any>(name, current, _ctx);
    outSteps.push({ name, ok: res.ok, meta: res.meta, error: res.error });
    if (!res.ok) {
      return { ok: false, error: res.error, data: current, steps: outSteps, meta: { failedAt: name } };
    }
    current = res.data;
  }

  return { ok: true, data: current, steps: outSteps, meta: { steps: steps.slice() } };
}

/**
 * Convenience for row pipelines:
 * Each adapter receives `{ rows, meta }` and returns the same shape.
 */
export type RowsPayload = { rows: any[]; meta?: Dict };

export async function runRowPipeline(
  steps: PipelineStep[],
  rows: any[],
  meta?: Dict,
  ctx?: AdapterContext
): Promise<PipelineResult<RowsPayload>> {
  return runPipeline<RowsPayload>(steps, { rows: ensureArray(rows), meta: meta || {} }, ctx);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Built-in utilities (optional helpers you can reuse inside adapters)
 * ────────────────────────────────────────────────────────────────────────── */

export function ensureArray<T = any>(v: any): T[] {
  if (Array.isArray(v)) return v as T[];
  if (v == null) return [];
  return [v as T];
}

export function mapFields(row: Dict, aliases: Dict<string>): Dict {
  const out: Dict = {};
  for (const k in row) {
    const canon = aliases && aliases[k] ? aliases[k] : k;
    out[canon] = row[k];
  }
  return out;
}

export function pick<T extends Dict>(obj: T, keys: string[]): Partial<T> {
  const o: Partial<T> = {};
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    if (k in obj) (o as any)[k] = obj[k];
  }
  return o;
}

export function omit<T extends Dict>(obj: T, keys: string[]): Partial<T> {
  const set: Dict<boolean> = {};
  for (let i = 0; i < keys.length; i++) set[keys[i]] = true;
  const o: Partial<T> = {};
  for (const k in obj) {
    if (!set[k]) (o as any)[k] = obj[k];
  }
  return o;
}

export function normalizeDateYYYYMMDD(v: any): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    const s = v.trim();
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[1]}-${m[2]}-${m[3]}`;
    const t = Date.parse(s);
    if (!isNaN(t)) {
      const d = new Date(t);
      const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
      const dd = String(d.getUTCDate()).padStart(2, "0");
      return `${d.getUTCFullYear()}-${mm}-${dd}`;
    }
    return null;
  }
  if (v instanceof Date) {
    const mm = String(v.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(v.getUTCDate()).padStart(2, "0");
    return `${v.getUTCFullYear()}-${mm}-${dd}`;
  }
  return null;
}

export function coercePercentFraction(n: any): number | null {
  if (n == null || n === "") return null;
  const val = typeof n === "number" ? n : Number(String(n).replace(/[_ ,]/g, ""));
  if (!isFinite(val)) return null;
  return Math.abs(val) > 1.5 ? val / 100 : val; // interpret 12 => 0.12
}

export function stableStringify(v: any): string {
  try {
    return JSON.stringify(sortKeysDeep(v));
  } catch {
    return String(v);
  }
}

function sortKeysDeep(v: any): any {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v && typeof v === "object") {
    const out: Dict = {};
    Object.keys(v).sort().forEach(k => { out[k] = sortKeysDeep(v[k]); });
    return out;
  }
  return v;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Default adapters (safe, minimal)
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * "identity" adapter: passes input through unchanged.
 * Useful as a placeholder in pipelines.
 */
registerAdapter("identity", function identity(input: any): AdapterResult<any> {
  return { ok: true, data: input, meta: { adapter: "identity" } };
});

/**
 * "rows/ensure-array": guarantees `{rows}` shape.
 * - input: any[] | {rows:any[]}
 * - output: {rows:any[]}
 */
registerAdapter("rows/ensure-array", function ensureRows(input: any): AdapterResult<RowsPayload> {
  const rows = Array.isArray(input) ? input : (input && Array.isArray(input.rows) ? input.rows : []);
  return { ok: true, data: { rows }, meta: { count: rows.length } };
});

/**
 * "rows/filter-truthy": removes falsy rows.
 */
registerAdapter("rows/filter-truthy", function filterTruthy(input: RowsPayload): AdapterResult<RowsPayload> {
  const rows = ensureArray(input && (input as any).rows).filter(Boolean);
  return { ok: true, data: { rows }, meta: { count: rows.length } };
});

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals
 * ────────────────────────────────────────────────────────────────────────── */

function withDefaults(ctx?: AdapterContext): AdapterContext {
  const d: AdapterContext = {
    startedAt: (ctx && ctx.startedAt) || new Date().toISOString(),
    env: (ctx && ctx.env) || {},
    version: (ctx && ctx.version) || "0.0.0",
    log: ctx && ctx.log ? ctx.log : {
      info: (...a: any[]) => { try { console.info?.(...a); } catch { /* noop */ } },
      warn: (...a: any[]) => { try { console.warn?.(...a); } catch { /* noop */ } },
      error: (...a: any[]) => { try { console.error?.(...a); } catch { /* noop */ } },
      debug: (...a: any[]) => { try { console.debug?.(...a); } catch { /* noop */ } }
    }
  };
  return d;
}

function stringifyError(e: any): string {
  if (!e) return "Unknown error";
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message || String(e);
  try { return stableStringify(e); } catch { return String(e); }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Usage (commented)
 *
 * // Register a custom adapter
 * registerAdapter("my/uppercase-names", (input: RowsPayload) => {
 *   const rows = ensureArray(input.rows).map(r => ({ ...r, name: String(r.name || "").toUpperCase() }));
 *   return { ok: true, data: { rows }, meta: { count: rows.length } };
 * });
 *
 * // Run single adapter
 * const single = await runAdapter<RowsPayload, RowsPayload>("my/uppercase-names", { rows: [{ name: "alice" }] });
 *
 * // Pipeline
 * const piped = await runRowPipeline(["rows/ensure-array", "my/uppercase-names"], [{ name: "bob" }]);
 * ────────────────────────────────────────────────────────────────────────── */