// loader/validator.ts
// Zero-dependency validation utilities for tabular data.
// Works with rows parsed by loader/parser.ts (but has no imports).
//
// Core idea:
//   validate(rows, { columns, rules, options? }) → ValidationReport
//
// • Field-level checks: required, type, allowNull, min/max (numbers & dates),
//   minLen/maxLen (strings), pattern (regex), enum(oneOf), unique.
// • Table-level checks: primaryKey (composite), uniqueTogether[][],
//   monotonic(column, "increasing" | "decreasing" | "nondecreasing" | "nonincreasing").
// • Summaries: null counts, distinct counts, type tallies.
// • Safe: no eval; accepts JavaScript values (numbers/strings/bools/dates).
//
// NOTE: This file intentionally duplicates tiny helpers (date/number sniffing)
// so it can stand alone without imports.

export type Dict<T = any> = { [k: string]: T };

export type ColumnType = "string" | "bool" | "int" | "float" | "date" | "timestamp" | "mixed";

export type Column = { name: string; type?: ColumnType };

export type FieldRule = {
  field: string;

  // Presence
  required?: boolean;       // default false
  allowNull?: boolean;      // default true

  // Typing
  type?: ColumnType;        // stronger than Column.type when present

  // Numeric bounds (applies to int/float/date/timestamp; for date/ts use ISO or Date)
  min?: number | string | Date;
  max?: number | string | Date;

  // String bounds
  minLen?: number;
  maxLen?: number;

  // String pattern (JavaScript regex, without or with delimiters)
  pattern?: string;

  // Enumerations
  enum?: Array<string | number | boolean | null>;

  // Uniqueness (single column)
  unique?: boolean;

  // Custom message prefixes (optional)
  label?: string;
};

export type TableRule =
  | { kind: "primaryKey"; fields: string[] }
  | { kind: "uniqueTogether"; fields: string[] }
  | { kind: "monotonic"; field: string; order: "increasing" | "decreasing" | "nondecreasing" | "nonincreasing" };

export type ValidationSpec = {
  fields?: FieldRule[];
  table?: TableRule[];
};

export type ValidateOptions = {
  maxErrors?: number;   // cap hard errors collected (default 10_000)
  treatEmptyStringAsNull?: boolean; // default true
};

export type Issue = {
  row: number;                // 0-based row index in the provided `rows`
  field?: string;             // optional field name
  code: string;               // e.g., "required", "type", "min", "max", "enum", "unique", "primary_key", ...
  message: string;            // human-friendly
  value?: any;                // offending value (if safe)
};

export type FieldSummary = {
  nulls: number;
  distinct: number;
  types: Partial<Record<ColumnType | "number", number>>; // "number" = generic if mixed
};

export type ValidationReport = {
  ok: boolean;
  errors: Issue[];
  warnings: Issue[];      // reserved (not used by default rules, but kept for symmetry)
  stats: {
    rowCount: number;
    fields: Record<string, FieldSummary>;
  };
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Entry point
 * ────────────────────────────────────────────────────────────────────────── */

export function validate(
  rows: Array<Dict>,
  spec: { columns?: Column[]; rules?: ValidationSpec; options?: ValidateOptions }
): ValidationReport {
  const columns = normalizeColumns(spec.columns, rows);
  const rules = spec.rules || {};
  const opts = withDefaults(spec.options);

  const errors: Issue[] = [];
  const warnings: Issue[] = [];

  // Precompile field rules map
  const fieldRules = new Map<string, FieldRule>();
  if (Array.isArray(rules.fields)) {
    for (let i = 0; i < rules.fields.length; i++) {
      const r = normalizeFieldRule(rules.fields[i]);
      fieldRules.set(r.field, r);
    }
  }

  // Pre-alloc summaries
  const stats: ValidationReport["stats"] = { rowCount: rows.length, fields: {} };
  for (let i = 0; i < columns.length; i++) {
    stats.fields[columns[i].name] = { nulls: 0, distinct: 0, types: {} };
  }

  // Distinct and unique trackers
  const distinctSets: Record<string, Set<string>> = {};
  const uniqueSets: Record<string, Set<string>> = {};
  for (let i = 0; i < columns.length; i++) {
    const c = columns[i].name;
    distinctSets[c] = new Set<string>();
    const fr = fieldRules.get(c);
    if (fr?.unique) uniqueSets[c] = new Set<string>();
  }

  // Pass 1: Field-level validations
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];

    for (let i = 0; i < columns.length; i++) {
      const col = columns[i];
      const name = col.name;
      const fr = fieldRules.get(name);

      const raw = row?.[name];
      const isNull = valueIsNull(raw, opts.treatEmptyStringAsNull !== false);

      // Summary: nulls
      if (isNull) stats.fields[name].nulls += 1;

      // Distinct tracking (stringify)
      distinctSets[name].add(keyOfValue(raw));

      // Required
      if (fr?.required && isNull) {
        pushErr(errors, { row: r, field: name, code: "required", message: prefix(fr, name) + "is required", value: raw }, opts);
        continue; // no further checks on null
      }

      // allowNull? If false and null provided
      if (fr && fr.allowNull === false && isNull) {
        pushErr(errors, { row: r, field: name, code: "not_null", message: prefix(fr, name) + "cannot be null", value: raw }, opts);
        continue;
      }

      if (isNull) continue; // nothing else to validate

      // Type check (prefer FieldRule.type, else column.type if defined)
      const targetType: ColumnType | undefined = fr?.type || col.type;
      if (targetType && targetType !== "mixed") {
        if (!checkType(raw, targetType)) {
          pushErr(errors, {
            row: r, field: name, code: "type",
            message: prefix(fr, name) + `must be ${targetType}`,
            value: raw
          }, opts);
          // still continue to try other checks cautiously
        }
      }

      // Record observed type for stats
      const t = classifyValue(raw);
      stats.fields[name].types[t] = (stats.fields[name].types[t] || 0) + 1;

      // Numeric/date bounds
      if (fr && (fr.min !== undefined || fr.max !== undefined)) {
        const cmp = asComparable(raw, fr.type || col.type);
        const minC = asComparable(fr.min, fr.type || col.type);
        const maxC = asComparable(fr.max, fr.type || col.type);
        if (minC != null && cmp != null && cmp < minC) {
          pushErr(errors, {
            row: r, field: name, code: "min",
            message: prefix(fr, name) + `must be ≥ ${fmtBound(fr.min)}`,
            value: raw
          }, opts);
        }
        if (maxC != null && cmp != null && cmp > maxC) {
          pushErr(errors, {
            row: r, field: name, code: "max",
            message: prefix(fr, name) + `must be ≤ ${fmtBound(fr.max)}`,
            value: raw
          }, opts);
        }
      }

      // String length
      if (fr && (fr.minLen != null || fr.maxLen != null)) {
        const s = String(raw);
        if (fr.minLen != null && s.length < fr.minLen) {
          pushErr(errors, {
            row: r, field: name, code: "min_len",
            message: prefix(fr, name) + `length must be ≥ ${fr.minLen}`,
            value: raw
          }, opts);
        }
        if (fr.maxLen != null && s.length > fr.maxLen) {
          pushErr(errors, {
            row: r, field: name, code: "max_len",
            message: prefix(fr, name) + `length must be ≤ ${fr.maxLen}`,
            value: raw
          }, opts);
        }
      }

      // Pattern
      if (fr?.pattern) {
        const re = compileRegex(fr.pattern);
        if (!re.test(String(raw))) {
          pushErr(errors, {
            row: r, field: name, code: "pattern",
            message: prefix(fr, name) + `does not match pattern ${re.source}`,
            value: raw
          }, opts);
        }
      }

      // Enum
      if (fr?.enum && fr.enum.length > 0) {
        if (!enumIncludes(fr.enum, raw)) {
          pushErr(errors, {
            row: r, field: name, code: "enum",
            message: prefix(fr, name) + `must be one of [${fr.enum.map(fmtBound).join(", ")}]`,
            value: raw
          }, opts);
        }
      }

      // Unique (single)
      if (fr?.unique) {
        const k = keyOfValue(raw);
        const seen = uniqueSets[name]!;
        if (seen.has(k)) {
          pushErr(errors, {
            row: r, field: name, code: "unique",
            message: prefix(fr, name) + "must be unique",
            value: raw
          }, opts);
        } else {
          seen.add(k);
        }
      }
    }

    if (errors.length >= (opts.maxErrors || 10_000)) break;
  }

  // Distinct counts
  for (let i = 0; i < columns.length; i++) {
    const c = columns[i].name;
    stats.fields[c].distinct = distinctSets[c].size;
  }

  // Table-level rules
  if (errors.length < (opts.maxErrors || 10_000) && Array.isArray(rules.table)) {
    for (let t = 0; t < rules.table.length; t++) {
      const tr = rules.table[t];
      if (tr.kind === "primaryKey") {
        validateUniqueCombo(rows, tr.fields, "primary_key", "Primary key must be unique & not null", errors, opts);
      } else if (tr.kind === "uniqueTogether") {
        validateUniqueCombo(rows, tr.fields, "unique_together", `Combination must be unique: [${tr.fields.join(", ")}]`, errors, opts, true);
      } else if (tr.kind === "monotonic") {
        validateMonotonic(rows, tr.field, tr.order, errors, opts);
      }
      if (errors.length >= (opts.maxErrors || 10_000)) break;
    }
  }

  return { ok: errors.length === 0, errors, warnings, stats };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Table-level validators
 * ────────────────────────────────────────────────────────────────────────── */

function validateUniqueCombo(
  rows: Array<Dict>,
  fields: string[],
  code: string,
  baseMsg: string,
  errors: Issue[],
  opts: Required<ValidateOptions>,
  allowNulls = false
) {
  const seen = new Map<string, number>(); // key -> firstRow
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    let anyNull = false;
    const parts: string[] = [];
    for (let i = 0; i < fields.length; i++) {
      const v = row?.[fields[i]];
      const isN = valueIsNull(v, true);
      if (isN) anyNull = true;
      parts.push(keyOfValue(v));
    }
    if (!allowNulls && anyNull) {
      errors.push({ row: r, code, message: `${baseMsg}: null in [${fields.join(", ")}]` });
      if (errors.length >= opts.maxErrors) return;
      continue;
    }
    const key = parts.join("¦");
    if (seen.has(key)) {
      errors.push({ row: r, code, message: `${baseMsg}: duplicate of row ${seen.get(key)}` });
      if (errors.length >= opts.maxErrors) return;
    } else {
      seen.set(key, r);
    }
  }
}

function validateMonotonic(
  rows: Array<Dict>,
  field: string,
  order: "increasing" | "decreasing" | "nondecreasing" | "nonincreasing",
  errors: Issue[],
  opts: Required<ValidateOptions>
) {
  let prevC: number | null = null;
  for (let r = 0; r < rows.length; r++) {
    const v = rows[r]?.[field];
    if (valueIsNull(v, true)) { prevC = null; continue; } // skip gaps
    
   

    if (prevC != null) {
      let ok = true;
      
      if (!ok) {
        errors.push({
          row: r,
          field,
          code: "monotonic",
          message: `${field} must be ${order.replace("non", "non-")}`,
          value: rows[r]?.[field]
        });
        if (errors.length >= opts.maxErrors) return;
      }
    }
    
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Helpers: rules, columns, typing
 * ────────────────────────────────────────────────────────────────────────── */

function normalizeColumns(cols: Column[] | undefined, rows: Array<Dict>): Column[] {
  if (Array.isArray(cols) && cols.length) {
    // ensure names and cleaned types
    return cols.map(c => ({ name: String(c.name), type: normalizeType(c.type) }));
  }
  // infer from first row
  const names = new Set<string>();
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (r && typeof r === "object") {
      for (const k in r) names.add(k);
    }
    if (names.size) break;
  }
  const inferred: Column[] = [];
  const arr = Array.from(names);
  for (let i = 0; i < arr.length; i++) {
    const name = arr[i];
    const t = inferTypeFromSamples(rows.slice(0, 50).map(ro => ro?.[name]));
   
  }
  return inferred;
}

function normalizeFieldRule(r: FieldRule): FieldRule {
  const nr: FieldRule = { ...r };
  nr.field = String(r.field);
  if (r.type) nr.type = normalizeType(r.type);
  if (typeof r.allowNull !== "boolean") nr.allowNull = true;
  return nr;
}

function withDefaults(o?: ValidateOptions): Required<ValidateOptions> {
  return {
    maxErrors: o?.maxErrors ?? 10_000,
    treatEmptyStringAsNull: o?.treatEmptyStringAsNull !== false
  };
}

function normalizeType(t?: string): ColumnType | undefined {
  if (!t) return undefined;
  const x = t.toLowerCase();
  if (x === "integer" || x === "int") return "int";
  if (x === "float" || x === "double" || x === "number") return "float";
  if (x === "string" || x === "text") return "string";
  if (x === "bool" || x === "boolean") return "bool";
  if (x === "date") return "date";
  if (x === "timestamp" || x === "datetime" || x === "iso") return "timestamp";
  if (x === "mixed") return "mixed";
  return "mixed";
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Helpers: value classification & comparisons
 * ────────────────────────────────────────────────────────────────────────── */

function valueIsNull(v: any, emptyIsNull: boolean): boolean {
  return v === null || v === undefined || (emptyIsNull && (v === "" || (typeof v === "string" && v.trim() === "")));
}

function classifyValue(v: any): ColumnType | "number" {
  if (v === null || v === undefined) return "mixed";
  if (typeof v === "boolean") return "bool";
  if (typeof v === "number") {
    return Number.isInteger(v) ? "int" : "float";
  }
  if (v instanceof Date) return "timestamp";
  const s = String(v).trim();
  if (looksTimestamp(s)) return "timestamp";
  if (looksDate(s)) return "date";
  if (looksInt(s)) return "int";
  if (looksFloat(s)) return "float";
  return "string";
}

function checkType(v: any, t: ColumnType): boolean {
  if (v === null || v === undefined) return true; // handled by allowNull/required earlier
  if (t === "mixed") return true;
  if (t === "bool") return typeof v === "boolean" || /^(true|false|yes|no|0|1)$/i.test(String(v).trim());
  if (t === "string") return typeof v === "string" || typeof v === "number" || typeof v === "boolean" || v instanceof Date;
  if (t === "int") {
    if (typeof v === "number") return Number.isInteger(v);
    return looksInt(String(v).trim());
  }
  if (t === "float") {
    if (typeof v === "number") return isFinite(v);
    return looksFloat(String(v).trim());
  }
  if (t === "date") return looksDate(String(v).trim()) || v instanceof Date;
  if (t === "timestamp") return looksTimestamp(String(v).trim()) || v instanceof Date;
  return true;
}

function asComparable(v: any, t?: ColumnType): number | null {
  if (v == null) return null;
  const tt = t || classifyValue(v);
  if (tt === "date") {
    const iso = normalizeDate(String(v));
    if (!iso) return null;
    return Date.parse(iso);
  }
  if (tt === "timestamp") {
    const iso = normalizeISO(String(v));
    if (!iso) return null;
    return Date.parse(iso);
  }
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (looksFloat(String(v).trim()) || looksInt(String(v).trim())) return Number(String(v).replace(/[_ ,]/g, ""));
  return null;
}

function keyOfValue(v: any): string {
  if (v === null) return "∅";
  if (v === undefined) return "∅u";
  if (v instanceof Date) return "d:" + v.toISOString();
  if (typeof v === "object") {
    try { return "o:" + JSON.stringify(v); } catch { return "o:[unstringifiable]"; }
  }
  return typeof v + ":" + String(v);
}

function compileRegex(pat: string): RegExp {
  // Accept "/.../flags" or "..." with default flags "i"
  const m = pat.match(/^\/(.*)\/([gimsuy]*)$/);
  if (m) return new RegExp(m[1], m[2] || "i");
  return new RegExp(pat, "i");
}

function enumIncludes(arr: any[], v: any): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (deepEq(arr[i], v)) return true;
    // loose match for number-like strings vs numbers
    if (typeof arr[i] === "number" && typeof v !== "number" && looksFloat(String(v))) {
      if (Number(String(v)) === arr[i]) return true;
    }
    if (typeof arr[i] === "string" && typeof v !== "string") {
      if (String(v) === arr[i]) return true;
    }
  }
  return false;
}

function deepEq(a: any, b: any): boolean {
  if (a === b) return true;
  if (a instanceof Date && b instanceof Date) return a.getTime() === b.getTime();
  if (typeof a !== typeof b) return false;
  if (a && b && typeof a === "object") {
    const ak = Object.keys(a), bk = Object.keys(b);
    if (ak.length !== bk.length) return false;
    for (let i = 0; i < ak.length; i++) {
      const k = ak[i];
      if (!deepEq(a[k], b[k])) return false;
    }
    return true;
  }
  return false;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Tiny parsers for numbers/dates (duplicated from other files intentionally)
 * ────────────────────────────────────────────────────────────────────────── */

function looksInt(s: string): boolean { return /^[-+]?\d{1,3}(?:_?\d{3})*$/.test(s); }
function looksFloat(s: string): boolean { return /^[-+]?(?:\d{1,3}(?:_?\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$/.test(s) || /^[-+]?\.\d+(?:[eE][-+]?\d+)?$/.test(s); }
function looksDate(s: string): boolean { return /^\d{4}-\d{2}-\d{2}$/.test(s); }
function looksTimestamp(s: string): boolean { return /^\d{4}-\d{2}-\d{2}[tT ][\d:.]+(?:[zZ]|[+-]\d{2}:?\d{2})?$/.test(s); }

function normalizeDate(v: string): string | null {
  const s = v.trim();
  if (looksDate(s)) return s;
  const t = Date.parse(s);
  if (isNaN(t)) return null;
  const d = new Date(t);
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}
function normalizeISO(v: string): string | null {
  const t = Date.parse(v);
  return isNaN(t) ? null : new Date(t).toISOString();
}
function pad2(n: number): string { return String(n).padStart(2, "0"); }

/* ────────────────────────────────────────────────────────────────────────── *
 * Formatting & limits
 * ────────────────────────────────────────────────────────────────────────── */

function fmtBound(v: any): string {
  if (v instanceof Date) return v.toISOString();
  if (typeof v === "string") {
    const nd = normalizeDate(v) || normalizeISO(v);
    if (nd) return nd;
  }
  return String(v);
}

function prefix(fr: FieldRule | undefined, field: string): string {
  const lbl = fr?.label || field;
  return lbl ? `${lbl}: ` : "";
}

function pushErr(list: Issue[], issue: Issue, opts: Required<ValidateOptions>) {
  if (list.length >= opts.maxErrors) return;
  list.push(issue);
}


function inferTypeFromSamples(arg0: any[]) {
    throw new Error("Function not implemented.");
}
/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */

/* ────────────────────────────────────────────────────────────────────────── *
 * Example (commented)
 *
 * // const rows = [
 * //   { ticker: "AAPL", date: "2025-01-02", px: 199.12 },
 * //   { ticker: "AAPL", date: "2025-01-02", px: 199.12 }, // duplicate PK
 * //   { ticker: "", date: "2025-01-03", px: -5 }          // required + min
 * // ];
 * //
 * // const report = validate(rows, {
 * //   columns: [
 * //     { name: "ticker", type: "string" },
 * //     { name: "date", type: "date" },
 * //     { name: "px", type: "float" }
 * //   ],
 * //   rules: {
 * //     fields: [
 * //       { field: "ticker", required: true, minLen: 1, unique: false },
 * //       { field: "date", required: true, type: "date" },
 * //       { field: "px", min: 0 }
 * //     ],
 * //     table: [
 * //       { kind: "primaryKey", fields: ["ticker", "date"] },
 * //       { kind: "monotonic", field: "date", order: "increasing" }
 * //     ]
 * //   }
 * // });
 * //
 * // console.log(report.ok, report.errors);
 * ────────────────────────────────────────────────────────────────────────── */