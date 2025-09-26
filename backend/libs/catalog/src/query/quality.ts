// query/quality.ts
// Zero-dependency data quality scoring & diagnostics.
//
// What this gives you (no imports required):
//   assessQuality(rows, { columns?, rules?, freshness?, weights?, baselineSchema? })
//     → QualityReport (overall score 0..100, per-dimension scores, issues, and rich stats)
//
// Dimensions (default weights in parentheses):
//   • completeness (30%)  – non-null ratios against targets / required fields
//   • validity     (30%)  – type/range/pattern/enum checks
//   • uniqueness   (20%)  – duplicate primary key / duplicate rows
//   • freshness    (20%)  – recency & continuity vs expected frequency
//
// Also includes:
//   • per-field metrics (nulls, distinct, type tallies, outlier rate via z-score)
//   • schema drift vs baseline (added/removed/type-changed columns)
//
// Everything is optional: pass only what you have. Rows are array of plain objects.

export type Dict<T = any> = { [k: string]: T };

export type ColumnType = "string" | "bool" | "int" | "float" | "date" | "timestamp" | "mixed";
export type Column = { name: string; type?: ColumnType };

export type FieldQualityRule = {
  field: string;

  // Completeness targets
  required?: boolean;                 // if true, target completeness = 1.0
  minCompleteness?: number;           // target 0..1 (default 0 for optional fields)

  // Validity: prefer to specify target type; min/max apply to numbers/dates
  type?: ColumnType;
  allowNull?: boolean;                // default true unless required
  min?: number | string | Date;
  max?: number | string | Date;
  minLen?: number;
  maxLen?: number;
  pattern?: string;                   // JS regex ("/.../flags" or bare)
  enum?: Array<string | number | boolean | null>;

  // Uniqueness (single column)
  unique?: boolean;

  // Outlier detection
  outlierZ?: number;                  // e.g. 3 → flag |z|>3 as outliers for numeric/timestamp

  // Optional display name in messages
  label?: string;
};

export type TableQualityRules = {
  primaryKey?: string[];              // fields that must be unique and non-null
  uniqueTogether?: string[][];        // additional unique combos (each array is a combo)
};

export type FreshnessSpec = {
  timeField: string;                  // field containing date or timestamp
  expectedFrequency?: "daily" | "weekly" | "monthly"; // for continuity/gaps
  maxAgeDays?: number;                // maximum age (days) for the most recent record
  // Optional: tolerate some missing periods (0..1). Example: 0.05 → allow 5% gaps.
  maxGapRate?: number;
};

export type QualityWeights = {
  completeness?: number; // default 0.30
  validity?: number;     // default 0.30
  uniqueness?: number;   // default 0.20
  freshness?: number;    // default 0.20
};

export type QualitySpec = {
  columns?: Column[];
  rules?: {
    fields?: FieldQualityRule[];
    table?: TableQualityRules;
  };
  freshness?: FreshnessSpec;
  weights?: QualityWeights;
  baselineSchema?: Column[]; // optional schema for drift detection
};

export type Issue = {
  code: string;           // e.g., "required", "type", "range", "pattern", "enum", "duplicate_pk", "stale", "gap"
  field?: string;
  row?: number;           // 0-based
  message: string;
  value?: any;
};

export type FieldStats = {
  nulls: number;
  completeness: number;           // 1 - nullRate
  distinct: number;
  types: Partial<Record<ColumnType | "number", number>>;
  outliers?: number;              // count flagged by z-score (if applied)
  validityErrors?: number;        // count of rule violations on this field
};

export type TableStats = {
  rowCount: number;
  duplicateRowCount: number;              // identical across all keys (stringified)
  duplicatePKCount?: number;              // for primary key if provided
  uniqueComboViolations?: Record<string, number>; // key "a,b,c" -> count of dups
};

export type FreshnessStats = {
  lastTimestamp?: string;         // ISO of max date/time
  ageDays?: number;               // days since lastTimestamp (UTC)
  stale?: boolean;                // ageDays > maxAgeDays?
  gapRate?: number;               // fraction of expected periods missing (0..1)
  gaps?: Array<{ from: string; to: string }>; // missing interval summaries
};

export type SchemaDrift = {
  added: string[];
  removed: string[];
  typeChanged: Array<{ name: string; from?: ColumnType; to?: ColumnType }>;
};

export type DimensionScores = {
  completeness: number;  // 0..1
  validity: number;      // 0..1
  uniqueness: number;    // 0..1
  freshness: number;     // 0..1
};

export type QualityReport = {
  ok: boolean;                  // overall score >= 0.9 by default (heuristic)
  score: number;                // 0..100 (weighted)
  scores: DimensionScores;      // per-dimension 0..1
  weights: Required<QualityWeights>;
  issues: Issue[];
  fields: Record<string, FieldStats>;
  table: TableStats;
  freshness?: FreshnessStats;
  schemaDrift?: SchemaDrift;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Entry
 * ────────────────────────────────────────────────────────────────────────── */

export function assessQuality(
  rows: Array<Dict>,
  spec?: QualitySpec
): QualityReport {
  const columns = normalizeColumns(spec?.columns, rows);
  const ruleMap = toRuleMap(spec?.rules?.fields);
  const pk = spec?.rules?.table?.primaryKey || [];
  const uniqCombos = (spec?.rules?.table?.uniqueTogether || []).map(a => a.slice().map(String));
  const weights = withWeights(spec?.weights);
  const issues: Issue[] = [];

  // Precompute field stats & validity errors
  const fieldStats: Record<string, FieldStats> = {};
  for (let i = 0; i < columns.length; i++) {
    fieldStats[columns[i].name] = { nulls: 0, completeness: 0, distinct: 0, types: {} };
  }

  // Distinct tracking, uniqueness trackers, outlier prep
  const distinctSets: Record<string, Set<string>> = {};
  const uniqueSingleSets: Record<string, Set<string>> = {};
  const numericBuffers: Record<string, number[]> = {};
  for (let i = 0; i < columns.length; i++) {
    const name = columns[i].name;
    distinctSets[name] = new Set<string>();
    const r = ruleMap.get(name);
    if (r?.unique) uniqueSingleSets[name] = new Set<string>();
    if (shouldZ(name, ruleMap)) numericBuffers[name] = [];
  }

  // First pass: completeness, distinct, type tallies, collect numeric values
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    for (let c = 0; c < columns.length; c++) {
      const col = columns[c];
      const name = col.name;
      const raw = row?.[name];
      const isNull = valueIsNull(raw, ruleMap.get(name)?.required ? false : true);
      if (isNull) fieldStats[name].nulls++;
      distinctSets[name].add(keyOfValue(raw));

      // type tallies
      const t = classifyValue(raw);
      fieldStats[name].types[t] = (fieldStats[name].types[t] || 0) + 1;

      // collect for outliers
      if (numericBuffers[name] && raw != null) {
        const v = asComparable(raw, ruleMap.get(name)?.type || col.type);
        if (v != null && isFinite(v)) numericBuffers[name].push(v);
      }
    }
  }

  // Compute completeness and distinct counts
  const N = rows.length || 0;
  for (let i = 0; i < columns.length; i++) {
    const name = columns[i].name;
    const st = fieldStats[name];
    st.completeness = N === 0 ? 0 : 1 - st.nulls / Math.max(1, N);
    st.distinct = distinctSets[name].size;
  }

  // Outlier detection (per rule)
  for (const name in numericBuffers) {
    const vals = numericBuffers[name];
    const r = ruleMap.get(name);
    const zThr = (r && typeof r.outlierZ === "number") ? Math.max(0, r.outlierZ) : 0;
    if (zThr > 0 && vals.length >= 5) {
      const mu = mean(vals);
      const sd = stdev(vals);
      if (sd > 0) {
        let outCount = 0;
        for (let i = 0; i < vals.length; i++) {
          const z = (vals[i] - mu) / sd;
          if (Math.abs(z) > zThr) outCount++;
        }
        fieldStats[name].outliers = outCount;
      } else {
        fieldStats[name].outliers = 0;
      }
    }
  }

  // Second pass: validity checks + single-column uniqueness
  let validityViolations = 0;
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];

    for (let c = 0; c < columns.length; c++) {
      const col = columns[c];
      const name = col.name;
      const rule = ruleMap.get(name);
      if (!rule) continue;

      const raw = row?.[name];
      const isNull = valueIsNull(raw, rule.allowNull !== false);

      // required
      if (rule.required && isNull) {
        issues.push(issue("required", name, r, msg(rule, name, "is required"), raw));
        fieldStats[name].validityErrors = (fieldStats[name].validityErrors || 0) + 1;
        validityViolations++;
        continue;
      }
      // allowNull?
      if (isNull) continue;

      // type
      const targetType: ColumnType | undefined = rule.type || col.type;
      if (targetType && targetType !== "mixed") {
        if (!checkType(raw, targetType)) {
          issues.push(issue("type", name, r, msg(rule, name, `must be ${targetType}`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
      }

      // numeric/date bounds
      if (rule.min !== undefined || rule.max !== undefined) {
        const cmp = asComparable(raw, rule.type || col.type);
        const minC = asComparable(rule.min, rule.type || col.type);
        const maxC = asComparable(rule.max, rule.type || col.type);
        if (minC != null && cmp != null && cmp < minC) {
          issues.push(issue("min", name, r, msg(rule, name, `must be ≥ ${fmtBound(rule.min)}`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
        if (maxC != null && cmp != null && cmp > maxC) {
          issues.push(issue("max", name, r, msg(rule, name, `must be ≤ ${fmtBound(rule.max)}`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
      }

      // length
      if (rule.minLen != null || rule.maxLen != null) {
        const s = String(raw);
        if (rule.minLen != null && s.length < rule.minLen) {
          issues.push(issue("min_len", name, r, msg(rule, name, `length must be ≥ ${rule.minLen}`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
        if (rule.maxLen != null && s.length > rule.maxLen) {
          issues.push(issue("max_len", name, r, msg(rule, name, `length must be ≤ ${rule.maxLen}`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
      }

      // pattern
      if (rule.pattern) {
        const re = compileRegex(rule.pattern);
        if (!re.test(String(raw))) {
          issues.push(issue("pattern", name, r, msg(rule, name, `does not match pattern ${re.source}`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
      }

      // enum
      if (rule.enum && rule.enum.length > 0) {
        if (!enumIncludes(rule.enum, raw)) {
          issues.push(issue("enum", name, r, msg(rule, name, `must be one of [${rule.enum.map(fmtBound).join(", ")}]`), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        }
      }

      // single-column unique
      if (rule.unique) {
        const set = uniqueSingleSets[name]!;
        const k = keyOfValue(raw);
        if (set.has(k)) {
          issues.push(issue("unique", name, r, msg(rule, name, "must be unique"), raw));
          inc(fieldStats, name, "validityErrors"); validityViolations++;
        } else set.add(k);
      }
    }
  }

  // Table-level uniqueness
  const tableStats: TableStats = {
    rowCount: N,
    duplicateRowCount: countDuplicateRows(rows)
  };

  if (pk.length) {
    tableStats.duplicatePKCount = countDuplicateCombo(rows, pk);
    if ((tableStats.duplicatePKCount || 0) > 0) {
      issues.push({ code: "duplicate_pk", message: `Primary key not unique: [${pk.join(", ")}] has ${tableStats.duplicatePKCount} duplicate row(s)` });
    }
    // also nulls in PK
    for (let i = 0; i < rows.length; i++) {
      let nullInPk = false;
      for (let j = 0; j < pk.length; j++) if (valueIsNull(rows[i]?.[pk[j]], true)) { nullInPk = true; break; }
      if (nullInPk) {
        issues.push({ code: "pk_null", row: i, message: `Primary key contains null in row ${i}` });
      }
    }
  }
  if (uniqCombos.length) {
    tableStats.uniqueComboViolations = {};
    for (let u = 0; u < uniqCombos.length; u++) {
      const combo = uniqCombos[u];
      const d = countDuplicateCombo(rows, combo);
      if (d > 0) {
        tableStats.uniqueComboViolations[combo.join(",")] = d;
        issues.push({ code: "unique_together", message: `Combination must be unique: [${combo.join(", ")}] has ${d} duplicate row(s)` });
      }
    }
  }

  // Freshness
  let freshnessStats: FreshnessStats | undefined;
  if (spec?.freshness?.timeField) {
    freshnessStats = calcFreshness(rows, spec.freshness);
    if (freshnessStats.stale) {
      issues.push({ code: "stale", field: spec.freshness.timeField, message: `Data is stale: age ${freshnessStats.ageDays} days exceeds max ${spec.freshness.maxAgeDays}` });
    }
    if (typeof freshnessStats.gapRate === "number" && spec.freshness.maxGapRate != null && freshnessStats.gapRate > spec.freshness.maxGapRate) {
      issues.push({ code: "gap", field: spec.freshness.timeField, message: `Gap rate ${pct(freshnessStats.gapRate)} exceeds max ${pct(spec.freshness.maxGapRate)}` });
    }
  }

  // Schema drift
  let drift: SchemaDrift | undefined;
  if (spec?.baselineSchema && spec.baselineSchema.length) {
    drift = schemaDrift(spec.baselineSchema, columns);
    if (drift.added.length) issues.push({ code: "schema_added", message: `New columns: ${drift.added.join(", ")}` });
    if (drift.removed.length) issues.push({ code: "schema_removed", message: `Removed columns: ${drift.removed.join(", ")}` });
    if (drift.typeChanged.length) {
      const txt = drift.typeChanged.map(x => `${x.name} (${x.from}→${x.to})`).join(", ");
      issues.push({ code: "schema_type_change", message: `Type changes: ${txt}` });
    }
  }

  // Dimension scoring (0..1)
  const completenessScore = scoreCompleteness(columns, fieldStats, ruleMap);
  const validityScore = scoreValidity(columns, fieldStats, validityViolations, N);
  const uniquenessScore = scoreUniqueness(tableStats, N, pk.length > 0);
  const freshnessScore = scoreFreshness(freshnessStats);

  const dim: DimensionScores = {
    completeness: completenessScore,
    validity: validityScore,
    uniqueness: uniquenessScore,
    freshness: freshnessScore
  };

  const overall01 =
    weights.completeness * dim.completeness +
    weights.validity * dim.validity +
    weights.uniqueness * dim.uniqueness +
    weights.freshness * dim.freshness;

  return {
    ok: overall01 >= 0.9,
    score: Math.round(overall01 * 1000) / 10, // 1 decimal
    scores: dim,
    weights,
    issues,
    fields: fieldStats,
    table: tableStats,
    freshness: freshnessStats,
    schemaDrift: drift
  };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Scoring helpers
 * ────────────────────────────────────────────────────────────────────────── */

function scoreCompleteness(cols: Column[], stats: Record<string, FieldStats>, rules: Map<string, FieldQualityRule>): number {
  if (cols.length === 0) return 0;
  let s = 0, w = 0;
  for (let i = 0; i < cols.length; i++) {
    const name = cols[i].name;
    const r = rules.get(name);
    const target = r?.required ? 1 : (r?.minCompleteness != null ? clamp01(r!.minCompleteness!) : 0);
    const c = stats[name]?.completeness ?? 0;
    // Score contribution: c relative to target; if c >= target, full credit
    const contrib = target === 0 ? c : clamp01(c / Math.max(1e-9, target));
    // Weight required fields higher
    const ww = r?.required ? 2 : 1;
    s += contrib * ww;
    w += ww;
  }
  return w ? clamp01(s / w) : 0;
}

function scoreValidity(cols: Column[], stats: Record<string, FieldStats>, totalViolations: number, N: number): number {
  if (N === 0) return 0;
  // Per-field validity errors normalized by rows
  let errRate = 0;
  for (let i = 0; i < cols.length; i++) {
    const name = cols[i].name;
    const v = stats[name]?.validityErrors || 0;
    errRate += v;
  }
  // also include collected totalViolations (same count)
  const totalErr = Math.max(errRate, totalViolations);
  const rate = clamp01(totalErr / Math.max(1, N * Math.max(1, cols.length * 0.3))); // heuristically scale
  return clamp01(1 - rate);
}

function scoreUniqueness(t: TableStats, N: number, hasPK: boolean): number {
  if (N === 0) return 0;
  const dupRows = (t.duplicateRowCount || 0) / N;
  const dupPk = hasPK ? ((t.duplicatePKCount || 0) / N) : 0;
  // Penalty if many duplicates; weight PK more if present
  const penalty = hasPK ? (0.7 * dupPk + 0.3 * dupRows) : dupRows;
  return clamp01(1 - penalty);
}

function scoreFreshness(f: FreshnessStats | undefined): number {
  if (!f) return 1; // if not specified, don't penalize
  let s = 1;
  if (typeof f.ageDays === "number" && typeof f.stale === "boolean") {
    s *= f.stale ? 0 : 1;
  }
  if (typeof f.gapRate === "number") {
    s *= clamp01(1 - f.gapRate);
  }
  return clamp01(s);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Freshness
 * ────────────────────────────────────────────────────────────────────────── */

function calcFreshness(rows: Array<Dict>, spec: FreshnessSpec): FreshnessStats {
  const tf = spec.timeField;
  const times: number[] = [];
  for (let i = 0; i < rows.length; i++) {
    const v = rows[i]?.[tf];
    if (v == null) continue;
    const t = toEpoch(v);
    if (t != null) times.push(t);
  }
  if (times.length === 0) return { ageDays: undefined, stale: false };
  times.sort((a, b) => a - b);
  const last = times[times.length - 1];
  const ageDays = Math.max(0, Math.floor((Date.now() - last) / DAY_MS));
  const stale = spec.maxAgeDays != null ? ageDays > Math.max(0, Math.floor(spec.maxAgeDays)) : false;

  // continuity/gap rate
  let gapRate: number | undefined = undefined;
  let gaps: Array<{ from: string; to: string }> = [];
  if (spec.expectedFrequency) {
    const step = freqStepDays(spec.expectedFrequency);
    if (step > 0 && times.length > 1) {
      let expected = 0;
      let missing = 0;
      for (let i = 1; i < times.length; i++) {
        const dd = Math.round((times[i] - times[i - 1]) / DAY_MS);
        if (dd <= 0) continue;
        expected += Math.max(0, Math.round(dd / step));
        const miss = Math.max(0, Math.round(dd / step) - 1);
        if (miss > 0) {
          missing += miss;
          gaps.push({ from: new Date(times[i - 1]).toISOString(), to: new Date(times[i]).toISOString() });
        }
      }
      if (expected > 0) gapRate = clamp01(missing / expected);
      else gapRate = 0;
    } else {
      gapRate = 0;
    }
  }

  return {
    lastTimestamp: new Date(last).toISOString(),
    ageDays,
    stale,
    gapRate,
    gaps
  };
}

function freqStepDays(f: NonNullable<FreshnessSpec["expectedFrequency"]>): number {
  if (f === "daily") return 1;
  if (f === "weekly") return 7;
  if (f === "monthly") return 30; // coarse heuristic
  return 0;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Schema drift
 * ────────────────────────────────────────────────────────────────────────── */

function schemaDrift(baseline: Column[], current: Column[]): SchemaDrift {
  const bmap: Dict<Column> = {};
  const cmap: Dict<Column> = {};
  for (let i = 0; i < baseline.length; i++) bmap[baseline[i].name] = baseline[i];
  for (let i = 0; i < current.length; i++) cmap[current[i].name] = current[i];

  const added: string[] = [];
  const removed: string[] = [];
  const typeChanged: Array<{ name: string; from?: ColumnType; to?: ColumnType }> = [];

  for (const name in cmap) if (!bmap[name]) added.push(name);
  for (const name in bmap) if (!cmap[name]) removed.push(name);
  for (const name in bmap) {
    const b = bmap[name], c = cmap[name];
    if (c && normalizeType(b.type) !== normalizeType(c.type)) {
      typeChanged.push({ name, from: normalizeType(b.type), to: normalizeType(c.type) });
    }
  }
  return { added, removed, typeChanged };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Utilities: columns, rules, typing, comparisons
 * ────────────────────────────────────────────────────────────────────────── */

function normalizeColumns(cols: Column[] | undefined, rows: Array<Dict>): Column[] {
  if (Array.isArray(cols) && cols.length) {
    return cols.map(c => ({ name: String(c.name), type: normalizeType(c.type) }));
  }
  // infer from first non-empty row
  const names = new Set<string>();
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (r && typeof r === "object") { for (const k in r) names.add(k); }
    if (names.size) break;
  }
  const out: Column[] = [];
  const arr = Array.from(names);
  for (let i = 0; i < arr.length; i++) {
    const name = arr[i];
    const t = inferTypeFromSamples(rows.slice(0, 50).map(ro => ro?.[name]));
    out.push({ name, type: t });
  }
  return out;
}

function toRuleMap(rules?: FieldQualityRule[]): Map<string, FieldQualityRule> {
  const m = new Map<string, FieldQualityRule>();
  if (!rules) return m;
  for (let i = 0; i < rules.length; i++) {
    const r = { ...rules[i] };
    r.field = String(r.field);
    if (r.type) r.type = normalizeType(r.type);
    if (typeof r.allowNull !== "boolean") r.allowNull = !r.required;
    m.set(r.field, r);
  }
  return m;
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

function valueIsNull(v: any, emptyIsNull: boolean): boolean {
  return v === null || v === undefined || (emptyIsNull && (v === "" || (typeof v === "string" && v.trim() === "")));
}

function classifyValue(v: any): ColumnType | "number" {
  if (v === null || v === undefined) return "mixed";
  if (typeof v === "boolean") return "bool";
  if (typeof v === "number") return Number.isInteger(v) ? "int" : "float";
  if (v instanceof Date) return "timestamp";
  const s = String(v).trim();
  if (looksTimestamp(s)) return "timestamp";
  if (looksDate(s)) return "date";
  if (looksInt(s)) return "int";
  if (looksFloat(s)) return "float";
  return "string";
}

function inferTypeFromSamples(values: any[]): ColumnType {
  let sb = false, si = false, sf = false, sd = false, st = false, ss = false;
  for (let i = 0; i < values.length; i++) {
    const t = classifyValue(values[i]);
    if (t === "bool") sb = true;
    else if (t === "int") si = true;
    else if (t === "float") sf = true;
    else if (t === "date") sd = true;
    else if (t === "timestamp") st = true;
    else if (t === "string") ss = true;
  }
  return ss ? "string" : st ? "timestamp" : sd ? "date" : sf ? "float" : si ? "int" : sb ? "bool" : "mixed";
}

function checkType(v: any, t: ColumnType): boolean {
  if (v === null || v === undefined) return true;
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
  const s = String(v).trim();
  if (looksFloat(s) || looksInt(s)) return Number(s.replace(/[_ ,]/g, ""));
  return null;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Small parsing helpers
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
function toEpoch(v: any): number | null {
  if (typeof v === "number" && isFinite(v)) return v > 1e12 ? v : v * 1000; // tolerate seconds
  if (v instanceof Date) return v.getTime();
  const s = String(v);
  const iso = looksTimestamp(s) ? normalizeISO(s) : looksDate(s) ? normalizeDate(s) : null;
  return iso ? Date.parse(iso) : null;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Misc helpers
 * ────────────────────────────────────────────────────────────────────────── */

function countDuplicateRows(rows: Array<Dict>): number {
  const seen = new Set<string>();
  const dups = new Set<string>();
  for (let i = 0; i < rows.length; i++) {
    const k = keyOfRow(rows[i]);
    if (seen.has(k)) dups.add(k); else seen.add(k);
  }
  return dups.size; // number of distinct duplicate rows
}

function countDuplicateCombo(rows: Array<Dict>, fields: string[]): number {
  const seen = new Set<string>();
  const dups = new Set<string>();
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const parts: string[] = [];
    for (let j = 0; j < fields.length; j++) parts.push(keyOfValue(row?.[fields[j]]));
    const k = parts.join("¦");
    if (seen.has(k)) dups.add(k); else seen.add(k);
  }
  return dups.size;
}

function keyOfRow(r: Dict): string {
  const keys = Object.keys(r).sort();
  const obj: any = {};
  for (let i = 0; i < keys.length; i++) obj[keys[i]] = r[keys[i]];
  try { return JSON.stringify(obj); } catch { return String(obj); }
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
  const m = pat.match(/^\/(.*)\/([gimsuy]*)$/);
  if (m) return new RegExp(m[1], m[2] || "i");
  return new RegExp(pat, "i");
}

function enumIncludes(arr: any[], v: any): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (deepEq(arr[i], v)) return true;
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
    for (let i = 0; i < ak.length; i++) if (!deepEq(a[ak[i]], b[ak[i]])) return false;
    return true;
  }
  return false;
}

function msg(rule: FieldQualityRule | undefined, field: string, m: string): string {
  const lbl = rule?.label || field;
  return `${lbl}: ${m}`;
}
function issue(code: string, field: string | undefined, row: number | undefined, message: string, value?: any): Issue {
  return { code, field, row, message, value };
}
function inc(fs: Record<string, FieldStats>, name: string, key: "validityErrors") {
  fs[name][key] = (fs[name][key] || 0) + 1;
}

function mean(xs: number[]): number { if (!xs.length) return NaN; let s = 0; for (let i = 0; i < xs.length; i++) s += xs[i]; return s / xs.length; }
function stdev(xs: number[]): number { if (xs.length < 2) return 0; const m = mean(xs); let v = 0; for (let i = 0; i < xs.length; i++) { const d = xs[i] - m; v += d * d; } return Math.sqrt(v / (xs.length - 1)); }

function withWeights(w?: QualityWeights): Required<QualityWeights> {
  const defaults = { completeness: 0.30, validity: 0.30, uniqueness: 0.20, freshness: 0.20 };
  const x = { ...defaults, ...(w || {}) };
  // normalize to sum 1
  const sum = (x.completeness || 0) + (x.validity || 0) + (x.uniqueness || 0) + (x.freshness || 0);
  if (sum <= 0) return defaults;
  return {
    completeness: x.completeness / sum,
    validity: x.validity / sum,
    uniqueness: x.uniqueness / sum,
    freshness: x.freshness / sum
  };
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
function pad2(n: number): string { return String(n).padStart(2, "0"); }
function fmtBound(v: any): string {
  if (v instanceof Date) return v.toISOString();
  if (typeof v === "string") {
    const nd = normalizeDate(v) || normalizeISO(v);
    if (nd) return nd;
  }
  return String(v);
}
function pct(x: number): string { return `${Math.round(clamp01(x) * 1000) / 10}%`; }
function shouldZ(name: string, rules: Map<string, FieldQualityRule>): boolean {
  const r = rules.get(name);
  return !!(r && typeof r.outlierZ === "number" && r.outlierZ > 0);
}

const DAY_MS = 24 * 60 * 60 * 1000;

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */