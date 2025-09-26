// src/types.ts
// Central, zero-dependency type definitions used across the tiny data toolkit.
// Pure types only — no imports, no runtime code.

export type Dict<T = any> = { [k: string]: T };

/* ──────────────────────────────────────────────────────────────────────────
 * Reader / Parser
 * ────────────────────────────────────────────────────────────────────────── */

export type ReadOptions = {
  encoding?: "utf-8" | "utf8";
  maxBytes?: number;
  timeoutMs?: number;
  asTextHint?: boolean;
};

export type ReadResult = {
  ok: boolean;
  sourceType?: "url" | "path" | "dataurl" | "blob" | "bytes" | "text" | "stream";
  bytes?: Uint8Array;
  text?: string;
  contentType?: string;
  filename?: string;
  size?: number;
  meta?: Dict;
  error?: string;
};

export type ColumnType =
  | "string"
  | "bool"
  | "int"
  | "float"
  | "date"
  | "timestamp"
  | "mixed";

export type Column = {
  name: string;
  type: ColumnType;
};

export type ParseOptions = {
  format?: "auto" | "json" | "ndjson" | "csv" | "tsv" | "yaml";
  filename?: string;
  delimiter?: string;
  header?: boolean;
  comment?: string;
  skipEmpty?: boolean;
  autoType?: boolean;
  nulls?: string[];
  sampleSize?: number;
};

export type ParseResult = {
  ok: boolean;
  format?: "json" | "ndjson" | "csv" | "tsv" | "yaml";
  rows?: any[];
  columns?: Column[];
  meta?: Dict;
  error?: string;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Loader registry
 * ────────────────────────────────────────────────────────────────────────── */

export type RegistryContext = {
  read: (src: any, opts?: ReadOptions) => Promise<ReadResult>;
  parse: (
    input: string | Uint8Array | ArrayBuffer,
    opts?: ParseOptions
  ) => ParseResult;
};

export type LoadResult = {
  ok: boolean;
  id?: string;
  format?: ParseResult["format"];
  rows?: any[];
  columns?: Column[];
  meta?: Dict;
  error?: string;
};

export type Loader = {
  id: string;
  match: (
    src: any,
    hint: { filename?: string; contentType?: string; sourceType?: string }
  ) => number;
  load: (
    src: any,
    ctx: RegistryContext,
    hint: { filename?: string; contentType?: string; sourceType?: string },
    opts?: { read?: ReadOptions; parse?: ParseOptions }
  ) => Promise<LoadResult>;
};

export type LoadOptions = { read?: ReadOptions; parse?: ParseOptions };

export type Registry = {
  context(): RegistryContext;
  list(): Loader[];
  register(loader: Loader): void;
  unregister(id: string): void;
  resolve(
    src: any,
    hint?: { filename?: string; contentType?: string; sourceType?: string }
  ): Loader;
  load(src: any, opts?: LoadOptions): Promise<LoadResult>;
  loadAll(sources: any[], opts?: LoadOptions): Promise<LoadResult[]>;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Resolver (classification only; no FS required)
 * ────────────────────────────────────────────────────────────────────────── */

export type ResolveOptions = {
  classifyStrings?: boolean; // default true
};

export type ResolvedSourceHint = {
  filename?: string;
  contentType?: string;
  sourceType?: "url" | "dataurl" | "path" | "blob" | "bytes" | "text" | "stream";
  path?: string;
};

export type ResolvedSource = {
  src: any;
  hint: ResolvedSourceHint;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Query / Select
 * ────────────────────────────────────────────────────────────────────────── */

export type Row = Dict<any>;

export type Expr<T = any> = (row: Row, index: number, rows: Row[]) => T;
export type Pred = (row: Row, index: number, rows: Row[]) => boolean;

export type ColumnSpec =
  | string
  | Expr<any>
  | { as: string; expr: Expr<any> }
  | [string, Expr<any>];

export type OrderKey =
  | string
  | Expr<any>
  | { key: string | Expr<any>; desc?: boolean; nulls?: "first" | "last" };

export type DistinctSpec = boolean | string[] | ((row: Row) => any | any[]);

export type SelectOptions = {
  from: Row[];
  where?: Pred;
  columns?: ColumnSpec[];
  distinct?: DistinctSpec;
  orderBy?: OrderKey[];
  limit?: number;
  offset?: number;
};

export type GroupKeySpec = string | Expr<any>;

export type GroupOptions = {
  from: Row[];
  groupBy: GroupKeySpec[];
  aggregates: Record<
    string,
    {
      fn:
        | "count"
        | "sum"
        | "avg"
        | "min"
        | "max"
        | "first"
        | "last"
        | "median"
        | "std"
        | "var";
      expr?: Expr<any>;
    }
  >;
  having?: (groupRow: Row, rows: Row[]) => boolean;
  orderBy?: OrderKey[];
  limit?: number;
  offset?: number;
};

export type Group = { key: any[]; rows: Row[] };
export type GroupIndex = { [key: string]: Group };

/* ──────────────────────────────────────────────────────────────────────────
 * Validator
 * ────────────────────────────────────────────────────────────────────────── */

export type FieldRule = {
  field: string;
  required?: boolean;
  allowNull?: boolean;
  type?: ColumnType;
  min?: number | string | Date;
  max?: number | string | Date;
  minLen?: number;
  maxLen?: number;
  pattern?: string;
  enum?: Array<string | number | boolean | null>;
  unique?: boolean;
  label?: string;
};

export type TableRule =
  | { kind: "primaryKey"; fields: string[] }
  | { kind: "uniqueTogether"; fields: string[] }
  | {
      kind: "monotonic";
      field: string;
      order: "increasing" | "decreasing" | "nondecreasing" | "nonincreasing";
    };

export type ValidationSpec = {
  fields?: FieldRule[];
  table?: TableRule[];
};

export type ValidateOptions = {
  maxErrors?: number;
  treatEmptyStringAsNull?: boolean;
};

export type FieldSummary = {
  nulls: number;
  distinct: number;
  types: Partial<Record<ColumnType | "number", number>>;
};

export type ValidationIssue = {
  row: number;
  field?: string;
  code: string;
  message: string;
  value?: any;
};

export type ValidationReport = {
  ok: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  stats: {
    rowCount: number;
    fields: Record<string, FieldSummary>;
  };
};

/* ──────────────────────────────────────────────────────────────────────────
 * Data Quality
 * ────────────────────────────────────────────────────────────────────────── */

export type FieldQualityRule = {
  field: string;
  required?: boolean;
  minCompleteness?: number;
  type?: ColumnType;
  allowNull?: boolean;
  min?: number | string | Date;
  max?: number | string | Date;
  minLen?: number;
  maxLen?: number;
  pattern?: string;
  enum?: Array<string | number | boolean | null>;
  unique?: boolean;
  outlierZ?: number;
  label?: string;
};

export type TableQualityRules = {
  primaryKey?: string[];
  uniqueTogether?: string[][];
};

export type FreshnessSpec = {
  timeField: string;
  expectedFrequency?: "daily" | "weekly" | "monthly";
  maxAgeDays?: number;
  maxGapRate?: number;
};

export type QualityWeights = {
  completeness?: number;
  validity?: number;
  uniqueness?: number;
  freshness?: number;
};

export type QualitySpec = {
  columns?: Column[];
  rules?: { fields?: FieldQualityRule[]; table?: TableQualityRules };
  freshness?: FreshnessSpec;
  weights?: QualityWeights;
  baselineSchema?: Column[];
};

export type FieldStats = {
  nulls: number;
  completeness: number;
  distinct: number;
  types: Partial<Record<ColumnType | "number", number>>;
  outliers?: number;
  validityErrors?: number;
};

export type TableStats = {
  rowCount: number;
  duplicateRowCount: number;
  duplicatePKCount?: number;
  uniqueComboViolations?: Record<string, number>;
};

export type FreshnessStats = {
  lastTimestamp?: string;
  ageDays?: number;
  stale?: boolean;
  gapRate?: number;
  gaps?: Array<{ from: string; to: string }>;
};

export type SchemaDrift = {
  added: string[];
  removed: string[];
  typeChanged: Array<{ name: string; from?: ColumnType; to?: ColumnType }>;
};

export type DimensionScores = {
  completeness: number;
  validity: number;
  uniqueness: number;
  freshness: number;
};

export type QualityIssue = {
  code: string;
  field?: string;
  row?: number;
  message: string;
  value?: any;
};

export type QualityReport = {
  ok: boolean;
  score: number; // 0..100
  scores: DimensionScores;
  weights: Required<QualityWeights>;
  issues: QualityIssue[];
  fields: Record<string, FieldStats>;
  table: TableStats;
  freshness?: FreshnessStats;
  schemaDrift?: SchemaDrift;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Lineage graph
 * ────────────────────────────────────────────────────────────────────────── */

export type NodeKind =
  | "dataset"
  | "field"
  | "table"
  | "view"
  | "column"
  | "provider"
  | "job"
  | "task"
  | "metric"
  | "model"
  | "source"
  | "sink"
  | "other";

export type EdgeKind =
  | "derives"
  | "reads"
  | "writes"
  | "joins"
  | "aggregates"
  | "transforms"
  | "maps"
  | "filters"
  | "loads"
  | "exports"
  | "depends";

export type LineageNode = {
  id: string;
  kind?: NodeKind;
  label?: string;
  tags?: string[];
  meta?: Dict;
};

export type LineageEdge = {
  from: string;
  to: string;
  type?: EdgeKind | string;
  label?: string;
  meta?: Dict;
};

export type LineageGraph = {
  nodes: Dict<LineageNode>;
  edges: LineageEdge[];
  outAdj: Dict<string[]>;
  inAdj: Dict<string[]>;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Catalog (lightweight types mirroring dataset.schema.json)
 * ────────────────────────────────────────────────────────────────────────── */

export type CatalogPrimitiveType =
  | "string"
  | "text"
  | "int"
  | "bigint"
  | "float"
  | "double"
  | "decimal"
  | "bool"
  | "boolean"
  | "date"
  | "timestamp"
  | "json"
  | "bytes";

export type CatalogType =
  | CatalogPrimitiveType
  | { array: CatalogType }
  | { map: { key: CatalogType; value: CatalogType } };

export type CatalogField = {
  name: string;
  type: CatalogType;
  description?: string;
  nullable?: boolean;
  unit?: string;
  semantic_type?:
    | "id"
    | "code"
    | "category"
    | "text"
    | "bool"
    | "date"
    | "timestamp"
    | "price"
    | "rate"
    | "quantity"
    | "json"
    | "other";
  format?: string;
  example?: string | number | boolean | null;
  enum?: Array<string | number | boolean | null>;
  min?: number | string;
  max?: number | string;
  unique?: boolean;
  index?: boolean;
  foreign_key?: { dataset?: string; field?: string };
  tags?: string[];
};

export type CatalogOwner = {
  name: string;
  email?: string;
  team?: string;
  slack?: string;
};

export type CatalogStorage = {
  system?:
    | "s3"
    | "gcs"
    | "azure"
    | "hdfs"
    | "local"
    | "bigquery"
    | "snowflake"
    | "redshift"
    | "postgres"
    | "duckdb";
  catalog?: string;
  database?: string;
  schema?: string;
  table?: string;
  bucket?: string;
  path?: string;
  region?: string;
};

export type CatalogPartition = {
  name: string;
  type?: "by_time" | "by_value" | "by_hash";
  expr?: string;
  granularity?: "hour" | "day" | "week" | "month" | "year";
};

export type CatalogSource = {
  provider?: "bloomberg" | "koyfin" | "refinitiv" | "yahoo" | "custom" | "internal" | string;
  symbol?: string;
  query?: string;
  params?: Dict;
  license?: string;
};

export type DatasetDescriptor = {
  id: string;
  title: string;
  description?: string;
  domain?: string;
  tags?: string[];
  owner?: CatalogOwner;
  visibility?: "private" | "internal" | "public";
  status?: "active" | "deprecated" | "experimental";
  version?: string;
  created_at?: string; // ISO
  updated_at?: string; // ISO
  fields: CatalogField[];
  primary_key?: string[];
  grain?: string[];
  constraints?: Array<
    | { kind: "check"; expr: string }
    | { kind: "unique"; fields: string[] }
    | { kind: "primary_key"; fields: string[] }
    | { kind: "foreign_key"; fields: string[]; reference?: { dataset?: string; fields?: string[] } }
  >;
  partitions?: CatalogPartition[];
  format?: "csv" | "tsv" | "parquet" | "orc" | "json" | "jsonl" | "delta" | "iceberg" | "bigquery" | "table";
  compression?: "none" | "gzip" | "snappy" | "zstd" | "bzip2" | "lz4";
  storage?: CatalogStorage;
  path?: string;
  sources?: CatalogSource[];
  lineage?: {
    inputs?: string[];
    outputs?: string[];
    transform?: string;
  };
  freshness?: {
    time_field: string;
    expected_frequency?: "event" | "hourly" | "daily" | "weekly" | "monthly" | "quarterly";
    max_age_days?: number;
    timezone?: string;
  };
  schedule?: string; // cron
  sla?: { max_lag_minutes?: number; availability?: number };
  expectations?: Array<{ field: string; check: string; params?: Dict; severity?: "error" | "warn" }>;
  quality?: {
    targets?: { completeness?: number; validity?: number; freshness?: number; uniqueness?: number };
    monitors?: string[];
  };
  docs?: string;
  sample?: { path?: string; rows?: number };
  license?: string;
};