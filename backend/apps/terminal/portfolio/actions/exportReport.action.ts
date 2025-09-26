// actions/exportreport.action.ts
// Rewritten: stricter typing, no undefineds in normalized config, no imports.
// Server Action to export rows as CSV/JSON (flattening objects, BOM for Excel, etc).

"use server";

/* ======================== Types ======================== */

type Format = "csv" | "json";

export type ExportInput = {
  rows?: Array<Record<string, any>> | readonly Record<string, any>[];
  format?: Format;            // default "csv"
  filename?: string;          // base name; extension auto-added
  fields?: string[];          // CSV column order; if omitted, inferred
  delimiter?: string;         // default ","
  includeHeader?: boolean;    // default true
  bom?: boolean;              // CSV BOM; default true
  nullAsEmpty?: boolean;      // default true
  flatten?: boolean;          // default true
};

type Normalized = {
  rows: Record<string, any>[];
  format: Format;
  filename: string;           // always present (no extension)
  fields: string[];           // may be []
  delimiter: string;
  includeHeader: boolean;
  bom: boolean;
  nullAsEmpty: boolean;
  flatten: boolean;
};

export type ExportResult =
  | {
      ok: true;
      filename: string;
      mime: "text/csv" | "application/json";
      contents: string; // UTF-8 text
      meta: { rows: number; columns?: string[]; bytes: number; generatedAt: string };
    }
  | { ok: false; error: string; fieldErrors?: Record<string, string> };

/* ======================== Action ======================== */

export async function exportReportAction(input: ExportInput | FormData): Promise<ExportResult> {
  try {
    const cfg = normalizeInput(input);

    const v = validate(cfg);
    if (!v.valid) return { ok: false, error: "Validation failed", fieldErrors: v.errors };

    const stamp = fmtTs(new Date());
    const base = cfg.filename ? sanitizeFilename(cfg.filename) : `report_${stamp}`;

    if (cfg.format === "json") {
      const text = jsonPretty(cfg.rows);
      const filename = ensureExt(base, ".json");
      return ok(filename, "application/json", text, cfg.rows.length);
    }

    // CSV
    const rows = cfg.flatten ? cfg.rows.map((r) => flatten(r)) : cfg.rows.slice();
    const columns = cfg.fields.length ? cfg.fields : inferColumns(rows);
    const csv = toCSV(rows, columns, {
      delimiter: cfg.delimiter,
      header: cfg.includeHeader,
      nullAsEmpty: cfg.nullAsEmpty,
    });
    const withBom = (cfg.bom ? "\uFEFF" : "") + csv;
    const filename = ensureExt(base, ".csv");
    return ok(filename, "text/csv", withBom, rows.length, columns);
  } catch (e: any) {
    return { ok: false, error: e?.message || "Export failed" };
  }
}

/* ======================== Parsing ======================== */

function normalizeInput(input: ExportInput | FormData): Normalized {
  if (isFormData(input)) {
    const fd = input as FormData;

    const rowsRaw = str(fd.get("rows")) || "[]";
    const parsed = safeJson(rowsRaw);
    const rows = Array.isArray(parsed) ? (parsed as Record<string, any>[]) : [];

    const format = ((str(fd.get("format")) as Format) || "csv") as Format;
    const filename = str(fd.get("filename")) || "";
    const fieldsCsv = str(fd.get("fields"));
    const fields = fieldsCsv ? fieldsCsv.split(",").map((t) => t.trim()).filter(Boolean) : [];

    return {
      rows,
      format,
      filename,
      fields,
      delimiter: str(fd.get("delimiter")) || ",",
      includeHeader: boolish(fd.get("includeHeader"), true),
      bom: boolish(fd.get("bom"), true),
      nullAsEmpty: boolish(fd.get("nullAsEmpty"), true),
      flatten: boolish(fd.get("flatten"), true),
    };
  }

  const obj = input as ExportInput;

  return {
    rows: Array.isArray(obj.rows) ? (obj.rows as Record<string, any>[]) : [],
    format: (obj.format as Format) || "csv",
    filename: obj.filename || "",
    fields: Array.isArray(obj.fields) ? obj.fields.filter(Boolean) : [],
    delimiter: obj.delimiter || ",",
    includeHeader: obj.includeHeader !== false,
    bom: obj.bom !== false,
    nullAsEmpty: obj.nullAsEmpty !== false,
    flatten: obj.flatten !== false,
  };
}

/* ======================== Core ======================== */

function toCSV(
  rows: Record<string, any>[],
  columns: string[],
  opts: { delimiter: string; header: boolean; nullAsEmpty: boolean },
): string {
  const D = opts.delimiter;

  const esc = (v: any): string => {
    if (v == null) return opts.nullAsEmpty ? "" : String(v);
    const s =
      typeof v === "string"
        ? v
        : typeof v === "number" || typeof v === "boolean"
        ? String(v)
        : JSON.stringify(v);
    const needsWrap = /[",\n\r]/.test(s) || s.includes(D) || /^\s|\s$/.test(s);
    return needsWrap ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const head = opts.header ? columns.map(esc).join(D) + "\n" : "";
  const body = rows.map((r) => columns.map((c) => esc(r[c])).join(D)).join("\n");
  return head + body;
}

function inferColumns(rows: Record<string, any>[]): string[] {
  const set = new Set<string>();
  if (rows[0]) Object.keys(rows[0]).forEach((k) => set.add(k));
  for (const r of rows) Object.keys(r).forEach((k) => set.add(k));
  return Array.from(set);
}

function flatten(obj: any, prefix = "", out: Record<string, any> = {}): Record<string, any> {
  if (obj == null || typeof obj !== "object") {
    out[prefix || "value"] = obj;
    return out;
  }
  if (Array.isArray(obj)) {
    out[prefix || "value"] = JSON.stringify(obj);
    return out;
  }
  for (const key of Object.keys(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const v = obj[key];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      flatten(v, path, out);
    } else if (Array.isArray(v)) {
      out[path] = JSON.stringify(v);
    } else {
      out[path] = v;
    }
  }
  return out;
}

/* ======================== Validation ======================== */

function validate(cfg: Normalized): { valid: boolean; errors: Record<string, string> } {
  const errors: Record<string, string> = {};
  if (!Array.isArray(cfg.rows)) errors.rows = "rows must be an array";
  if (cfg.format !== "csv" && cfg.format !== "json") errors.format = "format must be csv or json";
  if (cfg.format === "csv" && typeof cfg.delimiter !== "string") errors.delimiter = "delimiter must be a string";
  return { valid: Object.keys(errors).length === 0, errors };
}

/* ======================== Small utils ======================== */

function ok(
  filename: string,
  mime: "text/csv" | "application/json",
  contents: string,
  rows: number,
  columns?: string[],
): ExportResult {
  return {
    ok: true,
    filename,
    mime,
    contents,
    meta: {
      rows,
      columns,
      bytes: utf8Length(contents),
      generatedAt: new Date().toISOString(),
    },
  };
}

function isFormData(x: any): x is FormData {
  return typeof x === "object" && x?.constructor?.name === "FormData";
}

function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}

function boolish(v: any, fallback: boolean): boolean {
  if (v == null || v === "") return fallback;
  const s = String(v).toLowerCase();
  if (["1", "true", "yes", "y"].includes(s)) return true;
  if (["0", "false", "no", "n"].includes(s)) return false;
  return fallback;
}

function safeJson(s: string): any {
  try {
    return JSON.parse(s);
  } catch {
    return [];
  }
}

function jsonPretty(x: any): string {
  const seen = new WeakSet();
  const rep = (_k: string, v: any) => {
    if (typeof v === "object" && v !== null) {
      if (seen.has(v)) return "[Circular]";
      seen.add(v);
    }
    return v;
  };
  return JSON.stringify(x, rep, 2);
}

function ensureExt(name: string, ext: ".csv" | ".json"): string {
  return name.toLowerCase().endsWith(ext) ? name : name + ext;
}

function sanitizeFilename(s: string): string {
  return s.replace(/[\\/:*?"<>|]+/g, "_").slice(0, 120);
}

function fmtTs(d: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function utf8Length(s: string): number {
  let bytes = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x80) bytes += 1;
    else if (c < 0x800) bytes += 2;
    else if (c >= 0xd800 && c <= 0xdbff) {
      i++;
      bytes += 4;
    } else bytes += 3;
  }
  return bytes;
}

/* ======================== Convenience ======================== */

export async function exportCSV(rows: Record<string, any>[], filename?: string): Promise<ExportResult> {
  return exportReportAction({ rows, format: "csv", filename });
}

export async function exportJSON(rows: Record<string, any>[], filename?: string): Promise<ExportResult> {
  return exportReportAction({ rows, format: "json", filename });
}
