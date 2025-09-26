// loader/parser.ts
// Pure TypeScript parsers for JSON, NDJSON, CSV/TSV, and small-subset YAML.
// No imports. Safe in Node or browser.
//
// Exported API:
//   - parse(input, options?) -> { ok, rows?, format, columns?, meta?, error? }
//   - parseCSV / parseNDJSON / parseJSON / parseYAML (direct use if needed)
//
// Notes:
//   • CSV parser supports quotes, escaped quotes, newlines in quotes, comments.
//   • Auto-delimiter sniff: ',', '\t', ';', '|'.
//   • Header inference if options.header is undefined.
//   • Auto-typing (number, bool, null, date/timestamp as strings) unless disabled.
//   • YAML parser handles common config shapes (maps, lists of scalars/objects).

/* ────────────────────────────────────────────────────────────────────────── *
 * Types
 * ────────────────────────────────────────────────────────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type ParseOptions = {
  format?: "auto" | "json" | "ndjson" | "csv" | "tsv" | "yaml";
  filename?: string;
  delimiter?: string;           // for CSV/TSV (overrides sniff)
  header?: boolean;             // if undefined, inferred
  comment?: string;             // e.g. "#", "//"
  skipEmpty?: boolean;          // default true
  autoType?: boolean;           // default true
  nulls?: string[];             // treated as null (case-insensitive)
  sampleSize?: number;          // for header/type inference (default 200 lines)
};

export type Column = { name: string; type: "string" | "bool" | "int" | "float" | "date" | "timestamp" | "mixed" };

export type ParseResult = {
  ok: boolean;
  format?: "json" | "ndjson" | "csv" | "tsv" | "yaml";
  rows?: any[];
  columns?: Column[];
  meta?: Dict;
  error?: string;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Public: main entry
 * ────────────────────────────────────────────────────────────────────────── */

export function parse(input: string | Uint8Array | ArrayBuffer, options?: ParseOptions): ParseResult {
  const opts = withDefaults(options);
  const text = toText(input);
  const fmt = opts.format === "auto" ? sniffFormat(text, opts.filename) : opts.format;

  try {
    if (fmt === "json") {
      const res = parseJSON(text);
      return { ok: true, format: "json", rows: res.rows, meta: res.meta, columns: inferColumns(res.rows, opts) };
    }
    if (fmt === "ndjson") {
      const rows = parseNDJSON(text, opts);
      return { ok: true, format: "ndjson", rows, columns: inferColumns(rows, opts) };
    }
    if (fmt === "yaml") {
      const obj = parseYAML(text);
      // Prefer rows if top-level has "rows" or "datasets" array; otherwise wrap into single row
      let rows: any[] = [];
      if (obj && typeof obj === "object") {
        if (Array.isArray((obj as any).rows)) rows = (obj as any).rows;
        else if (Array.isArray((obj as any).datasets)) rows = (obj as any).datasets;
        else rows = [obj];
      } else {
        rows = [{ value: obj }];
      }
      return { ok: true, format: "yaml", rows, columns: inferColumns(rows, opts) };
    }
    // CSV / TSV
    const delim = fmt === "tsv" ? "\t" : (opts.delimiter || sniffDelimiter(text));
    const out = parseCSV(text, { ...opts, delimiter: delim });
    return { ok: true, format: fmt === "tsv" ? "tsv" : "csv", rows: out.rows, columns: out.columns, meta: out.meta };
  } catch (e: any) {
    return { ok: false, error: stringifyError(e) };
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * JSON + NDJSON
 * ────────────────────────────────────────────────────────────────────────── */

export function parseJSON(text: string): { rows: any[]; meta?: Dict } {
  const t = stripBOM(text).trim();
  if (t === "") return { rows: [] };
  const obj = JSON.parse(t);
  if (Array.isArray(obj)) return { rows: obj };
  if (obj && typeof obj === "object") {
    // try common containers
    if (Array.isArray((obj as any).rows)) return { rows: (obj as any).rows, meta: pick(obj as any, ["meta", "schema", "columns"]) };
    if (Array.isArray((obj as any).data)) return { rows: (obj as any).data, meta: pick(obj as any, ["meta", "schema", "columns"]) };
    return { rows: [obj] };
  }
  return { rows: [{ value: obj }] };
}

export function parseNDJSON(text: string, options?: ParseOptions): any[] {
  const opts = withDefaults(options);
  const lines = stripBOM(text).split(/\r?\n/);
  const out: any[] = [];
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (!raw) continue;
    if (opts.comment && startsWith(raw, opts.comment)) continue;
    const s = raw.trim();
    if (s === "") continue;
    try {
      out.push(JSON.parse(s));
    } catch (e) {
      // tolerate trailing commas or loose JSON by attempting fix
      throw new Error(`NDJSON parse error at line ${i + 1}: ${String(e && (e as any).message || e)}`);
    }
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * CSV / TSV (RFC 4180-ish with extras)
 * ────────────────────────────────────────────────────────────────────────── */

export function parseCSV(
  text: string,
  options?: ParseOptions & { delimiter?: string }
): { rows: any[]; columns: Column[]; meta?: Dict } {
  const opts = withDefaults(options);
  const delimiter = (options && options.delimiter) || ",";
  const comment = opts.comment;
  const autoType = opts.autoType;
  const nulls = opts.nulls;

  const s = stripBOM(text);
  if (s.trim() === "") return { rows: [], columns: [] };

  // Tokenize into records
  const records: string[][] = [];
  let i = 0;
  const N = s.length;
  const row: string[] = [];
  let field = "";
  let inQuotes = false;

  function pushField() {
    row.push(field);
    field = "";
  }
  function pushRow() {
    // Comment row?
    const joined = row.join("").trim();
    if (comment && startsWith(trimLeft(joined), comment)) {
      row.length = 0; field = "";
      return;
    }
    // Skip fully empty row
    if (opts.skipEmpty && row.length === 1 && row[0].trim() === "") {
      row.length = 0; field = "";
      return;
    }
    records.push(row.slice());
    row.length = 0;
  }

  while (i < N) {
    const ch = s[i];
    if (inQuotes) {
      if (ch === '"') {
        // peek next
        if (i + 1 < N && s[i + 1] === '"') {
          // escaped quote
          field += '"'; i += 2; continue;
        } else {
          inQuotes = false; i++; continue;
        }
      } else {
        field += ch; i++; continue;
      }
    } else {
      if (ch === '"') {
        inQuotes = true; i++; continue;
      }
      if (ch === delimiter) {
        pushField(); i++; continue;
      }
      if (ch === "\r") { i++; continue; }
      if (ch === "\n") {
        pushField(); pushRow(); i++; continue;
      }
      field += ch; i++; continue;
    }
  }
  // last field/row
  pushField();
  pushRow();

  // Infer header?
  let headerPresent = typeof opts.header === "boolean" ? opts.header : inferHeader(records);
  let header: string[];
  const dataRows: string[][] = [];

  if (headerPresent) {
    header = normalizeHeader(records[0]);
    for (let r = 1; r < records.length; r++) dataRows.push(padTo(records[r], header.length));
  } else {
    const maxLen = records.reduce((m, r) => Math.max(m, r.length), 0);
    header = autoHeaders(maxLen);
    for (let r = 0; r < records.length; r++) dataRows.push(padTo(records[r], header.length));
  }

  // Build rows with auto typing
  const rows: any[] = [];
  const cols: Column[] = header.map(h => ({ name: h, type: "mixed" as Column["type"] }));

  // Pre-infer types on a sample (light)
  const sample = (opts.sampleSize && opts.sampleSize > 0) ? dataRows.slice(0, opts.sampleSize) : dataRows;
  const inferred = autoType ? inferTypesFromSample(sample, nulls) : header.map(() => "string" as Column["type"]);
  for (let c = 0; c < cols.length; c++) cols[c].type = inferred[c] || "mixed";

  for (let r = 0; r < dataRows.length; r++) {
    const rec = dataRows[r];
    const obj: Dict = {};
    for (let c = 0; c < header.length; c++) {
      const raw = rec[c] ?? "";
      obj[header[c]] = autoType ? coerce(raw, inferred[c], nulls) : raw;
    }
    rows.push(obj);
  }

  return { rows, columns: cols, meta: { delimiter, header: headerPresent, records: records.length } };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * YAML (tiny subset parser: maps, arrays, scalars)
 *   - Supports:
 *       key: value
 *       key:
 *         nested: value
 *       list:
 *         - item
 *         - key: value
 *     Strings can be quoted or bare; # comments supported.
 *     Indentation with spaces (2+). Tabs are not supported.
 * ────────────────────────────────────────────────────────────────────────── */

export function parseYAML(text: string): any {
  const src = stripBOM(text).replace(/\r/g, "");
  const lines = src.split("\n");

  type Frame = { container: any; kind: "obj" | "arr"; key?: string };
  const root: any = {};
  const stack: Frame[] = [{ container: root, kind: "obj" }];
  const keyWaiting: { [level: number]: string | undefined } = {};
  let prevIndent = 0;

  function setInParent(level: number, key: string, value: any) {
    const parent = stack[level].container;
    parent[key] = value;
  }

  for (let idx = 0; idx < lines.length; idx++) {
    let line = lines[idx];
    if (!line) continue;
    // Strip comments (ignore '#' inside quotes)
    line = stripComment(line);
    if (line.trim() === "") continue;

    const indent = countIndent(line);
    const level = indent;

    // pop to level
    while (stack.length - 1 > level) { stack.pop(); delete keyWaiting[stack.length - 1]; }

    // normalize content
    const content = line.trim();

    if (content.startsWith("- ")) {
      // Ensure current parent is an array. If parent is object and we had a pending key, transform its value to array.
      const parent = stack[stack.length - 1];
      if (parent.kind === "obj") {
        const pendingKey = keyWaiting[level] || keyWaiting[level - 1];
        if (pendingKey) {
          if (!parent.container[pendingKey]) parent.container[pendingKey] = [];
          // Switch context to that array
          stack.push({ container: parent.container[pendingKey], kind: "arr" });
          delete keyWaiting[level];
        } else {
          // If no pending key, create anonymous list at root (rare)
          if (!Array.isArray(parent.container["_"])) parent.container["_"] = [];
          stack.push({ container: parent.container["_"], kind: "arr" });
        }
      }
      const arr = stack[stack.length - 1].container as any[];

      const after = content.slice(2).trim();
      if (after === "") {
        // Empty item which will be filled by nested lines
        const obj: any = {};
        arr.push(obj);
        stack.push({ container: obj, kind: "obj" });
      } else if (after.includes(":")) {
        // Inline object start: "key: value [more?]" (we only parse first pair; nested will follow)
        const { key, value, hasNested } = parseKeyValue(after);
        const obj: any = {};
        obj[key] = value;
        arr.push(obj);
        if (hasNested) {
          stack.push({ container: obj, kind: "obj" });
          keyWaiting[stack.length - 1] = key;
        }
      } else {
        // Scalar item
        arr.push(parseScalar(after));
      }
      continue;
    }

    // key: value  or key:
    if (content.includes(":")) {
      const { key, value, hasNested } = parseKeyValue(content);
      const parent = stack[stack.length - 1];
      if (hasNested) {
        // allocate object (or preserve if array desired later)
        if (parent.kind !== "obj") {
          // inserting into array item object (ensure last item is object)
          const arr = parent.container as any[];
          if (typeof arr[arr.length - 1] !== "object" || arr[arr.length - 1] == null) {
            arr.push({});
          }
          (arr[arr.length - 1] as any)[key] = {};
          stack.push({ container: (arr[arr.length - 1] as any), kind: "obj" });
          keyWaiting[stack.length - 1] = key;
        } else {
          parent.container[key] = {};
          keyWaiting[stack.length - 1] = key;
        }
      } else {
        if (parent.kind === "obj") {
          parent.container[key] = value;
        } else {
          // parent is array → assign into the last object item
          const arr = parent.container as any[];
          if (arr.length === 0 || typeof arr[arr.length - 1] !== "object") arr.push({});
          (arr[arr.length - 1] as any)[key] = value;
        }
      }
      continue;
    }

    // Bare scalar line at root (rare) → push into anonymous list
    const top = stack[stack.length - 1];
    if (top.kind === "arr") {
      (top.container as any[]).push(parseScalar(content));
    } else {
      if (!Array.isArray(top.container["_"])) top.container["_"] = [];
      top.container["_"].push(parseScalar(content));
    }
    prevIndent = indent;
  }

  // Clean anonymous keys if any
  if (Array.isArray(root["_"]) && Object.keys(root).length === 1) return root["_"];
  return root;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Helpers: CSV header & typing
 * ────────────────────────────────────────────────────────────────────────── */

function inferHeader(records: string[][]): boolean {
  if (records.length === 0) return false;
  const head = records[0] || [];
  if (head.length === 0) return false;
  // Heuristic: if all cells are non-numeric-ish and unique, treat as header
  const seen: Dict<boolean> = {};
  let nonNumeric = 0;
  for (let i = 0; i < head.length; i++) {
    const c = (head[i] || "").trim();
    if (c === "") return false;
    if (!looksNumeric(c) && !looksDateTime(c)) nonNumeric++;
    if (seen[c]) return false;
    seen[c] = true;
  }
  return nonNumeric >= Math.ceil(head.length * 0.7);
}

function normalizeHeader(h: string[]): string[] {
  const out: string[] = [];
  const used: Dict<number> = {};
  for (let i = 0; i < h.length; i++) {
    let k = (h[i] || "").trim();
    if (k === "") k = `col_${i + 1}`;
    k = k.replace(/\s+/g, "_").replace(/[^A-Za-z0-9_]/g, "_");
    if (used[k]) { used[k]++; k = `${k}_${used[k]}`; } else used[k] = 1;
    out.push(k);
  }
  return out;
}
function autoHeaders(n: number): string[] { const a: string[] = []; for (let i = 1; i <= n; i++) a.push(`c${i}`); return a; }
function padTo(arr: string[], n: number): string[] { const a = arr.slice(); while (a.length < n) a.push(""); return a; }

function inferTypesFromSample(sample: string[][], nulls: string[] | undefined): Column["type"][] {
  const w = sample.length ? sample[0].length : 0;
  const types: Column["type"][] = new Array(w).fill("mixed");
  for (let c = 0; c < w; c++) {
    let seenStr = false, seenBool = false, seenInt = false, seenFloat = false, seenDate = false, seenTs = false;
    for (let r = 0; r < sample.length; r++) {
      const raw = sample[r][c] ?? "";
      if (isNullToken(raw, nulls)) continue;
      const s = raw.trim();
      if (looksBool(s)) { seenBool = true; continue; }
      if (looksInt(s)) { seenInt = true; continue; }
      if (looksFloat(s)) { seenFloat = true; continue; }
      if (looksTimestamp(s)) { seenTs = true; continue; }
      if (looksDate(s)) { seenDate = true; continue; }
      seenStr = true;
    }
    types[c] =
      seenStr ? "string" :
      seenTs ? "timestamp" :
      seenDate ? "date" :
      seenFloat ? "float" :
      seenInt ? "int" :
      seenBool ? "bool" : "mixed";
  }
  return types;
}

function coerce(raw: string, t: Column["type"], nulls: string[] | undefined): any {
  if (isNullToken(raw, nulls)) return null;
  const s = raw.trim();
  switch (t) {
    case "bool": return toBool(s);
    case "int": return toInt(s);
    case "float": return toFloat(s);
    case "date": return normalizeDate(s) ?? s;
    case "timestamp": return normalizeISO(s) ?? s;
    default: return s;
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Format sniffers
 * ────────────────────────────────────────────────────────────────────────── */

function sniffFormat(text: string, filename?: string): Required<ParseOptions>["format"] {
  // filename first
  if (filename) {
    const f = filename.toLowerCase();
    if (ends(f, ".json")) return "json";
    if (ends(f, ".ndjson") || ends(f, ".jsonl")) return "ndjson";
    if (ends(f, ".yaml") || ends(f, ".yml")) return "yaml";
    if (ends(f, ".tsv")) return "tsv";
    if (ends(f, ".csv")) return "csv";
  }
  const s = stripBOM(text).trimStart();
  if (s.startsWith("{") || s.startsWith("[")) return "json";
  if (/^\s*[{[]/.test(s) && s.split(/\r?\n/, 3).length === 1) return "json";
  if (/^\s*[-\w]+\s*:/.test(s) || /^\s*-\s+/.test(s)) return "yaml";
  if (s.indexOf("\n") > -1 && /^\s*[{[]/.test(s.split(/\r?\n/)[0]) === false && /{/.test(s) === false) {
    // likely CSV/TSV
    const d = sniffDelimiter(s);
    return d === "\t" ? "tsv" : "csv";
  }
  // fallback
  return "csv";
}

function sniffDelimiter(text: string): string {
  const head = stripBOM(text).split(/\r?\n/).slice(0, 10);
  const counts = { ",": 0, "\t": 0, ";": 0, "|": 0 };
  for (let i = 0; i < head.length; i++) {
    const line = head[i];
    if (!line) continue;
    counts[","] += countCharOutsideQuotes(line, ",");
    counts["\t"] += countCharOutsideQuotes(line, "\t");
    counts[";"] += countCharOutsideQuotes(line, ";");
    counts["|"] += countCharOutsideQuotes(line, "|");
  }
  let best = ",";
  let max = -1;
  for (const k in counts) { if ((counts as any)[k] > max) { max = (counts as any)[k]; best = k; } }
  return best;
}

function countCharOutsideQuotes(line: string, ch: string): number {
  let n = 0; let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') inQ = !inQ;
    else if (!inQ && c === ch) n++;
  }
  return n;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Scalar parsing & detection
 * ────────────────────────────────────────────────────────────────────────── */

function isNullToken(s: string, nulls?: string[]): boolean {
  const tok = s.trim().toLowerCase();
  const defaults = ["", "na", "n/a", "null", "nil", "~"];
  const set: Dict<boolean> = {};
  for (let i = 0; i < defaults.length; i++) set[defaults[i]] = true;
  if (Array.isArray(nulls)) for (let i = 0; i < nulls.length; i++) set[String(nulls[i]).toLowerCase()] = true;
  return !!set[tok];
}

function looksNumeric(s: string): boolean { return looksInt(s) || looksFloat(s); }
function looksInt(s: string): boolean { return /^[-+]?\d{1,3}(_?\d{3})*$/.test(s.trim()); }
function looksFloat(s: string): boolean { return /^[-+]?\d{1,3}(_?\d{3})*(\.\d+)?([eE][-+]?\d+)?$/.test(s.trim()) || /^[-+]?\.\d+([eE][-+]?\d+)?$/.test(s.trim()); }
function looksBool(s: string): boolean { const t = s.trim().toLowerCase(); return t === "true" || t === "false" || t === "yes" || t === "no"; }
function looksDate(s: string): boolean { return /^\d{4}-\d{2}-\d{2}$/.test(s.trim()); }
function looksTimestamp(s: string): boolean { return /^\d{4}-\d{2}-\d{2}[tT ][\d:.]+([zZ]|[+-]\d{2}:?\d{2})?$/.test(s.trim()); }
function looksDateTime(s: string): boolean { return looksDate(s) || looksTimestamp(s); }

function toBool(s: string): boolean | null {
  const t = s.trim().toLowerCase();
  if (t === "true" || t === "yes" || t === "y" || t === "1") return true;
  if (t === "false" || t === "no" || t === "n" || t === "0") return false;
  return null;
}
function toInt(s: string): number | null {
  const m = s.trim().replace(/_/g, "");
  const n = Number(m);
  if (!isFinite(n)) return null;
  return Math.trunc(n);
}
function toFloat(s: string): number | null {
  const m = s.trim().replace(/_/g, "");
  const n = Number(m);
  return isFinite(n) ? n : null;
}
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

/* ────────────────────────────────────────────────────────────────────────── *
 * YAML helpers (subset)
 * ────────────────────────────────────────────────────────────────────────── */

function parseKeyValue(content: string): { key: string; value: any; hasNested: boolean } {
  // Split only on first ':'
  const i = content.indexOf(":");
  const rawKey = content.slice(0, i).trim();
  const rest = content.slice(i + 1).trim();
  const key = unquote(rawKey);
  if (rest === "" || rest === "|") {
    return { key, value: {}, hasNested: true };
  }
  return { key, value: parseScalar(rest), hasNested: false };
}

function parseScalar(raw: string): any {
  const s = raw.trim();
  // quoted
  if ((startsWith(s, '"') && ends(s, '"')) || (startsWith(s, "'") && ends(s, "'"))) {
    return unquote(s);
  }
  // null-ish
  if (isNullToken(s)) return null;
  // bool
  const b = toBool(s);
  if (b !== null) return b;
  // numbers
  if (looksInt(s)) return toInt(s);
  if (looksFloat(s)) return toFloat(s);
  // date/timestamp → keep as string but normalized if possible
  if (looksTimestamp(s)) return normalizeISO(s) || s;
  if (looksDate(s)) return normalizeDate(s) || s;
  // flow-style arrays: [a, b, c]
  if (startsWith(s, "[") && ends(s, "]")) {
    const inner = s.slice(1, -1).trim();
    if (inner === "") return [];
    // naive split on commas not inside quotes (simple)
    const parts = splitCSVLike(inner, ",");
    return parts.map(p => parseScalar(p));
  }
  // flow-style object: {k: v, ...}
  if (startsWith(s, "{") && ends(s, "}")) {
    const inner = s.slice(1, -1).trim();
    if (inner === "") return {};
    const pairs = splitCSVLike(inner, ",");
    const o: Dict = {};
    for (let i = 0; i < pairs.length; i++) {
      const kv = pairs[i].split(":");
      if (kv.length < 2) continue;
      const k = unquote(kv.shift()!.trim());
      const v = parseScalar(kv.join(":"));
      o[k] = v;
    }
    return o;
  }
  return s;
}

function splitCSVLike(s: string, delim: string): string[] {
  const out: string[] = [];
  let buf = "";
  let q: string | null = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) {
      if (c === q) q = null;
      buf += c;
    } else {
      if (c === "'" || c === '"') { q = c; buf += c; }
      else if (c === delim) { out.push(buf.trim()); buf = ""; }
      else buf += c;
    }
  }
  if (buf !== "") out.push(buf.trim());
  return out;
}

function stripComment(line: string): string {
  // remove '#' that is not inside quotes
  let q: string | null = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === q) q = null;
    } else {
      if (c === "'" || c === '"') q = c;
      else if (c === "#") return line.slice(0, i);
    }
  }
  return line;
}

function countIndent(line: string): number {
  if (/^\t/.test(line)) throw new Error("YAML: tabs not supported; use spaces for indentation.");
  const m = line.match(/^[ ]+/);
  const n = m ? m[0].length : 0;
  // use 2-space blocks as levels
  return Math.floor(n / 2);
}

function unquote(s: string): string {
  if (!s) return s;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
    const body = s.slice(1, -1);
    if (s.startsWith('"')) return body.replace(/\\"/g, '"');
    return body.replace(/\\'/g, "'");
  }
  return s;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Utilities
 * ────────────────────────────────────────────────────────────────────────── */

function withDefaults(opts?: ParseOptions): Required<Pick<ParseOptions, "format" | "skipEmpty" | "autoType" | "nulls" | "sampleSize">> & ParseOptions {
  return {
    format: opts?.format || "auto",
    skipEmpty: opts?.skipEmpty !== false,
    autoType: opts?.autoType !== false,
    nulls: opts?.nulls || ["", "NA", "N/A", "null", "Nil", "~"],
    sampleSize: opts?.sampleSize || 200,
    ...opts
  };
}

function toText(input: string | Uint8Array | ArrayBuffer): string {
  if (typeof input === "string") return input;
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
  return utf8Decode(u8);
}

function stripBOM(s: string): string { return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s; }

function utf8Decode(u8: Uint8Array): string {
  if (typeof (globalThis as any).TextDecoder !== "undefined") {
    try { return new (globalThis as any).TextDecoder("utf-8").decode(u8); } catch {}
  }
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return s;
}

function pad2(n: number): string { return String(n).padStart(2, "0"); }
function pick<T extends Dict>(obj: T, keys: string[]): Partial<T> { const o: Partial<T> = {}; for (let i = 0; i < keys.length; i++) if (keys[i] in obj) (o as any)[keys[i]] = obj[keys[i]]; return o; }
function startsWith(s: string, p: string): boolean { return s.slice(0, p.length) === p; }
function ends(s: string, suf: string): boolean { return s.slice(-suf.length) === suf; }
function trimLeft(s: string): string { return s.replace(/^\s+/, ""); }
function stringifyError(e: any): string { if (!e) return "Unknown error"; if (typeof e === "string") return e; if (e instanceof Error) return e.message || String(e); try { return JSON.stringify(e); } catch { return String(e); } }

/* ────────────────────────────────────────────────────────────────────────── *
 * Column inference for JSON/NDJSON/YAML rows
 * ────────────────────────────────────────────────────────────────────────── */

function inferColumns(rows: any[], opts: ParseOptions): Column[] {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  const sample = rows.slice(0, Math.min(rows.length, opts.sampleSize || 200));
  const keys: Dict<boolean> = {};
  for (let i = 0; i < sample.length; i++) {
    const r = sample[i];
    if (r && typeof r === "object" && !Array.isArray(r)) {
      for (const k in r) keys[k] = true;
    } else {
      keys["value"] = true;
    }
  }
  const cols: Column[] = [];
  const names = Object.keys(keys).sort();
  for (let i = 0; i < names.length; i++) {
    const k = names[i];
    let t: Column["type"] = "mixed";
    if (opts.autoType !== false) {
      let sb = false, si = false, sf = false, sd = false, st = false, ss = false;
      for (let j = 0; j < sample.length; j++) {
        const v = (sample[j] && typeof sample[j] === "object" && !Array.isArray(sample[j])) ? sample[j][k] : (k === "value" ? sample[j] : undefined);
        if (v == null) continue;
        const type = typeof v;
        if (type === "boolean") sb = true;
        else if (type === "number") {
          if (Number.isInteger(v as number)) si = true; else sf = true;
        } else if (type === "string") {
          const s = String(v);
          if (looksTimestamp(s)) st = true;
          else if (looksDate(s)) sd = true;
          else if (looksInt(s)) si = true;
          else if (looksFloat(s)) sf = true;
          else ss = true;
        } else {
          ss = true;
        }
      }
      t = ss ? "string" : st ? "timestamp" : sd ? "date" : sf ? "float" : si ? "int" : sb ? "bool" : "mixed";
    } else {
      t = "string";
    }
    cols.push({ name: k, type: t });
  }
  return cols;
}