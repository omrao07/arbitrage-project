// data/loaders/csv.ts
// Minimal, import-free CSV utilities with robust parsing, type inference,
// delimiter auto-detect, BOM handling, comments, and optional fetch/load.
//
// Usage:
//   const { parse, toObjects, fetchCsv, load } = createCsv();
//   const rows = parse("a,b\n1,2");
//   const objs = toObjects(rows); // [{a:"1", b:"2"}]
//
//   // From URL (Node>=18 or browsers with global fetch):
//   const r = await fetchCsv("https://example.com/file.csv");
//   if (r.ok) console.log(toObjects(r.rows, { inferTypes: true }));
//
//   // Generic load() accepts URL or raw CSV text
//   const r2 = await load("a,b\n1,2"); // treats as inline CSV

export type CsvParseOptions = {
  delimiter?: "," | ";" | "\t" | "|" | string;
  comment?: string;            // lines starting with this are skipped (e.g., "#")
  hasHeader?: boolean;         // default: auto if first row looks like header
  trim?: boolean;              // trim cells (default true)
  relax?: boolean;             // tolerate unbalanced quotes (default true)
  skipEmpty?: boolean;         // drop empty rows (default true)
  maxRows?: number;            // stop after N rows (0 = unlimited)
};

export type CsvToObjectsOptions = {
  headers?: string[];          // override headers
  inferTypes?: boolean;        // convert to number/boolean/date when sensible
  nullish?: (string | null | undefined)[]; // values to coerce to null
  dateFmt?: RegExp;            // custom date detector (default ISO-ish)
};

export type CsvFetchResult = {
  ok: boolean;
  url?: string;
  status?: number;
  rows?: string[][];
  error?: string;
};

export function createCsv() {
  /* ---------------------------- Public API ---------------------------- */

  function parse(input: string, opts: CsvParseOptions = {}): string[][] {
    if (typeof input !== "string") return [];
    let s = stripBOM(input);
    const delimiter = opts.delimiter || detectDelimiter(s);
    const comment = opts.comment;
    const trim = opts.trim !== false; // default true
    const relax = opts.relax !== false; // default true
    const skipEmpty = opts.skipEmpty !== false; // default true
    const maxRows = Math.max(0, opts.maxRows || 0);

    const rows: string[][] = [];
    let row: string[] = [];
    let field = "";
    let i = 0;
    const N = s.length;
    let inQuotes = false;

    function endField() {
      row.push(trim ? field.trim() : field);
      field = "";
    }
    function endRow() {
      // drop comment/empty rows if needed
      if (row.length === 1 && row[0] === "" && skipEmpty) { row = []; return; }
      rows.push(row);
      row = [];
    }

    while (i < N) {
      const c = s[i];

      // Line comment (only if at field start and not inside quotes)
      if (!inQuotes && comment && field.length === 0 && c === comment) {
        // consume to end of line
        while (i < N && s[i] !== "\n") i++;
        // consume newline
        if (i < N && s[i] === "\n") { i++; }
        continue;
      }

      if (c === '"') {
        if (inQuotes) {
          // possible escaped quote
          if (i + 1 < N && s[i + 1] === '"') {
            field += '"';
            i += 2;
            continue;
          } else {
            inQuotes = false;
            i++;
            continue;
          }
        } else {
          // if field is empty (or only whitespace), start quoted field
          if (field === "" || /^\s*$/.test(field)) {
            inQuotes = true;
            i++;
            continue;
          } else {
            // stray quote inside unquoted field
            if (!relax) throw new Error("CSV parse error: unexpected quote");
            // treat as literal
            field += c;
            i++;
            continue;
          }
        }
      }

      if (!inQuotes && c === delimiter) {
        endField();
        i++;
        continue;
      }

      if (!inQuotes && (c === "\n" || c === "\r")) {
        // handle CRLF / LF
        // finalize field
        endField();
        // swallow CRLF
        if (c === "\r" && i + 1 < N && s[i + 1] === "\n") i += 2; else i++;
        endRow();
        if (maxRows && rows.length >= maxRows) break;
        continue;
      }

      field += c;
      i++;
    }

    // last field / row (if any)
    if (inQuotes && !relax) {
      throw new Error("CSV parse error: unterminated quotes");
    }
    // push final field
    if (i === N) endField();
    // push last row if any content or if not empty
    if (row.length > 1 || (row.length === 1 && row[0] !== "" || !skipEmpty)) {
      endRow();
    }

    return rows;
  }

  function toObjects(rows: string[][], opts: CsvToObjectsOptions = {}) {
    if (!Array.isArray(rows) || rows.length === 0) return [];

    const headerRow = opts.headers || (looksLikeHeader(rows[0]) ? rows.shift()! : makeHeaders(rows[0]?.length || 0));
    const headers = headerRow.map((h, i) => (String(h || "").trim() || `col_${i + 1}`));

    const infer = !!opts.inferTypes;
    const nullish = Array.isArray(opts.nullish) ? new Set(opts.nullish.map(String)) : new Set(["", "NA", "N/A", "NULL", "NaN", "na", "null"]);
    const dateRx = opts.dateFmt || ISO_DATE_RX;

    const out: any[] = [];
    for (const r of rows) {
      const obj: any = {};
      for (let i = 0; i < headers.length; i++) {
        const key = headers[i];
        const raw = (r && r[i] != null) ? String(r[i]) : "";
        obj[key] = infer ? inferValue(raw, nullish, dateRx) : raw;
      }
      out.push(obj);
    }
    return out;
  }

  async function fetchCsv(url: string, parseOpts?: CsvParseOptions): Promise<CsvFetchResult> {
    if (typeof (globalThis as any).fetch !== "function") {
      return { ok: false, url, status: 0, error: "fetch not available" };
    }
    try {
      const resp = await (globalThis as any).fetch(url);
      const text = await resp.text();
      if (!resp.ok) return { ok: false, url, status: resp.status, error: `HTTP ${resp.status}` };
      const rows = parse(text, parseOpts || {});
      return { ok: true, url, status: resp.status, rows };
    } catch (err: any) {
      return { ok: false, url, status: 0, error: err?.message || String(err) };
    }
  }

  /**
   * load(input):
   * - If input looks like a URL (http/https), fetches and parses CSV.
   * - Otherwise treats input as CSV text and parses directly.
   */
  async function load(input: string, parseOpts?: CsvParseOptions): Promise<{ ok: true; rows: string[][] } | { ok: false; error: string }> {
    if (/^https?:\/\//i.test(input)) {
      const r = await fetchCsv(input, parseOpts);
      if (!r.ok) return { ok: false, error: r.error || "fetch failed" };
      return { ok: true, rows: r.rows || [] };
    }
    try {
      return { ok: true, rows: parse(input, parseOpts || {}) };
    } catch (err: any) {
      return { ok: false, error: err?.message || String(err) };
    }
  }

  return { parse, toObjects, fetchCsv, load };
}

/* ------------------------------- Helpers ------------------------------- */

const ISO_DATE_RX = /^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?$/;

function stripBOM(s: string) {
  if (s.charCodeAt(0) === 0xfeff) return s.slice(1);
  if (s.startsWith("\uFEFF")) return s.slice(1);
  return s;
}

function detectDelimiter(s: string): string {
  // Inspect first few lines and choose the delimiter with highest consistency.
  const lines = s.split(/\r?\n/).slice(0, 5).filter(Boolean);
  const cands = [",", ";", "\t", "|"];
  let best = ",", bestScore = -1;
  for (const d of cands) {
    const counts = lines.map((ln) => splitCount(ln, d));
    const score = variance(counts) === 0 ? counts[0] : 0; // prefer consistent columns
    if (score > bestScore) { bestScore = score; best = d; }
  }
  return best;
}

function splitCount(line: string, d: string) {
  // very rough: count delimiters not inside quotes
  let cnt = 0, inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQ && i + 1 < line.length && line[i + 1] === '"') { i++; continue; }
      inQ = !inQ;
      continue;
    }
    if (!inQ && c === d) cnt++;
  }
  return cnt + 1; // columns = delimiters + 1
}

function variance(nums: number[]) {
  if (nums.length === 0) return 0;
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  const sq = nums.reduce((a, b) => a + (b - mean) ** 2, 0) / nums.length;
  return sq;
}

function looksLikeHeader(row?: string[]) {
  if (!row || row.length === 0) return false;
  // If any cell has non-alnum/underscore or has spaces, treat as header-ish.
  // Or if cells aren’t numeric at all.
  let alphaish = 0, numericish = 0;
  for (const c of row) {
    if (/^[a-zA-Z_][\w\s\.\-/]*$/.test(c || "")) alphaish++;
    if (/^[\s+-]?\d+(\.\d+)?$/.test(c || "")) numericish++;
  }
  return alphaish >= Math.max(1, Math.ceil(row.length / 2)) && numericish === 0;
}

function makeHeaders(n: number) {
  const a: string[] = [];
  for (let i = 0; i < n; i++) a.push(`col_${i + 1}`);
  return a;
}

function inferValue(raw: string, nullish: Set<string>, dateRx: RegExp) {
  const v = raw == null ? "" : String(raw).trim();
  if (v.length === 0 || nullish.has(v)) return null;

  // boolean
  const vl = v.toLowerCase();
  if (vl === "true" || vl === "false") return vl === "true";

  // number
  if (/^[\s+-]?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(v)) {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }

  // date-ish (ISO)
  if (dateRx.test(v)) {
    const t = Date.parse(v);
    if (!Number.isNaN(t)) return new Date(t);
  }

  return v;
}