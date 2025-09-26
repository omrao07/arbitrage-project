// Server-only fetcher that returns heatmap rows compatible with Nivo:
//   [{ id: "AAPL", MSFT: 0.82, AMZN: 0.76, ... }, ...]
import "server-only";

/** Request sent to your backend */
export type FetchHeatmapReq = {
  tickers: string[];
  metric?: "correlation" | "covariance" | "distance" | "beta";
  method?: "pearson" | "spearman" | "kendall";
  startDate?: string;
  endDate?: string;
  periodicity?: "daily" | "weekly" | "monthly";
  fill?: "drop" | "ffill" | "zeros";
  universe?: string;
  groupBy?: string;
};

/** Nivo-compatible row (id is string; other keys are numeric cells) */
export interface NivoHeatmapRow {
  id: string;
  [key: string]: number | null | string | undefined;
}

/** Accepted backend shapes */
type BackendShape =
  | { rows: NivoHeatmapRow[] }
  | NivoHeatmapRow[]
  | { matrix: number[][]; keys: string[] }
  | { cells: Array<{ x: string; y: string; value: number | null }>; keys?: string[] };

/* endpoint resolution */
const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.API_BASE_URL ||
  "";

function endpoint(): string {
  return BACKEND_URL
    ? `${BACKEND_URL.replace(/\/+$/, "")}/analytics/heatmap`
    : "/api/analytics/heatmap";
}

/* main fetcher */
export async function fetchHeatmap(req: FetchHeatmapReq): Promise<NivoHeatmapRow[]> {
  if (!req?.tickers || req.tickers.length < 2) {
    throw new Error("fetchHeatmap: provide at least two tickers.");
  }

  const payload: FetchHeatmapReq = {
    metric: "correlation",
    method: "pearson",
    periodicity: "daily",
    fill: "drop",
    ...req,
  };

  const res = await fetch(endpoint(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`fetchHeatmap: ${res.status} ${res.statusText} ${text}`);
  }

  const raw: BackendShape = await res.json();
  return normalizeToRows(raw, payload.tickers);
}

/* normalization */
function normalizeToRows(raw: BackendShape, keysHint: string[]): NivoHeatmapRow[] {
  // 1) already rows
  if (Array.isArray(raw)) return ensureAllKeys(raw, guessKeys(raw, keysHint));
  if ("rows" in raw && Array.isArray(raw.rows)) return ensureAllKeys(raw.rows, guessKeys(raw.rows, keysHint));

  // 2) matrix + keys
  if ("matrix" in raw && Array.isArray(raw.matrix) && "keys" in raw && Array.isArray(raw.keys)) {
    const keys = raw.keys.map(String);
    const rows: NivoHeatmapRow[] = raw.matrix.map((line, i) => {
      const rowKey = keys[i] ?? `row_${i}`;
      const row: NivoHeatmapRow = { id: rowKey };
      keys.forEach((k, j) => (row[k] = toNum(line[j])));
      return row;
    });
    return ensureAllKeys(rows, keys);
  }

  // 3) cell list
  if ("cells" in raw && Array.isArray(raw.cells)) {
    const keys =
      raw.keys?.map(String) ??
      Array.from(new Set(raw.cells.flatMap((c) => [String(c.x), String(c.y)])));

    const byRow: Record<string, NivoHeatmapRow> = {};
    keys.forEach((k) => (byRow[k] = { id: k }));

    raw.cells.forEach((c) => {
      const x = String(c.x);
      const y = String(c.y);
      if (!byRow[y]) byRow[y] = { id: y };
      byRow[y][x] = toNum(c.value);
    });

    const rows = keys.map((k) => byRow[k]);
    return ensureAllKeys(rows, keys);
  }

  // fallback: identity matrix with zeros off-diagonal
  const keys = keysHint.map(String);
  return keys.map((r, i) => {
    const row: NivoHeatmapRow = { id: r };
    keys.forEach((c, j) => (row[c] = i === j ? 1 : 0));
    return row;
  });
}

function guessKeys(rows: NivoHeatmapRow[], keysHint: string[]): string[] {
  if (rows.length === 0) return keysHint.map(String);
  const cols = new Set<string>();
  rows.forEach((r) => Object.keys(r).forEach((k) => k !== "id" && cols.add(k)));
  if (keysHint.length && keysHint.every((k) => cols.has(k))) return keysHint.map(String);
  return Array.from(cols);
}

function ensureAllKeys(rows: NivoHeatmapRow[], keys: string[]): NivoHeatmapRow[] {
  return rows.map((r) => {
    const row: NivoHeatmapRow = { id: String(r.id) };
    keys.forEach((k) => (row[k] = toNum((r as any)[k])));
    return row;
  });
}

function toNum(v: any): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}