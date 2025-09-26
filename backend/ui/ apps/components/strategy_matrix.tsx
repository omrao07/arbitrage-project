import React from "react";

export type StrategyStatus = "idle" | "queued" | "running" | "succeeded" | "failed";

export type KPIs = Partial<{
  cagr: number;        // fraction (0.12 = 12%)
  sharpe: number;
  max_dd: number;      // negative fraction
  vol_annual: number;  // fraction
}>;

export type StrategyRow = {
  id: string;
  name: string;
  kind: "momentum" | "mean_reversion" | "pairs_trading" | "mc" | "stress" | string;
  params?: Record<string, unknown>;
  kpis?: KPIs | null;
  updatedAt?: string;     // ISO
  status?: StrategyStatus;
  reportUrl?: string | null;
};

export type StrategyMatrixProps = {
  rows: StrategyRow[];
  onRun?: (row: StrategyRow) => void | Promise<void>;
  onStop?: (row: StrategyRow) => void | Promise<void>;
  onOpenReport?: (row: StrategyRow) => void;
  loading?: boolean;
  defaultSearch?: string;
  pageSize?: number;
  headerActions?: React.ReactNode;
  showKindFilter?: boolean;
};

type SortKey = "name" | "kind" | "cagr" | "sharpe" | "max_dd" | "vol_annual" | "updatedAt" | "status";
type SortDir = "asc" | "desc";

const pct = (v?: number | null) =>
  v == null || !isFinite(v) ? "—" : (v * 100).toFixed(2) + "%";
const num = (v?: number | null) =>
  v == null || !isFinite(v) ? "—" : v.toFixed(2);

const StatusBadge: React.FC<{ s: StrategyStatus | undefined }> = ({ s }) => {
  const map: Record<StrategyStatus, string> = {
    idle: "#8e9aaf",
    queued: "#4dabf7",
    running: "#ffd43b",
    succeeded: "#51cf66",
    failed: "#ff6b6b",
  } as const;
  const label = s ?? "idle";
  return (
    <span
      style={{
        padding: "2px 8px",
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 600,
        color: "var(--foreground, #eaeaea)",
        border: "1px solid var(--border, rgba(255,255,255,.12))",
        background: `color-mix(in oklab, ${map[label as StrategyStatus]} 22%, transparent)`,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
};

const Pill: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span
    style={{
      padding: "2px 6px",
      borderRadius: 8,
      background: "color-mix(in oklab, var(--muted, #999) 18%, transparent)",
      border: "1px solid var(--border, rgba(255,255,255,.08))",
      fontSize: 12,
      whiteSpace: "nowrap",
    }}
  >
    {children}
  </span>
);

const HeaderCell: React.FC<{
  label: string;
  sortKey?: SortKey;
  curSort: { key: SortKey; dir: SortDir } | null;
  setSort: (k: SortKey) => void;
  width?: number | string;
  align?: "left" | "right" | "center";
}> = ({ label, sortKey, curSort, setSort, width, align = "left" }) => {
  const active = curSort?.key === sortKey;
  const arrow = active ? (curSort!.dir === "asc" ? "▲" : "▼") : "↕";
  return (
    <th
      style={{
        width,
        textAlign: align,
        padding: "10px 12px",
        fontSize: 12,
        color: "var(--muted-foreground, #9aa1a9)",
        borderBottom: "1px solid var(--border, rgba(255,255,255,.12))",
        userSelect: "none",
      }}
    >
      {sortKey ? (
        <button
          onClick={() => setSort(sortKey)}
          style={{
            all: "unset",
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
          }}
          aria-label={`Sort by ${label}`}
        >
          <span>{label}</span>
          <span style={{ opacity: active ? 1 : 0.35, fontSize: 10 }}>{arrow}</span>
        </button>
      ) : (
        label
      )}
    </th>
  );
};

const StrategyMatrix: React.FC<StrategyMatrixProps> = ({
  rows,
  onRun,
  onStop,
  onOpenReport,
  loading = false,
  defaultSearch = "",
  pageSize = 10,
  headerActions = null,
  showKindFilter = true,
}) => {
  const [search, setSearch] = React.useState(defaultSearch);
  const [kindFilter, setKindFilter] = React.useState<string>("all");
  const [expanded, setExpanded] = React.useState<Record<string, boolean>>({});
  const [sort, setSort] = React.useState<{ key: SortKey; dir: SortDir } | null>({
    key: "updatedAt",
    dir: "desc",
  });
  const [page, setPage] = React.useState(1);
  const [busyId, setBusyId] = React.useState<string | null>(null);

  const kinds = React.useMemo(() => {
    const set = new Set<string>();
    rows.forEach((r) => set.add(r.kind));
    return ["all", ...Array.from(set).sort()];
  }, [rows]);

  const filtered = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    return rows.filter((r) => {
      const matchesText =
        !q ||
        r.name.toLowerCase().includes(q) ||
        r.kind.toLowerCase().includes(q) ||
        (r.params && JSON.stringify(r.params).toLowerCase().includes(q));
      const matchesKind = kindFilter === "all" || r.kind === kindFilter;
      return matchesText && matchesKind;
    });
  }, [rows, search, kindFilter]);

  const sorted = React.useMemo(() => {
    if (!sort) return filtered;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const va = getSortValue(a, sort.key);
      const vb = getSortValue(b, sort.key);
      if (typeof va === "string" || typeof vb === "string") {
        return String(va).localeCompare(String(vb)) * dir;
      }
      const na = Number.isFinite(va as any) ? (va as number) : -Infinity;
      const nb = Number.isFinite(vb as any) ? (vb as number) : -Infinity;
      return (na - nb) * dir;
    });
  }, [filtered, sort]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageRows = sorted.slice((page - 1) * pageSize, page * pageSize);

  function toggleExpand(id: string) {
    setExpanded((s) => ({ ...s, [id]: !s[id] }));
  }
  function setSortKey(k: SortKey) {
    setPage(1);
    setSort((s) => (!s || s.key !== k ? { key: k, dir: "asc" } : { key: k, dir: s.dir === "asc" ? "desc" : "asc" }));
  }

  async function handleRun(row: StrategyRow) {
    if (!onRun) return;
    setBusyId(row.id);
    try {
      await onRun(row);
    } finally {
      setBusyId(null);
    }
  }
  async function handleStop(row: StrategyRow) {
    if (!onStop) return;
    setBusyId(row.id);
    try {
      await onStop(row);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <section
      aria-label="strategy matrix"
      style={{
        border: "1px solid var(--border, rgba(255,255,255,.14))",
        borderRadius: 14,
        overflow: "hidden",
        background: "color-mix(in oklab, var(--panel, transparent) 80%, transparent)",
      }}
    >
      {/* Header / controls */}
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          padding: 12,
          borderBottom: "1px solid var(--border, rgba(255,255,255,.12))",
          flexWrap: "wrap",
        }}
      >
        <div style={{ fontWeight: 700, color: "var(--foreground, #eaeaea)", marginRight: "auto" }}>
          Strategy Matrix <span style={{ opacity: 0.6, fontWeight: 500 }}>({rows.length})</span>
        </div>

        {showKindFilter && (
          <select
            value={kindFilter}
            onChange={(e) => {
              setKindFilter(e.target.value);
              setPage(1);
            }}
            style={selectStyle}
            aria-label="Filter by kind"
          >
            {kinds.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        )}

        <input
          type="search"
          placeholder="Search name, kind, params…"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          style={searchStyle}
          aria-label="Search strategies"
        />

        {headerActions}
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <HeaderCell label="Name" sortKey="name" curSort={sort} setSort={setSortKey} width="24%" />
              <HeaderCell label="Kind" sortKey="kind" curSort={sort} setSort={setSortKey} width="13%" />
              <HeaderCell label="CAGR" sortKey="cagr" curSort={sort} setSort={setSortKey} width="10%" align="right" />
              <HeaderCell label="Sharpe" sortKey="sharpe" curSort={sort} setSort={setSortKey} width="10%" align="right" />
              <HeaderCell label="Max DD" sortKey="max_dd" curSort={sort} setSort={setSortKey} width="10%" align="right" />
              <HeaderCell label="Vol" sortKey="vol_annual" curSort={sort} setSort={setSortKey} width="9%" align="right" />
              <HeaderCell label="Updated" sortKey="updatedAt" curSort={sort} setSort={setSortKey} width="14%" />
              <HeaderCell label="Status" sortKey="status" curSort={sort} setSort={setSortKey} width="9%" />
              <HeaderCell label="Actions" curSort={null} setSort={() => {}} width="11%" align="center" />
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={9} style={emptyCellStyle}>
                  Loading…
                </td>
              </tr>
            ) : pageRows.length === 0 ? (
              <tr>
                <td colSpan={9} style={emptyCellStyle}>
                  No strategies match your filters.
                </td>
              </tr>
            ) : (
              pageRows.map((r) => (
                <React.Fragment key={r.id}>
                  <tr
                    style={{
                      borderTop: "1px solid var(--border, rgba(255,255,255,.06))",
                      borderBottom: "1px solid var(--border, rgba(255,255,255,.06))",
                    }}
                  >
                    <td style={cellStyleLeft}>
                      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <button onClick={() => toggleExpand(r.id)} style={iconBtnStyle} aria-label="Toggle parameters">
                          {expanded[r.id] ? "▾" : "▸"}
                        </button>
                        <div style={{ fontWeight: 600, color: "var(--foreground, #eaeaea)" }}>{r.name}</div>
                      </div>
                    </td>
                    <td style={cellStyle}>
                      <Pill>{r.kind}</Pill>
                    </td>
                    <td style={cellStyleNum}>{pct(r.kpis?.cagr)}</td>
                    <td style={cellStyleNum}>{num(r.kpis?.sharpe)}</td>
                    <td style={cellStyleNum} title={pct(r.kpis?.max_dd)}>
                      <span style={{ color: "var(--negative, #ff6b6b)" }}>{pct(r.kpis?.max_dd)}</span>
                    </td>
                    <td style={cellStyleNum}>{pct(r.kpis?.vol_annual)}</td>
                    <td style={cellStyle}>{r.updatedAt ? new Date(r.updatedAt).toLocaleString() : "—"}</td>
                    <td style={cellStyle}>
                      <StatusBadge s={r.status} />
                    </td>
                    <td style={{ ...cellStyle, textAlign: "center" }}>
                      <div style={{ display: "inline-flex", gap: 8 }}>
                        <button
                          onClick={() => handleRun(r)}
                          disabled={!onRun || busyId === r.id || r.status === "running" || r.status === "queued"}
                          style={btnStylePrimary}
                          title="Run"
                        >
                          {busyId === r.id && (r.status === "running" || r.status === "queued") ? "…" : "Run"}
                        </button>
                        <button
                          onClick={() => handleStop(r)}
                          disabled={!onStop || busyId === r.id || (r.status !== "running" && r.status !== "queued")}
                          style={btnStyle}
                          title="Stop"
                        >
                          Stop
                        </button>
                        <button
                          onClick={() => (onOpenReport ? onOpenReport(r) : r.reportUrl && window.open(r.reportUrl, "_blank"))}
                          disabled={!r.reportUrl}
                          style={btnStyle}
                          title="Open report"
                        >
                          Report
                        </button>
                      </div>
                    </td>
                  </tr>

                  {expanded[r.id] && (
                    <tr>
                      <td colSpan={9} style={{ padding: 0 }}>
                        <div
                          style={{
                            padding: "12px 16px 16px 44px",
                            borderTop: "1px dashed var(--border, rgba(255,255,255,.1))",
                            background: "color-mix(in oklab, var(--panel, transparent) 70%, transparent)",
                          }}
                        >
                          {r.params && Object.keys(r.params).length ? (
                            <code
                              style={{
                                display: "block",
                                overflowX: "auto",
                                whiteSpace: "pre-wrap",
                                wordBreak: "break-word",
                                fontFamily:
                                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                                fontSize: 12.5,
                                color: "var(--muted-foreground, #9aa1a9)",
                              }}
                            >
                              {pretty(r.params)}
                            </code>
                          ) : (
                            <span style={{ color: "var(--muted-foreground, #9aa1a9)" }}>No parameters.</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          gap: 8,
          padding: 12,
          borderTop: "1px solid var(--border, rgba(255,255,255,.12))",
        }}
      >
        <span style={{ fontSize: 12, color: "var(--muted-foreground, #9aa1a9)" }}>
          Page {page} / {totalPages}
        </span>
        <button onClick={() => setPage(1)} disabled={page === 1} style={btnStyle}>
          «
        </button>
        <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} style={btnStyle}>
          ‹
        </button>
        <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page === totalPages} style={btnStyle}>
          ›
        </button>
        <button onClick={() => setPage(totalPages)} disabled={page === totalPages} style={btnStyle}>
          »
        </button>
      </div>
    </section>
  );
};

// ---------- utilities / styles ----------

function getSortValue(row: StrategyRow, key: SortKey): string | number {
  switch (key) {
    case "name":
      return row.name.toLowerCase();
    case "kind":
      return row.kind.toLowerCase();
    case "cagr":
      return row.kpis?.cagr ?? -Infinity;
    case "sharpe":
      return row.kpis?.sharpe ?? -Infinity;
    case "max_dd":
      return row.kpis?.max_dd ?? Infinity; // more negative is "worse"
    case "vol_annual":
      return row.kpis?.vol_annual ?? -Infinity;
    case "updatedAt":
      return row.updatedAt ? Date.parse(row.updatedAt) : 0;
    case "status":
      return statusRank(row.status);
    default:
      return 0;
  }
}

function statusRank(s?: StrategyStatus) {
  const order: StrategyStatus[] = ["running", "queued", "failed", "succeeded", "idle"];
  return s ? order.indexOf(s) : order.length;
}

function pretty(obj: unknown) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

const searchStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderRadius: 10,
  border: "1px solid var(--border, rgba(255,255,255,.16))",
  background: "var(--panel, transparent)",
  color: "var(--foreground, #eaeaea)",
  minWidth: 240,
  outline: "none",
};

const selectStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderRadius: 10,
  border: "1px solid var(--border, rgba(255,255,255,.16))",
  background: "var(--panel, transparent)",
  color: "var(--foreground, #eaeaea)",
  outline: "none",
};

const cellStyleBase: React.CSSProperties = {
  padding: "10px 12px",
  fontSize: 13,
  color: "var(--foreground, #dedede)",
  verticalAlign: "middle",
};
const cellStyle: React.CSSProperties = { ...cellStyleBase, textAlign: "left" };
const cellStyleLeft: React.CSSProperties = { ...cellStyleBase, textAlign: "left", whiteSpace: "nowrap" };
const cellStyleNum: React.CSSProperties = { ...cellStyleBase, textAlign: "right", fontVariantNumeric: "tabular-nums" };
const emptyCellStyle: React.CSSProperties = {
  padding: 28,
  textAlign: "center",
  color: "var(--muted-foreground, #9aa1a9)",
};

const btnBase: React.CSSProperties = {
  borderRadius: 10,
  padding: "6px 10px",
  border: "1px solid var(--border, rgba(255,255,255,.16))",
  background: "var(--panel, transparent)",
  color: "var(--foreground, #eaeaea)",
  cursor: "pointer",
  fontSize: 12,
};
const btnStyle: React.CSSProperties = { ...btnBase };
const btnStylePrimary: React.CSSProperties = {
  ...btnBase,
  background: "color-mix(in oklab, var(--accent, #4ea8de) 24%, transparent)",
  border: "1px solid color-mix(in oklab, var(--accent, #4ea8de) 32%, var(--border, rgba(255,255,255,.24)))",
  fontWeight: 700,
};
const iconBtnStyle: React.CSSProperties = { ...btnBase, padding: "2px 6px" };

export default React.memo(StrategyMatrix);