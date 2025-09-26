import React from "react";
import { ResponsiveContainer, LineChart, Line, Tooltip as RTooltip } from "recharts";

type KPIs = Partial<{
  cagr: number;            // fraction (0.12 = 12%)
  sharpe: number;
  sortino: number;
  vol_annual: number;      // fraction
  max_dd: number;          // negative fraction (e.g., -0.23)
  calmar: number;
  win_rate: number;        // fraction
  avg_win: number;         // fraction
  avg_loss: number;        // negative fraction
}>;

export type MetricsPanelProps = {
  kpis?: KPIs | null;
  /**
   * Optional tiny sparkline (typically daily PnL or equity % change).
   * Values are fractional returns (0.01 = +1%).
   */
  sparkline?: number[];    // chronological order
  /** Optional title shown above KPIs */
  title?: string;
  /** If true, show a compact layout (fewer metrics) */
  compact?: boolean;
  /** Whether to show tiny sparkline beside the title */
  showSparkline?: boolean;
  /** Custom formatting for percents (e.g., Intl.NumberFormat) */
  formatPercent?: (v: number) => string;
  /** Custom formatting for raw ratios (Sharpe/Sortino/Calmar) */
  formatNumber?: (v: number) => string;
  /** Loading state */
  loading?: boolean;
  /** Error message (takes precedence over loading) */
  error?: string | null;
};

const defaultFmtPercent = (v: number) =>
  (isFinite(v) ? (v * 100).toFixed(2) : "—") + "%";

const defaultFmtNumber = (v: number) =>
  isFinite(v) ? v.toFixed(2) : "—";

/** Small badge for positive/negative values */
const Badge: React.FC<{ value: number; kind?: "pct" | "num" }> = ({ value, kind = "pct" }) => {
  const isPos = value >= 0;
  const txt = kind === "pct" ? defaultFmtPercent(value) : defaultFmtNumber(value);
  return (
    <span
      style={{
        fontVariantNumeric: "tabular-nums",
        padding: "2px 6px",
        borderRadius: 6,
        border: "1px solid var(--border, rgba(255,255,255,.1))",
        background: isPos ? "color-mix(in oklab, var(--positive, #1f8a3b) 20%, transparent)" :
                            "color-mix(in oklab, var(--negative, #d64545) 20%, transparent)",
        color: "var(--foreground, #e8e8e8)",
      }}
      aria-label={txt}
      title={txt}
    >
      {txt}
    </span>
  );
};

/** A single KPI tile */
const Tile: React.FC<{
  label: string;
  value?: number | null;
  fmt?: (v: number) => string;
  emphasize?: boolean;
  help?: string;
}> = ({ label, value, fmt = defaultFmtNumber, emphasize = false, help }) => (
  <div
    style={{
      border: "1px solid var(--border, rgba(255,255,255,.08))",
      background: "var(--panel, transparent)",
      borderRadius: 12,
      padding: "12px 14px",
      display: "flex",
      flexDirection: "column",
      gap: 6,
      minHeight: 72,
    }}
    title={help}
  >
    <div style={{ fontSize: 12, color: "var(--muted-foreground, #9aa1a9)" }}>{label}</div>
    <div
      style={{
        fontSize: emphasize ? 20 : 16,
        fontWeight: emphasize ? 700 : 600,
        color: "var(--foreground, #eaeaea)",
        fontVariantNumeric: "tabular-nums",
      }}
      aria-label={value == null ? "—" : fmt(value)}
    >
      {value == null || !isFinite(value) ? "—" : fmt(value)}
    </div>
  </div>
);

/** Minimal sparkline used near the title (no axes) */
const TinySparkline: React.FC<{ data: number[] }> = ({ data }) => {
  if (!data?.length) return null;
  // convert to chart-friendly points; show cumulative product for smoother equity-like curve
  let cum = 1;
  const rows = data.map((r, i) => {
    cum = cum * (1 + (isFinite(r) ? r : 0));
    return { i, y: cum };
  });
  return (
    <div style={{ width: 160, height: 36 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 6, right: 0, bottom: 0, left: 0 }}>
          <Line
            type="monotone"
            dataKey="y"
            stroke="var(--chart-1, #4ea8de)"
            strokeWidth={1.8}
            dot={false}
            isAnimationActive={false}
          />
          <RTooltip
            content={({ active, payload }) =>
              active && payload?.length ? (
                <div
                  style={{
                    background: "var(--card, #111)",
                    color: "var(--card-foreground, #ddd)",
                    border: "1px solid var(--border, #333)",
                    padding: "6px 8px",
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                >
                  Equity×: <b>{payload[0].value.toFixed(3)}</b>
                </div>
              ) : null
            }
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const MetricsPanel: React.FC<MetricsPanelProps> = ({
  kpis,
  sparkline,
  title = "Performance",
  compact = false,
  showSparkline = true,
  formatPercent = defaultFmtPercent,
  formatNumber = defaultFmtNumber,
  loading = false,
  error = null,
}) => {
  // Layout matrix (compact shows the most important KPIs)
  const grid = compact
    ? ([
        { key: "cagr", label: "CAGR", fmt: formatPercent, em: true, help: "Annualized compounded return" },
        { key: "sharpe", label: "Sharpe", fmt: formatNumber, help: "Risk-adjusted return (annualized)" },
        { key: "max_dd", label: "Max DD", fmt: formatPercent, help: "Worst peak-to-trough drawdown" },
        { key: "vol_annual", label: "Volatility", fmt: formatPercent, help: "Annualized standard deviation" },
      ] as const)
    : ([
        { key: "cagr", label: "CAGR", fmt: formatPercent, em: true, help: "Annualized compounded return" },
        { key: "sharpe", label: "Sharpe", fmt: formatNumber },
        { key: "sortino", label: "Sortino", fmt: formatNumber },
        { key: "max_dd", label: "Max DD", fmt: formatPercent },
        { key: "vol_annual", label: "Volatility", fmt: formatPercent },
        { key: "calmar", label: "Calmar", fmt: formatNumber },
        { key: "win_rate", label: "Win Rate", fmt: formatPercent },
        { key: "avg_win", label: "Avg Win", fmt: formatPercent },
        { key: "avg_loss", label: "Avg Loss", fmt: formatPercent },
      ] as const);

  return (
    <section
      aria-label="metrics panel"
      style={{
        border: "1px solid var(--border, rgba(255,255,255,.12))",
        borderRadius: 14,
        padding: 14,
        background: "color-mix(in oklab, var(--panel, transparent) 75%, transparent)",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 12,
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontWeight: 700, color: "var(--foreground, #eaeaea)" }}>{title}</div>
          {showSparkline && sparkline?.length ? <TinySparkline data={sparkline} /> : null}
        </div>
        {/* Optional actions could go here (download CSV, open full report, etc.) */}
      </header>

      {error ? (
        <div
          role="alert"
          style={{
            padding: 12,
            borderRadius: 10,
            background: "color-mix(in oklab, var(--negative, #d64545) 20%, transparent)",
            border: "1px solid var(--border, rgba(255,255,255,.14))",
            color: "var(--foreground, #eee)",
            fontSize: 13,
          }}
        >
          {error}
        </div>
      ) : loading ? (
        <div
          style={{
            display: "grid",
            gap: 12,
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
          }}
        >
          {Array.from({ length: compact ? 4 : 8 }).map((_, i) => (
            <div
              key={i}
              aria-busy="true"
              style={{
                height: 72,
                borderRadius: 12,
                background:
                  "linear-gradient(90deg, rgba(255,255,255,.05), rgba(255,255,255,.12), rgba(255,255,255,.05))",
                backgroundSize: "200% 100%",
                animation: "shine 1.2s linear infinite",
              }}
            />
          ))}
          <style>{`@keyframes shine { 0%{background-position: 200% 0} 100%{background-position: -200% 0} }`}</style>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gap: 12,
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
          }}
        >

        </div       
        >   
       
      )}
    </section>
  );
};

export default React.memo(MetricsPanel);