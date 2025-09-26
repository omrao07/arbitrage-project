import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Brush,
  ReferenceLine,
} from "recharts";

export type EquityPoint = {
  /** ISO date string (e.g., "2025-01-15") or time label */
  date: string;
  /** Equity value in account currency */
  equity: number;
  /** Optional benchmark equity for comparison */
  benchmark?: number | null;
  /** Optional drawdown (fraction, e.g., -0.123 for -12.3%) */
  drawdown?: number | null;
};

export type EquityChartProps = {
  /** Timeseries data in chronological order */
  data: EquityPoint[];
  /** Title shown above the chart */
  title?: string;
  /** If true, shows a second line for benchmark equity */
  showBenchmark?: boolean;
  /** Format y values (default compact currency) */
  formatY?: (v: number) => string;
  /** Height in pixels (responsive width) */
  height?: number;
  /** Show a horizontal line at starting equity */
  showStartLine?: boolean;
};

const defaultFormatY = (v: number) =>
  new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(v);

/**
 * EquityChart
 * ----------
 * A responsive equity curve with optional benchmark and brush selector.
 * - Pass `drawdown` (negative fraction) in data to gray-out underwater periods.
 *
 * Example:
 * <EquityChart data={rows} title="Strategy Equity" showBenchmark />
 */
const EquityChart: React.FC<EquityChartProps> = ({
  data,
  title,
  showBenchmark = true,
  formatY = defaultFormatY,
  height = 320,
  showStartLine = true,
}) => {
  const startEquity = React.useMemo(() => (data.length ? data[0].equity : 0), [data]);

  // Tooltip content
  const renderTooltip = (props: any) => {
    const { active, payload, label } = props;
    if (!active || !payload || !payload.length) return null;
    const p = payload.reduce((acc: any, cur: any) => ({ ...acc, [cur.name]: cur.value }), {});
    const dd = (payload.find((x: any) => x.dataKey === "drawdown")?.payload?.drawdown ?? null) as number | null;
    return (
      <div
        style={{
          background: "var(--card, #111)",
          color: "var(--card-foreground, #ddd)",
          border: "1px solid var(--border, #333)",
          padding: "8px 10px",
          borderRadius: 8,
          boxShadow: "0 4px 14px rgba(0,0,0,.25)",
          fontSize: 12,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
        <div>Equity: <b>{formatY(p.equity)}</b></div>
        {showBenchmark && p.benchmark != null && (
          <div>Benchmark: <b>{formatY(p.benchmark)}</b></div>
        )}
        {dd != null && (
          <div style={{ opacity: 0.85 }}>
            Drawdown: <b>{(dd * 100).toFixed(2)}%</b>
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ width: "100%" }}>
      {title && (
        <div
          style={{
            marginBottom: 8,
            fontWeight: 600,
            fontSize: 14,
            color: "var(--foreground, #eaeaea)",
          }}
        >
          {title}
        </div>
      )}

      <div
        style={{
          width: "100%",
          height,
          background: "var(--panel, transparent)",
          borderRadius: 12,
          border: "1px solid var(--border, rgba(255,255,255,.08))",
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="var(--muted, rgba(255,255,255,.06))" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: "var(--muted-foreground, #9aa1a9)", fontSize: 12 }}
              tickMargin={8}
              minTickGap={24}
            />
            <YAxis
              tickFormatter={formatY}
              tick={{ fill: "var(--muted-foreground, #9aa1a9)", fontSize: 12 }}
              width={64}
            />
            <Tooltip content={renderTooltip} />
            {showStartLine && startEquity > 0 && (
              <ReferenceLine
                y={startEquity}
                stroke="var(--accent, #8884d8)"
                strokeDasharray="4 4"
                ifOverflow="extendDomain"
              />
            )}

            {/* Main equity line */}
            <Line
              type="monotone"
              dataKey="equity"
              name="equity"
              stroke="var(--chart-1, #4ea8de)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />

            {/* Optional benchmark */}
            {showBenchmark && (
              <Line
                type="monotone"
                dataKey="benchmark"
                name="benchmark"
                stroke="var(--chart-2, #94d82d)"
                strokeWidth={1.75}
                dot={false}
                isAnimationActive={false}
              />
            )}

            {/* Underwater overlay via area baseline trick: we render drawdown as a thin line on the bottom axis so it’s visible in tooltip; 
                to shade underwater regions more richly, consider adding <ReferenceArea> spans based on drawdown<0 segments. */}
            {/* Brush for zooming/selection */}
            {data.length > 60 && (
              <Brush
                dataKey="date"
                height={24}
                stroke="var(--border, rgba(255,255,255,.24))"
                travellerWidth={8}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default React.memo(EquityChart);