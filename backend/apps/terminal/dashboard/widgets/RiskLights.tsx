"use client";

/**
 * RiskLights.tsx
 * - compact traffic-light widgets for portfolio risk & limits
 * - no external libs, no hooks
 * - fully controlled via props
 */

export type LightStatus = "ok" | "warn" | "alert";

export type Thresholds = {
  /** inclusive max for OK; anything above moves to warn/alert depending on direction */
  ok: number;
  /** inclusive max for WARN (values >ok and <=warn) */
  warn: number;
  /**
   * direction of risk: "higher" means higher values are riskier,
   * "lower" means lower values are riskier (e.g., liquidity days lower=bad)
   */
  direction?: "higher" | "lower";
};

export type RiskMetric = {
  id: string;
  label: string;
  value: number;
  unit?: string;            // "%", "$", "d", "x", etc.
  thresholds: Thresholds;   // ok/warn/alert bands
  hint?: string;            // tooltip text
  format?: (v: number) => string; // custom formatter
  weight?: number;          // optional sort weight (desc)
  onClickId?: string;       // optional deep-link id
};

export type RiskLightsProps = {
  title?: string;
  items: RiskMetric[];
  /** show small bar under each light indicating band position */
  showBands?: boolean;
  /** show unit after formatted value */
  showUnit?: boolean;
  /** max grid columns */
  columns?: 3 | 4 | 5 | 6;
  /** click handler when a card is clicked */
  onOpen?: (metricId: string) => void;
};

export default function RiskLights({
  title = "Risk Lights",
  items,
  showBands = true,
  showUnit = true,
  columns = 4,
  onOpen,
}: RiskLightsProps) {
  const ordered = [...items].sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0));

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      <div className="px-4 py-2 border-b border-[#222] flex items-center justify-between">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">traffic-lights</div>
      </div>

      <div
        className={`p-3 grid gap-3`}
        style={{
          gridTemplateColumns:
            columns === 6
              ? "repeat(6, minmax(0, 1fr))"
              : columns === 5
              ? "repeat(5, minmax(0, 1fr))"
              : columns === 4
              ? "repeat(4, minmax(0, 1fr))"
              : "repeat(3, minmax(0, 1fr))",
        }}
      >
        {ordered.map((m) => {
          const status = grade(m.value, m.thresholds);
          const { ring, chipBg, chipTxt } = colors(status);
          const val = m.format ? m.format(m.value) : defaultFormat(m.value);
          const unit = m.unit && showUnit ? ` ${m.unit}` : "";
          const tip =
            m.hint ??
            `${m.label} • ${val}${unit} (${status.toUpperCase()}) ` +
              bandText(m.thresholds);

        return (
          <button
            key={m.id}
            onClick={() => onOpen?.(m.onClickId ?? m.id)}
            title={tip}
            className="text-left bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg p-3 hover:bg-[#101010] transition-colors"
          >
            <div className="flex items-start gap-3">
              {/* traffic light */}
              <div className={`mt-[2px] w-3.5 h-3.5 rounded-full ${ring}`} />
              {/* label + value */}
              <div className="min-w-0 flex-1">
                <div className="text-[12px] text-gray-300 truncate">{m.label}</div>
                <div className="mt-0.5 flex items-center gap-2">
                  <span className="text-sm font-semibold text-gray-100">
                    {val}
                    {unit}
                  </span>
                  <span className={`text-[10px] px-1.5 py-[1px] rounded border ${chipBg} ${chipTxt}`}>
                    {status.toUpperCase()}
                  </span>
                </div>

                {/* band ruler */}
                {showBands && (
                  <BandGauge metric={m} status={status} />
                )}
              </div>
            </div>
          </button>
        );
        })}
      </div>
    </div>
  );
}

/* ---------------- helpers & tiny subcomponents ---------------- */

function defaultFormat(n: number) {
  // smart-ish formatter
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (a >= 1_000_000_000) return `${sign}${(a / 1_000_000_000).toFixed(2)}B`;
  if (a >= 1_000_000) return `${sign}${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 1_000) return `${sign}${(a / 1_000).toFixed(2)}K`;
  return `${sign}${a.toFixed(2)}`;
}

function grade(value: number, th: Thresholds): LightStatus {
  const dir = th.direction ?? "higher"; // default: higher is riskier
  if (dir === "higher") {
    if (value <= th.ok) return "ok";
    if (value <= th.warn) return "warn";
    return "alert";
  } else {
    // lower is riskier
    if (value >= th.ok) return "ok";
    if (value >= th.warn) return "warn";
    return "alert";
  }
}

function colors(s: LightStatus) {
  if (s === "ok")
    return {
      ring: "bg-emerald-500/70 shadow-[0_0_8px_rgba(16,185,129,0.6)]",
      chipBg: "border-emerald-700/60 bg-emerald-700/20",
      chipTxt: "text-emerald-300",
    };
  if (s === "warn")
    return {
      ring: "bg-amber-500/70 shadow-[0_0_8px_rgba(245,158,11,0.6)]",
      chipBg: "border-amber-700/60 bg-amber-700/20",
      chipTxt: "text-amber-300",
    };
  return {
    ring: "bg-red-500/70 shadow-[0_0_8px_rgba(239,68,68,0.6)]",
    chipBg: "border-red-700/60 bg-red-700/20",
    chipTxt: "text-red-300",
  };
}

function bandText(th: Thresholds) {
  const dir = th.direction ?? "higher";
  if (dir === "higher") {
    return `OK ≤ ${th.ok}, WARN ≤ ${th.warn}, ALERT > ${th.warn}`;
  }
  return `OK ≥ ${th.ok}, WARN ≥ ${th.warn}, ALERT < ${th.warn}`;
}

function normalizeTo01(value: number, th: Thresholds): number {
  const dir = th.direction ?? "higher";
  if (dir === "higher") {
    // 0 at 0, 1 at warn (cap at warn+)
    if (th.warn === 0) return 1;
    return Math.max(0, Math.min(1, value / th.warn));
  } else {
    // 0 at warn (bad low), 1 at ok/high
    if (th.ok === th.warn) return value >= th.ok ? 1 : 0;
    const span = Math.max(1e-6, th.ok - th.warn);
    return Math.max(0, Math.min(1, (value - th.warn) / span));
  }
}

function BandGauge({ metric, status }: { metric: RiskMetric; status: LightStatus }) {
  const v = normalizeTo01(metric.value, metric.thresholds);
  const bar =
    status === "ok"
      ? "bg-emerald-500/40"
      : status === "warn"
      ? "bg-amber-500/40"
      : "bg-red-500/50";
  return (
    <div className="mt-2">
      <div className="h-1.5 w-full rounded bg-[#141414] overflow-hidden border border-[#1f1f1f]">
        <div className={`h-full ${bar}`} style={{ width: `${Math.round(v * 100)}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-gray-500">
        <span>{metric.thresholds.direction === "lower" ? "Low (risk)" : "Low"}</span>
        <span>{metric.thresholds.direction === "lower" ? "High (safe)" : "High (risk)"}</span>
      </div>
    </div>
  );
}

/* ---------------- sample usage ----------------
<RiskLights
  columns={4}
  onOpen={(id) => console.log("open", id)}
  items={[
    { id: "var", label: "1d VaR", value: 1.25, unit: "%", thresholds: { ok: 1.5, warn: 2.5, direction: "higher" }, weight: 10, hint: "Portfolio 1-day VaR at 99%." },
    { id: "gross", label: "Gross Exposure", value: 2.1, unit: "x", thresholds: { ok: 2.5, warn: 3.5, direction: "higher" }, weight: 9 },
    { id: "net", label: "Net Exposure", value: 0.35, unit: "x", thresholds: { ok: 0.7, warn: 1, direction: "higher" }, weight: 8 },
    { id: "lev", label: "Leverage", value: 1.8, unit: "x", thresholds: { ok: 2.0, warn: 3.0, direction: "higher" }, weight: 7 },
    { id: "dd", label: "MTD Drawdown", value: -2.4, unit: "%", thresholds: { ok: 0, warn: -3, direction: "lower" }, weight: 6, format: (v)=>`${v.toFixed(2)}%` },
    { id: "liq", label: "Liquidity (days)", value: 4.2, unit: "d", thresholds: { ok: 3, warn: 1, direction: "lower" }, weight: 5 },
    { id: "conc", label: "Top 5 Concentration", value: 42, unit: "%", thresholds: { ok: 45, warn: 60, direction: "higher" }, weight: 4 },
    { id: "mr", label: "Margin Utilization", value: 61, unit: "%", thresholds: { ok: 70, warn: 85, direction: "higher" }, weight: 3 },
    { id: "beta", label: "Beta (SPY)", value: 0.92, unit: "", thresholds: { ok: 1.2, warn: 1.5, direction: "higher" }, weight: 2 },
    { id: "corr", label: "Avg Correlation", value: 0.38, unit: "", thresholds: { ok: 0.5, warn: 0.7, direction: "higher" }, weight: 1 },
  ]}
/>
------------------------------------------------- */