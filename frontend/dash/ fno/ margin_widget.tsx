'use client';
import React from 'react';

type MarginProps = {
  balance?: number;          // total account equity
  used?: number;             // margin currently used
  currency?: string;         // display currency code
  warningPct?: number;       // color turns amber above this %
  dangerPct?: number;        // color turns red above this %
  title?: string;
};

export default function MarginWidget({
  balance = 100000,
  used = 25000,
  currency = 'USD',
  warningPct = 70,
  dangerPct = 90,
  title = 'Margin Overview',
}: MarginProps) {
  const free = Math.max(0, balance - used);
  const ratio = balance > 0 ? (used / balance) * 100 : 0;

  const color = ratio < warningPct ? '#10b981' : ratio < dangerPct ? '#f59e0b' : '#ef4444';

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.titleRow}>
          <span style={styles.title}>{title}</span>
          <span style={{ ...styles.badge, background: '#eef2ff', color: '#3730a3' }}>
            {currency}
          </span>
        </div>
      </div>

      <div style={styles.body}>
        {/* KPI grid */}
        <div style={styles.grid}>
          <KPI label="Account Balance" value={balance} currency={currency} />
          <KPI label="Used Margin" value={used} currency={currency} />
          <KPI label="Free Margin" value={free} currency={currency} />
          <KPI label="Margin Ratio" value={`${ratio.toFixed(1)}%`} />
        </div>

        {/* Progress bar */}
        <div style={{ marginTop: 18 }}>
          <div style={styles.rowHeader}>
            <span style={styles.rowLabel}>Utilization</span>
            <span style={{ ...styles.rowValue, color }}>{ratio.toFixed(1)}%</span>
          </div>
          <ProgressBar ratio={ratio} color={color} />
          <div style={styles.legend}>
            <Legend swatch="#10b981" text={`< ${warningPct}%`} />
            <Legend swatch="#f59e0b" text={`${warningPct}–${dangerPct}%`} />
            <Legend swatch="#ef4444" text={`≥ ${dangerPct}%`} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- components ---------------- */

function KPI({ label, value, currency }: { label: string; value: number | string; currency?: string }) {
  const formatted =
    typeof value === 'number'
      ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
      : value;
  const showCurr = currency && typeof value === 'number';
  return (
    <div style={styles.kpiCard}>
      <div style={styles.kpiLabel}>{label}</div>
      <div style={styles.kpiValue}>
        {formatted}
        {showCurr ? ` ${currency}` : ''}
      </div>
    </div>
  );
}

function ProgressBar({ ratio, color }: { ratio: number; color: string }) {
  const width = Math.max(0, Math.min(100, ratio));
  return (
    <div style={styles.progressOuter}>
      <div style={{ ...styles.progressInner, width: `${width}%`, background: color }} />
    </div>
  );
}

function Legend({ swatch, text }: { swatch: string; text: string }) {
  return (
    <div style={styles.legendItem}>
      <span style={{ ...styles.swatch, background: swatch }} />
      <span>{text}</span>
    </div>
  );
}

/* ---------------- styles ---------------- */

const styles: Record<string, React.CSSProperties> = {
  card: {
    border: '1px solid #e5e7eb',
    borderRadius: 16,
    background: '#fff',
    boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
    width: '100%',
    maxWidth: '100%',          // fill container
    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
  },
  header: {
    padding: '16px 20px',
    borderBottom: '1px solid #eee',
  },
  titleRow: { display: 'flex', alignItems: 'center', gap: 10 },
  title: { fontWeight: 700, fontSize: 20, color: '#111827' },
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    borderRadius: 999,
    padding: '4px 10px',
    fontSize: 12,
    fontWeight: 600,
  },
  body: { padding: 20 },

  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
    gap: 12,
  },
  kpiCard: {
    border: '1px solid #eef2f7',
    borderRadius: 12,
    padding: '14px 12px',
    background: '#fafafa',
  },
  kpiLabel: { fontSize: 12, color: '#6b7280', marginBottom: 6 },
  kpiValue: { fontSize: 18, fontWeight: 700, color: '#111827' },

  rowHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  rowLabel: { fontSize: 14, color: '#374151', fontWeight: 600 },
  rowValue: { fontSize: 14, fontWeight: 700 },

  progressOuter: {
    position: 'relative',
    height: 16,
    borderRadius: 10,
    overflow: 'hidden',
    background: '#f3f4f6',
    border: '1px solid #e5e7eb',
  },
  progressInner: {
    height: '100%',
    borderRadius: 10,
    transition: 'width 0.3s ease',
  },

  legend: {
    display: 'flex',
    gap: 16,
    alignItems: 'center',
    marginTop: 8,
    color: '#6b7280',
    fontSize: 12,
  },
  legendItem: { display: 'flex', alignItems: 'center', gap: 8 },
  swatch: {
    width: 10,
    height: 10,
    borderRadius: 3,
    display: 'inline-block',
    border: '1px solid rgba(0,0,0,0.1)',
  },
};