// app/fx/loading.tsx
export default function Loading() {
  return (
    <div style={wrap} aria-busy="true" aria-live="polite">
      <style>{shimmerCss}</style>

      {/* Header skeleton */}
      <div style={row}>
        <div style={rowLeft}>
          <div style={pill} className="shimmer" />
          <div style={{ ...pill, width: 140 }} className="shimmer" />
        </div>
        <div style={rowRight}>
          <div style={{ ...btn, width: 90 }} className="shimmer" />
          <div style={{ ...btn, width: 110 }} className="shimmer" />
        </div>
      </div>

      {/* Heatmap skeleton */}
      <div style={card}>
        <div style={cardHeader}>
          <div style={{ ...line, width: 160 }} className="shimmer" />
          <div style={{ ...line, width: 220, height: 10 }} className="shimmer" />
        </div>
        <div style={heatmapBox} className="shimmer" />
        <div style={axisRow}>
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} style={{ ...tick, width: 24 }} className="shimmer" />
          ))}
        </div>
      </div>

      {/* Charts skeleton */}
      <div style={grid}>
        <div style={card}>
          <div style={cardHeader}>
            <div style={{ ...line, width: 100 }} className="shimmer" />
            <div style={{ ...pill, width: 72 }} className="shimmer" />
          </div>
          <div style={chartBox} className="shimmer" />
        </div>

        <div style={card}>
          <div style={cardHeader}>
            <div style={{ ...line, width: 120 }} className="shimmer" />
            <div style={{ ...pill, width: 60 }} className="shimmer" />
          </div>
          <div style={chartBox} className="shimmer" />
        </div>
      </div>
    </div>
  );
}

/* ---------- inline styles ---------- */
const wrap: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 16,
  padding: 16,
};

const row: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 12,
};

const rowLeft: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
};

const rowRight: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
};

const pill: React.CSSProperties = {
  height: 34,
  width: 120,
  borderRadius: 10,
  background: "#eee",
};

const btn: React.CSSProperties = {
  height: 34,
  width: 100,
  borderRadius: 8,
  background: "#eee",
};

const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 16,
};

const cardHeader: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  marginBottom: 12,
};

const line: React.CSSProperties = {
  height: 14,
  width: 140,
  borderRadius: 6,
  background: "#eee",
};

const heatmapBox: React.CSSProperties = {
  height: 260,
  borderRadius: 12,
  background: "#eee",
};

const axisRow: React.CSSProperties = {
  marginTop: 10,
  display: "grid",
  gridTemplateColumns: "repeat(10, 1fr)",
  gap: 8,
};

const tick: React.CSSProperties = {
  height: 8,
  borderRadius: 4,
  background: "#eee",
};

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr",
  gap: 16,
};

const chartBox: React.CSSProperties = {
  height: 240,
  borderRadius: 12,
  background: "#eee",
};

/* ---------- shimmer animation ---------- */
const shimmerCss = `
  @keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
  }
  .shimmer {
    background: linear-gradient(90deg, #f2f2f2 25%, #e9e9e9 37%, #f2f2f2 63%);
    background-size: 400px 100%;
    animation: shimmer 1.2s infinite linear;
  }
  @media (min-width: 1024px) {
    div[style*="grid-template-columns: 1fr"] {
      grid-template-columns: 1fr 1fr !important;
    }
  }
  @media (prefers-color-scheme: dark) {
    .shimmer {
      background: linear-gradient(90deg, #1d1d1f 25%, #2a2a2e 37%, #1d1d1f 63%);
      background-size: 400px 100%;
    }
    div[style*="box-shadow"][style*="border-radius"] {
      background: #0b0b0c !important;
      border-color: rgba(255,255,255,0.08) !important;
      box-shadow: 0 6px 24px rgba(0,0,0,0.5) !important;
    }
  }
`;
