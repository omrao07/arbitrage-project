// app/market/loading.tsx
export default function Loading() {
  return (
    <div style={wrap} aria-busy="true" aria-live="polite">
      <style>{shimmerCss}</style>

      {/* Header skeleton */}
      <div style={header}>
        <div style={{ ...line, width: 120, height: 20 }} className="shimmer" />
        <div style={nav}>
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              style={{ ...pill, width: 80, height: 24 }}
              className="shimmer"
            />
          ))}
        </div>
      </div>

      {/* Cards row */}
      <div style={cards}>
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} style={card}>
            <div
              style={{ ...line, width: "60%", height: 16 }}
              className="shimmer"
            />
            <div
              style={{ ...line, width: "40%", height: 12, marginTop: 10 }}
              className="shimmer"
            />
            <div
              style={{ ...box, height: 80, marginTop: 14 }}
              className="shimmer"
            />
          </div>
        ))}
      </div>

      {/* Table skeleton */}
      <div style={card}>
        <div
          style={{ ...line, width: 160, height: 16, marginBottom: 12 }}
          className="shimmer"
        />
        <div style={table}>
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} style={row}>
              <div
                style={{ ...line, width: "30%", height: 12 }}
                className="shimmer"
              />
              <div
                style={{ ...line, width: "20%", height: 12 }}
                className="shimmer"
              />
              <div
                style={{ ...line, width: "20%", height: 12 }}
                className="shimmer"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------- inline styles ---------- */
const wrap: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 20,
  padding: 16,
};

const header: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const nav: React.CSSProperties = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
};

const cards: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr",
  gap: 12,
};

const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 12,
  padding: 16,
  boxShadow: "0 4px 12px rgba(0,0,0,0.04)",
};

const line: React.CSSProperties = {
  background: "#eee",
  borderRadius: 6,
};

const pill: React.CSSProperties = {
  background: "#eee",
  borderRadius: 12,
};

const box: React.CSSProperties = {
  background: "#eee",
  borderRadius: 8,
};

const table: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const row: React.CSSProperties = {
  display: "flex",
  gap: 20,
  justifyContent: "space-between",
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
  @media (min-width: 768px) {
    div[style*="grid-template-columns: 1fr"] {
      grid-template-columns: repeat(3, 1fr) !important;
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
      box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
  }
`;
