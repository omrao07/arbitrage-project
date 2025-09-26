// app/news/loading.tsx  (or app/market/news/loading.tsx)
// No imports. Lightweight shimmer skeleton for the News page.

export default function Loading() {
  const items = Array.from({ length: 8 });

  return (
    <section role="status" aria-busy="true" style={wrap}>
      <style>{css}</style>

      {/* Header skeleton */}
      <div style={controls}>
        <div style={searchWrap}>
          <div className="shimmer" style={{ height: 34, borderRadius: 10 }} />
        </div>
        <div style={tabs}>
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="shimmer" style={tabShimmer} />
          ))}
        </div>
      </div>

      {/* Cards */}
      <div style={list}>
        {items.map((_, i) => (
          <article key={i} style={card}>
            <div className="shimmer" style={{ height: 16, width: "70%", borderRadius: 8, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 12, width: "95%", borderRadius: 8, marginBottom: 6 }} />
            <div className="shimmer" style={{ height: 12, width: "85%", borderRadius: 8, marginBottom: 6 }} />
            <div className="shimmer" style={{ height: 10, width: "30%", borderRadius: 8, marginTop: 10 }} />
          </article>
        ))}
      </div>
    </section>
  );
}

/* ---------------- styles (no imports) ---------------- */
const wrap: any = { display: "flex", flexDirection: "column", gap: 12, padding: 16 };

const controls: any = {
  display: "flex",
  gap: 12,
  justifyContent: "space-between",
  alignItems: "center",
  flexWrap: "wrap",
};

const searchWrap: any = { flex: "1 1 320px", minWidth: 240 };

const tabs: any = { display: "flex", gap: 8, flexWrap: "wrap" };
const tabShimmer: any = { height: 28, width: 90, borderRadius: 999 };

const list: any = { display: "grid", gridTemplateColumns: "1fr", gap: 12 };

const card: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 14,
};

/* shimmer + dark mode */
const css = `
  @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
  .shimmer {
    background: linear-gradient(90deg, #f2f2f2 25%, #e9e9e9 37%, #f2f2f2 63%);
    background-size: 400px 100%;
    animation: shimmer 1.2s infinite linear;
  }
  @media (prefers-color-scheme: dark) {
    .shimmer {
      background: linear-gradient(90deg, #1d1d1f 25%, #2a2a2e 37%, #1d1d1f 63%);
    }
    article[style] {
      background: #0b0b0c !important;
      border-color: rgba(255,255,255,0.08) !important;
      box-shadow: 0 6px 24px rgba(0,0,0,0.6) !important;
    }
  }
`;
