// app/market/layout.tsx
export default function MarketLayout({ children }: { children: any }) {
  return (
    <section style={wrap}>
      <style>{css}</style>

      <header style={header}>
        <div style={container}>
          <h1 style={title}>Market</h1>
          <nav style={nav}>
            <a href="/market" style={link}>Overview</a>
            <a href="/market/equities" style={link}>Equities</a>
            <a href="/market/fixed-income" style={link}>Fixed Income</a>
            <a href="/market/fx" style={link}>FX</a>
            <a href="/market/derivatives" style={link}>Derivatives</a>
          </nav>
        </div>
      </header>

      <main style={main}>
        <div style={container}>{children}</div>
      </main>

      <footer style={footer}>
        <div style={container}>
          <small style={muted}>Data shown is for informational purposes only.</small>
        </div>
      </footer>
    </section>
  );
}

/* ---------------- inline styles (no imports) ---------------- */
const wrap: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  minHeight: "100vh",
};

const header: React.CSSProperties = {
  borderBottom: "1px solid rgba(0,0,0,0.08)",
  background: "#fff",
  position: "sticky",
  top: 0,
  zIndex: 10,
};

const container: React.CSSProperties = {
  maxWidth: 1200,
  margin: "0 auto",
  padding: "12px 16px",
};

const title: React.CSSProperties = {
  margin: 0,
  fontSize: 18,
  lineHeight: "24px",
};

const nav: React.CSSProperties = {
  marginTop: 8,
  display: "flex",
  gap: 12,
  flexWrap: "wrap",
};

const link: React.CSSProperties = {
  display: "inline-block",
  padding: "6px 10px",
  borderRadius: 8,
  background: "#f5f5f7",
  color: "#111",
  textDecoration: "none",
  fontSize: 13,
};

const main: React.CSSProperties = {
  flex: 1,
  padding: "16px 0",
  background: "linear-gradient(180deg, rgba(0,0,0,0.02) 0%, rgba(0,0,0,0.04) 100%)",
};

const footer: React.CSSProperties = {
  borderTop: "1px solid rgba(0,0,0,0.08)",
  background: "#fff",
};

const muted: React.CSSProperties = {
  color: "#666",
  fontSize: 12,
};

/* Dark mode + active link hint (pure CSS) */
const css = `
  @media (prefers-color-scheme: dark) {
    header { background: #0b0b0c; border-bottom-color: rgba(255,255,255,0.08); }
    footer { background: #0b0b0c; border-top-color: rgba(255,255,255,0.08); }
    a[href^="/market"] { background: #1a1a1b; color: #e6e6e6; }
    body { color: #e6e6e6; }
    main { background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.04) 100%); }
  }

  /* Hover/active styles */
  a[href^="/market"]:hover { filter: brightness(0.96); }
  /* Optional: simple "active" highlight via attribute contains current path segment */
  a[href="/market"],
  a[href="/market/"]:where(:any-link) {
    outline: 2px solid transparent;
  }
`;
