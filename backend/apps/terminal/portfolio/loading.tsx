// app/portfolio/loading.tsx
// Minimal skeleton loader for the /portfolio route.
// No imports. No hooks. Inline CSS shimmer.

export default function Loading() {
  const rows = Array.from({ length: 10 });

  return (
    <section style={wrap} aria-busy="true" aria-live="polite">
      <style>{css}</style>

      <header style={head}>
        <div style={{ display: "grid", gap: 6 }}>
          <div style={sk(160, 18)} />
          <div style={sk(280, 12)} />
        </div>

        <div style={ctrls}>
          <div style={{ position: "relative" }}>
            <div style={{ ...sk(220, 30), borderRadius: 10 }} />
          </div>
          <div style={{ ...sk(92, 30), borderRadius: 10 }} />
        </div>
      </header>

      <div style={{ overflow: "auto" }}>
        <div style={table}>
          {/* header */}
          <div style={thead}>
            {["Symbol", "Sector", "Qty", "Price", "Value", "P&L", "P&L%"].map((_, i) => (
              <div key={i} style={thCell}>
                <div style={sk(i % 2 === 0 ? 64 : 48, 10)} />
              </div>
            ))}
          </div>

          {/* rows */}
          <div>
            {rows.map((_, i) => (
              <div key={i} style={tr}>
                <div style={tdLeft}>
                  <div style={{ display: "grid", gap: 4 }}>
                    <div style={sk(72 + ((i * 7) % 24), 12)} />
                    <div style={{ ...sk(90, 8), opacity: 0.6 }} />
                  </div>
                </div>
                <div style={td}><div style={sk(60, 10)} /></div>
                <div style={td}><div style={sk(40, 10)} /></div>
                <div style={td}><div style={sk(56, 10)} /></div>
                <div style={td}><div style={sk(76, 10)} /></div>
                <div style={td}><div style={sk(68, 10)} /></div>
                <div style={td}><div style={sk(48, 10)} /></div>
              </div>
            ))}
          </div>

          {/* footer */}
          <div style={tfoot}>
            <div style={{ gridColumn: "1 / span 2" }} />
            <div />
            <div />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <div style={sk(90, 12)} />
            </div>
            <div />
            <div />
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------------- helpers & styles ---------------- */

function sk(w: number, h: number): any {
  return {
    width: w,
    height: h,
    borderRadius: 6,
    background:
      "linear-gradient(90deg, var(--sk-a) 25%, var(--sk-b) 37%, var(--sk-a) 63%)",
    backgroundSize: "400% 100%",
    animation: "sk-shimmer 1.1s ease-in-out infinite",
  };
}

const wrap: any = { display: "flex", flexDirection: "column", gap: 12, padding: 12 };
const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const ctrls: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };

const table: any = {
  minWidth: 760,
  border: "1px solid var(--tbl-border)",
  borderRadius: 14,
  overflow: "hidden",
  background: "var(--tbl-bg)",
};

const thead: any = {
  display: "grid",
  gridTemplateColumns: "minmax(180px, 2fr) 1.2fr .8fr 1fr 1.2fr 1fr .8fr",
  gap: 0,
  padding: "8px 10px",
  borderBottom: "1px solid var(--tbl-border)",
  background: "var(--tbl-head)",
};
const thCell: any = { display: "flex", alignItems: "center", justifyContent: "flex-start" };

const tr: any = {
  display: "grid",
  gridTemplateColumns: "minmax(180px, 2fr) 1.2fr .8fr 1fr 1.2fr 1fr .8fr",
  gap: 0,
  padding: "10px",
  borderBottom: "1px solid var(--tbl-row)",
};
const tdLeft: any = { display: "flex", alignItems: "center", gap: 10 };
const td: any = { display: "flex", alignItems: "center", justifyContent: "flex-end" };

const tfoot: any = {
  display: "grid",
  gridTemplateColumns: "minmax(180px, 2fr) 1.2fr .8fr 1fr 1.2fr 1fr .8fr",
  gap: 0,
  padding: "10px",
  background: "var(--tbl-foot)",
};

const css = `
  :root {
    --sk-a:#f3f4f6; --sk-b:#e5e7eb;
    --tbl-bg:#fff; --tbl-head:#fafafa; --tbl-foot:#fafafa;
    --tbl-border:#e5e7eb; --tbl-row:#f0f0f1;
  }
  @keyframes sk-shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --sk-a:#1a1b1e; --sk-b:#24262a;
      --tbl-bg:#0b0b0c; --tbl-head:#0f0f11; --tbl-foot:#0f0f11;
      --tbl-border:rgba(255,255,255,.08); --tbl-row:rgba(255,255,255,.06);
    }
  }
`;
