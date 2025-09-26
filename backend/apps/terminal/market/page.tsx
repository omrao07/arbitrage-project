// app/market/page.tsx
// Server component: no client hooks, no external imports.
export const dynamic = "force-dynamic";

type IndexRow = { name: string; level: number; chg: number; chgPct: number };
type SectorRow = { name: string; pct: number };
type Mover = { symbol: string; name: string; last: number; pct: number; vol?: number };

const indices: IndexRow[] = [
  { name: "NIFTY 50", level: 24482.3, chg: -68.4, chgPct: -0.28 },
  { name: "SENSEX", level: 80591.7, chg: -112.1, chgPct: -0.14 },
  { name: "BANK NIFTY", level: 49425.9, chg: 185.5, chgPct: 0.38 },
];

const breadth = { adv: 892, dec: 1015, unch: 152 }; // placeholder

const sectors: SectorRow[] = [
  { name: "IT", pct: 0.8 },
  { name: "Financials", pct: 0.5 },
  { name: "Energy", pct: -0.2 },
  { name: "FMCG", pct: -0.6 },
  { name: "Autos", pct: 0.1 },
  { name: "Pharma", pct: -0.4 },
];

const gainers: Mover[] = [
  { symbol: "INFY", name: "Infosys", last: 1675.2, pct: 3.1, vol: 12.1 },
  { symbol: "HCLTECH", name: "HCL Tech", last: 1590.0, pct: 2.7, vol: 8.4 },
  { symbol: "TCS", name: "TCS", last: 4122.5, pct: 2.3, vol: 6.2 },
];

const losers: Mover[] = [
  { symbol: "ITC", name: "ITC Ltd", last: 447.9, pct: -2.4, vol: 10.5 },
  { symbol: "RELIANCE", name: "Reliance", last: 2898.7, pct: -1.9, vol: 9.2 },
  { symbol: "HINDUNILVR", name: "HUL", last: 2482.4, pct: -1.5, vol: 5.1 },
];

export default async function Page() {
  // Wire up your real server fetches here; placeholders above keep this file drop-in ready.
  return (
    <main style={wrap}>
      <header style={header}>
        <h1 style={h1}>Market Overview</h1>
        <p style={sub}>
          Snapshot of indices, sector performance, and movers. Navigate to{" "}
          <a href="/market/equities" style={a}>Equities</a>,{" "}
          <a href="/market/fixed-income" style={a}>Fixed Income</a>,{" "}
          <a href="/market/fx" style={a}>FX</a>,{" "}
          <a href="/market/derivatives" style={a}>Derivatives</a>.
        </p>
      </header>

      {/* Indices snapshot */}
      <section style={grid3}>
        {indices.map((idx) => (
          <div key={idx.name} style={card}>
            <div style={rowBetween}>
              <div style={title}>{idx.name}</div>
              <Badge up={idx.chg >= 0} />
            </div>
            <div style={big}>{fmt(idx.level)}</div>
            <div style={mutedRow}>
              <span style={{ color: idx.chg >= 0 ? "#057a55" : "#b42318" }}>
                {sign(idx.chg)} ({sign(idx.chgPct)}%)
              </span>
            </div>
          </div>
        ))}
      </section>

      {/* Breadth + VIX + Volume */}
      <section style={grid3}>
        <div style={card}>
          <div style={title}>Market Breadth</div>
          <div style={{ marginTop: 10 }}>
            <Bar label="Advancers" value={breadth.adv} max={breadth.adv + breadth.dec + breadth.unch} color="#0ea5e9" />
            <Bar label="Decliners" value={breadth.dec} max={breadth.adv + breadth.dec + breadth.unch} color="#ef4444" />
            <Bar label="Unchanged" value={breadth.unch} max={breadth.adv + breadth.dec + breadth.unch} color="#a1a1aa" />
          </div>
        </div>
        <div style={card}>
          <div style={title}>Implied Vol (VIX)</div>
          <div style={big}>13.8</div>
          <div style={mutedRow}>-0.6 pts (-4.2%)</div>
          <MiniSparkline data={[16, 15.4, 15.8, 14.9, 14.4, 13.8]} />
        </div>
        <div style={card}>
          <div style={title}>Turnover (₹ Cr)</div>
          <div style={big}>78,250</div>
          <div style={mutedRow}>+6% vs 20D avg</div>
          <MiniSparkline data={[62, 65, 70, 68, 71, 78]} />
        </div>
      </section>

      {/* Sector performance */}
      <section style={card}>
        <div style={title}>Sector Performance (Day)</div>
        <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "1fr", gap: 8 }}>
          {sectors.map((s) => (
            <div key={s.name} style={rowBetween}>
              <div style={muted}>{s.name}</div>
                <div style={{ color: s.pct >= 0 ? "#057a55" : "#b42318" }}>
                {s.pct >= 0 ? "▲" : "▼"} {Math.abs(s.pct).toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Movers */}
      <section style={grid2}>
        <div style={card}>
          <div style={title}>Top Gainers</div>
          <MoversTable rows={gainers} />
        </div>
        <div style={card}>
          <div style={title}>Top Losers</div>
          <MoversTable rows={losers} />
        </div>
      </section>

      {/* Links to deep dives */}
      <section style={linksRow}>
        <a href="/market/equities" style={linkBtn}>Explore Equities</a>
        <a href="/market/fixed-income" style={linkBtn}>Explore Fixed Income</a>
        <a href="/market/fx" style={linkBtn}>Explore FX</a>
        <a href="/market/derivatives" style={linkBtn}>Explore Derivatives</a>
      </section>
    </main>
  );
}

/* ---------------- helpers (no imports) ---------------- */
function sign(n: number) {
  const s = n.toFixed(Math.abs(n) < 10 ? 2 : 1);
  return n >= 0 ? `+${s}` : s;
}
function fmt(n: number) {
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function Badge({ up }: { up: boolean }) {
  const bg = up ? "#ecfdf5" : "#fef2f2";
  const fg = up ? "#067647" : "#b42318";
  return (
    <span style={{ background: bg, color: fg, borderRadius: 999, fontSize: 12, padding: "2px 8px" }}>
      {up ? "▲ Up" : "▼ Down"}
    </span>
  );
}

function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.max(0.04, Math.min(1, value / Math.max(1, max)));
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#666" }}>
        <span>{label}</span>
        <span>{value.toLocaleString()}</span>
      </div>
      <div style={{ height: 10, background: "#eee", borderRadius: 8, overflow: "hidden" }}>
        <div style={{ width: `${pct * 100}%`, height: "100%", background: color }} />
      </div>
    </div>
  );
}

function MiniSparkline({ data }: { data: number[] }) {
  const w = 220, h = 48;
  const min = Math.min(...data), max = Math.max(...data);
  const norm = (v: number) => (h - 6) * (1 - (v - min) / Math.max(1e-9, max - min)) + 3;
  const step = (w - 6) / Math.max(1, data.length - 1);
  let d = "";
  data.forEach((v, i) => {
    const x = 3 + i * step;
    const y = norm(v);
    d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
  });
  return (
    <svg width={w} height={h} style={{ marginTop: 10 }}>
      <path d={d} fill="none" stroke="#111" strokeWidth="2" />
    </svg>
  );
}

function MoversTable({ rows }: { rows: Mover[] }) {
  return (
    <div style={{ marginTop: 8 }}>
      <div style={tableHead}>
        <div style={{ ...th, flex: 1.2 }}>Symbol</div>
        <div style={{ ...th, flex: 2.4 }}>Name</div>
        <div style={{ ...th, flex: 1, textAlign: "right" }}>Last</div>
        <div style={{ ...th, flex: 1, textAlign: "right" }}>% Chg</div>
        <div style={{ ...th, flex: 1, textAlign: "right" }}>Vol (M)</div>
      </div>
      <div>
        {rows.map((r) => (
          <div key={r.symbol} style={tr}>
            <div style={{ ...td, flex: 1.2, fontWeight: 600 }}>{r.symbol}</div>
            <div style={{ ...td, flex: 2.4, color: "#555" }}>{r.name}</div>
            <div style={{ ...td, flex: 1, textAlign: "right" }}>{fmt(r.last)}</div>
            <div style={{ ...td, flex: 1, textAlign: "right", color: r.pct >= 0 ? "#067647" : "#b42318" }}>
              {sign(r.pct)}%
            </div>
            <div style={{ ...td, flex: 1, textAlign: "right" }}>{r.vol?.toFixed(1) ?? "—"}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------------- inline styles ---------------- */
const wrap: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 16,
  padding: 16,
};

const header: React.CSSProperties = { marginBottom: 4 };
const h1: React.CSSProperties = { margin: 0, fontSize: 22, lineHeight: "28px" };
const sub: React.CSSProperties = { margin: "4px 0 0", color: "#555", fontSize: 13 };
const a: React.CSSProperties = { color: "#0f62fe", textDecoration: "none" };

const grid3: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr",
  gap: 12,
};

const grid2: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr",
  gap: 12,
};

const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 16,
};

const rowBetween: React.CSSProperties = { display: "flex", alignItems: "center", justifyContent: "space-between" };
const title: React.CSSProperties = { fontSize: 14, fontWeight: 600 };
const big: React.CSSProperties = { fontSize: 20, fontWeight: 700, marginTop: 6 };
const mutedRow: React.CSSProperties = { marginTop: 4, color: "#666", fontSize: 13 };
const muted: React.CSSProperties = { color: "#666", fontSize: 13 };

const tableHead: React.CSSProperties = { display: "flex", gap: 10, padding: "6px 0", borderBottom: "1px solid #eee", color: "#666", fontSize: 12 };
const th: React.CSSProperties = { };
const tr: React.CSSProperties = { display: "flex", gap: 10, padding: "10px 0", borderBottom: "1px dashed #eee" };
const td: React.CSSProperties = { fontSize: 13 };

const linksRow: React.CSSProperties = { display: "flex", gap: 10, flexWrap: "wrap" };
const linkBtn: React.CSSProperties = {
  background: "#f4f4f5",
  color: "#111",
  borderRadius: 10,
  padding: "8px 12px",
  textDecoration: "none",
  border: "1px solid rgba(0,0,0,0.12)",
};

/* Basic responsiveness via attribute selector trick (no CSS file) */
if (typeof window !== "undefined") {
  // no-op; layout rules are purely inline for SSR
}
