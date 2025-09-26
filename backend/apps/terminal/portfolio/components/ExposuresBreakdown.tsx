// app/components/xposuresbreakdown.tsx
// No imports. No hooks. Self-contained exposures breakdown renderer with bars.
// Pass any of the buckets (asset / sector / country / currency) as Record<label, value>.
// Interactions are done with plain DOM (form onChange).

"use client";

type Buckets = Record<string, number>;

type Props = {
  title?: string;
  unit?: string;                 // e.g. "$" or "₹" (prefix for absolute)
  decimals?: number;             // value decimals for absolute, default 2
  percentDecimals?: number;      // value decimals for percent, default 1
  maxRows?: number;              // show top N then roll-up into "Others"
  asset?: Buckets;               // e.g. { Equity: 650000, Debt: 120000, Cash: 30000 }
  sector?: Buckets;              // e.g. { IT: 0.32, Banks: 0.28, ... } (percent or absolute—treated as absolute)
  country?: Buckets;             // e.g. { India: 0.7, US: 0.2, ... }
  currency?: Buckets;            // e.g. { INR: 0.85, USD: 0.1, ... }
  note?: string;
};

export default function XposuresBreakdown({
  title = "Exposures Breakdown",
  unit = "",
  decimals = 2,
  percentDecimals = 1,
  maxRows = 8,
  asset,
  sector,
  country,
  currency,
  note,
}: Props) {
  const sections: Array<{ id: string; name: string; data?: Buckets }> = [
    { id: "asset",   name: "By Asset Class", data: asset },
    { id: "sector",  name: "By Sector",      data: sector },
    { id: "country", name: "By Country",     data: country },
    { id: "ccy",     name: "By Currency",    data: currency },
  ].filter((s) => s.data && Object.keys(s.data as Buckets).length > 0);

  function onModeChange(e: any) {
    const form = e.currentTarget as HTMLFormElement;
    const mode = (form.elements.namedItem("mode") as RadioNodeList).value; // "percent" | "absolute"
    const sec = form.closest('[data-sec]') as HTMLElement;
    if (!sec) return;

    const rows = Array.from(sec.querySelectorAll<HTMLElement>('[data-row]'));
    const maxAbs = Number(sec.getAttribute("data-maxabs") || "1") || 1;

    rows.forEach((row) => {
      const pct = Number(row.getAttribute("data-pct") || "0");
      const abs = Number(row.getAttribute("data-abs") || "0");
      const sign = abs < 0 ? -1 : 1;

      const bar = row.querySelector<HTMLElement>('[data-bar]');
      const val = row.querySelector<HTMLElement>('[data-val]');
      if (!bar || !val) return;

      if (mode === "percent") {
        bar.style.setProperty("--w", `${Math.max(0, Math.min(100, Math.abs(pct)))}%`);
        bar.setAttribute("aria-valuenow", String(Math.round(pct)));
        val.textContent = fmtPct(pct, percentDecimals);
      } else {
        const rel = Math.abs(abs) / maxAbs * 100;
        bar.style.setProperty("--w", `${Math.max(0, Math.min(100, rel))}%`);
        bar.setAttribute("aria-valuenow", String(Math.round(rel)));
        val.textContent = fmtAbs(abs, unit, decimals);
      }

      // color hint for negatives
      bar.style.setProperty("--bar-bg", sign < 0 ? "#fee2e2" : "#ecfeff");
      bar.style.setProperty("--bar-fg", sign < 0 ? "#b42318" : "#0e7490");
    });
  }

  // Initialize bars after hydration (no hooks).
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      document.querySelectorAll<HTMLFormElement>('form[data-mode]').forEach((f) => onModeChange({ currentTarget: f }));
    });
  }

  return (
    <section style={wrap} aria-label="Exposures breakdown">
      <style>{css}</style>

      <header style={header}>
        <h3 style={h3}>{title}</h3>
        {note ? <p style={sub}>{note}</p> : null}
      </header>

      {sections.length === 0 ? (
        <div style={emptyBox}>No exposure data.</div>
      ) : (
        <div style={grid}>
          {sections.map((sec) => {
            const rows = normalize(sec.data!, maxRows);
            const { sumPos, maxAbs } = totals(rows);
            return (
              <article key={sec.id} data-sec={sec.id} data-maxabs={maxAbs} style={card}>
                <header style={cardHead}>
                  <div style={{ fontWeight: 600 }}>{sec.name}</div>
                  <form data-mode onChange={onModeChange} style={toggleRow}>
                    <label style={toggleLbl}>
                      <input type="radio" name="mode" value="percent" defaultChecked /> %
                    </label>
                    <label style={toggleLbl}>
                      <input type="radio" name="mode" value="absolute" /> {unit || "Value"}
                    </label>
                  </form>
                </header>

                <ul style={list}>
                  {rows.map((r) => {
                    const pct = share(r.value, sumPos);
                    return (
                      <li key={r.label} data-row data-abs={r.value} data-pct={pct} style={li}>
                        <div style={labelCell} title={r.label}>{r.label}</div>
                        <div style={barCell}>
                          <div data-bar role="progressbar" aria-valuemin={0} aria-valuemax={100} style={bar} />
                        </div>
                        <div data-val style={valCell}>{fmtPct(pct, percentDecimals)}</div>
                      </li>
                    );
                  })}
                </ul>

                <footer style={foot}>
                  <span style={footNote}>Positive base: {fmtAbs(sumPos, unit, decimals)}</span>
                  <span style={hint}>Toggle to see {unit ? unit : "absolute"} values</span>
                </footer>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

/* ---------------- helpers ---------------- */
function normalize(b: Buckets, maxRows: number): Array<{ label: string; value: number }> {
  const rows = Object.entries(b)
    .map(([k, v]) => ({ label: String(k), value: Number(v) || 0 }))
    .filter((r) => Number.isFinite(r.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  if (rows.length > maxRows) {
    const head = rows.slice(0, maxRows);
    const tail = rows.slice(maxRows);
    const others = tail.reduce((s, r) => s + r.value, 0);
    head.push({ label: "Others", value: others });
    return head;
  }
  return rows;
}

function totals(rows: Array<{ value: number }>) {
  const sumPos = rows.reduce((s, r) => s + Math.max(0, r.value), 0);
  const fallback = rows.reduce((s, r) => s + Math.abs(r.value), 0);
  const base = sumPos > 0 ? sumPos : fallback || 1;
  const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(r.value)), 1);
  return { sumPos: base, maxAbs };
}

function share(val: number, base: number) {
  return base > 0 ? (val / base) * 100 : 0;
}

function fmtPct(x: number, d = 1) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x);
  return `${sign}${v.toFixed(d)}%`;
}
function fmtAbs(x: number, unit: string, d = 2) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x);
  return `${sign}${unit}${nfmt(v, d)}`;
}
function nfmt(n: number, d: number) {
  // compact-ish formatting with fixed decimals for small numbers
  if (Math.abs(n) >= 1_000_000_000) return (n / 1_000_000_000).toFixed(d) + "B";
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(d) + "M";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(d) + "k";
  return n.toFixed(d);
}

/* ---------------- styles ---------------- */
const wrap: any = { display: "flex", flexDirection: "column", gap: 12, padding: 12 };
const header: any = { marginBottom: 2 };
const h3: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const grid: any = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 };
const card: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 12,
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const cardHead: any = { display: "flex", alignItems: "center", justifyContent: "space-between" };
const toggleRow: any = { display: "inline-flex", gap: 8, alignItems: "center" };
const toggleLbl: any = { display: "inline-flex", gap: 4, alignItems: "center", fontSize: 12, color: "#444" };

const list: any = { listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 6 };
const li: any = { display: "grid", gridTemplateColumns: "1fr 140px 80px", alignItems: "center", gap: 8 };
const labelCell: any = { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 13 };
const barCell: any = { height: 18, background: "#f4f4f5", borderRadius: 999, position: "relative", overflow: "hidden" };
const bar: any = {
  position: "absolute",
  left: 0,
  top: 0,
  bottom: 0,
  width: "var(--w, 0%)",
  background: "var(--bar-bg, #ecfeff)",
  borderRight: "2px solid var(--bar-fg, #0e7490)",
  transition: "width .25s ease",
  borderRadius: 999,
};
const valCell: any = { fontVariantNumeric: "tabular-nums", textAlign: "right", fontSize: 12.5 };

const foot: any = { display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 };
const footNote: any = { color: "#666", fontSize: 12 };
const hint: any = { color: "#777", fontSize: 11 };

const emptyBox: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.04)",
  padding: 16,
  textAlign: "center",
  color: "#555",
};

const css = `
  @media (prefers-color-scheme: dark) {
    section > div > article[style], div[style*="empty"] {
      background: #0b0b0c !important;
      border-color: rgba(255,255,255,0.08) !important;
      box-shadow: 0 6px 24px rgba(0,0,0,0.6) !important;
    }
  }
`;
