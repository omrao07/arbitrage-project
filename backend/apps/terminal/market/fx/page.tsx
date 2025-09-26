// app/fx/page.tsx

import FxVolSurface from "./panes/FxVolSurface";


export const dynamic = "force-dynamic"; // avoid caching if you fetch live data downstream

type SearchParams = Record<string, string | string[]>;

function getParam(sp: SearchParams | undefined, key: string) {
  const v = sp?.[key];
  return Array.isArray(v) ? v[0] : v;
}

export default function Page({
  searchParams,
}: {
  searchParams?: SearchParams;
}) {
  // URL usage examples:
  // /fx?pair=USD/INR
  // /fx?pair=EUR/USD&atm=vega
  const pairParam = getParam(searchParams, "pair");
  const atmParam = getParam(searchParams, "atm");

  // Validate/normalize ATM method from URL (?atm=vega | delta)
  const atmMethod =
    atmParam?.toLowerCase().startsWith("vega")
      ? ("vega-weighted" as const)
      : ("delta-neutral" as const);

  // basic inline layout
  return (
    <main style={wrap}>
      <header style={header}>
        <h1 style={h1}>FX Volatility</h1>
        <p style={sub}>
          View FX implied volatility surface (heatmap), with optional URL params{" "}
          <code>?pair=USD/INR&amp;atm=vega</code>
        </p>
      </header>

      {/* Client component (pure React, no external imports) */}
      <FxVolSurface
        initialPair={(pairParam as any) || "EUR/USD"}
        supportedPairs={["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/INR"]}
      />

      <footer style={foot}>
        <small>
          Tip: switch pair via URL, e.g. <code>/fx?pair=USD/INR</code>. Set ATM
          method via <code>atm=vega</code> or default delta-neutral.
        </small>
      </footer>
    </main>
  );
}

/* -------- inline styles -------- */
const wrap: React.CSSProperties = {
  padding: 16,
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

const header: React.CSSProperties = {
  marginBottom: 4,
};

const h1: React.CSSProperties = {
  margin: 0,
  fontSize: 22,
  lineHeight: "28px",
};

const sub: React.CSSProperties = {
  margin: "6px 0 0",
  color: "#555",
  fontSize: 13,
};

const foot: React.CSSProperties = {
  marginTop: 6,
  color: "#666",
  fontSize: 12,
};
