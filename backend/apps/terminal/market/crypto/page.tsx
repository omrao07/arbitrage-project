"use client";

/**
 * app/crypto/page.tsx
 * Zero-import, self-contained page component for crypto dashboard.
 * - No external imports: only React ambient
 * - Shows crypto market overview using mocked data
 * - Sections: header, tiles, and simple table
 */

export default async function CryptoPage() {
  // Mocked snapshot data
  const snaps: CryptoSnap[] = [
    { symbol: "BTC", price: 65000, change24h: 2.1, volume24h: 32000000000, marketCap: 1.27e12 },
    { symbol: "ETH", price: 3400, change24h: -1.4, volume24h: 18000000000, marketCap: 420000000000 },
    { symbol: "SOL", price: 160, change24h: 5.3, volume24h: 5000000000, marketCap: 72000000000 },
    { symbol: "XRP", price: 0.6, change24h: 0.4, volume24h: 1500000000, marketCap: 31000000000 },
  ];

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      {/* Header */}
      <div className="sticky top-0 z-20 border-b border-neutral-800 bg-neutral-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
          <h1 className="text-lg font-semibold tracking-tight">Crypto Dashboard</h1>
          <p className="text-xs text-neutral-400">Updated just now (mock)</p>
        </div>
      </div>

      {/* Tiles */}
      <div className="mx-auto grid max-w-7xl grid-cols-2 gap-4 px-4 py-6 md:grid-cols-4">
        {snaps.map((c) => (
          <Tile
            key={c.symbol}
            label={c.symbol}
            value={`$${fmtNum(c.price)}`}
            accent={c.change24h >= 0 ? "pos" : "neg"}
            sub={`${c.change24h >= 0 ? "+" : ""}${c.change24h.toFixed(2)}%`}
          />
        ))}
      </div>

      {/* Table */}
      <div className="mx-auto max-w-7xl px-4 pb-10">
        <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
          <table className="min-w-full text-sm">
            <thead className="bg-neutral-800/60 text-neutral-400">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Symbol</th>
                <th className="px-3 py-2 text-right font-medium">Price</th>
                <th className="px-3 py-2 text-right font-medium">24h %</th>
                <th className="px-3 py-2 text-right font-medium">24h Vol (USD)</th>
                <th className="px-3 py-2 text-right font-medium">Market Cap</th>
              </tr>
            </thead>
            <tbody>
              {snaps.map((c, i) => (
                <tr key={i} className="border-t border-neutral-800">
                  <td className="px-3 py-2">{c.symbol}</td>
                  <td className="px-3 py-2 text-right">${fmtNum(c.price)}</td>
                  <td
                    className={`px-3 py-2 text-right ${
                      c.change24h >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {c.change24h >= 0 ? "+" : ""}
                    {c.change24h.toFixed(2)}%
                  </td>
                  <td className="px-3 py-2 text-right">${fmtNum(c.volume24h)}</td>
                  <td className="px-3 py-2 text-right">${fmtNum(c.marketCap)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* --------------------------- UI helpers --------------------------- */

type CryptoSnap = {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  marketCap: number;
};

function Tile({
  label,
  value,
  sub,
  accent = "mut",
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: "pos" | "neg" | "mut";
}) {
  const color =
    accent === "pos"
      ? "text-emerald-400"
      : accent === "neg"
      ? "text-rose-400"
      : "text-neutral-100";
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-neutral-500">{sub}</div>}
    </div>
  );
}

function fmtNum(n: number, d = 2) {
  if (n >= 1e12) return (n / 1e12).toFixed(d) + "T";
  if (n >= 1e9) return (n / 1e9).toFixed(d) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(d) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(d) + "K";
  return n.toLocaleString("en-US", { maximumFractionDigits: d });
}

/* ---------------- Ambient React (no imports) ---------------- */
// Remove if you prefer real imports.
declare const React: any;