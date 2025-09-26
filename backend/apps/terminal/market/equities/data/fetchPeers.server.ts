// fetchPeers.server.ts
// Zero-import, deterministic mock “peer group” generator for equities.
// Can be used server-side or edge. No external libs.
//
// Exports:
//   - EquityPeer type
//   - fetchPeers(symbol)
//   - optional handleGET(url) for Next.js route handler

/* ------------------------------- Types ------------------------------- */

export type EquityPeer = {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  marketCap: number;    // $
  price: number;        // last close
  changePct: number;    // daily % change
  beta: number;
  pe: number;
};

/* ------------------------------- API -------------------------------- */

export async function fetchPeers(symbol: string): Promise<EquityPeer[]> {
  const key = symbol.toUpperCase();
  const rng = mulberry32(hash(key));

  const sector = guessSector(key);
  const industry = guessIndustry(key);
  const basePrice = 50 + (hash(key) % 500);

  // choose 5–8 peers
  const n = 5 + Math.floor(rng() * 4);
  const peers: EquityPeer[] = [];
  for (let i = 0; i < n; i++) {
    const ticker = fakeTicker(key, i);
    const price = round(basePrice * (0.5 + rng()), 2);
    const mc = Math.round(price * (1e8 + rng() * 9e8));
    peers.push({
      symbol: ticker,
      name: fakeName(ticker),
      sector,
      industry,
      marketCap: mc,
      price,
      changePct: round((rng() - 0.5) * 0.04, 4), // ±2%
      beta: round(0.5 + rng() * 1.5, 2),
      pe: round(5 + rng() * 40, 1),
    });
  }
  return peers;
}

/**
 * Optional helper to expose as Next.js route handler.
 * export const dynamic = "force-dynamic";
 * export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const symbol = url.searchParams.get("symbol") || "AAPL";
  const peers = await fetchPeers(symbol);
  return new Response(JSON.stringify(peers, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* ----------------------------- Generators ---------------------------- */

function fakeTicker(base: string, i: number) {
  // Derive a plausible ticker from base symbol + index
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const code = letters[(hash(base + i) % 26)] + letters[(hash(base + i + 7) % 26)];
  return (base.slice(0, 2) + code + (i + 1)).slice(0, 4).toUpperCase();
}

function fakeName(ticker: string) {
  const suffix = ["Corp", "Inc", "Ltd", "Group", "Holdings"];
  return ticker + " " + suffix[hash(ticker) % suffix.length];
}

/* ----------------------------- Sector/Industry ----------------------- */

function guessSector(symbol: string) {
  if (/AAPL|MSFT|NVDA/.test(symbol)) return "Information Technology";
  if (/AMZN|WMT|TGT/.test(symbol)) return "Consumer Discretionary";
  if (/XOM|CVX/.test(symbol)) return "Energy";
  if (/JPM|GS|BAC/.test(symbol)) return "Financials";
  return "Industrials";
}

function guessIndustry(symbol: string) {
  if (/AAPL|MSFT/.test(symbol)) return "Consumer Electronics";
  if (/NVDA/.test(symbol)) return "Semiconductors";
  if (/AMZN/.test(symbol)) return "Internet & Direct Marketing";
  return "Diversified";
}

/* ------------------------------ Utilities ---------------------------- */

function round(n: number, d = 2) { const p = 10 ** d; return Math.round(n * p) / p; }

// Deterministic RNG
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}
function hash(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}