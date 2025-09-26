// fetchriskstate.server.ts
// Zero-import, synthetic Risk State generator for dashboards.
// Provides portfolio risk metrics, limits, and stress-test placeholders.
// Works on server/edge without external libs.
//
// Exports:
//   - types
//   - fetchRiskState(params)
//   - handleGET(url) helper for Next.js API Routes

/* --------------------------------- Types --------------------------------- */

export type RiskMetric = {
  id: string;
  label: string;
  value: number;
  unit: string;
  limit?: number;
  breached?: boolean;
};

export type StressScenario = {
  id: string;
  label: string;
  pnlImpact: number; // PnL % impact
  shockedVars: string[];
};

export type RiskState = {
  asOf: string;
  accountId: string;
  metrics: RiskMetric[];
  stresses: StressScenario[];
  notes?: string;
};

export type RiskParams = {
  accountId?: string;
  seed?: number;
};

/* ---------------------------------- API ---------------------------------- */

export async function fetchRiskState(p?: RiskParams): Promise<RiskState> {
  const cfg = { accountId: p?.accountId || "DEMO", seed: p?.seed ?? 1 };
  const now = new Date().toISOString();
  const rng = mulberry32(hash(`risk|${cfg.accountId}|${cfg.seed}|${now.slice(0,10)}`));

  // Metrics
  const metrics: RiskMetric[] = [
    mkMetric("VaR", "Value-at-Risk (95%)", rng, 5e6, 9e6, "USD", 1e7),
    mkMetric("ES", "Expected Shortfall (95%)", rng, 6e6, 1.1e7, "USD", 1.2e7),
    mkMetric("Leverage", "Gross Leverage", rng, 2.5, 4.5, "×", 5),
    mkMetric("Liquidity", "1d Liquidity Coverage", rng, 0.8, 1.1, "×", 1),
    mkMetric("Concentration", "Top 5 names concentration", rng, 0.25, 0.55, "%", 0.5),
    mkMetric("IR01", "Rates DV01", rng, -150000, 150000, "USD"),
    mkMetric("FXDelta", "FX Delta", rng, -1.5e6, 1.5e6, "USD"),
    mkMetric("Beta", "Portfolio Beta", rng, 0.7, 1.3, "β"),
  ];

  // Stress scenarios
  const stresses: StressScenario[] = [
    { id: "eq_down", label: "Equities -20%", pnlImpact: -0.12 + (rng()-0.5)*0.02, shockedVars: ["Equities","Credit"] },
    { id: "rates_up", label: "Rates +100bps", pnlImpact: -0.05 + (rng()-0.5)*0.01, shockedVars: ["Rates"] },
    { id: "usd_up", label: "USD +5%", pnlImpact: -0.02 + (rng()-0.5)*0.01, shockedVars: ["FX"] },
    { id: "crash08", label: "2008-like Shock", pnlImpact: -0.25 + (rng()-0.5)*0.03, shockedVars: ["Equities","Credit","FX","Rates"] },
  ];

  return { asOf: now, accountId: cfg.accountId, metrics, stresses };
}

/**
 * Optional helper for Next.js Route Handlers:
 *   export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const accountId = url.searchParams.get("accountId") || undefined;
  const seed = num(url.searchParams.get("seed"));
  const snap = await fetchRiskState({ accountId, seed });
  return new Response(JSON.stringify(snap, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* ------------------------------ Metric Helper ----------------------------- */

function mkMetric(id: string, label: string, rng: () => number, min: number, max: number, unit: string, limit?: number): RiskMetric {
  const val = min + rng() * (max - min);
  const breached = limit != null && ((id==="Concentration") ? val > limit : val >= limit);
  return { id, label, value: round(val, 2), unit, limit, breached };
}

/* --------------------------------- Utils --------------------------------- */

function round(x: number, d = 2) { const p = 10**d; return Math.round(x*p)/p; }
function num(s: string | null) { if (!s) return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined; }

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}
function hash(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

/* ---------------------------------- Notes ----------------------------------
- All data here is synthetic and for UI testing only.
- Integrate with real risk engines for production.
---------------------------------------------------------------------------- */