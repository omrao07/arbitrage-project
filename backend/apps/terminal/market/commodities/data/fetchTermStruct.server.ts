// backend/apps/terminal/market/commodities/data/fetchTermStruct.server.ts
// Futures term structure utility (pure TypeScript, server-friendly)

export type TermPoint = {
  symbol: string;                 // e.g., "NIFTY"
  expiry: string;                 // ISO date YYYY-MM-DD
  daysToExp: number;              // days from today
  futPrice: number;               // futures price for that expiry
  spot: number;                   // reference spot used for calc
  basisPct: number;               // (F/S - 1) * 100
  annualizedCarryPct: number;     // ((F/S)^(365/days) - 1) * 100
  state: "CONTANGO" | "BACKWARDATION";
};

type GenerateParams = {
  symbol?: string;
  spot?: number;            // spot reference
  carryAnnualPct?: number;  // expected annual carry (+contango / -backwardation)
  noisePct?: number;        // small jitter percent for realism
  count?: number;           // number of expiries
  cadence?: "WEEKLY" | "MONTHLY";
};

const round = (n: number, d = 2) => {
  const p = 10 ** d;
  return Math.round(n * p) / p;
};

const addDays = (date: Date, days: number) => {
  const d = new Date(date);
  d.setDate(d.getDate() + days);
  return d;
};

const genExpiries = (count: number, mode: "WEEKLY" | "MONTHLY"): Date[] => {
  const today = new Date();
  const out: Date[] = [];
  if (mode === "WEEKLY") {
    for (let i = 1; i <= count; i++) out.push(addDays(today, i * 7));
  } else {
    for (let i = 1; i <= count; i++) {
      // end of month i months ahead
      const dt = new Date(today.getFullYear(), today.getMonth() + i + 1, 0);
      out.push(dt);
    }
  }
  return out;
};

/**
 * Mock term structure generator using a simple cost-of-carry model:
 *   F ≈ S * (1 + r * days/365)
 * with a small multiplicative jitter.
 */
export function generateTermStructure({
  symbol = "NIFTY",
  spot = 22000,
  carryAnnualPct = 6,
  noisePct = 0.6,
  count = 6,
  cadence = "WEEKLY",
}: GenerateParams = {}): TermPoint[] {
  const expiries = genExpiries(count, cadence);
  const today = new Date();

  return expiries.map((dt) => {
    const days = Math.max(1, Math.ceil((dt.getTime() - today.getTime()) / 86_400_000));
    const linearCarry = (carryAnnualPct / 100) * (days / 365);
    const jitter = 1 + ((Math.random() - 0.5) * noisePct) / 100 * 10; // ~±3% for noise
    const fut = spot * (1 + linearCarry) * jitter;

    const basisPct = (fut / spot - 1) * 100;
    const annualizedCarryPct = (Math.pow(fut / spot, 365 / days) - 1) * 100;

    return {
      symbol,
      expiry: dt.toISOString().slice(0, 10),
      daysToExp: days,
      futPrice: round(fut, 2),
      spot: round(spot, 2),
      basisPct: round(basisPct, 3),
      annualizedCarryPct: round(annualizedCarryPct, 3),
      state: fut >= spot ? "CONTANGO" : "BACKWARDATION",
    };
  });
}

/**
 * Public fetch function (kept async to allow easy swap with real data later).
 * Replace generateTermStructure(...) with your broker/NSE/IBKR data wiring when ready.
 */
export async function fetchTermStructure(params?: GenerateParams): Promise<TermPoint[]> {
  return generateTermStructure(params);
}