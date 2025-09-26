// runbondscreen.server.ts
// Zero-import synthetic “bond screener” that filters a sample universe on the server.
// Intended for wiring to your ScreenerForm. Replace the MOCK_UNIVERSE with your data/feed.
//
// Exports:
//   - types
//   - runBondScreen(input)
//   - handleGET(url) helper for Next.js Route Handlers

/* ---------------------------------- Types ---------------------------------- */

export type BondRow = {
  cusip: string;
  isin?: string;
  issuer: string;
  sector: string;         // e.g., "IG Corp", "HY Corp", "Sovereign", "Muni"
  country: string;        // ISO-2 (US, GB, DE…)
  currency: string;       // USD, EUR, JPY…
  coupon: number;         // % (e.g., 5.25)
  couponType: "Fixed" | "Float" | "Zero";
  issueDate: string;      // ISO date
  maturity: string;       // ISO date
  nextCall?: string;      // ISO date optional
  callable?: boolean;
  price: number;          // clean price
  yieldToMaturity: number;// % (e.g., 6.12)
  yieldToWorst?: number;  // % (optional)
  spreadOAS: number;      // bps
  durationMod: number;    // years
  rating: "AAA"|"AA"|"A"|"BBB"|"BB"|"B"|"CCC"|"NR";
  amountOutstanding: number; // notional in USD
};

export type BondScreenInput = {
  symbols?: string[];           // filter by CUSIP/ISIN exact
  issuers?: string[];           // issuer names (case-insensitive contains)
  sectors?: string[];           // sector whitelist
  countries?: string[];         // ISO-2 whitelist
  currencies?: string[];        // currency whitelist
  ratingMin?: BondRow["rating"]; // minimum rating (floor by rank)
  ratingMax?: BondRow["rating"]; // maximum rating (cap by rank)
  price?: [number | "", number | ""];            // clean price
  ytm?: [number | "", number | ""];              // %
  ytw?: [number | "", number | ""];              // %
  oas?: [number | "", number | ""];              // bps
  duration?: [number | "", number | ""];         // years
  coupon?: [number | "", number | ""];           // %
  mtyFrom?: string;             // ISO date (>=)
  mtyTo?: string;               // ISO date (<=)
  onlyCallable?: boolean;       // if true, callable==true
  sort?: { key: keyof BondRow; dir: "asc" | "desc" };
  limit?: number;               // default 500
  seed?: number;                // for mock universe generation stability
};

export type BondScreenResult = {
  asOf: string;
  total: number;
  rows: BondRow[];
  query: BondScreenInput;
};

/* ---------------------------------- API ---------------------------------- */

export async function runBondScreen(input: BondScreenInput = {}): Promise<BondScreenResult> {
  const now = new Date().toISOString();
  const seed = input.seed ?? 7;
  const rng = mulberry32(seed);
  const universe = buildMockUniverse(1500, rng); // synth universe

  // 1) Precompute rating order
  const R = ["AAA","AA","A","BBB","BB","B","CCC","NR"] as const;
  const rIndex = (r: BondRow["rating"]) => R.indexOf(r as any);

  // 2) Normalize filters
  const syms = new Set((input.symbols || []).map((s) => s.trim().toUpperCase()).filter(Boolean));
  const issLC = (input.issuers || []).map((s) => s.toLowerCase());
  const secSet = new Set((input.sectors || []).map((s) => s.toLowerCase()));
  const ctySet = new Set((input.countries || []).map((s) => s.toUpperCase()));
  const ccySet = new Set((input.currencies || []).map((s) => s.toUpperCase()));
  const minRat = input.ratingMin ? rIndex(input.ratingMin) : -Infinity;
  const maxRat = input.ratingMax ? rIndex(input.ratingMax) : +Infinity;

  // 3) Run filters
  let rows = universe.filter((b) => {
    if (syms.size) {
      const key = (b.cusip + "|" + (b.isin || "")).toUpperCase();
      let ok = false;
      for (const s of syms) if (key.includes(s)) { ok = true; break; }
      if (!ok) return false;
    }
    if (issLC.length) {
      const hay = b.issuer.toLowerCase();
      if (!issLC.some((s) => hay.includes(s))) return false;
    }
    if (secSet.size && !secSet.has(b.sector.toLowerCase())) return false;
    if (ctySet.size && !ctySet.has(b.country.toUpperCase())) return false;
    if (ccySet.size && !ccySet.has(b.currency.toUpperCase())) return false;

    const ri = rIndex(b.rating);
    if (ri < (Number.isFinite(minRat) ? minRat : -1e9)) return false;
    if (ri > (Number.isFinite(maxRat) ? maxRat : 1e9)) return false;

    if (!rangeOk(b.price, input.price)) return false;
    if (!rangeOk(b.yieldToMaturity, input.ytm)) return false;
    if (input.ytw && input.ytw.some((x) => x !== "") && b.yieldToWorst != null) {
      if (!rangeOk(b.yieldToWorst!, input.ytw)) return false;
    }
    if (!rangeOk(b.spreadOAS, input.oas)) return false;
    if (!rangeOk(b.durationMod, input.duration)) return false;
    if (!rangeOk(b.coupon, input.coupon)) return false;

    if (input.mtyFrom && +new Date(b.maturity) < +new Date(input.mtyFrom)) return false;
    if (input.mtyTo && +new Date(b.maturity) > +new Date(input.mtyTo)) return false;

    if (input.onlyCallable && !b.callable) return false;

    return true;
  });

  // 4) Sort
  const sortKey = input.sort?.key || "spreadOAS";
  const sortDir = input.sort?.dir || "desc";
  rows.sort((a, b) => cmp((a as any)[sortKey], (b as any)[sortKey], sortDir));

  // 5) Limit
  const limit = Math.max(1, Math.min(input.limit ?? 500, 5000));
  const out = rows.slice(0, limit);

  return { asOf: now, total: rows.length, rows: out, query: input };
}

/**
 * Optional helper for Next.js Route Handlers:
 *   export async function GET(req: Request) { return handleGET(req.url); }
 *
 * Query params (all optional):
 *   ?symbols=AAPL,US0003M &sectors=IG%20Corp,HY%20Corp &countries=US,GB
 *   &ratingMin=BBB&ratingMax=A &price=80,110 &ytm=5,12 &oas=150,600
 *   &duration=1,7 &coupon=3,10 &mtyFrom=2026-01-01 &mtyTo=2032-12-31
 *   &currencies=USD,EUR &onlyCallable=1 &sort=yieldToMaturity:desc &limit=200
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const getArr = (k: string) => (url.searchParams.get(k) || "").split(",").map((s) => s.trim()).filter(Boolean);
  const getPair = (k: string): [number | "", number | ""] | undefined => {
    const v = getArr(k);
    if (!v.length) return undefined;
    const lo = v[0] === "" ? "" : num(v[0]);
    const hi = v[1] === "" ? "" : num(v[1]);
    return [lo ?? "", hi ?? ""];
  };
  const sortRaw = url.searchParams.get("sort") || "";
  const sort = sortRaw ? ({ key: sortRaw.split(":")[0] as keyof BondRow, dir: (sortRaw.split(":")[1] as "asc"|"desc") || "desc" }) : undefined;

  const input: BondScreenInput = {
    symbols: getArr("symbols"),
    issuers: getArr("issuers"),
    sectors: getArr("sectors"),
    countries: getArr("countries"),
    currencies: getArr("currencies"),
    ratingMin: (url.searchParams.get("ratingMin") as any) || undefined,
    ratingMax: (url.searchParams.get("ratingMax") as any) || undefined,
    price: getPair("price"),
    ytm: getPair("ytm"),
    ytw: getPair("ytw"),
    oas: getPair("oas"),
    duration: getPair("duration"),
    coupon: getPair("coupon"),
    mtyFrom: url.searchParams.get("mtyFrom") || undefined,
    mtyTo: url.searchParams.get("mtyTo") || undefined,
    onlyCallable: ["1","true","yes"].includes((url.searchParams.get("onlyCallable") || "").toLowerCase()),
    sort,
    limit: num(url.searchParams.get("limit")),
    seed: num(url.searchParams.get("seed")),
  };

  const res = await runBondScreen(input);
  return new Response(JSON.stringify(res, null, 2), { headers: { "content-type": "application/json; charset=utf-8" } });
}

/* ------------------------------ Mock Universe ------------------------------ */

function buildMockUniverse(n: number, rng: () => number): BondRow[] {
  const sectors = ["IG Corp","HY Corp","Sovereign","Muni"];
  const countries = ["US","GB","DE","FR","JP","CN","IN","BR","CA","AU"];
  const currencies = ["USD","EUR","JPY","GBP"];
  const ratings: BondRow["rating"][] = ["AAA","AA","A","BBB","BB","B","CCC","NR"];
  const issuers = [
    "Globex Corp","Initech","Vandelay Industries","Acme Holdings","Stark Industries",
    "Wayne Enterprises","Umbrella Corp","Hooli Inc","Massive Dynamic","Soylent Corp",
  ];

  const out: BondRow[] = [];
  for (let i = 0; i < n; i++) {
    const sector = pick(sectors, rng);
    const issuer = pick(issuers, rng);
    const country = pick(countries, rng);
    const currency = pick(currencies, rng);
    const rating = pick(ratings, rng);
    const couponType: BondRow["couponType"] = rnd(rng) < 0.1 ? "Zero" : rnd(rng) < 0.3 ? "Float" : "Fixed";

    const issueYear = 2008 + Math.floor(rnd(rng) * 15);
    const issueMonth = 1 + Math.floor(rnd(rng) * 12);
    const issueDay = 1 + Math.floor(rnd(rng) * 28);
    const issueDate = isoDate(issueYear, issueMonth, issueDay);

    const mtyYears = 2 + Math.floor(rnd(rng) * 25);
    const maturity = isoDate(issueYear + mtyYears, issueMonth, issueDay);

    const callable = rnd(rng) < 0.35;
    const nextCall = callable ? shiftMonths(issueDate, 60 + Math.floor(rnd(rng) * 60)) : undefined;

    const coupon = couponType === "Zero" ? 0 : round(2 + rnd(rng) * 7, 2);
    const price = round(70 + rnd(rng) * 50, 2); // 70–120
    const ytm = round(1.5 + (120 - price) * 0.12 + (ratingPenalty(rating)) + rnd(rng) * 0.7, 2);
    const ytw = callable ? round(ytm - rnd(rng) * 0.5, 2) : undefined;
    const oas = Math.max(20, Math.floor(30 + (ytm - 2) * 40 + rnd(rng) * 50)); // bps
    const duration = round(1 + mtyYears * (0.5 + rnd(rng) * 0.6), 2);
    const amountOutstanding = Math.floor(100e6 + rnd(rng) * 1900e6);

    const cusip = genCUSIP(i, rng);
    const isin = "US" + cusip + "0";

    out.push({
      cusip, isin, issuer, sector, country, currency,
      coupon, couponType, issueDate, maturity, nextCall, callable,
      price, yieldToMaturity: ytm, yieldToWorst: ytw, spreadOAS: oas,
      durationMod: duration, rating, amountOutstanding,
    });
  }
  return out;
}

/* --------------------------------- Helpers --------------------------------- */

function rangeOk(v: number, p?: [number | "", number | ""]) {
  if (!p) return true;
  const [lo, hi] = p;
  if (lo !== "" && v < Number(lo)) return false;
  if (hi !== "" && v > Number(hi)) return false;
  return true;
}

function cmp(a: any, b: any, dir: "asc" | "desc") {
  const ax = a ?? -Infinity, bx = b ?? -Infinity;
  if (ax === bx) return 0;
  const r = ax > bx ? 1 : -1;
  return dir === "asc" ? r : -r;
}

function pick<T>(arr: T[], rng: () => number): T { return arr[Math.floor(rng() * arr.length)]; }
function rnd(rng: () => number) { return rng(); }
function isoDate(y: number, m: number, d: number) { return `${y}-${String(m).padStart(2,"0")}-${String(d).padStart(2,"0")}`; }
function shiftMonths(iso: string, n: number) {
  const dt = new Date(iso + "T00:00:00Z");
  dt.setUTCMonth(dt.getUTCMonth() + n);
  return dt.toISOString().slice(0, 10);
}
function ratingPenalty(r: BondRow["rating"]) {
  const map: Record<BondRow["rating"], number> = { AAA: 0, AA: 0.2, A: 0.4, BBB: 0.7, BB: 1.3, B: 2.0, CCC: 3.0, NR: 0.8 };
  return map[r] ?? 0.8;
}
function round(x: number, d = 2) { const p = 10 ** d; return Math.round(x * p) / p; }
function num(s: string | null) { if (!s) return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined; }

function genCUSIP(i: number, rng: () => number) {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const digits = "0123456789";
  const part = (len: number, pool: string) => Array.from({ length: len }, () => pool[Math.floor(rng() * pool.length)]).join("");
  return part(3, letters) + part(5, digits) + String(i % 10);
}

// Small, fast RNG
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------------------------- Notes -----------------------------------
- All numbers and bonds here are synthetic and for UI development only.
- Replace buildMockUniverse() with your data provider for production.
------------------------------------------------------------------------------- */