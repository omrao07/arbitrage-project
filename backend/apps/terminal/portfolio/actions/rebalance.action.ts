// actions/rebalance.action.ts
// No imports. Self-contained Server Action to compute portfolio rebalance trades.
// - Accepts either a typed object or FormData (from <form action={rebalanceAction}>).
// - Produces round-lot trades honoring cash, fees, min trade value, and turnover cap.
// - Never shorts: cannot sell below 0, cannot buy without cash.
// - Weights can sum to <= 1. Any remainder is treated as target cash reserve.
//
// Usage (typed):
//   const res = await simulateRebalance({ positions, prices, cash, targets });
// Usage (FormData):
//   <form action={rebalanceAction}> with fields:
//     positions, prices, targets  -> JSON strings
//     cash, feeBps, minTradeValue, maxTurnoverPct, tolerancePct, reserveCashPct -> numbers
//     lotSizeMap -> JSON like {"INFY":1, "NIFTY":25}
//     blacklist -> CSV string "ABC,XYZ"
//     sidePreference -> "neutral" | "sellFirst" | "buyFirst"
//     notes -> optional string
//
// Returns { ok, trades, before, after, summary, warnings }.

"use server";

/* ---------------- Types ---------------- */
type Position = { symbol: string; qty: number };
type PriceMap = Record<string, number>;
type Target = { symbol: string; weight: number }; // 0..1
type LotSizeMap = Record<string, number>;

export type RebalanceInput = {
  positions: Position[];        // current holdings (qty >= 0)
  prices: PriceMap;             // last prices, same symbols as positions/targets (price > 0)
  cash: number;                 // available cash (>= 0)
  targets: Target[];            // desired weights
  feeBps?: number;              // per-trade fee in basis points, default 5 (0.05%)
  minTradeValue?: number;       // skip trades below this notional, default 0
  lotSizeMap?: LotSizeMap;      // per-symbol trading lot, default 1
  tolerancePct?: number;        // drift tolerance per symbol (pct points), default 0.10 (%)
  maxTurnoverPct?: number;      // cap gross turnover (% of starting MV), default 100
  reserveCashPct?: number;      // keep this % of total in cash, reduces target MV, default 0
  blacklist?: string[];         // symbols to NEVER trade
  sidePreference?: "neutral" | "sellFirst" | "buyFirst"; // tie-breaker, default "sellFirst"
  notes?: string;               // optional, echoed back
};

type Trade = {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;                  // integer, respects lot size
  price: number;
  value: number;                // qty * price
  fee: number;                  // absolute currency
};

export type RebalanceResult =
  | {
      ok: true;
      trades: Trade[];
      before: {
        total: number;
        cash: number;
        positions: Array<{ symbol: string; qty: number; price: number; value: number; weight: number }>;
      };
      after: {
        total: number;
        cash: number;
        positions: Array<{ symbol: string; qty: number; price: number; value: number; weight: number }>;
      };
      drift: Array<{ symbol: string; current: number; target: number; deltaPct: number }>;
      summary: {
        startTotal: number;
        endTotal: number;
        grossTurnover: number; // %
        fees: number;
        buyNotional: number;
        sellNotional: number;
        cashChange: number;
        constraintsApplied: string[];
        notes?: string;
      };
      warnings: string[];
    }
  | { ok: false; error: string; fieldErrors?: Record<string, string> };

/* ---------------- Public Server Action ---------------- */
export async function rebalanceAction(
  input: Partial<RebalanceInput> | FormData,
): Promise<RebalanceResult> {
  try {
    const cfg = parseInput(input);
    const v = validate(cfg);
    if (!v.valid) return { ok: false, error: "Validation failed", fieldErrors: v.errors };
    return simulateRebalance(cfg);
  } catch (e: any) {
    return { ok: false, error: e?.message || "Rebalance failed" };
  }
}

/** Programmatic API (typed input). */
export function simulateRebalance(cfg: RequiredCfg): RebalanceResult {
  // ---------- setup ----------
  const prices = cfg.prices;
  const lots = cfg.lotSizeMap;
  const feePct = cfg.feeBps / 10_000;
  const minNotional = Math.max(0, cfg.minTradeValue);
  const black = new Set((cfg.blacklist || []).map((s) => s.toUpperCase()));
  const tol = cfg.tolerancePct / 100; // convert to fraction
  const pref = cfg.sidePreference;

  const posMap = new Map<string, number>();
  cfg.positions.forEach((p) => posMap.set(p.symbol.toUpperCase(), clampNonNeg(p.qty)));

  const targetW = new Map<string, number>();
  cfg.targets.forEach((t) => targetW.set(t.symbol.toUpperCase(), Math.max(0, t.weight)));

  // Universe = union of positions & targets with known prices (ignore symbols without price)
  const symSet = new Set<string>();
  for (const s of posMap.keys()) if (validPrice(prices[s])) symSet.add(s);
  for (const s of targetW.keys()) if (validPrice(prices[s])) symSet.add(s);

  // Compute current MV + cash
  let mvPositions = 0;
  const beforePositions: Array<{ symbol: string; qty: number; price: number; value: number; weight: number }> = [];
  for (const s of Array.from(symSet).sort()) {
    const qty = posMap.get(s) || 0;
    const px = prices[s];
    const val = qty * px;
    mvPositions += val;
    beforePositions.push({ symbol: s, qty, price: px, value: val, weight: 0 });
  }
  const startTotal = mvPositions + cfg.cash;
  if (!(startTotal > 0)) return { ok: false, error: "Portfolio total must be > 0" };

  // Fill before weights
  for (const row of beforePositions) row.weight = row.value / startTotal;

  // Effective investable target total after reserving cash
  const targetCashWeight = Math.max(0, Math.min(1, 1 - sumWeights(targetW) + cfg.reserveCashPct / 100));
  const investableTarget = startTotal * (1 - targetCashWeight);

  // ---------- build desired values & raw deltas ----------
  type Delta = { symbol: string; px: number; curQty: number; curVal: number; tgtVal: number; deltaVal: number };
  const deltas: Delta[] = [];

  for (const s of symSet) {
    const px = prices[s];
    const curQty = posMap.get(s) || 0;
    const curVal = curQty * px;
    const tgtVal = investableTarget * (targetW.get(s) || 0);
    const deltaVal = tgtVal - curVal;
    deltas.push({ symbol: s, px, curQty, curVal, tgtVal, deltaVal });
  }

  // Skip tiny drifts within tolerance
  const actionable: Delta[] = deltas.filter((d) => {
    const curW = d.curVal / startTotal;
    const tgtW = (targetW.get(d.symbol) || 0);
    const drift = Math.abs(curW - tgtW);
    return drift > tol && !black.has(d.symbol);
  });

  // Order preference: usually sell first to raise cash
  actionable.sort((a, b) => {
    if (pref === "buyFirst") return (b.deltaVal > 0 ? 1 : -1) - (a.deltaVal > 0 ? 1 : -1);
    if (pref === "neutral") return Math.abs(b.deltaVal) - Math.abs(a.deltaVal);
    // default sellFirst
    const sa = a.deltaVal < 0 ? 0 : 1;
    const sb = b.deltaVal < 0 ? 0 : 1;
    if (sa !== sb) return sa - sb;
    return Math.abs(b.deltaVal) - Math.abs(a.deltaVal);
  });

  // ---------- stage 1: compute sells (hard constraint: cannot sell below 0) ----------
  let cash = cfg.cash;
  const trades: Trade[] = [];
  let sellNotional = 0;
  let buyDemand = 0; // desired total buy notional before scaling

  for (const d of actionable) {
    if (d.deltaVal < 0) {
      // Need to SELL |delta|. Round down to lots and <= current qty
      const lot = Math.max(1, Math.floor(lots[d.symbol] || 1));
      const rawQty = Math.floor(Math.abs(d.deltaVal) / d.px / lot) * lot;
      const qty = Math.min(rawQty, Math.floor(d.curQty / lot) * lot);
      if (qty > 0) {
        const notional = qty * d.px;
        if (notional >= minNotional) {
          const fee = notional * feePct;
          trades.push(mkTrade(d.symbol, "SELL", qty, d.px, notional, fee));
          sellNotional += notional;
          cash += notional - fee;
          posMap.set(d.symbol, d.curQty - qty);
        }
      }
    } else if (d.deltaVal > 0) {
      buyDemand += d.deltaVal;
    }
  }

  // ---------- stage 2: compute buys within cash & turnover caps ----------
  // Available cash for buys (respect reserveCashPct as weight already set)
  let availCash = Math.max(0, cash);

  // If turnover cap set < 100%, compute cap on gross notional and scale buys if needed
  const turnoverCap = Math.max(0, cfg.maxTurnoverPct) / 100 * startTotal;
  const provisionalGross = sellNotional + buyDemand;
  let turnoverLimited = false;
  if (turnoverCap > 0 && provisionalGross > turnoverCap) {
    const maxBuyGivenCap = Math.max(0, turnoverCap - sellNotional);
    const scale = maxBuyGivenCap / Math.max(1e-9, buyDemand);
    buyDemand *= scale;
    turnoverLimited = true;
  }

  // Also scale by available cash after fees
  const cashMaxBuy = availCash / (1 + feePct);
  if (buyDemand > cashMaxBuy) {
    const scale = cashMaxBuy / Math.max(1e-9, buyDemand);
    buyDemand *= scale;
  }

  // Distribute buyDemand proportional to positive deltas
  const buyDeltas = actionable.filter((d) => d.deltaVal > 0);
  const totalPosDelta = buyDeltas.reduce((s, d) => s + d.deltaVal, 0);
  let buyNotional = 0;
  let buyFees = 0;

  for (const d of buyDeltas) {
    const share = totalPosDelta > 0 ? d.deltaVal / totalPosDelta : 0;
    const targetBuyVal = buyDemand * share;

    // Convert to qty with lot rounding
    const lot = Math.max(1, Math.floor(lots[d.symbol] || 1));
    let qty = Math.floor(targetBuyVal / d.px / lot) * lot;
    // Ensure min notional
    if (qty > 0 && qty * d.px < minNotional) {
      qty = Math.ceil(minNotional / d.px / lot) * lot;
    }
    if (qty <= 0) continue;

    let notional = qty * d.px;
    let fee = notional * feePct;

    // Guard: don't overspend cash
    if (notional + fee > availCash + 1e-6) {
      // reduce qty to fit cash
      qty = Math.floor((availCash / (1 + feePct)) / d.px / lot) * lot;
      if (qty <= 0) continue;
      notional = qty * d.px;
      fee = notional * feePct;
    }

    if (notional >= minNotional) {
      trades.push(mkTrade(d.symbol, "BUY", qty, d.px, notional, fee));
      buyNotional += notional;
      buyFees += fee;
      availCash -= notional + fee;
      posMap.set(d.symbol, (posMap.get(d.symbol) || 0) + qty);
    }
  }

  // ---------- finalize portfolio ----------
  // Compute after state (mark-to-market at same prices; fees already deducted in cash)
  const afterPositions: Array<{ symbol: string; qty: number; price: number; value: number; weight: number }> = [];
  let afterMV = 0;
  for (const s of Array.from(symSet).sort()) {
    const px = prices[s];
    const qty = posMap.get(s) || 0;
    const val = qty * px;
    afterMV += val;
    afterPositions.push({ symbol: s, qty, price: px, value: val, weight: 0 });
  }
  const endTotal = afterMV + availCash;
  for (const row of afterPositions) row.weight = endTotal > 0 ? row.value / endTotal : 0;

  // Drift report
  const drift = Array.from(symSet).sort().map((s) => {
    const tgtW = targetW.get(s) || 0;
    const curW = beforePositions.find((r) => r.symbol === s)!.weight;
    return { symbol: s, current: round4(curW * 100), target: round4(tgtW * 100), deltaPct: round4((curW - tgtW) * 100) };
  });

  // Summary / constraints
  const fees = trades.reduce((s, t) => s + t.fee, 0);
  const grossTurnover = ((sellNotional + buyNotional) / startTotal) * 100;
  const warnings: string[] = [];
  const constraintsApplied: string[] = [];
  if (turnoverLimited) constraintsApplied.push("turnover_cap");
  if (cfg.reserveCashPct > 0) constraintsApplied.push("reserve_cash");
  if (cfg.tolerancePct > 0) constraintsApplied.push("drift_tolerance");
  if (buyDemand === 0 && actionable.some((d) => d.deltaVal > 0)) warnings.push("Insufficient cash to execute all buys.");
  if (trades.length === 0) warnings.push("No actionable trades (within tolerance / constraints).");

  // Order: sells first, then buys (deterministic)
  trades.sort((a, b) => (a.side === b.side ? a.symbol.localeCompare(b.symbol) : a.side === "SELL" ? -1 : 1));

  return {
    ok: true,
    trades,
    before: {
      total: round2(startTotal),
      cash: round2(cfg.cash),
      positions: beforePositions.map((r) => ({ ...r, value: round2(r.value), weight: round4(r.weight * 100) })),
    },
    after: {
      total: round2(endTotal),
      cash: round2(availCash),
      positions: afterPositions.map((r) => ({ ...r, value: round2(r.value), weight: round4(r.weight * 100) })),
    },
    drift,
    summary: {
      startTotal: round2(startTotal),
      endTotal: round2(endTotal),
      grossTurnover: round4(grossTurnover),
      fees: round2(fees),
      buyNotional: round2(buyNotional),
      sellNotional: round2(sellNotional),
      cashChange: round2(availCash - cfg.cash),
      constraintsApplied,
      notes: cfg.notes,
    },
    warnings,
  };
}

/* ---------------- Parsing & Validation ---------------- */
type RequiredCfg = {
  positions: Position[];
  prices: PriceMap;
  cash: number;
  targets: Target[];
  feeBps: number;
  minTradeValue: number;
  lotSizeMap: LotSizeMap;
  tolerancePct: number;
  maxTurnoverPct: number;
  reserveCashPct: number;
  blacklist: string[];
  sidePreference: "neutral" | "sellFirst" | "buyFirst";
  notes?: string;
};

function parseInput(input: Partial<RebalanceInput> | FormData): RequiredCfg {
  if (isFormData(input)) {
    const fd = input as FormData;
    const positions = json<Position[]>(fd.get("positions")) ?? [];
    const prices = json<PriceMap>(fd.get("prices")) ?? {};
    const targets = json<Target[]>(fd.get("targets")) ?? [];
    const lotSizeMap = json<LotSizeMap>(fd.get("lotSizeMap")) ?? {};
    const blacklist = csv(fd.get("blacklist"));
    return {
      positions,
      prices,
      targets,
      cash: num(fd.get("cash")) ?? 0,
      feeBps: num(fd.get("feeBps")) ?? 5,
      minTradeValue: num(fd.get("minTradeValue")) ?? 0,
      lotSizeMap,
      tolerancePct: num(fd.get("tolerancePct")) ?? 0.1,
      maxTurnoverPct: num(fd.get("maxTurnoverPct")) ?? 100,
      reserveCashPct: num(fd.get("reserveCashPct")) ?? 0,
      blacklist,
      sidePreference: (str(fd.get("sidePreference")) as any) || "sellFirst",
      notes: str(fd.get("notes")),
    };
  }
  const obj = input as Partial<RebalanceInput>;
  return {
    positions: (obj.positions || []).map((p) => ({ symbol: p.symbol.toUpperCase(), qty: clampNonNeg(p.qty) })),
    prices: spreadKeysUpper(obj.prices || {}),
    targets: (obj.targets || []).map((t) => ({ symbol: t.symbol.toUpperCase(), weight: Math.max(0, t.weight) })),
    cash: Math.max(0, obj.cash || 0),
    feeBps: obj.feeBps ?? 5,
    minTradeValue: obj.minTradeValue ?? 0,
    lotSizeMap: spreadKeysUpper(obj.lotSizeMap || {}),
    tolerancePct: obj.tolerancePct ?? 0.1,
    maxTurnoverPct: obj.maxTurnoverPct ?? 100,
    reserveCashPct: obj.reserveCashPct ?? 0,
    blacklist: (obj.blacklist || []).map((s) => s.toUpperCase()),
    sidePreference: obj.sidePreference || "sellFirst",
    notes: obj.notes,
  };
}

function validate(cfg: RequiredCfg): { valid: boolean; errors: Record<string, string> } {
  const errors: Record<string, string> = {};
  if (!Array.isArray(cfg.positions)) errors.positions = "positions must be an array";
  if (Object.keys(cfg.prices).length === 0) errors.prices = "prices required";
  if (!Array.isArray(cfg.targets)) errors.targets = "targets must be an array";
  if (!(cfg.cash >= 0)) errors.cash = "cash must be >= 0";
  const totalW = sumWeights(new Map(cfg.targets.map((t) => [t.symbol, t.weight])));
  if (!(totalW <= 1.0001)) errors.targets_sum = "sum(weights) must be <= 1";
  for (const [s, px] of Object.entries(cfg.prices)) {
    if (!validPrice(px)) { errors[`price_${s}`] = "price must be > 0"; }
  }
  return { valid: Object.keys(errors).length === 0, errors };
}

/* ---------------- Helpers ---------------- */
function mkTrade(symbol: string, side: "BUY" | "SELL", qty: number, px: number, notional: number, fee: number): Trade {
  return {
    id: uid(),
    symbol,
    side,
    qty,
    price: round4(px),
    value: round2(notional),
    fee: round2(fee),
  };
}

function validPrice(px: any): px is number {
  return typeof px === "number" && isFinite(px) && px > 0;
}
function clampNonNeg(n: number) { return Math.max(0, Number(n || 0)); }

function sumWeights(m: Map<string, number>): number {
  let s = 0;
  for (const v of m.values()) s += v || 0;
  return s;
}

function round2(n: number) { return Math.round((n + Number.EPSILON) * 100) / 100; }
function round4(n: number) { return Math.round((n + Number.EPSILON) * 10000) / 10000; }

function isFormData(x: any): x is FormData {
  return typeof x === "object" && x?.constructor?.name === "FormData";
}
function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}
function num(v: any): number | undefined {
  if (v == null || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
function json<T>(v: any): T | undefined {
  const s = str(v);
  if (!s) return undefined;
  try { return JSON.parse(s) as T; } catch { return undefined; }
}
function csv(v: any): string[] {
  const s = str(v);
  if (!s) return [];
  return s.split(/[,\s]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
}
function spreadKeysUpper<T extends Record<string, any>>(obj: T): T {
  const out: any = {};
  for (const k of Object.keys(obj)) out[k.toUpperCase()] = obj[k];
  return out as T;
}
function uid(): string {
  // Deterministic-ish ID without imports
  if ((globalThis as any).crypto?.randomUUID) return (globalThis as any).crypto.randomUUID();
  const t = Date.now().toString(36);
  const r = Math.random().toString(36).slice(2, 8);
  return `tr_${t}_${r}`;
}
