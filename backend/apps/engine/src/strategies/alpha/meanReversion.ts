// alpha/meanreversion.ts
// Import-free mean-reversion utilities: z-scores, OU half-life, pairs hedge,
// Bollinger channels, simple stat-arb backtests.
//
// What you get
// - rollMean/rollStdev/rollZScore
// - bollBands({prices, win, k}) → { upper[], mid[], lower[] }
// - halfLife(prices) via AR(1) OLS; ouKappa(), ouSigma()
// - pairsHedge(x, y) (OLS beta, intercept, residuals, z-score spread)
// - mrSignalZ({series, win, k}) → latest z and band status
// - backtestBand({prices, win, k, feesBps}) → channel reversion P&L
// - backtestPairs({x, y, mode, win, enter, exit, feesBps}) → pairs P&L
//
// Notes
// - All arrays are chronological (older → newer).
// - Returns and P&L are in the *same units* as the input (default equity=1).

/* ------------------------------- Types ---------------------------------- */

export type Num = number;

export type Bands = {
  upper: Num[];
  mid: Num[];
  lower: Num[];
};

export type MRSignal = {
  z: number;
  state: "long" | "short" | "flat"; // suggested side for mean reversion
  tsIndex: number;                  // index of the latest observation
};

export type PairsHedge = {
  beta: number;        // hedge ratio for y ~ a + beta*x
  alpha: number;       // intercept
  spread: number[];    // y - (a + beta*x)
  z: number[];         // z-scored spread (rolling window)
  win: number;         // window used for z
};

export type BandBTOptions = {
  win?: number;      // rolling window
  k?: number;        // band width multiplier
  feesBps?: number;  // per-turnover fees in bps
  startEquity?: number;
  slippageBps?: number;
};

export type BandBTOut = {
  equity: { i: number; equity: number }[];
  trades: { i: number; side: "long" | "short" | "flat"; px: number }[];
  stats: { ret: number; vol: number; sharpe: number; maxDD: number; trades: number };
};

export type PairsBTOptions = {
  mode?: "price" | "log";  // hedge on price level or log-price
  win?: number;            // rolling window for z
  lookbackHedge?: number;  // window to re-estimate hedge (default=win)
  enter?: number;          // z entry threshold
  exit?: number;           // z exit threshold
  feesBps?: number;
  slippageBps?: number;
  startEquity?: number;
  maxGross?: number;       // cap |wX|+|wY|
};

export type PairsBTOut = {
  equity: { i: number; equity: number }[];
  z: number[];
  beta: number[];
  trades: { i: number; side: "long" | "short" | "flat"; wX: number; wY: number; pxX: number; pxY: number }[];
  stats: { ret: number; vol: number; sharpe: number; maxDD: number; trades: number };
};

/* ----------------------------- Core utils ------------------------------- */

export function rollMean(x: Num[], win: number) {
  const w = Math.max(1, win | 0), out: Num[] = [];
  let s = 0, q: Num[] = [];
  for (let i = 0; i < x.length; i++) {
    const v = num(x[i]); q.push(v); s += v;
    if (q.length > w) s -= q.shift()!;
    if (q.length === w) out.push(s / w);
    else out.push(NaN);
  }
  return out;
}

export function rollStdev(x: Num[], win: number) {
  const w = Math.max(2, win | 0), out: Num[] = [];
  const q: Num[] = [];
  for (let i = 0; i < x.length; i++) {
    const v = num(x[i]); q.push(v);
    if (q.length > w) q.shift();
    if (q.length === w) out.push(stdev(q));
    else out.push(NaN);
  }
  return out;
}

export function rollZScore(x: Num[], win: number) {
  const mu = rollMean(x, win);
  const sd = rollStdev(x, win);
  const out: Num[] = [];
  for (let i = 0; i < x.length; i++) {
    const s = sd[i];
    out.push(isFinite(s) && s > 0 ? (x[i] - mu[i]) / s : NaN);
  }
  return out;
}

export function bollBands(args: { prices: Num[]; win?: number; k?: number }): Bands {
  const p = args.prices, w = Math.max(2, num(args.win, 20)), k = num(args.k, 2);
  const m = rollMean(p, w), s = rollStdev(p, w);
  const upper: Num[] = [], lower: Num[] = [], mid: Num[] = [];
  for (let i = 0; i < p.length; i++) {
    const u = isFinite(s[i]) ? m[i] + k * s[i] : NaN;
    const l = isFinite(s[i]) ? m[i] - k * s[i] : NaN;
    upper.push(u); lower.push(l); mid.push(m[i]);
  }
  return { upper, mid, lower };
}

/* ----------------------- OU / Half-life estimation ---------------------- */

export function halfLife(prices: Num[]) {
  // AR(1): Δp_t = φ * p_{t-1} + ε  → half-life = -ln(2)/ln(1+φ)
  const p = prices.slice();
  if (p.length < 3) return { phi: 0, halflife: 0 };
  const y: Num[] = [], x: Num[] = [];
  for (let t = 1; t < p.length; t++) { y.push(p[t] - p[t - 1]); x.push(p[t - 1]); }
  const { beta: phi } = ols(x, y);
  const hl = (1 + phi) > 0 ? -Math.log(2) / Math.log(Math.max(1e-9, 1 + phi)) : 0;
  return { phi, halflife: hl };
}

export function ouKappa(prices: Num[], dt = 1) {
  // Using AR(1) mapping: p_t = a + b p_{t-1} + ε; b = e^{-kappa*dt} ⇒ kappa = -ln(b)/dt
  if (prices.length < 3) return { kappa: 0, sigma: 0 };
  const x: Num[] = [], y: Num[] = [];
  for (let t = 1; t < prices.length; t++) { x.push(prices[t - 1]); y.push(prices[t]); }
  const { alpha, beta, resid } = ols(x, y); // y ≈ alpha + beta x
  const b = beta;
  const kappa = b > 0 ? -Math.log(Math.max(1e-9, b)) / dt : 0;
  // diffusion sigma via residual variance: Var(ε) ≈ σ^2 * (1 - e^{-2kappa dt})/(2kappa)
  const varE = variance(resid);
  const denom = (1 - Math.exp(-2 * kappa * dt)) / Math.max(1e-9, 2 * kappa);
  const sigma = denom > 0 ? Math.sqrt(Math.max(0, varE / denom)) : 0;
  return { kappa, sigma };
}

/* ------------------------------- Pairs ---------------------------------- */

export function pairsHedge(x: Num[], y: Num[], opts: { mode?: "price" | "log"; win?: number } = {}): PairsHedge {
  const mode = opts.mode || "price";
  const X = mode === "log" ? logArr(x) : x.slice();
  const Y = mode === "log" ? logArr(y) : y.slice();
  const { alpha, beta } = ols(X, Y);
  const spread: Num[] = [];
  for (let i = 0; i < X.length; i++) spread.push(Y[i] - (alpha + beta * X[i]));
  const win = Math.max(20, num(opts.win, 60));
  const z = rollZScore(spread, win);
  return { alpha, beta, spread, z, win };
}

export function mrSignalZ(args: { series: Num[]; win?: number; k?: number }): MRSignal {
  const z = rollZScore(args.series, Math.max(10, num(args.win, 20)));
  const latest = z[z.length - 1];
  const k = num(args.k, 2);
  let state: MRSignal["state"] = "flat";
  if (latest <= -k) state = "long"; else if (latest >= k) state = "short";
  return { z: latest, state, tsIndex: z.length - 1 };
}

/* ------------------------------- Backtests ------------------------------- */

export function backtestBand(prices: Num[], opts: BandBTOptions = {}): BandBTOut {
  const w = Math.max(10, num(opts.win, 20));
  const k = num(opts.k, 2);
  const fee = Math.max(0, num(opts.feesBps, 0)) / 10_000;
  const slip = Math.max(0, num(opts.slippageBps, 0)) / 10_000;
  const startEq = num(opts.startEquity, 1);

  const bands = bollBands({ prices, win: w, k });
  let side: "long" | "short" | "flat" = "flat";
  let equity = startEq;
  const curve: { i: number; equity: number }[] = [];
  const trades: BandBTOut["trades"] = [];
  let tradeCount = 0;

  for (let i = 1; i < prices.length; i++) {
    const px = prices[i];
    const u = bands.upper[i], l = bands.lower[i], m = bands.mid[i];
    const pxPrev = prices[i - 1];

    // Signals: cross above upper → short, below lower → long; revert to mean → exit.
    if (isFinite(u) && isFinite(l) && isFinite(m)) {
      if (side === "flat") {
        if (px >= u) { side = "short"; trades.push({ i, side, px }); tradeCount++; equity -= equity * (fee + slip); }
        else if (px <= l) { side = "long"; trades.push({ i, side, px }); tradeCount++; equity -= equity * (fee + slip); }
      } else if (side === "long" && px >= m) {
        side = "flat"; trades.push({ i, side, px }); tradeCount++; equity -= equity * (fee + slip);
      } else if (side === "short" && px <= m) {
        side = "flat"; trades.push({ i, side, px }); tradeCount++; equity -= equity * (fee + slip);
      }
    }

    // P&L accrual using daily return times position (+1/-1/0)
    const ret = (px / pxPrev) - 1;
    const pos = side === "long" ? 1 : side === "short" ? -1 : 0;
    equity *= (1 + pos * ret);
    curve.push({ i, equity: round6(equity) });
  }

  const rets = toRets(curve.map(c => c.equity));
  const { annVol, sharpe } = quickStats(rets);
  const { maxDD } = drawdownFromEquity(curve.map(c => c.equity));

  return {
    equity: curve,
    trades,
    stats: {
      ret: round6(curve.length ? curve[curve.length - 1].equity / startEq - 1 : 0),
      vol: round6(annVol),
      sharpe: round6(sharpe),
      maxDD: round6(maxDD),
      trades: tradeCount,
    },
  };
}

export function backtestPairs(x: Num[], y: Num[], opts: PairsBTOptions = {}): PairsBTOut {
  const mode = opts.mode || "price";
  const win = Math.max(30, num(opts.win, 60));
  const look = Math.max(10, num(opts.lookbackHedge, win));
  const enter = num(opts.enter, 2);
  const exit = num(opts.exit, 0.5);
  const fee = Math.max(0, num(opts.feesBps, 0)) / 10_000;
  const slip = Math.max(0, num(opts.slippageBps, 0)) / 10_000;
  const startEq = num(opts.startEquity, 1);
  const cap = isFiniteNum(opts.maxGross) ? Math.abs(opts.maxGross!) : 2; // |wX|+|wY| cap

  const pxX = mode === "log" ? logArr(x) : x.slice();
  const pxY = mode === "log" ? logArr(y) : y.slice();

  let equity = startEq;
  const eqCurve: { i: number; equity: number }[] = [];
  const zPath: number[] = [];
  const betaPath: number[] = [];
  const trades: PairsBTOut["trades"] = [];

  let side: "long" | "short" | "flat" = "flat";
  let wX = 0, wY = 0; // notional weights such that spread = Y - (a + beta X); long spread = long Y, short X*beta
  let a = 0, b = 1;

  for (let i = 1; i < pxX.length; i++) {
    // Re-estimate hedge on rolling window
    const s = Math.max(0, i - look + 1), e = i + 1;
    const { alpha, beta, spread } = pairsHedge(pxX.slice(s, e), pxY.slice(s, e), { mode: "price", win });
    a = alpha; b = beta;
    betaPath.push(b);

    // Z-score on current window of spread
    const zAll = rollZScore(spread, Math.min(win, spread.length));
    const z = zAll[zAll.length - 1];
    zPath.push(z);

    // Signals
    if (side === "flat") {
      if (z >= enter) { side = "short"; wY = -1; wX = +b; trades.push({ i, side, wX, wY, pxX: x[i], pxY: y[i] }); equity -= equity * (fee + slip); }
      else if (z <= -enter) { side = "long"; wY = +1; wX = -b; trades.push({ i, side, wX, wY, pxX: x[i], pxY: y[i] }); equity -= equity * (fee + slip); }
    } else if (side === "long" && z >= exit) {
      side = "flat"; wX = 0; wY = 0; trades.push({ i, side, wX, wY, pxX: x[i], pxY: y[i] }); equity -= equity * (fee + slip);
    } else if (side === "short" && z <= -exit) {
      side = "flat"; wX = 0; wY = 0; trades.push({ i, side, wX, wY, pxX: x[i], pxY: y[i] }); equity -= equity * (fee + slip);
    }

    // Cap gross
    const gross = Math.abs(wX) + Math.abs(wY);
    if (gross > cap && gross > 0) { wX *= cap / gross; wY *= cap / gross; }

    // Daily mark-to-market using raw price changes (on original scale)
    const rX = x[i] / x[i - 1] - 1;
    const rY = y[i] / y[i - 1] - 1;
    const pnl = equity * (wY * rY + wX * rX); // wX multiplies *raw* return
    equity = Math.max(0, equity + pnl);
    eqCurve.push({ i, equity: round6(equity) });
  }

  const rets = toRets(eqCurve.map(c => c.equity));
  const { annVol, sharpe } = quickStats(rets);
  const { maxDD } = drawdownFromEquity(eqCurve.map(c => c.equity));

  return {
    equity: eqCurve,
    z: zPath,
    beta: betaPath,
    trades,
    stats: {
      ret: round6(eqCurve.length ? eqCurve[eqCurve.length - 1].equity / startEq - 1 : 0),
      vol: round6(annVol),
      sharpe: round6(sharpe),
      maxDD: round6(maxDD),
      trades: trades.length,
    },
  };
}

/* ------------------------------- Math bits ------------------------------- */

function ols(x: Num[], y: Num[]) {
  // Regress y ≈ a + b x
  let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const xi = num(x[i]), yi = num(y[i]);
    if (!isFinite(xi) || !isFinite(yi)) continue;
    sx += xi; sy += yi; sxx += xi * xi; sxy += xi * yi; n++;
  }
  if (n < 2) return { alpha: 0, beta: 0, resid: [] as Num[] };
  const den = n * sxx - sx * sx;
  const beta = den !== 0 ? (n * sxy - sx * sy) / den : 0;
  const alpha = (sy - beta * sx) / n;

  const resid: Num[] = [];
  for (let i = 0; i < n; i++) {
    const xi = num(x[i]), yi = num(y[i]);
    resid.push(yi - (alpha + beta * xi));
  }
  return { alpha, beta, resid };
}

function variance(a: Num[]) { if (a.length < 2) return 0; const m = mean(a); let s = 0; for (let i=0;i<a.length;i++){ const d=a[i]-m; s+=d*d;} return s/(a.length-1); }
function mean(a: Num[]) { let s=0,n=0; for(let i=0;i<a.length;i++){ const v=a[i]; if(isFinite(v)){ s+=v;n++; } } return n? s/n:0; }
function stdev(a: Num[]) { const v=variance(a); return v>0? Math.sqrt(v):0; }
function toRets(eq: Num[]) { const out: Num[] = []; for (let i=1;i<eq.length;i++) out.push(eq[i]/eq[i-1]-1); return out; }
function quickStats(rets: Num[], ann = 252) { const m=mean(rets), s=stdev(rets); const annVol=s*Math.sqrt(ann); const sharpe= s!==0? (m/s)*Math.sqrt(ann):0; return { annVol, sharpe }; }
function drawdownFromEquity(eq: Num[]) { let peak=-Infinity, maxDD=0; for(let i=0;i<eq.length;i++){ const e=eq[i]; peak=Math.max(peak,e); if(peak>0){ const dd=(peak-e)/peak; if(dd>maxDD) maxDD=dd; } } return { maxDD }; }

function logArr(a: Num[]) { const out: Num[] = []; for (let i=0;i<a.length;i++){ const v=num(a[i]); out.push(v>0? Math.log(v): NaN); } return out; }
function num(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function isFiniteNum(v: any) { const n = Number(v); return Number.isFinite(n); }
function round6(n: number) { return Math.round(n * 1e-6) * 1e-6 + Math.round(n * 1e6) / 1e6 - Math.round(n * 1e-6) * 1e-6; } // stable