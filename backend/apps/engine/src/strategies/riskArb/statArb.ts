// riskarb/statarb.ts
// No imports. Pure TypeScript utilities for stat-arb (pairs) research.
// - OLS hedge ratio (y ~ alpha + beta x)
// - Engle–Granger cointegration test (ADF on residuals)
// - ADF test (constant, no trend) with classic critical values
// - Spread builder, rolling z-score, half-life of mean reversion
// - Pairs backtester with z-score entry/exit and costs
// All helpers are deterministic and side-effect free.
//
// Usage (prices aligned, same length):
//   const beta = hedgeRatio(y, x).beta;
//   const eg = engleGranger(y, x);
//   const bt = backtestPairs({ y, x, options: { zWindow: 60, entryZ: 2, exitZ: 0.5 } });

/* ============================ Public Types ============================ */

export type HedgeRatio = {
  alpha: number;
  beta: number;
  r2: number;
  stderrAlpha: number;
  stderrBeta: number;
  n: number;
};

export type ADFResult = {
  tStat: number;              // tau statistic on lagged level
  lags: number;               // number of Δ lags used
  n: number;                  // effective obs
  critical: { "1%": number; "5%": number; "10%": number };
  pApprox: number;            // very rough p-value (<=)
  stationaryAt: "1%" | "5%" | "10%" | "none";
};

export type EngleGrangerResult = {
  ok: boolean;                // cointegrated at alpha (default 5%)
  alphaLevel: 0.01 | 0.05 | 0.1;
  adf: ADFResult;
  hedge: HedgeRatio;
  residuals: number[];
  spread: number[];           // y - beta x - alpha (same as residuals)
};

export type BacktestOptions = {
  zWindow?: number;           // rolling window for z, default 60
  entryZ?: number;            // default 2
  exitZ?: number;             // default 0.5
  lookbackBeta?: number;      // rolling OLS window for beta; if undefined, use full sample static beta
  costBps?: number;           // per-leg cost when position changes; default 1 (0.01%)
  useAlpha?: boolean;         // include α in spread; default true
  allowShort?: boolean;       // both directions allowed; default true
};

export type Trade = {
  dir: "longSpread" | "shortSpread";
  openIdx: number;
  closeIdx: number;
  openZ: number;
  closeZ: number;
  pnl: number;                // net P&L over trade (per $ gross)
  ret: number;                // same as pnl (weights are per $1 gross)
};

export type BacktestResult = {
  equity: number[];           // cumulative equity curve (starts at 1)
  returns: number[];          // per-step returns (includes costs)
  positions: Array<0 | 1 | -1>; // signal state: 1=longSpread, -1=shortSpread
  trades: Trade[];
  summary: {
    n: number;
    cagr: number;
    vol: number;
    sharpe: number;
    hitRate: number;
    avgTrade: number;
    maxDD: number;
    grossLeverage: number;    // constant 1 by construction
    costBps: number;
  };
  params: Required<BacktestOptions> & { beta: number; alpha: number };
};

/* ============================== OLS / Hedge ============================== */

export function hedgeRatio(y: number[], x: number[]): HedgeRatio {
  const { n, sx, sy, sxx, sxy, syy } = moments2(y, x);
  if (n < 2) return { alpha: 0, beta: 0, r2: 0, stderrAlpha: NaN, stderrBeta: NaN, n };
  const beta = (sxy - (sx * sy) / n) / (sxx - (sx * sx) / n);
  const alpha = (sy - beta * sx) / n;

  // Residuals, R^2, standard errors
  let sse = 0;
  let sst = 0;
  const meanY = sy / n;
  for (let i = 0; i < n; i++) {
    const yi = y[i];
    const xi = x[i];
    const yhat = alpha + beta * xi;
    const e = yi - yhat;
    sse += e * e;
    const d = yi - meanY;
    sst += d * d;
  }
  const r2 = sst > 0 ? Math.max(0, 1 - sse / sst) : 0;
  const sigma2 = sse / Math.max(1, n - 2);
  const sxxAdj = sxx - (sx * sx) / n;
  const stderrBeta = Math.sqrt(sigma2 / Math.max(1e-12, sxxAdj));
  const stderrAlpha = Math.sqrt(sigma2 * (1 / n + (sx * sx) / (n * Math.max(1e-12, sxxAdj))));

  return { alpha, beta, r2, stderrAlpha, stderrBeta, n };
}

/* ============================== ADF Test ============================== */
/**
 * Augmented Dickey–Fuller (constant, no trend):
 *   Δy_t = a + γ y_{t-1} + Σ_{i=1..k} φ_i Δy_{t-i} + ε_t
 * Returns t-stat on γ; criticals from MacKinnon (approx).
 */
export function adf(series: number[], maxLags: number = autoLags(series.length)): ADFResult {
  const y = dropNa(series.slice());
  const n = y.length;
  const crit = { "1%": -3.43, "5%": -2.86, "10%": -2.57 }; // with constant, no trend (large n)
  if (n < 20) {
    return { tStat: NaN, lags: 0, n, critical: crit, pApprox: 1, stationaryAt: "none" };
  }

  // Build Δy and lag matrix
  const dy: number[] = new Array(n - 1);
  for (let i = 1; i < n; i++) dy[i - 1] = y[i] - y[i - 1];

  // Choose lags via BIC over [0..maxLags]
  let best = { k: 0, bic: Infinity, t: NaN };
  for (let k = 0; k <= Math.min(maxLags, n - 2); k++) {
    const reg = adfRegression(y, dy, k);
    if (isNaN(reg.tStat)) continue;
    if (reg.bic < best.bic) best = { k, bic: reg.bic, t: reg.tStat };
  }

  const tStat = best.t;
  const lags = best.k;
  const pApprox = approxP(tStat, crit);
  const stationaryAt =
    tStat <= crit["1%"] ? "1%" :
    tStat <= crit["5%"] ? "5%" :
    tStat <= crit["10%"] ? "10%" : "none";

  return { tStat, lags, n: n - 1 - lags, critical: crit, pApprox, stationaryAt };
}

function adfRegression(y: number[], dy: number[], k: number) {
  // Build X with columns: [1, y_{t-1}, d1..dk], response: dy_t
  const T = dy.length - k;
  if (T <= 2) return { tStat: NaN, bic: Infinity };
  const Y = new Array(T);
  const X0 = new Array(T); // intercept
  const X1 = new Array(T); // y_{t-1}
  const Xd: number[][] = [];
  for (let j = 0; j < k; j++) Xd[j] = new Array(T);

  for (let t = 0; t < T; t++) {
    Y[t] = dy[k + t];
    X0[t] = 1;
    X1[t] = y[k + t]; // y_{t-1} because dy index shifted
    for (let j = 0; j < k; j++) Xd[j][t] = dy[k + t - (j + 1)];
  }

  // OLS via normal equations (small k, fine)
  const p = 2 + k;
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const XtY: number[] = new Array(p).fill(0);
  for (let t = 0; t < T; t++) {
    const row: number[] = [X0[t], X1[t], ...Xd.map((col) => col[t])];
    for (let i = 0; i < p; i++) {
      XtY[i] += row[i] * Y[t];
      for (let j = 0; j < p; j++) XtX[i][j] += row[i] * row[j];
    }
  }
  const theta = solveSymmetric(XtX, XtY); // [a, γ, φ...]
  if (!theta) return { tStat: NaN, bic: Infinity };

  // Residual variance and (XtX)^-1
  const inv = invSymmetric(XtX);
  if (!inv) return { tStat: NaN, bic: Infinity };

  let sse = 0;
  for (let t = 0; t < T; t++) {
    const yhat = theta[0] + theta[1] * X1[t] + Xd.reduce((s, col, j) => s + theta[2 + j] * col[t], 0);
    const e = Y[t] - yhat;
    sse += e * e;
  }
  const sigma2 = sse / Math.max(1, T - p);
  const varGamma = sigma2 * inv[1][1];
  const tStat = theta[1] / Math.sqrt(Math.max(1e-12, varGamma));

  // BIC = n*ln(sigma2) + p*ln(n)
  const bic = T * Math.log(Math.max(1e-12, sigma2)) + p * Math.log(T);

  return { tStat, bic };
}

function autoLags(n: number) {
  // Schwert rule of thumb
  return Math.min(12, Math.floor(Math.pow(n, 1 / 3)));
}

function approxP(t: number, crit: { "1%": number; "5%": number; "10%": number }) {
  // crude, monotonic interpolation across criticals (left tail)
  if (!isFinite(t)) return 1;
  if (t <= crit["1%"]) return 0.01;
  if (t <= crit["5%"]) return 0.05;
  if (t <= crit["10%"]) return 0.1;
  return 0.5; // not significant
}

/* =========================== Engle–Granger =========================== */

export function engleGranger(
  y: number[],
  x: number[],
  alphaLevel: 0.01 | 0.05 | 0.1 = 0.05,
  maxLags?: number,
): EngleGrangerResult {
  const H = hedgeRatio(y, x);
  const res: number[] = y.map((yi, i) => yi - (H.alpha + H.beta * x[i]));
  const A = adf(res, maxLags);
  const ok =
    (alphaLevel === 0.01 && A.tStat <= A.critical["1%"]) ||
    (alphaLevel === 0.05 && A.tStat <= A.critical["5%"]) ||
    (alphaLevel === 0.1 && A.tStat <= A.critical["10%"]);

  return { ok, alphaLevel, adf: A, hedge: H, residuals: res.slice(1), spread: res.slice(1) };
}

/* ============================ Spread, Z-score ============================ */

export function buildSpread(y: number[], x: number[], beta: number, alpha: number = 0): number[] {
  const n = Math.min(y.length, x.length);
  const s = new Array(n);
  for (let i = 0; i < n; i++) s[i] = y[i] - (alpha + beta * x[i]);
  return s;
}

export function rollingZ(arr: number[], window: number): number[] {
  const n = arr.length;
  const out = new Array(n).fill(NaN);
  if (window <= 1) return out;
  let sum = 0, sum2 = 0;
  const q: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    q.push(v); sum += v; sum2 += v * v;
    if (q.length > window) {
      const u = q.shift()!;
      sum -= u; sum2 -= u * u;
    }
    if (q.length === window) {
      const mean = sum / window;
      const varr = Math.max(1e-12, sum2 / window - mean * mean);
      out[i] = (v - mean) / Math.sqrt(varr);
    }
  }
  return out;
}

export function halflife(series: number[]): number {
  // Δs = α + ρ s_{t-1} + ε  => HL = -ln(2)/ln(1+ρ)
  const y = dropNa(series);
  if (y.length < 2) return NaN;
  const dy = new Array(y.length - 1);
  for (let i = 1; i < y.length; i++) dy[i - 1] = y[i] - y[i - 1];
  const x = y.slice(0, -1);
  const { beta: rho } = hedgeRatio(dy, x); // dy ~ α + ρ s_{t-1}
  const lam = Math.log(1 + rho);
  if (!isFinite(lam) || lam >= 0) return NaN;
  return -Math.log(2) / lam;
}

/* ============================== Backtester ============================== */

export function backtestPairs(args: {
  y: number[];               // price of asset Y
  x: number[];               // price of asset X
  options?: BacktestOptions;
}): BacktestResult {
  const { y, x } = args;
  const n = Math.min(y.length, x.length);
  const opt: Required<BacktestOptions> = {
    zWindow: args.options?.zWindow ?? 60,
    entryZ: args.options?.entryZ ?? 2,
    exitZ: args.options?.exitZ ?? 0.5,
    lookbackBeta: args.options?.lookbackBeta ?? 0,
    costBps: args.options?.costBps ?? 1,
    useAlpha: args.options?.useAlpha ?? true,
    allowShort: args.options?.allowShort ?? true,
  };

  if (n < Math.max(2, opt.zWindow + 1)) {
    return emptyBT(n, opt);
  }

  // Optionally rolling beta/alpha
  const betas = new Array(n).fill(NaN);
  const alphas = new Array(n).fill(NaN);
  const fullHedge = hedgeRatio(y, x);
  for (let i = 0; i < n; i++) {
    if (opt.lookbackBeta && i + 1 >= opt.lookbackBeta) {
      const s = i + 1 - opt.lookbackBeta;
      const h = hedgeRatio(y.slice(s, i + 1), x.slice(s, i + 1));
      betas[i] = h.beta;
      alphas[i] = h.alpha;
    } else {
      betas[i] = fullHedge.beta;
      alphas[i] = fullHedge.alpha;
    }
  }

  // Spread and z-score
  const spread = new Array(n);
  for (let i = 0; i < n; i++) spread[i] = y[i] - ((opt.useAlpha ? alphas[i] : 0) + betas[i] * x[i]);
  const z = rollingZ(spread, opt.zWindow);

  // Price returns
  const rY = ret(y), rX = ret(x);

  // Sim loop
  const pos: Array<0 | 1 | -1> = new Array(n).fill(0);
  const rets: number[] = new Array(n).fill(0);
  const trades: Trade[] = [];
  let state: 0 | 1 | -1 = 0;
  let openIdx = -1;
  let openZ = NaN;

  const bps = opt.costBps * 1e-4;

  for (let t = 1; t < n; t++) {
    const zt = z[t - 1]; // decide based on yesterday's close (avoid look-ahead)
    const beta = betas[t - 1];
    if (!isFinite(zt) || !isFinite(beta)) { pos[t] = state; rets[t] = 0; continue; }

    // Entry rules
    let desired: 0 | 1 | -1 = state;
    if (state === 0) {
      if (opt.allowShort && zt > opt.entryZ) desired = -1;      // short spread: short y, long beta*x
      else if (zt < -opt.entryZ) desired = 1;                    // long spread: long y, short beta*x
    } else if (state === 1 && zt >= -opt.exitZ) {
      desired = 1; // hold until cross inside band (we'll exit when |z| <= exitZ)
      if (Math.abs(zt) <= opt.exitZ) desired = 0;
    } else if (state === -1 && zt <= opt.exitZ) {
      desired = -1;
      if (Math.abs(zt) <= opt.exitZ) desired = 0;
    }

    // Costs on position change (per leg, per $ gross)
    const prevW = weights(state, beta);
    const nextW = weights(desired, beta);
    const turnover = Math.abs(prevW.wY - nextW.wY) + Math.abs(prevW.wX - nextW.wX); // gross per $1
    const cost = turnover * bps;

    // Realized return at t from positions held over (t-1,t]
    const pnl = prevW.wY * rY[t] + prevW.wX * rX[t] - cost;
    rets[t] = pnl;

    // Track trades
    if (state === 0 && desired !== 0) {
      openIdx = t; openZ = zt;
    } else if (state !== 0 && desired === 0) {
      const tr: Trade = {
        dir: state === 1 ? "longSpread" : "shortSpread",
        openIdx,
        closeIdx: t,
        openZ,
        closeZ: zt,
        pnl: cum(rets, openIdx + 1, t), // from entry step to close step
        ret: cum(rets, openIdx + 1, t),
      };
      trades.push(tr);
      openIdx = -1; openZ = NaN;
    }

    state = desired;
    pos[t] = state;
  }

  // Build equity
  const equity: number[] = new Array(n).fill(1);
  for (let t = 1; t < n; t++) equity[t] = equity[t - 1] * (1 + rets[t]);

  const summary = summarizeBT(rets, equity, trades, opt);

  return {
    equity,
    returns: rets,
    positions: pos,
    trades,
    summary,
    params: { ...opt, beta: fullHedge.beta, alpha: fullHedge.alpha },
  };
}

/* ============================== Helpers ============================== */

// Per-$1 gross weights for spread trade.
// Long spread (dir=+1): long y, short beta*x
// Short spread (dir=-1): short y, long beta*x
function weights(dir: 0 | 1 | -1, beta: number) {
  if (dir === 0 || !isFinite(beta)) return { wY: 0, wX: 0 };
  const rawY = dir === 1 ? 1 : -1;
  const rawX = -beta * rawY;
  const scale = 1 / (Math.abs(rawY) + Math.abs(rawX)); // keep gross = 1
  return { wY: rawY * scale, wX: rawX * scale };
}

function ret(p: number[]): number[] {
  const n = p.length, r = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const prev = p[i - 1], cur = p[i];
    r[i] = prev > 0 && isFinite(prev) && isFinite(cur) ? cur / prev - 1 : 0;
  }
  return r;
}

function summarizeBT(rets: number[], eq: number[], trades: Trade[], opt: Required<BacktestOptions>) {
  const n = rets.length;
  const avgR = mean(rets.slice(1));
  const sd = stdev(rets.slice(1), avgR);
  const sharpe = sd > 0 ? (avgR * 252) / (sd * Math.sqrt(252)) : 0;
  const cagr = n > 1 ? Math.pow(eq[eq.length - 1], 252 / Math.max(1, n - 1)) - 1 : 0;
  const vol = sd * Math.sqrt(252);
  const hitRate = trades.length ? trades.filter((t) => t.pnl > 0).length / trades.length : 0;
  const avgTrade = trades.length ? trades.reduce((s, t) => s + t.pnl, 0) / trades.length : 0;
  const maxDD = maxDrawdown(eq);
  return {
    n,
    cagr,
    vol,
    sharpe,
    hitRate,
    avgTrade,
    maxDD,
    grossLeverage: 1,
    costBps: opt.costBps,
  };
}

function cum(a: number[], i0: number, i1: number) {
  let s = 0;
  for (let i = i0; i <= i1; i++) s += a[i] || 0;
  return s;
}

function maxDrawdown(curve: number[]) {
  let peak = curve[0] || 1, maxDD = 0;
  for (let i = 1; i < curve.length; i++) {
    peak = Math.max(peak, curve[i]);
    const dd = 1 - (curve[i] / peak);
    if (dd > maxDD) maxDD = dd;
  }
  return maxDD;
}

/* ============================== Math Core ============================== */

function dropNa(a: number[]) {
  return a.filter((v) => isFinite(v));
}
function moments2(y: number[], x: number[]) {
  const n = Math.min(y.length, x.length);
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i], yi = y[i];
    sx += xi; sy += yi;
    sxx += xi * xi; syy += yi * yi; sxy += xi * yi;
  }
  return { n, sx, sy, sxx, syy, sxy };
}
function mean(a: number[]) {
  if (a.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}
function stdev(a: number[], m = mean(a)) {
  const n = a.length;
  if (n < 2) return 0;
  let acc = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - m; acc += d * d; }
  return Math.sqrt(acc / (n - 1));
}

// Solve symmetric positive-definite system via Cholesky
function solveSymmetric(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 1e-12) return null;
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  // Solve Ly = b
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let k = 0; k < i; k++) sum -= L[i][k] * y[k];
    y[i] = sum / L[i][i];
  }
  // Solve L^T x = y
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let k = i + 1; k < n; k++) sum -= L[k][i] * x[k];
    x[i] = sum / L[i][i];
  }
  return x;
}

// Inverse via Cholesky
function invSymmetric(A: number[][]): number[][] | null {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 1e-12) return null;
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  // Compute inv using L
  const inv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  // Solve for columns of inverse
  for (let col = 0; col < n; col++) {
    // Ly = e_col
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = (i === col ? 1 : 0);
      for (let k = 0; k < i; k++) sum -= L[i][k] * y[k];
      y[i] = sum / L[i][i];
    }
    // L^T x = y
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i];
      for (let k = i + 1; k < n; k++) sum -= L[k][i] * x[k];
      x[i] = sum / L[i][i];
    }
    for (let i = 0; i < n; i++) inv[i][col] = x[i];
  }
  return inv;
}

/* ============================== Defaults ============================== */

function emptyBT(n: number, opt: Required<BacktestOptions>): BacktestResult {
  return {
    equity: new Array(n).fill(1),
    returns: new Array(n).fill(0),
    positions: new Array(n).fill(0) as Array<0 | 1 | -1>,
    trades: [],
    summary: {
      n,
      cagr: 0,
      vol: 0,
      sharpe: 0,
      hitRate: 0,
      avgTrade: 0,
      maxDD: 0,
      grossLeverage: 1,
      costBps: opt.costBps,
    },
    params: { ...opt, beta: 0, alpha: 0 },
  };
}

/* ============================== Export default ============================== */

const statarb = {
  hedgeRatio,
  adf,
  engleGranger,
  buildSpread,
  rollingZ,
  halflife,
  backtestPairs,
};

export default statarb;