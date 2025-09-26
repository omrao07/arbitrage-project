// riskarb/pairstrade.ts
// Import-free utilities for statistical pairs trading tailored for risk-arb / relative value.
// Focus: robust hedge estimation, (quasi) cointegration check, z-spread signals,
// dynamic sizing with stop/TP, and a compact backtester.
//
// What you get
// - hedgeOLS(x,y,{mode}) → { alpha, beta, spread[], residVar }
// - egADF(spread,{lags}) → simple Engle–Granger step with ADF-like t-stat (approx)
// - zSpread(spread, win) → rolling z-score of spread
// - signalFromZ(z, {enter, exit}) → long/short/flat
// - targetWeights({beta, z, enter, exit, maxGross}) → wX,wY for dollar-neutral leg
// - rebalance(port, weights, feeBps) → {trades, after, cost}
// - backtestPairs({x,y,win,enter,exit,lookbackHedge,feeBps,slipBps,capDD}) → path & stats
//
// Conventions
// - Arrays are chronological (oldest → newest).
// - x = reference (hedge) asset; y = trade asset; spread = y - (a + b x).
// - Positive position is long asset (weight +1 means fully notional to equity).

/* ------------------------------- Types ---------------------------------- */

export type Num = number;

export type Hedge = {
  alpha: number;
  beta: number;
  spread: number[];
  residVar: number;
};

export type ADF = {
  tStat: number;    // approximate ADF t-stat on Δu_t = ρ u_{t-1} + Σ φ_i Δu_{t-i} + ε_t
  rho: number;     // ρ
  lags: number;
};

export type Weights = { wX: number; wY: number }; // exposure on x & y legs

export type Signal = "long" | "short" | "flat";

/* ----------------------------- Estimation -------------------------------- */

/**
 * Ordinary Least Squares hedge y ≈ a + b x.
 * mode: "price" (default) or "log" to regress on log-prices.
 */
export function hedgeOLS(
  x: Num[],
  y: Num[],
  opts: { mode?: "price" | "log" } = {}
): Hedge {
  const mode = opts.mode || "price";
  const X = mode === "log" ? toLog(x) : x.slice();
  const Y = mode === "log" ? toLog(y) : y.slice();

  let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  const m = Math.min(X.length, Y.length);
  for (let i = 0; i < m; i++) {
    const xi = num(X[i]), yi = num(Y[i]);
    if (!isFinite(xi) || !isFinite(yi)) continue;
    sx += xi; sy += yi; sxx += xi * xi; sxy += xi * yi; n++;
  }
  if (n < 2) return { alpha: 0, beta: 0, spread: [], residVar: 0 };

  const den = n * sxx - sx * sx;
  const beta = den !== 0 ? (n * sxy - sx * sy) / den : 0;
  const alpha = (sy - beta * sx) / n;

  const spread: number[] = [];
  let ss = 0, k = 0;
  for (let i = 0; i < m; i++) {
    const xi = num(X[i]), yi = num(Y[i]);
    const u = yi - (alpha + beta * xi);
    spread.push(u);
    if (isFinite(u)) { ss += u * u; k++; }
  }
  const residVar = k > 1 ? ss / (k - 1) : 0;

  return { alpha, beta, spread, residVar };
}

/**
 * Lightweight Engle–Granger step: run ADF on residuals (spread).
 * Returns an approximate t-stat for ρ in Δu_t = ρ u_{t-1} + Σ φ_i Δu_{t-i} + ε_t.
 * If tStat < ~ -2.8 (heuristic), you often have mean-reverting behavior.
 */
export function egADF(spread: Num[], opts: { lags?: number } = {}): ADF {
  const u = spread.filter(isFiniteNum);
  const p = u.length;
  const lags = Math.max(0, (opts.lags ?? Math.floor(Math.sqrt(p)) - 1) | 0);
  if (p < Math.max(10, lags + 5)) return { tStat: 0, rho: 0, lags };

  // Build regression vectors
  // y = Δu_t ; x0 = u_{t-1}; x1..xk = Δu_{t-1}..Δu_{t-k}
  const du: number[] = [];                // Δu_t
  const uLag: number[] = [];              // u_{t-1}
  const dLags: number[][] = [];           // lagged Δu
  for (let i = 0; i < lags; i++) dLags.push([]);

  for (let t = 1 + lags; t < p; t++) {
    const d = u[t] - u[t - 1];
    du.push(d);
    uLag.push(u[t - 1]);
    for (let j = 0; j < lags; j++) {
      const dj = u[t - 1 - j] - u[t - 2 - j];
      dLags[j].push(dj);
    }
  }

  // OLS for du ~ ρ*uLag + Σ φ_j dLags_j
  const N = du.length;
  const K = 1 + lags;
  const XtX: number[][] = Array.from({ length: K }, () => Array(K).fill(0));
  const XtY: number[] = Array(K).fill(0);

  for (let i = 0; i < N; i++) {
    const row: number[] = [uLag[i]];
    for (let j = 0; j < lags; j++) row.push(dLags[j][i]);
    for (let a = 0; a < K; a++) {
      XtY[a] += row[a] * du[i];
      for (let b = 0; b < K; b++) XtX[a][b] += row[a] * row[b];
    }
  }

  const coef = solveSymmetric(XtX, XtY); // [ρ, φ1..φk]
  const rho = coef.length ? coef[0] : 0;

  // Residual variance and s.e. for ρ (sandwich via (X'X)^{-1} σ^2)
  let sse = 0;
  for (let i = 0; i < N; i++) {
    const row: number[] = [uLag[i]];
    for (let j = 0; j < lags; j++) row.push(dLags[j][i]);
    let yhat = 0;
    for (let a = 0; a < K; a++) yhat += (coef[a] || 0) * row[a];
    const e = du[i] - yhat;
    sse += e * e;
  }
  const sigma2 = N > K ? sse / (N - K) : 0;
  const inv = invSymmetric(XtX);
  const varRho = inv && inv[0] ? sigma2 * inv[0][0] : Infinity;
  const seRho = varRho > 0 ? Math.sqrt(varRho) : Infinity;
  const tStat = isFinite(seRho) && seRho > 0 ? rho / seRho : 0;

  return { tStat, rho, lags };
}

/* ------------------------------ Signals --------------------------------- */

export function zSpread(spread: Num[], win: number) {
  const w = Math.max(10, win | 0);
  const out: number[] = [];
  const q: number[] = [];
  for (let i = 0; i < spread.length; i++) {
    const v = num(spread[i]);
    q.push(v);
    if (q.length > w) q.shift();
    if (q.length === w) {
      const m = mean(q), s = stdev(q);
      out.push(s > 0 ? (v - m) / s : 0);
    } else {
      out.push(NaN);
    }
  }
  return out;
}

export function signalFromZ(z: number, cfg: { enter?: number; exit?: number } = {}): Signal {
  const enter = num(cfg.enter, 2);
  const exit = num(cfg.exit, 0.5);
  if (z >= enter) return "short";
  if (z <= -enter) return "long";
  if (Math.abs(z) <= exit) return "flat";
  return "flat";
}

/**
 * Dollar-neutral target weights from z-signal and hedge beta.
 * Caps gross exposure and keeps wY = -sign(z)*1 scaled; wX = -beta*wY.
 */
export function targetWeights(args: {
  beta: number; z: number; enter?: number; exit?: number; maxGross?: number; kScale?: number;
}): Weights {
  const enter = num(args.enter, 2);
  const exit = num(args.exit, 0.5);
  const grossCap = Math.abs(num(args.maxGross, 2)); // |wX|+|wY| cap
  const kScale = Math.max(0, num(args.kScale, 1));  // scale by |z|/enter (<= kScale)

  let wX = 0, wY = 0;
  let side: Signal = "flat";
  if (args.z >= enter) side = "short";
  else if (args.z <= -enter) side = "long";
  else if (Math.abs(args.z) <= exit) side = "flat";

  if (side !== "flat") {
    const mag = Math.min(kScale, Math.max(0.2, Math.abs(args.z) / enter)); // 0.2 floor
    // spread = y - (a + b x). Long spread → long y, short x*b
    if (side === "long") { wY = +mag; wX = -mag * args.beta; }
    if (side === "short") { wY = -mag; wX = +mag * args.beta; }
  }

  // cap gross
  const gross = Math.abs(wX) + Math.abs(wY);
  if (gross > grossCap && gross > 0) {
    const s = grossCap / gross;
    wX *= s; wY *= s;
  }
  return { wX: round6(wX), wY: round6(wY) };
}

/* ------------------------------ Rebalance -------------------------------- */

export function rebalance(
  current: Weights,
  target: Weights,
  equity: number,
  feeBps = 0
) {
  const fee = Math.max(0, feeBps) / 10_000;
  const trades = {
    dX: round6(target.wX - num(current.wX)),
    dY: round6(target.wY - num(current.wY)),
  };
  const turnover = Math.abs(trades.dX) + Math.abs(trades.dY);
  const cost = round6(turnover * fee * equity);
  const after: Weights = { wX: round6(num(current.wX) + trades.dX), wY: round6(num(current.wY) + trades.dY) };
  return { trades, after, cost };
}

/* ------------------------------- Backtest -------------------------------- */

export function backtestPairs(args: {
  x: Num[]; y: Num[];
  win?: number; enter?: number; exit?: number;
  lookbackHedge?: number; feeBps?: number; slipBps?: number;
  capDD?: number; startEquity?: number;
}) {
  const x = args.x.slice();
  const y = args.y.slice();
  const N = Math.min(x.length, y.length);
  const win = Math.max(30, num(args.win, 60));
  const look = Math.max(win, num(args.lookbackHedge, 120));
  const enter = num(args.enter, 2);
  const exit = num(args.exit, 0.5);
  const fee = Math.max(0, num(args.feeBps, 0)) / 10_000;
  const slip = Math.max(0, num(args.slipBps, 0)) / 10_000;
  const start = num(args.startEquity, 1);
  const capDD = Math.max(0, num(args.capDD, 0)); // stop if drawdown exceeds this fraction

  if (N < look + 5) return { equity: [] as { i: number; equity: number }[], stats: emptyStats(), trace: [] as any[] };

  let eq = start, peak = start;
  let w: Weights = { wX: 0, wY: 0 };
  const curve: { i: number; equity: number }[] = [];
  const trace: any[] = [];

  // Precompute z-spread progressively
  let a = 0, b = 1;
  let spreadWin: number[] = [];

  for (let i = 1; i < N; i++) {
    // re-estimate hedge on rolling window [i-look, i]
    const s = Math.max(0, i - look);
    const e = i + 1;
    const { alpha, beta, spread } = hedgeOLS(x.slice(s, e), y.slice(s, e));
    a = alpha; b = beta;

    // z-score on 'win' window
    spreadWin = spread;
    const zArr = zSpread(spreadWin, win);
    const z = zArr[zArr.length - 1];

    // target weights from z
    const tgt = targetWeights({ beta: b, z, enter, exit, maxGross: 2, kScale: 1.5 });

    // apply rebalance with costs
    const { after, cost } = rebalance(w, tgt, eq, (fee + slip) * 10_000);
    eq = Math.max(0, eq - cost);
    w = after;

    // daily P&L using raw returns of legs
    const rX = x[i] / x[i - 1] - 1;
    const rY = y[i] / y[i - 1] - 1;
    const pnl = eq * (w.wY * rY + w.wX * rX);
    eq = Math.max(0, eq + pnl);

    // risk control: drawdown stop
    peak = Math.max(peak, eq);
    const dd = peak > 0 ? (peak - eq) / peak : 0;
    if (capDD > 0 && dd > capDD) {
      // flatten
      const { after: flatAfter, cost: flatCost } = rebalance(w, { wX: 0, wY: 0 }, eq, (fee + slip) * 10_000);
      eq = Math.max(0, eq - flatCost);
      w = flatAfter;
    }

    curve.push({ i, equity: round6(eq) });
    trace.push({ i, z, beta: b, w: { ...w } });
  }

  const rets = toRets(curve.map(c => c.equity));
  const st = stats(rets);
  const dd = drawdown(curve.map(c => c.equity));

  return {
    equity: curve,
    stats: {
      ret: round6(curve.length ? curve[curve.length - 1].equity / start - 1 : 0),
      vol: round6(st.annVol),
      sharpe: round6(st.sharpe),
      maxDD: round6(dd.maxDD),
      trades: approxTrades(trace),
    },
    trace,
  };
}

/* --------------------------------- Math ---------------------------------- */

function solveSymmetric(A: number[][], b: number[]) {
  const inv = invSymmetric(A);
  if (!inv) return [];
  const k = b.length;
  const x = Array(k).fill(0);
  for (let i = 0; i < k; i++) {
    let s = 0;
    for (let j = 0; j < k; j++) s += inv[i][j] * b[j];
    x[i] = s;
  }
  return x;
}

function invSymmetric(M: number[][]) {
  const n = M.length;
  // Copy
  const A = Array.from({ length: n }, (_, i) => M[i].slice());
  const I = eye(n);

  // Gaussian elimination (not pivot-robust; OK for small K)
  for (let i = 0; i < n; i++) {
    let piv = A[i][i];
    if (Math.abs(piv) < 1e-12) return null;
    const invP = 1 / piv;
    for (let j = 0; j < n; j++) {
      A[i][j] *= invP;
      I[i][j] *= invP;
    }
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = A[r][i];
      for (let c = 0; c < n; c++) {
        A[r][c] -= f * A[i][c];
        I[r][c] -= f * I[i][c];
      }
    }
  }
  return I;
}

function eye(n: number) { const M: number[][] = []; for (let i=0;i<n;i++){ const r:Array<number>=[]; for(let j=0;j<n;j++) r.push(i===j?1:0); M.push(r);} return M; }

function toLog(a: number[]) { const out: number[] = []; for (let i=0;i<a.length;i++){ const v=num(a[i]); out.push(v>0? Math.log(v): NaN); } return out; }

function mean(a: number[]) { let s=0,n=0; for(let i=0;i<a.length;i++){ const v=a[i]; if(isFinite(v)){ s+=v; n++; } } return n? s/n:0; }
function variance(a: number[]) { if (a.length<2) return 0; const m=mean(a); let s=0; for(let i=0;i<a.length;i++){ const d=a[i]-m; s+=d*d; } return s/(a.length-1); }
function stdev(a: number[]) { const v=variance(a); return v>0? Math.sqrt(v):0; }
function toRets(p: number[]) { const out: number[] = []; for (let i = 1; i < p.length; i++) out.push(p[i]/p[i-1]-1); return out; }
function stats(rets: number[], ann=252) { const m=mean(rets), s=stdev(rets); return { annVol: s*Math.sqrt(ann), sharpe: s!==0? (m/s)*Math.sqrt(ann):0 }; }
function drawdown(eq: number[]) { let peak=-Infinity, maxDD=0; for (let i=0;i<eq.length;i++){ const e=eq[i]; peak=Math.max(peak,e); if(peak>0){ const dd=(peak-e)/peak; if(dd>maxDD) maxDD=dd; } } return { maxDD }; }
function approxTrades(trace: any[]) { // count changes in sign of weights as proxy
  let cnt=0; let prev=0;
  for (let i=0;i<trace.length;i++){ const w = (Math.abs(trace[i].w.wX)+Math.abs(trace[i].w.wY))>1e-9 ? Math.sign(trace[i].w.wY - trace[i].beta*trace[i].w.wX) : 0; if (w!==0 && prev!==0 && w!==prev) cnt++; if (w!==0) prev=w; }
  return cnt;
}

function num(v:any, d=0){ const n=Number(v); return Number.isFinite(n)? n: d; }
function isFiniteNum(v:any){ const n=Number(v); return Number.isFinite(n); }
function round6(n:number){ return Math.round(n*1e6)/1e6; }
function emptyStats(){ return { ret: 0, vol: 0, sharpe: 0, maxDD: 0, trades: 0 }; }

/* ---------------------------------- Note ---------------------------------- */
// The ADF implementation here is a compact approximation for desk screening.
// For rigorous inference (critical values, p-values), rely on a stats package
// or robust time-series library. This file purposely avoids all imports.