// risk/metrics.ts
// Pure, import-free risk & performance metrics utilities.
//
// What you get (all functions are deterministic & side-effect free):
// - returnsFromPrices(prices) → simple log/arith returns
// - volatility(returns, {annFactor}) → stdev and annualized vol
// - sharpe/sortino/informationRatio
// - maxDrawdown(drawdownSeries?) and full drawdown path
// - betaAlpha(asset, bench, {annFactor})
// - covMatrix / corrMatrix for a matrix of series
// - portfolioVol(weights, cov) and contribution to risk
// - VaR/CVaR: historical & Gaussian (parametric)
// - exposure({positions, prices, equity}) → gross, net, leverage
//
// All inputs are plain arrays/records; no dependencies.

export type Num = number;

export type ReturnsOptions = {
  method?: "log" | "simple"; // default "simple"
  dropNaN?: boolean;         // default true
};

export function returnsFromPrices(prices: Num[], opts: ReturnsOptions = {}) {
  const method = opts.method || "simple";
  const dropNaN = opts.dropNaN !== false;
  const out: Num[] = [];
  for (let i = 1; i < prices.length; i++) {
    const p0 = Number(prices[i - 1]), p1 = Number(prices[i]);
    if (!(isFinite(p0) && isFinite(p1)) || p0 <= 0 || p1 <= 0) {
      if (!dropNaN) out.push(NaN);
      continue;
    }
    out.push(method === "log" ? Math.log(p1 / p0) : (p1 - p0) / p0);
  }
  return out;
}

export function mean(arr: Num[]) {
  let s = 0, n = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]; if (!isFinite(v)) continue; s += v; n++;
  }
  return n ? s / n : 0;
}

export function stdev(arr: Num[]) {
  let m = mean(arr), s = 0, n = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]; if (!isFinite(v)) continue; const d = v - m; s += d * d; n++;
  }
  return n > 1 ? Math.sqrt(s / (n - 1)) : 0;
}

export function volatility(returns: Num[], opts: { annFactor?: number } = {}) {
  const sigma = stdev(returns);
  const ann = sigma * (opts.annFactor ?? 252) ** 0.5;
  return { stdev: sigma, annVol: ann };
}

export function sharpe(returns: Num[], opts: { rf?: number; annFactor?: number } = {}) {
  const rf = opts.rf ?? 0;
  const ex = returns.map(r => r - rf / (opts.annFactor ?? 252));
  const m = mean(ex);
  const v = stdev(ex);
  const annF = Math.sqrt(opts.annFactor ?? 252);
  return v === 0 ? 0 : (m / v) * annF;
}

export function sortino(returns: Num[], opts: { rf?: number; annFactor?: number } = {}) {
  const rf = opts.rf ?? 0;
  const f = opts.annFactor ?? 252;
  const dr: Num[] = [];
  for (let i = 0; i < returns.length; i++) {
    const ex = returns[i] - rf / f;
    if (ex < 0) dr.push(ex);
  }
  const downside = stdev(dr);
  const m = mean(returns) - rf / f;
  return downside === 0 ? 0 : (m / downside) * Math.sqrt(f);
}

export function informationRatio(asset: Num[], bench: Num[], annFactor = 252) {
  const n = Math.min(asset.length, bench.length);
  if (n === 0) return 0;
  const diff: Num[] = [];
  for (let i = 0; i < n; i++) diff.push(asset[i] - bench[i]);
  const m = mean(diff);
  const s = stdev(diff);
  return s === 0 ? 0 : (m / s) * Math.sqrt(annFactor);
}

export function drawdownFromEquity(equity: Num[]) {
  let peak = -Infinity;
  const dd: Num[] = [];
  for (let i = 0; i < equity.length; i++) {
    const x = equity[i];
    peak = Math.max(peak, x);
    const d = peak > 0 ? (peak - x) / peak : 0;
    dd.push(Math.max(0, d));
  }
  const maxDD = dd.reduce((a, b) => Math.max(a, b), 0);
  return { ddSeries: dd, maxDD };
}

export function drawdownFromReturns(rets: Num[], startEquity = 1) {
  const eq: Num[] = [startEquity];
  for (let i = 0; i < rets.length; i++) eq.push(eq[eq.length - 1] * (1 + rets[i]));
  return drawdownFromEquity(eq);
}

export function betaAlpha(asset: Num[], bench: Num[], opts: { annFactor?: number } = {}) {
  const n = Math.min(asset.length, bench.length);
  if (n === 0) return { beta: 0, alpha: 0 };
  const ax = asset.slice(-n), bx = bench.slice(-n);
  const mA = mean(ax), mB = mean(bx);
  let cov = 0, varB = 0, k = 0;
  for (let i = 0; i < n; i++) {
    const da = ax[i] - mA, db = bx[i] - mB;
    if (!isFinite(da) || !isFinite(db)) continue;
    cov += da * db;
    varB += db * db;
    k++;
  }
  const beta = k > 1 ? cov / varB : 0;
  const alphaDaily = mA - beta * mB;
  const annF = opts.annFactor ?? 252;
  return { beta, alpha: alphaDaily * annF };
}

export function covMatrix(series: Num[][]) {
  const n = series.length;
  const m: Num[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const means = series.map(mean);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0, k = 0;
      const a = series[i], b = series[j];
      const mi = means[i], mj = means[j];
      const len = Math.min(a.length, b.length);
      for (let t = 0; t < len; t++) {
        const ai = a[t], bj = b[t];
        if (!isFinite(ai) || !isFinite(bj)) continue;
        s += (ai - mi) * (bj - mj);
        k++;
      }
      const c = k > 1 ? s / (k - 1) : 0;
      m[i][j] = m[j][i] = c;
    }
  }
  return m;
}

export function corrMatrix(series: Num[][]) {
  const Σ = covMatrix(series);
  const n = Σ.length;
  const out: Num[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const sd = Σ.map((row, i) => Math.sqrt(row[i]));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = sd[i] * sd[j];
      out[i][j] = d === 0 ? 0 : Σ[i][j] / d;
    }
  }
  return out;
}

export function portfolioVol(weights: Num[], cov: Num[][]) {
  const n = weights.length;
  let v = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      v += weights[i] * weights[j] * (cov[i]?.[j] ?? 0);
    }
  }
  return Math.sqrt(Math.max(0, v));
}

export function riskContributions(weights: Num[], cov: Num[][]) {
  // Marginal contribution: (Σ w)_i ; %RC = w_i * (Σ w)_i / (w^T Σ w)
  const n = weights.length;
  const sig = portfolioVol(weights, cov);
  if (sig === 0) return weights.map(() => 0);
  const grad: Num[] = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let g = 0;
    for (let j = 0; j < n; j++) g += cov[i][j] * weights[j];
    grad[i] = g;
  }
  const rc: Num[] = [];
  for (let i = 0; i < n; i++) rc.push((weights[i] * grad[i]) / (sig * sig));
  return rc; // sums to 1
}

/* ----------------------------- VaR / CVaR ----------------------------- */

export function histVaR(returns: Num[], cl = 0.99) {
  const clean = returns.filter((x) => isFinite(x)).slice().sort((a, b) => a - b);
  if (clean.length === 0) return 0;
  const idx = Math.max(0, Math.min(clean.length - 1, Math.floor((1 - cl) * clean.length)));
  // VaR is positive loss number
  return Math.max(0, -clean[idx]);
}

export function histCVaR(returns: Num[], cl = 0.99) {
  const clean = returns.filter((x) => isFinite(x)).slice().sort((a, b) => a - b);
  if (clean.length === 0) return 0;
  const k = Math.max(1, Math.floor((1 - cl) * clean.length));
  let s = 0;
  for (let i = 0; i < k; i++) s += clean[i];
  return Math.max(0, -s / k);
}

export function gaussianVaR(mu: number, sigma: number, cl = 0.99) {
  // For normal dist: VaR = -(mu + z*sigma) with z = Φ^-1(1-cl)
  const z = invNorm(1 - cl);
  return Math.max(0, -(mu + z * sigma));
}

export function gaussianCVaR(mu: number, sigma: number, cl = 0.99) {
  // CVaR (a.k.a. ES) for normal: = -(mu + sigma * φ(z)/(1-cl)), z = Φ^-1(1-cl)
  const z = invNorm(1 - cl);
  const c = phi(z) / (1 - cl);
  return Math.max(0, -(mu + sigma * c));
}

/* ------------------------------- Exposure ------------------------------- */

export function exposure(input: {
  positions: Record<string, number>; // qty
  prices: Record<string, number>;    // px
  equity?: number;
}) {
  let longMV = 0, shortMV = 0;
  for (const k in input.positions) {
    const q = Number(input.positions[k]);
    const p = Number(input.prices[k]);
    if (!isFinite(q) || !isFinite(p)) continue;
    const mv = q * p;
    if (mv > 0) longMV += mv;
    if (mv < 0) shortMV += -mv;
  }
  const gross = longMV + shortMV;
  const net = longMV - shortMV;
  const lev = (input.equity && input.equity > 0) ? gross / input.equity : 0;
  return {
    longMV, shortMV, gross, net, leverage: lev,
  };
}

/* ----------------------------- Math helpers ---------------------------- */

function phi(x: number) { // standard normal PDF
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function invNorm(p: number) {
  // Acklam's approximation for inverse normal CDF
  // Valid for 0<p<1. For tails used here (p small), this is sufficient.
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285104469687e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00,
  ];
  const b = [
    -5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01,
  ];
  const c = [
    -7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
     4.374664141464968e+00,  2.938163982698783e+00,
  ];
  const d = [
     7.784695709041462e-03,  3.224671290700398e-01,
     2.445134137142996e+00,  3.754408661907416e+00,
  ];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
           ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
         (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
}