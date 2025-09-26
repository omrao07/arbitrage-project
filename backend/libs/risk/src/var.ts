// risk/src/var.ts
// Pure, zero-import VaR/ES utilities for portfolios and factor models.
// Standalone: duplicate tiny math helpers so there are NO imports.

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type Dict<T = any> = { [k: string]: T };

export type FactorHistory = {
  factors: string[];
  series: Record<string, number[]>; // factor -> returns[]
};

export type Exposure = { factor: string; value: number };

export type GaussianRisk = {
  sigma: number;       // daily stdev of portfolio P&L
  var: Record<"95" | "99", number>;
  es: Record<"95" | "99", number>;
  componentVaR?: Record<string, number>; // factor -> contribution (VaR decomposition, sums ≈ var[CL])
  componentES?: Record<string, number>;  // same scaling as VaR under Gaussian
};

export type HistoricalRisk = {
  sigma?: number;      // stdev of historical P&L
  var: Record<"95" | "99", number>;
  es: Record<"95" | "99", number>;
  pnl: number[];       // historical portfolio P&L path used
};

export type BacktestResult = {
  cl: number;              // e.g., 0.99
  n: number;               // sample size
  violations: number;      // count of P&L < -VaR
  violRate: number;        // violations / n
  kupiec: { LR: number; pValue: number }; // Unconditional coverage test
};

export type AnnualizeOptions = { days?: number; tradingDaysPerYear?: number };

// ────────────────────────────────────────────────────────────────────────────
// Historical VaR / ES
// ────────────────────────────────────────────────────────────────────────────

/** Compute historical portfolio P&L from factor exposures and factor history. */
export function pnlFromExposures(
  exposures: Exposure[],                 // monetary exposure per factor
  history: FactorHistory,                // factor returns history
  order?: string[]                       // optional explicit factor order
): number[] {
  const facOrder = order && order.length ? order.slice() : history.factors.slice();
  const map: Record<string, number> = {};
  for (let i = 0; i < exposures.length; i++) map[exposures[i].factor] = exposures[i].value;

  const T = lengthFromHistory(history);
  if (!T) return [];
  const pnl = new Array(T).fill(0);
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let f = 0; f < facOrder.length; f++) {
      const fac = facOrder[f];
      const r = Number(history.series[fac]?.[t] || 0);
      const E = Number(map[fac] || 0);
      s += E * r;
    }
    pnl[t] = s;
  }
  return pnl;
}

/** Historical risk stats (VaR/ES) from a P&L series. */
export function historicalRisk(pnl: number[]): HistoricalRisk {
  const clean = (pnl || []).map(Number).filter((x) => isFinite(x));
  const sig = clean.length >= 2 ? stdev(clean) : undefined;
  return {
    sigma: sig,
    var: {
      "95": -percentile(clean, 0.05),
      "99": -percentile(clean, 0.01)
    },
    es: {
      "95": -expectedShortfall(clean, 0.05),
      "99": -expectedShortfall(clean, 0.01)
    },
    pnl: clean
  };
}

/** One-liner: historical risk from exposures + factor history. */
export function historicalRiskFromFactors(exposures: Exposure[], history: FactorHistory): HistoricalRisk {
  const pnl = pnlFromExposures(exposures, history);
  return historicalRisk(pnl);
}

// ────────────────────────────────────────────────────────────────────────────
/** Parametric (Gaussian) VaR/ES */
// ────────────────────────────────────────────────────────────────────────────

/** Portfolio sigma from exposures vector and covariance matrix (E' Σ E)^0.5. */
export function portfolioSigma(E: number[], cov: number[][]): number {
  let s = 0;
  for (let i = 0; i < E.length; i++) {
    const Ei = E[i] || 0;
    const row = cov[i] || [];
    for (let j = 0; j < E.length; j++) s += Ei * (row[j] || 0) * (E[j] || 0);
  }
  return Math.sqrt(Math.max(0, s));
}

/** Gaussian VaR for sigma and confidence level (e.g., 0.99). */
export function gaussianVaR(sigma: number, cl: number): number {
  const z = invStdNormal(cl);
  return Math.max(0, z * Math.max(0, sigma));
}

/** Gaussian ES for sigma and confidence level (e.g., 0.99). */
export function gaussianES(sigma: number, cl: number): number {
  const z = invStdNormal(cl);
  const phi = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const tail = Math.max(1e-12, 1 - cl);
  return Math.max(0, (sigma * phi) / tail);
}

/** Gaussian portfolio risk with component breakdown by factor (contributions sum ≈ VaR). */
export function gaussianRiskFromCov(
  exposures: Exposure[] | { names: string[]; vector: number[] },
  cov: number[][],
  cls: number[] = [0.95, 0.99]
): GaussianRisk {
  const { names, vector } = normalizeExposures(exposures);
  const sigma = portfolioSigma(vector, cov);
  const res: GaussianRisk = {
    sigma,
    var: { "95": 0, "99": 0 },
    es: { "95": 0, "99": 0 }
  };
  // totals
  for (let i = 0; i < cls.length; i++) {
    const k = keyCL(cls[i]);
    (res.var as any)[k] = round2(gaussianVaR(sigma, cls[i]));
    (res.es as any)[k] = round2(gaussianES(sigma, cls[i]));
  }
  // component VaR (delta-normal): c_i = (E_i * (Σ E)_i) / (E'ΣE) * TotalVaR
  const SE: number[] = matVec(cov, vector);
  const denom = Math.max(1e-12, dot(vector, SE));
  const shares: number[] = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) shares[i] = (vector[i] * SE[i]) / denom;

  const compVaR: Record<string, number> = {};
  const compES: Record<string, number> = {};
  for (let i = 0; i < names.length; i++) {
    compVaR[names[i]] = round2(shares[i] * (res.var["99"] || 0)); // default show 99% breakdown
    compES[names[i]] = round2(shares[i] * (res.es["99"] || 0));
  }
  res.componentVaR = compVaR;
  res.componentES = compES;
  return res;
}

/** Cornish–Fisher adjusted VaR given sigma, skew, excess kurtosis and confidence level. */
export function cornishFisherVaR(sigma: number, cl: number, skew: number, exKurt: number): number {
  const z = invStdNormal(cl);
  const z2 = z * z, z3 = z2 * z;
  // CF expansion up to kurtosis term
  const adj = z + (1/6)*skew*(z2 - 1) + (1/24)*exKurt*(z3 - 3*z) - (1/36)*skew*skew*(2*z3 - 5*z);
  return Math.max(0, Math.abs(adj) * Math.max(0, sigma));
}

// ────────────────────────────────────────────────────────────────────────────
/** Covariance from factor history */
// ────────────────────────────────────────────────────────────────────────────

export function covarianceFromHistory(history: FactorHistory): number[][] {
  const F = history.factors.length;
  const T = lengthFromHistory(history);
  const X: number[][] = new Array(F);
  for (let i = 0; i < F; i++) {
    const f = history.factors[i];
    X[i] = (history.series[f] || []).slice(-T).map(Number);
  }
  // demean
  const mu = new Array(F).fill(0);
  for (let i = 0; i < F; i++) mu[i] = mean(X[i]);
  for (let i = 0; i < F; i++) for (let t = 0; t < T; t++) X[i][t] -= mu[i];

  const C: number[][] = new Array(F);
  for (let i = 0; i < F; i++) {
    C[i] = new Array(F).fill(0);
    for (let j = i; j < F; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += X[i][t] * X[j][t];
      const v = s / Math.max(1, T - 1);
      C[i][j] = v;
      if (i !== j) C[j][i] = v;
    }
  }
  return C;
}

// ────────────────────────────────────────────────────────────────────────────
/** Backtesting (Kupiec unconditional coverage) */
// ────────────────────────────────────────────────────────────────────────────

/**
 * Backtest a VaR model by counting breaches on a P&L path.
 * @param pnl  Array of realized daily P&L
 * @param var  Array of modeled VaR for each day (positive number)
 * @param cl   Confidence (e.g., 0.99)
 */
export function backtestVaR(pnl: number[], vaR: number[], cl: number): BacktestResult {
  const n = Math.min(pnl.length, vaR.length);
  let x = 0;
  for (let i = 0; i < n; i++) {
    const loss = -Number(pnl[i] || 0);
    const thr = Number(vaR[i] || 0);
    if (isFinite(loss) && isFinite(thr) && loss > thr) x++;
  }
  const p = 1 - cl;
  const LR = kupiecLR(n, x, p);
  const pValue = 1 - chiSqCDF(LR, 1); // 1 d.o.f.
  return { cl, n, violations: x, violRate: n ? x / n : 0, kupiec: { LR, pValue } };
}

// ────────────────────────────────────────────────────────────────────────────
/** Scaling & aggregation helpers */
// ────────────────────────────────────────────────────────────────────────────

export function scaleSigmaDailyToHorizon(sigmaDaily: number, days: number): number {
  return Math.max(0, sigmaDaily) * Math.sqrt(Math.max(0, days));
}
export function scaleVaRDailyToHorizon(varDaily: number, days: number): number {
  return Math.max(0, varDaily) * Math.sqrt(Math.max(0, days));
}
export function annualizeSigma(sigmaDaily: number, opts?: AnnualizeOptions): number {
  const N = Math.max(1, opts?.tradingDaysPerYear ?? 252);
  const days = Math.max(1, opts?.days ?? N);
  return scaleSigmaDailyToHorizon(sigmaDaily, days);
}

// ────────────────────────────────────────────────────────────────────────────
// Small utilities
// ────────────────────────────────────────────────────────────────────────────

function normalizeExposures(src: Exposure[] | { names: string[]; vector: number[] }): { names: string[]; vector: number[] } {
  if (Array.isArray(src)) {
    const names = src.map((e) => e.factor);
    const vector = src.map((e) => Number(e.value) || 0);
    return { names, vector };
  }
  return { names: src.names.slice(), vector: src.vector.slice() };
}

function matVec(A: number[][], x: number[]): number[] {
  const n = x.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = A[i] || [];
    for (let j = 0; j < n; j++) s += (row[j] || 0) * (x[j] || 0);
    y[i] = s;
  }
  return y;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] || 0) * (b[i] || 0);
  return s;
}

function keyCL(cl: number): "95" | "99" {
  if (Math.abs(cl - 0.95) < 1e-6) return "95";
  if (Math.abs(cl - 0.99) < 1e-6) return "99";
  // snap to nearest of the two
  return cl < 0.97 ? "95" : "99";
}

function lengthFromHistory(h: FactorHistory): number {
  if (!h?.factors?.length) return 0;
  const f0 = h.factors[0];
  return Array.isArray(h.series[f0]) ? h.series[f0].length : 0;
}

// Stats
function percentile(arr: number[], p: number): number {
  const a = (arr || []).map(Number).filter((x) => isFinite(x)).sort((x, y) => x - y);
  if (!a.length) return 0;
  const r = clamp01(p);
  const idx = r * (a.length - 1);
  const i = Math.floor(idx);
  const frac = idx - i;
  if (i + 1 < a.length) return a[i] * (1 - frac) + a[i + 1] * frac;
  return a[i];
}

function expectedShortfall(arr: number[], alpha: number): number {
  const a = (arr || []).map(Number).filter((x) => isFinite(x));
  if (!a.length) return 0;
  const thr = percentile(a, alpha);
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] <= thr) { s += a[i]; n++; }
  return n ? s / n : thr;
}

function stdev(xs: number[]): number {
  const a = xs.map(Number).filter((x) => isFinite(x));
  if (a.length < 2) return 0;
  const m = mean(a);
  let v = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; v += d * d; }
  return Math.sqrt(v / (a.length - 1));
}
function mean(xs: number[]): number {
  if (!xs.length) return 0;
  let s = 0; for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

// Inverse standard normal (Acklam)
function invStdNormal(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425, ph = 1 - pl;
  if (p <= 0 || p >= 1) return NaN;
  let x = 0;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  } else if (ph < p) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  } else {
    const q = p - 0.5; const r = q*q;
    x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
        (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
  }
  // one Halley step
  const e = 0.5 * (1 + erf(x / Math.SQRT2)) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
  return x - u / (1 + x * u / 2);
}
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * Math.exp(-x*x);
  return sign * y;
}

// Kupiec LR statistic
function kupiecLR(n: number, x: number, p: number): number {
  if (n <= 0) return 0;
  if (x <= 0) return -2 * Math.log(Math.pow(1 - p, n));
  if (x >= n) return -2 * Math.log(Math.pow(p, n));
  const pi = x / n;
  return -2 * (logPow(1 - p, n - x) + logPow(p, x) - logPow(1 - pi, n - x) - logPow(pi, x));
}
function logPow(base: number, exp: number): number {
  if (base <= 0) return -Infinity;
  return exp * Math.log(base);
}

// Chi-square CDF (k=1,2 small support; here generic via regularized gamma)
function chiSqCDF(x: number, k: number): number {
  if (x <= 0) return 0;
  const a = k / 2;
  const g = lowerRegGamma(a, x / 2);
  return clamp01(g);
}
function lowerRegGamma(s: number, x: number): number {
  // series expansion for x < s+1, continued fraction otherwise (Lanczos-ish light)
  if (x === 0) return 0;
  if (x < s + 1) {
    let sum = 1 / s, term = sum;
    for (let n = 1; n < 100; n++) {
      term *= x / (s + n);
      sum += term;
      if (Math.abs(term) < 1e-12) break;
    }
    return sum * Math.exp(-x + s * Math.log(x) - logGamma(s));
  } else {
    // continued fraction
    let a0 = 1, a1 = x, b0 = 0, b1 = 1, fac = 1, n = 1;
    let gOld = a1 / b1;
    for (; n < 200; n++) {
      const an = n % 2 === 0 ? (n / 2) * (s - (n / 2)) : ((n - 1) / 2);
      const bn = x + (n - s);
      a0 = a1 + an * a0;
      b0 = b1 + an * b0;
      a1 = bn * a0 + (n ? (n - s) : 0) * a1; // light stabilization
      b1 = bn * b0 + (n ? (n - s) : 0) * b1;
      if (b1 !== 0) {
        const g = a1 / b1;
        if (Math.abs((g - gOld) / g) < 1e-10) break;
        gOld = g;
      }
      // scale to avoid overflow
      if (Math.abs(a1) > 1e100) { a0 /= 1e100; b0 /= 1e100; a1 /= 1e100; b1 /= 1e100; }
    }
    const cf = a1 / b1;
    return 1 - Math.exp(-x + s * Math.log(x) - logGamma(s)) * cf;
  }
}
// log Γ(s) via Lanczos approximation
function logGamma(z: number): number {
  const g = 7;
  const p = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = p[0];
  for (let i = 1; i < p.length; i++) x += p[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// misc
function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
function round2(x: number): number { return Math.round(x * 100) / 100; }

// ────────────────────────────────────────────────────────────────────────────
// END
// ────────────────────────────────────────────────────────────────────────────