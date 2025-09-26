// portfolio/allocator.ts
// Portfolio allocation engine (pure Node, no imports).
//
// Supports several allocation schemes:
// - equalWeight()       → equally weights all assets
// - riskParity()        → inverse volatility weights
// - maxSharpe()         → mean-variance optimization for max Sharpe
// - targetVol()         → scale weights to hit target portfolio volatility
// - custom()            → user-supplied weights normalized
//
// Inputs: asset returns/vols/cov matrix, target vol, risk-free rate, etc.
// Outputs: allocation map (symbol → weight)
//
// NOTE: Stub math; replace with proper linear algebra/optimizers for production.

export type AssetStat = {
  symbol: string;
  mean: number;   // expected return
  vol: number;    // stdev
};

export type AllocatorInput = {
  stats: AssetStat[];
  cov?: number[][];   // covariance matrix (N x N)
  rf?: number;        // risk-free rate
  targetVol?: number; // for targetVol scheme
};

export type Weights = Record<string, number>;

export function Allocator() {
  /* ---------------------------- Helpers ---------------------------- */

  function normalize(w: Weights): Weights {
    const sum = Object.values(w).reduce((a, b) => a + b, 0);
    if (!sum) return w;
    const out: Weights = {};
    for (const k in w) out[k] = w[k] / sum;
    return out;
  }

  /* ------------------------- Strategies ---------------------------- */

  function equalWeight(input: AllocatorInput): Weights {
    const n = input.stats.length;
    const w: Weights = {};
    for (const a of input.stats) w[a.symbol] = 1 / n;
    return w;
  }

  function riskParity(input: AllocatorInput): Weights {
    const w: Weights = {};
    const vols = input.stats.map(s => s.vol || 1);
    const inv = vols.map(v => (v > 0 ? 1 / v : 0));
    const sum = inv.reduce((a, b) => a + b, 0);
    input.stats.forEach((s, i) => {
      w[s.symbol] = sum ? inv[i] / sum : 0;
    });
    return w;
  }

  function maxSharpe(input: AllocatorInput): Weights {
    // crude approximation: weight ∝ (mean-rf)/vol^2
    const rf = input.rf ?? 0;
    const w: Weights = {};
    const scores = input.stats.map(s => {
      const ex = s.mean - rf;
      return ex > 0 && s.vol > 0 ? ex / (s.vol * s.vol) : 0;
    });
    const sum = scores.reduce((a, b) => a + b, 0);
    input.stats.forEach((s, i) => {
      w[s.symbol] = sum ? scores[i] / sum : 0;
    });
    return w;
  }

  function targetVol(input: AllocatorInput): Weights {
    // naive: start with risk parity, then scale
    const rp = riskParity(input);
    const tgt = input.targetVol ?? 0.1;
    const vol = portVol(rp, input);
    if (!vol || vol <= 0) return rp;
    const scale = tgt / vol;
    const w: Weights = {};
    for (const k in rp) w[k] = rp[k] * scale;
    return normalize(w);
  }

  function custom(weights: Weights): Weights {
    return normalize({ ...weights });
  }

  /* ------------------------- Metrics ---------------------------- */

  function portVol(w: Weights, input: AllocatorInput): number {
    // var = w^T Σ w ; simple diag approx if no cov provided
    const syms = input.stats.map(s => s.symbol);
    const ws = syms.map(s => w[s] ?? 0);
    if (!input.cov) {
      // diag only
      let v = 0;
      for (let i = 0; i < syms.length; i++) v += ws[i] * ws[i] * (input.stats[i].vol ** 2);
      return Math.sqrt(v);
    }
    const Σ = input.cov;
    let sum = 0;
    for (let i = 0; i < ws.length; i++) {
      for (let j = 0; j < ws.length; j++) {
        sum += ws[i] * ws[j] * Σ[i][j];
      }
    }
    return Math.sqrt(Math.max(0, sum));
  }

  return {
    equalWeight,
    riskParity,
    maxSharpe,
    targetVol,
    custom,
  };
}