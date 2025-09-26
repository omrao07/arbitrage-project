// core/actions.ts
// Corporate actions utilities (pure TypeScript, zero imports).
// - Split adjustment (forward/backward)
// - Cash dividend back-adjustment (close-only)
// - Total-return series (reinvested dividends)

/*──────────── Types ────────────*/
export type Bar = { t: string; o: number; h: number; l: number; c: number; v?: number };
export type Split = { t: string; ratio: number };               // e.g., 2-for-1 -> 2.0
export type Dividend = { t: string; cash: number; currency?: string };

export type AdjustMode = "backward" | "forward";

export type AdjustResult = {
  bars: Bar[];                               // adjusted OHLC bars
  tr?: { t: string; tr: number }[];          // optional total return index
};

/*──────────── Public API ────────────*/

/**
 * Adjust bars for splits (and optionally dividends), returning adjusted bars.
 * - mode="backward": back-adjust history so the series is smooth (common for charts/analysis).
 * - mode="forward": apply splits going forward (keeps history raw; future bars scaled).
 */
export function adjustSplits(bars: Bar[], splits: Split[], mode: AdjustMode = "backward"): Bar[] {
  const S = normalizeSplits(splits);
  if (!S.length) return cloneBars(bars);

  return mode === "backward"
    ? adjustSplitsBackward(bars, S)
    : adjustSplitsForward(bars, S);
}

/**
 * Back-adjust close prices for cash dividends (close-only), producing a smooth price series.
 * Typical for computing continuous returns on a price-only index.
 */
export function adjustDividendsCloseOnlyBackward(bars: Bar[], dividends: Dividend[]): Bar[] {
  if (!dividends?.length) return cloneBars(bars);
  const B = sortAsc(bars.slice());
  const D = dividends.slice().sort((a, b) => cmpDate(a.t, b.t));

  // For each bar at time t, subtract all dividends that occur *after* t (back-adjustment).
  const out = B.map(b => ({ ...b }));
  for (let i = 0; i < out.length; i++) {
    let adj = 0;
    for (let j = 0; j < D.length; j++) {
      if (cmpDate(D[j].t, out[i].t) > 0) adj += D[j].cash;
    }
    out[i].c = Math.max(0, out[i].c - adj);
    // Leave o/h/l as-is (close-only adjustment); if desired, mirror to o/h/l.
  }
  return out;
}

/**
 * Build a total-return index from close prices and cash dividends
 * by reinvesting dividends at the close on (or just after) the dividend date.
 * Returns a series normalized to base = close[0] * 1 unit.
 */
export function buildTotalReturn(close: { t: string; c: number }[], dividends: Dividend[]): { t: string; tr: number }[] {
  if (!close?.length) return [];
  const C = close.slice().sort((a, b) => cmpDate(a.t, b.t));
  const D = (dividends || []).slice().sort((a, b) => cmpDate(a.t, b.t));

  let units = 1;                               // start with 1 unit
  const base = C[0].c || 1;
  const out: { t: string; tr: number }[] = [];

  let dj = 0;
  for (let i = 0; i < C.length; i++) {
    const ct = C[i].t;
    const cp = C[i].c;
    while (dj < D.length && cmpDate(D[dj].t, ct) <= 0) {
      // reinvest dividend at current close
      const addUnits = (D[dj].cash || 0) / Math.max(1e-9, cp);
      units += addUnits;
      dj++;
    }
    const trVal = (cp * units) / Math.max(1e-9, base);
    out.push({ t: ct, tr: trVal });
  }
  return out;
}

/**
 * Convenience: apply both split adjustment and produce a TR index.
 * - mode: "backward" or "forward" for split adjustment.
 * - dividends: used only to compute TR (and for optional back-adjust close if `divBackAdjust=true`).
 */
export function adjustBars(
  bars: Bar[],
  splits: Split[] = [],
  dividends: Dividend[] = [],
  mode: AdjustMode = "backward",
  divBackAdjust = false
): AdjustResult {
  let adj = adjustSplits(bars, splits, mode);

  if (divBackAdjust && dividends?.length) {
    // Back-adjust close for dividends; keep o/h/l unchanged.
    const adjDiv = adjustDividendsCloseOnlyBackward(adj, dividends);
    // Merge back adjusted closes into OHLC
    adj = adj.map((b, i) => ({ ...b, c: adjDiv[i].c }));
  }

  const tr = dividends?.length ? buildTotalReturn(adj.map(b => ({ t: b.t, c: b.c })), dividends) : undefined;
  return { bars: adj, tr };
}

/*──────────── Split Adjustment Implementations ────────────*/

// Backward adjustment: historical bars BEFORE a split are divided by the split ratio.
// For multiple splits, multiply all ratios that occur AFTER the bar's date.
function adjustSplitsBackward(bars: Bar[], splits: Split[]): Bar[] {
  const B = sortAsc(bars.slice());
  const S = splits.slice().sort((a, b) => cmpDate(a.t, b.t));
  const out = B.map(b => ({ ...b }));

  for (let i = 0; i < out.length; i++) {
    const bt = out[i].t;
    let factor = 1;
    for (let j = 0; j < S.length; j++) {
      if (cmpDate(S[j].t, bt) > 0) factor *= S[j].ratio; // future (after bar) splits
    }
    if (factor !== 1) out[i] = scaleBar(out[i], 1 / factor);
  }
  return out;
}

// Forward adjustment: bars AT/AFTER split date are divided by the ratio.
// Historical bars remain raw.
function adjustSplitsForward(bars: Bar[], splits: Split[]): Bar[] {
  const B = sortAsc(bars.slice());
  const S = splits.slice().sort((a, b) => cmpDate(a.t, b.t));
  const out = B.map(b => ({ ...b }));

  let idx = 0;
  let factor = 1; // cumulative for future portion
  for (let i = 0; i < out.length; i++) {
    const bt = out[i].t;
    while (idx < S.length && cmpDate(S[idx].t, bt) <= 0) {
      factor *= S[idx].ratio;
      idx++;
    }
    if (factor !== 1) out[i] = scaleBar(out[i], 1 / factor);
  }
  return out;
}

/*──────────── Helpers ────────────*/

function normalizeSplits(splits: Split[]): Split[] {
  return (splits || []).filter(s => isFiniteNum(s.ratio) && s.ratio > 0);
}

function cloneBars(bars: Bar[]): Bar[] {
  return bars.map(b => ({ ...b }));
}

function scaleBar(b: Bar, k: number): Bar {
  return { t: b.t, o: b.o * k, h: b.h * k, l: b.l * k, c: b.c * k, v: b.v };
}

function cmpDate(a: string, b: string): number {
  const ta = Date.parse(a), tb = Date.parse(b);
  return ta === tb ? 0 : ta < tb ? -1 : 1;
}

function sortAsc<T extends { t: string }>(arr: T[]): T[] {
  return arr.sort((a, b) => cmpDate(a.t, b.t));
}

function isFiniteNum(x: any): x is number {
  return typeof x === "number" && isFinite(x);
}