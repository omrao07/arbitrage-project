// macro/globalmacro.ts
// Import-free utilities for maintaining and querying a lightweight
// global macro database (time series store + derived indicators).
//
// What you get
// - Series registry by key (e.g., "US:CPI_YoY", "EU:PMI", "CN:EXPORTS_YoY")
// - Helpers to add/replace observations and compute MoM/YoY/rolling stats
// - Cross-country comparison and z-score normalization
// - Composite “risk regime” index (risk-on/off) from common market proxies
// - Simple GDP nowcast (linear + weights) by country
// - Recession probability using a logistic on yield-curve slope + PMI
// - Dump/Load for persistence
//
// Notes
// - You control units: store raw levels or pre-computed YoY, etc.
// - Timestamps are epoch ms (number). Monthly series: use month-end dates.
// - Everything is deterministic and side-effect free except the internal store.

export type Obs = { ts: number; v: number };
export type Series = { key: string; unit?: string; meta?: Record<string, any>; data: Obs[] };

export type MacroAPI = {
  // data management
  upsertSeries(s: { key: string; unit?: string; meta?: Record<string, any> }): void;
  addObs(key: string, obs: Obs | Obs[], replaceSameTs?: boolean): number;  // returns count added
  setSeries(key: string, data: Obs[], unit?: string): void;
  getSeries(key: string): Series | undefined;
  list(prefix?: string): string[];
  remove(key: string): boolean;
  clear(): void;

  // transforms
  slice(key: string, fromTs?: number, toTs?: number): Obs[];
  resampleMonthlyAvg(key: string): Obs[];     // naive monthly average by calendar month
  pctChange(key: string, periods: number): Obs[]; // (x_t/x_{t-n} - 1)
  diff(key: string, periods: number): Obs[];       // x_t - x_{t-n}
  rollMean(key: string, win: number): Obs[];       // simple MA
  rollStdev(key: string, win: number): Obs[];      // sample stdev

  // normalization & cross section
  zScore(key: string, win: number): Obs[];   // rolling z vs window
  latest(key: string): Obs | undefined;
  latestValue(key: string, fallback?: number): number;

  // composites
  riskRegime(input: Partial<RiskInputs>): { ts: number; score: number; bucket: "risk-on"|"neutral"|"risk-off"; detail: RiskDetail };
  gdpNowcast(country: string, spec?: Partial<NowcastSpec>): { ts: number; nowcast: number; detail: Record<string, number> };
  recessionProb(country: string, spec?: Partial<RecessionSpec>): { ts: number; prob: number; features: Record<string, number> };

  // persistence
  dump(): string;
  load(json: string): boolean;
};

export type RiskInputs = {
  vixKey: string;           // e.g., "US:VIX"
  creditKey: string;        // e.g., "US:BBB_OAS" (bp)
  slopeKey: string;         // e.g., "US:YC_10Y2Y" (bp)
  momKey?: string;          // e.g., "US:PMI" (level) or "US:RETAIL_YoY" (%)
  lookback?: number;        // window for z-scores (default 252)
};

export type RiskDetail = {
  zVix: number; zCredit: number; zSlope: number; zMom: number;
  weights: { vix: number; credit: number; slope: number; mom: number };
};

export type NowcastSpec = {
  // Linear combo of standardized features → quarterly real GDP annualized %
  features: Array<{ key: string; weight: number; stdWin?: number; transform?: "level"|"yoy"|"mom" }>;
  bias: number;
};

export type RecessionSpec = {
  // Logistic( b0 + b1*slope + b2*PMI + b3*credit + b4*equityRet )
  slopeKey: string;    // yield curve slope bp (+ steeper = less recession risk)
  pmiKey: string;      // PMI level
  creditKey?: string;  // credit spread bp
  eqRetKey?: string;   // equity 6m return (%)
  horizonMonths?: number; // calibration notion (unused except for doc)
  coefs?: { b0: number; b1: number; b2: number; b3: number; b4: number; scale?: number };
};


  const db = new Map<string, Series>(); // key -> Series

  /* ------------------------------- Store ------------------------------- */

  function upsertSeries(s: { key: string; unit?: string; meta?: Record<string, any> }) {
    const k = String(s.key);
    const cur = db.get(k);
    if (cur) {
      if (s.unit != null) cur.unit = s.unit;
      if (s.meta) cur.meta = Object.assign({}, cur.meta || {}, s.meta || {});
      return;
    }
    db.set(k, { key: k, unit: s.unit, meta: s.meta || {}, data: [] });
  }

  function addObs(key: string, obs: Obs | Obs[], replaceSameTs = true) {
    const k = String(key);
    if (!db.has(k)) db.set(k, { key: k, unit: "", meta: {}, data: [] });
    const s = db.get(k)!;
    const arr = Array.isArray(obs) ? obs : [obs];
    let added = 0;
    for (let i = 0; i < arr.length; i++) {
      const o = normalizeObs(arr[i]);
      if (!o) continue;
      const idx = binSearchTs(s.data, o.ts);
      if (idx.found) {
        if (replaceSameTs) { s.data[idx.idx] = o; added++; }
      } else {
        s.data.splice(idx.idx, 0, o);
        added++;
      }
    }
    return added;
  }

  function setSeries(key: string, data: Obs[], unit?: string) {
    const clean = cleanObs(data);
    db.set(String(key), { key: String(key), unit, meta: {}, data: clean });
  }

  function getSeries(key: string) { return cloneSeries(db.get(String(key))); }
  function list(prefix?: string) {
    const out: string[] = [];
    db.forEach((_, k) => { if (!prefix || k.startsWith(prefix)) out.push(k); });
    out.sort();
    return out;
  }
  function remove(key: string) { return db.delete(String(key)); }
  function clear() { db.clear(); }

  /* ----------------------------- Transforms ---------------------------- */

  function slice(key: string, fromTs?: number, toTs?: number): Obs[] {
    const s = db.get(String(key)); if (!s) return [];
    const a = s.data;
    const lo = fromTs == null ? 0 : lowerBound(a, Number(fromTs));
    const hi = toTs == null ? a.length : upperBound(a, Number(toTs));
    return cloneObs(a.slice(lo, hi));
  }

  function resampleMonthlyAvg(key: string): Obs[] {
    const s = db.get(String(key)); if (!s || s.data.length === 0) return [];
    const buckets = new Map<string, { sum: number; n: number; ts: number }>();
    for (let i = 0; i < s.data.length; i++) {
      const d = new Date(s.data[i].ts);
      const tag = d.getUTCFullYear() + "-" + pad2(d.getUTCMonth() + 1);
      const tEnd = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0); // month end
      const b = buckets.get(tag) || { sum: 0, n: 0, ts: tEnd };
      b.sum += s.data[i].v; b.n++; b.ts = tEnd;
      buckets.set(tag, b);
    }
    const out: Obs[] = [];
    buckets.forEach((b, tag) => { out.push({ ts: b.ts, v: b.sum / Math.max(1, b.n) }); });
    out.sort((a,b)=>a.ts-b.ts);
    return out;
  }

  function pctChange(key: string, periods: number): Obs[] {
    const xs = seriesValues(key);
    const out: Obs[] = [];
    for (let i = periods; i < xs.length; i++) {
      const prev = xs[i - periods]; const cur = xs[i];
      if (!isFinite(prev.v) || prev.v === 0) continue;
      out.push({ ts: cur.ts, v: cur.v / prev.v - 1 });
    }
    return out;
  }

  function diff(key: string, periods: number): Obs[] {
    const xs = seriesValues(key);
    const out: Obs[] = [];
    for (let i = periods; i < xs.length; i++) {
      const prev = xs[i - periods]; const cur = xs[i];
      out.push({ ts: cur.ts, v: cur.v - prev.v });
    }
    return out;
  }

  function rollMean(key: string, win: number): Obs[] {
    const xs = seriesValues(key);
    const out: Obs[] = [];
    const w = Math.max(1, win | 0);
    let sum = 0; let q: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      sum += xs[i].v; q.push(xs[i].v);
      if (q.length > w) sum -= q.shift()!;
      if (q.length === w) out.push({ ts: xs[i].ts, v: sum / w });
    }
    return out;
  }

  function rollStdev(key: string, win: number): Obs[] {
    const xs = seriesValues(key);
    const out: Obs[] = [];
    const w = Math.max(2, win | 0);
    const q: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      q.push(xs[i].v);
      if (q.length > w) q.shift();
      if (q.length === w) out.push({ ts: xs[i].ts, v: stdev(q) });
    }
    return out;
  }

  function zScore(key: string, win: number): Obs[] {
    const xs = seriesValues(key);
    const out: Obs[] = [];
    const w = Math.max(3, win | 0);
    const q: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      q.push(xs[i].v);
      if (q.length > w) q.shift();
      if (q.length === w) {
        const m = mean(q);
        const sd = stdev(q) || 1e-9;
        out.push({ ts: xs[i].ts, v: (q[q.length - 1] - m) / sd });
      }
    }
    return out;
  }

  function latest(key: string) {
    const s = db.get(String(key));
    return s && s.data.length ? { ...s.data[s.data.length - 1] } : undefined;
  }
  function latestValue(key: string, fallback = NaN) {
    const r = latest(key); return r ? r.v : fallback;
  }

  /* ----------------------------- Composites ----------------------------- */

  function riskRegime(input: Partial<RiskInputs>) {
    const ts = Date.now();
    const look = input.lookback ?? 252;
    const w = { vix: 0.35, credit: 0.30, slope: 0.25, mom: 0.10 }; // defaults

    // z-scores (higher = more risk-off for VIX/credit; slope sign inverted)
    const zVix = lastVal(zScore(req(input.vixKey, "vix"), look));
    const zCredit = lastVal(zScore(req(input.creditKey, "credit"), look));
    const zSlopeRaw = lastVal(zScore(req(input.slopeKey, "slope"), look));
    const zMom = input.momKey ? lastVal(zScore(req(input.momKey, "mom"), look)) : 0;

    const score = clamp(
      w.vix * pos(zVix) +
      w.credit * pos(zCredit) +
      w.slope * pos(-zSlopeRaw) +   // inverted: steepening → risk-on
      w.mom * neg(zMom),            // negative momentum → risk-off
      -3, 3
    );

    const bucket = score > 0.6 ? "risk-off" : score < -0.6 ? "risk-on" : "neutral";
    return { ts, score, bucket, detail: { zVix, zCredit, zSlope: -zSlopeRaw, zMom, weights: w } };
  }

  function gdpNowcast(country: string, spec?: Partial<NowcastSpec>) {
    // Minimal defaults per-availability
    const base: NowcastSpec = {
      bias: 0.0,
      features: [
        { key: `${country}:PMI`, weight: 0.8, stdWin: 60, transform: "level" },
        { key: `${country}:RETAIL_YoY`, weight: 0.6, stdWin: 60, transform: "yoy" },
        { key: `${country}:IP_YoY`, weight: 0.6, stdWin: 60, transform: "yoy" },
        { key: `${country}:UNEMP_RATE`, weight: -0.7, stdWin: 60, transform: "level" },
        { key: `${country}:YC_10Y2Y`, weight: 0.3, stdWin: 60, transform: "level" },
      ],
    };
    const cfg = mergeNowcast(base, spec || {});
    const featVals: Record<string, number> = {};
    let sum = cfg.bias;

    for (let i = 0; i < cfg.features.length; i++) {
      const f = cfg.features[i];
      const val = featureValue(f);
      featVals[f.key] = val * f.weight;
      sum += featVals[f.key];
    }
    return { ts: Date.now(), nowcast: sum, detail: featVals };
  }

  function recessionProb(country: string, spec?: Partial<RecessionSpec>) {
    const def: RecessionSpec = {
      slopeKey: `${country}:YC_10Y2Y`,
      pmiKey: `${country}:PMI`,
      creditKey: `${country}:BBB_OAS`,
      eqRetKey: `${country}:EQ_6M_RET`,
      horizonMonths: 12,
      coefs: { b0: -0.5, b1: -0.006, b2: -0.08, b3: 0.003, b4: -0.8, scale: 1 }, // heuristic
    };
    const cfg = Object.assign({}, def, spec || {});
    const C = cfg.coefs!;
    const slope = latestValue(cfg.slopeKey, 0);   // bp
    const pmi = latestValue(cfg.pmiKey, 50);      // level
    const credit = cfg.creditKey ? latestValue(cfg.creditKey, 150) : 150; // bp
    const eqr = cfg.eqRetKey ? latestValue(cfg.eqRetKey, 0) : 0;          // %

    const lin = C.b0 + C.b1 * slope + C.b2 * pmi + C.b3 * credit + C.b4 * eqr;
    const z = (C.scale ?? 1) * lin;
    const prob = 1 / (1 + Math.exp(-z));
    return { ts: Date.now(), prob, features: { slope, pmi, credit, eqr } };
  }

  /* ----------------------------- Persistence ---------------------------- */

  function dump() {
    const obj: any = { series: [] as any[] };
    db.forEach((s) => obj.series.push(s));
    try { return JSON.stringify(obj); } catch { return "{}"; }
  }

  function load(json: string) {
    try {
      const o = JSON.parse(json || "{}");
      if (!o || !Array.isArray(o.series)) return false;
      db.clear();
      for (let i = 0; i < o.series.length; i++) {
        const s = o.series[i];
        if (!s || !s.key || !Array.isArray(s.data)) continue;
        db.set(String(s.key), { key: String(s.key), unit: s.unit, meta: s.meta || {}, data: cleanObs(s.data) });
      }
      return true;
    } catch { return false; }
  }

  /* -------------------------------- Helpers ----------------------------- */

  function req(key?: string, label?: string): string {
    if (!key) return "";
    if (!db.has(key)) db.set(key, { key, unit: "", meta: {}, data: [] });
    return key;
  }

  function seriesValues(key: string): Obs[] {
    const s = db.get(String(key)); return s ? s.data : [];
  }

  function featureValue(f: { key: string; weight: number; stdWin?: number; transform?: "level"|"yoy"|"mom" }) {
    const t = f.transform || "level";
    if (t === "yoy") {
      const y = pctChange(f.key, 12); return lastVal(y);
    } else if (t === "mom") {
      const m = pctChange(f.key, 1); return lastVal(m);
    } else {
      // standardized level
      const z = zScore(f.key, Math.max(24, f.stdWin ?? 60));
      return lastVal(z);
    }
  }

  function lastVal(obs: Obs[] | number): number {
    if (typeof obs === "number") return obs;
    return obs.length ? obs[obs.length - 1].v : 0;
  }

  


/* ------------------------------- Utilities ------------------------------- */

function normalizeObs(o?: Obs): Obs | null {
  if (!o || !Number.isFinite(o.ts) || !Number.isFinite(o.v)) return null;
  return { ts: Number(o.ts), v: Number(o.v) };
}

function cleanObs(arr: Obs[]): Obs[] {
  const out: Obs[] = [];
  for (let i = 0; i < arr.length; i++) {
    const o = normalizeObs(arr[i]); if (o) out.push(o);
  }
  out.sort((a,b)=>a.ts-b.ts);
  // de-dup same ts keep last
  const ded: Obs[] = [];
  for (let i = 0; i < out.length; i++) {
    if (ded.length && ded[ded.length-1].ts === out[i].ts) ded[ded.length-1] = out[i];
    else ded.push(out[i]);
  }
  return ded;
}

function binSearchTs(a: Obs[], ts: number): { idx: number; found: boolean } {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m].ts === ts) return { idx: m, found: true };
    if (a[m].ts < ts) lo = m + 1; else hi = m;
  }
  return { idx: lo, found: false };
}

function lowerBound(a: Obs[], ts: number) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m].ts < ts) lo = m + 1; else hi = m;
  }
  return lo;
}
function upperBound(a: Obs[], ts: number) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m].ts <= ts) lo = m + 1; else hi = m;
  }
  return lo;
}

function mean(a: number[]) {
  let s = 0, n = 0; for (let i=0;i<a.length;i++){ const v=a[i]; if (isFinite(v)) { s+=v; n++; } }
  return n ? s / n : 0;
}
function stdev(a: number[]) {
  const m = mean(a); let s = 0, n = 0;
  for (let i=0;i<a.length;i++){ const v=a[i]; if (isFinite(v)) { const d=v-m; s+=d*d; n++; } }
  return n > 1 ? Math.sqrt(s/(n-1)) : 0;
}
function clamp(x: number, lo: number, hi: number) { return Math.min(hi, Math.max(lo, x)); }
function pos(x: number) { return x > 0 ? x : 0; }
function neg(x: number) { return x < 0 ? -x : 0; }
function pad2(n: number) { return n < 10 ? "0" + n : "" + n; }
function mergeNowcast(base: NowcastSpec, o: Partial<NowcastSpec>): NowcastSpec {
  const out: NowcastSpec = { bias: base.bias, features: base.features.slice() };
  if (typeof o.bias === "number") out.bias = o.bias;
  if (Array.isArray(o.features)) out.features = o.features.slice();
  return out;
}
function cloneSeries(s?: Series): Series | undefined {
  if (!s) return undefined;
  return { key: s.key, unit: s.unit, meta: s.meta ? { ...s.meta } : undefined, data: s.data.map(x => ({ ts: x.ts, v: x.v })) };
}

function cloneObs(arg0: Obs[]): Obs[] {
    throw new Error("Function not implemented.");
}
