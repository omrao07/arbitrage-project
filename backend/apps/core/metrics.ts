// core/metrics.ts
// Zero-import metrics primitives with a tiny Prometheus text exporter.
// Features:
//  • Counter, Gauge, Histogram (ms buckets by default)
//  • Labels (fast, stable key serialization)
//  • Timers (observe duration in ms) + helper registry.timer()
//  • Snapshot + reset + approximate percentiles
//  • EWMA rate helper for counters (1m/5m/15m)
// Pure TypeScript; no external deps.

/* ───────────────────────── Types ───────────────────────── */

export type Labels = Record<string, string>;

export type MetricType = "counter" | "gauge" | "histogram";

export type MetricBase = {
  name: string;
  help?: string;
  type: MetricType;
  labelNames?: string[];
};

export type CounterSnapshot = { value: number };
export type GaugeSnapshot = { value: number };
export type HistogramSnapshot = {
  buckets: { le: number, count: number }[];
  sum: number;
  count: number;
};

export type MetricSnapshot =
  | { type: "counter"; name: string; labels: Labels; data: CounterSnapshot }
  | { type: "gauge"; name: string; labels: Labels; data: GaugeSnapshot }
  | { type: "histogram"; name: string; labels: Labels; data: HistogramSnapshot };

export type RegistrySnapshot = {
  ts: string;
  metrics: MetricSnapshot[];
};

/* ───────────────────────── Utils ───────────────────────── */

function nowMs(): number {
  // @ts-ignore
  const perf = (typeof performance !== "undefined" && performance?.now) ? performance : null;
  if (perf) {
    const base = Date.now() - perf.now();
    return Math.floor(base + perf.now());
  }
  return Date.now();
}

function stableLabelKey(labels: Labels): string {
  if (!labels) return "";
  const keys = Object.keys(labels).sort();
  let s = "";
  for (const k of keys) s += k + "=" + (labels[k] ?? "") + "|";
  return s;
}

function pickLabels(names: string[] | undefined, labels: Labels | undefined): Labels {
  if (!names || !names.length) return {};
  const out: Labels = {};
  const src = labels || {};
  for (const n of names) out[n] = src[n] ?? "";
  return out;
}

function clone<T>(x: T): T { return JSON.parse(JSON.stringify(x)); }

function defaultMsBuckets(): number[] {
  // Prometheus-style ms buckets (suitable for API latencies)
  return [5, 10, 25, 50, 75, 100, 150, 250, 400, 600, 1000, 1500, 2500, 4000, 6000, 10_000];
}

/* ───────────────────────── Metrics ───────────────────────── */

class CounterImpl {
  readonly meta: MetricBase;
  private byLabels = new Map<string, { labels: Labels; value: number; ewma: EWMASet }>();

  constructor(meta: MetricBase) {
    this.meta = { ...meta, type: "counter" };
  }

  inc(labels?: Labels, delta = 1): void {
    const lbl = pickLabels(this.meta.labelNames, labels);
    const key = stableLabelKey(lbl);
    let row = this.byLabels.get(key);
    if (!row) {
      row = { labels: lbl, value: 0, ewma: new EWMASet(nowMs()) };
      this.byLabels.set(key, row);
    }
    row.value += delta;
    row.ewma.update(delta, nowMs());
  }

  add(labels?: Labels, delta = 1): void { this.inc(labels, delta); }

  get(labels?: Labels): number {
    const key = stableLabelKey(pickLabels(this.meta.labelNames, labels));
    return this.byLabels.get(key)?.value ?? 0;
  }

  rate(labels?: Labels): { m1: number; m5: number; m15: number } {
    const key = stableLabelKey(pickLabels(this.meta.labelNames, labels));
    const row = this.byLabels.get(key);
    return row ? row.ewma.rates(nowMs()) : { m1: 0, m5: 0, m15: 0 };
  }

  snapshot(): MetricSnapshot[] {
    const out: MetricSnapshot[] = [];
    for (const row of this.byLabels.values()) {
      out.push({ type: "counter", name: this.meta.name, labels: row.labels, data: { value: row.value } });
    }
    return out;
  }

  reset(): void { this.byLabels.clear(); }
}

class GaugeImpl {
  readonly meta: MetricBase;
  private byLabels = new Map<string, { labels: Labels; value: number }>();

  constructor(meta: MetricBase) {
    this.meta = { ...meta, type: "gauge" };
  }

  set(labels: Labels | undefined, value: number): void {
    const lbl = pickLabels(this.meta.labelNames, labels);
    const key = stableLabelKey(lbl);
    let row = this.byLabels.get(key);
    if (!row) { row = { labels: lbl, value: 0 }; this.byLabels.set(key, row); }
    row.value = value;
  }

  inc(labels?: Labels, delta = 1): void { this.add(labels, delta); }
  dec(labels?: Labels, delta = 1): void { this.add(labels, -delta); }
  add(labels: Labels | undefined, delta: number): void {
    const lbl = pickLabels(this.meta.labelNames, labels);
    const key = stableLabelKey(lbl);
    let row = this.byLabels.get(key);
    if (!row) { row = { labels: lbl, value: 0 }; this.byLabels.set(key, row); }
    row.value += delta;
  }

  get(labels?: Labels): number {
    const key = stableLabelKey(pickLabels(this.meta.labelNames, labels));
    return this.byLabels.get(key)?.value ?? 0;
  }

  snapshot(): MetricSnapshot[] {
    const out: MetricSnapshot[] = [];
    for (const row of this.byLabels.values()) {
      out.push({ type: "gauge", name: this.meta.name, labels: row.labels, data: { value: row.value } });
    }
    return out;
  }

  reset(): void { this.byLabels.clear(); }
}

class HistogramImpl {
  readonly meta: MetricBase;
  private bounds: number[];
  private byLabels = new Map<string, HRow>();

  constructor(meta: MetricBase, buckets?: number[]) {
    this.meta = { ...meta, type: "histogram" };
    const b = (buckets && buckets.slice().sort((a, b) => a - b)) || defaultMsBuckets();
    // Ensure +Inf virtual bucket handled during snapshot
    this.bounds = b;
  }

  observe(value: number, labels?: Labels): void {
    const v = Number(value);
    if (!isFinite(v)) return;
    const lbl = pickLabels(this.meta.labelNames, labels);
    const key = stableLabelKey(lbl);
    let row = this.byLabels.get(key);
    if (!row) {
      row = { labels: lbl, buckets: new Array(this.bounds.length).fill(0), sum: 0, count: 0, reservoir: new Reservoir(1024) };
      this.byLabels.set(key, row);
    }
    let i = 0;
    while (i < this.bounds.length && v > this.bounds[i]) i++;
    if (i < this.bounds.length) row.buckets[i] += 1;
    row.sum += v;
    row.count += 1;
    row.reservoir.push(v);
  }

  percentiles(pcts: number[], labels?: Labels): Record<string, number> {
    const key = stableLabelKey(pickLabels(this.meta.labelNames, labels));
    const row = this.byLabels.get(key);
    const out: Record<string, number> = Object.create(null);
    if (!row || row.count === 0) {
      for (const p of pcts) out[String(p)] = NaN;
      return out;
    }
    for (const p of pcts) out[String(p)] = row.reservoir.quantile(p);
    return out;
  }

  snapshot(): MetricSnapshot[] {
    const out: MetricSnapshot[] = [];
    for (const row of this.byLabels.values()) {
      const buckets: { le: number, count: number }[] = [];
      let acc = 0;
      for (let i = 0; i < this.bounds.length; i++) {
        acc += row.buckets[i];
        buckets.push({ le: this.bounds[i], count: acc });
      }
      // +Inf bucket
      buckets.push({ le: Number.POSITIVE_INFINITY, count: row.count });
      out.push({
        type: "histogram",
        name: this.meta.name,
        labels: row.labels,
        data: { buckets, sum: row.sum, count: row.count }
      });
    }
    return out;
  }

  reset(): void { this.byLabels.clear(); }
}

type HRow = {
  labels: Labels;
  buckets: number[];
  sum: number;
  count: number;
  reservoir: Reservoir;
};

/* ───────────────────────── Reservoir (quantiles) ───────────────────────── */

class Reservoir {
  private cap: number;
  private n = 0;
  private buf: number[];

  constructor(cap = 1024) {
    this.cap = Math.max(8, cap|0);
    this.buf = new Array<number>(0);
  }

  push(v: number) {
    if (this.n < this.cap) {
      this.buf.push(v);
      this.n++;
    } else {
      // reservoir sampling: replace with decreasing probability
      const j = Math.floor(Math.random() * (this.n + 1));
      if (j < this.cap) this.buf[j] = v;
      this.n++;
    }
  }

  quantile(p: number): number {
    if (!this.buf.length) return NaN;
    const a = this.buf.slice().sort((x, y) => x - y);
    const pos = Math.min(a.length - 1, Math.max(0, (a.length - 1) * p));
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return a[lo];
    const w = pos - lo;
    return a[lo] * (1 - w) + a[hi] * w;
  }
}

/* ───────────────────────── EWMA for rates ───────────────────────── */

class EWMASet {
  private lastTs: number;
  private m1: number;
  private m5: number;
  private m15: number;

  constructor(ts: number) {
    this.lastTs = ts;
    this.m1 = 0; this.m5 = 0; this.m15 = 0;
  }

  update(incr: number, ts: number) {
    const dt = Math.max(1, ts - this.lastTs) / 1000; // seconds
    const rate = incr / dt; // inst rate
    // alphas for ~1m, 5m, 15m EWMAs per second step
    const a1  = 1 - Math.exp(-dt / 60);
    const a5  = 1 - Math.exp(-dt / (5*60));
    const a15 = 1 - Math.exp(-dt / (15*60));
    this.m1  = this.m1  + a1  * (rate - this.m1);
    this.m5  = this.m5  + a5  * (rate - this.m5);
    this.m15 = this.m15 + a15 * (rate - this.m15);
    this.lastTs = ts;
  }

  rates(_ts: number): { m1: number; m5: number; m15: number } {
    return { m1: this.m1, m5: this.m5, m15: this.m15 };
  }
}

/* ───────────────────────── Registry ───────────────────────── */

export class Registry {
  private counters = new Map<string, CounterImpl>();
  private gauges = new Map<string, GaugeImpl>();
  private hists = new Map<string, HistogramImpl>();
  private help = new Map<string, string>();

  counter(name: string, help?: string, labelNames?: string[]): CounterImpl {
    let m = this.counters.get(name);
    if (!m) {
      m = new CounterImpl({ name, help, type: "counter", labelNames: labelNames?.slice() });
      this.counters.set(name, m); if (help) this.help.set(name, help);
    }
    return m;
  }

  gauge(name: string, help?: string, labelNames?: string[]): GaugeImpl {
    let m = this.gauges.get(name);
    if (!m) {
      m = new GaugeImpl({ name, help, type: "gauge", labelNames: labelNames?.slice() });
      this.gauges.set(name, m); if (help) this.help.set(name, help);
    }
    return m;
  }

  histogram(name: string, help?: string, buckets?: number[], labelNames?: string[]): HistogramImpl {
    let m = this.hists.get(name);
    if (!m) {
      m = new HistogramImpl({ name, help, type: "histogram", labelNames: labelNames?.slice() }, buckets);
      this.hists.set(name, m); if (help) this.help.set(name, help);
    }
    return m;
  }

  /** Helper: start a timer that will observe duration (ms) into a histogram on done(). */
  timer(name: string, labels?: Labels, help?: string, buckets?: number[], labelNames?: string[]): () => number {
    const h = this.histogram(name, help, buckets, labelNames);
    const t0 = nowMs();
    const lbl = labels || {};
    return () => {
      const dur = Math.max(0, nowMs() - t0);
      h.observe(dur, lbl);
      return dur;
    };
  }

  snapshot(): RegistrySnapshot {
    const out: MetricSnapshot[] = [];
    for (const m of this.counters.values()) out.push(...m.snapshot());
    for (const m of this.gauges.values()) out.push(...m.snapshot());
    for (const m of this.hists.values()) out.push(...m.snapshot());
    return { ts: new Date().toISOString(), metrics: out };
  }

  reset(): void {
    for (const m of this.counters.values()) m.reset();
    for (const m of this.gauges.values()) m.reset();
    for (const m of this.hists.values()) m.reset();
  }

  /** Export all metrics in Prometheus text exposition format. */
  toPrometheus(): string {
    const lines: string[] = [];
    // metadata
    for (const [name, help] of this.help) {
      lines.push(`# HELP ${sanitizeName(name)} ${escapeHelp(help)}`);
      const t = this.hists.has(name) ? "histogram" : this.gauges.has(name) ? "gauge" : "counter";
      lines.push(`# TYPE ${sanitizeName(name)} ${t}`);
    }

    // values
    for (const snap of this.snapshot().metrics) {
      const baseName = sanitizeName(snap.name);
      if (snap.type === "counter") {
        lines.push(`${baseName}${fmtLabels(snap.labels)} ${num(snap.data.value)}`);
      } else if (snap.type === "gauge") {
        lines.push(`${baseName}${fmtLabels(snap.labels)} ${num(snap.data.value)}`);
      } else if (snap.type === "histogram") {
        let acc = 0;
        for (const b of snap.data.buckets) {
          const le = isFinite(b.le) ? b.le : "+Inf";
          acc = b.count; // already cumulative
          lines.push(`${baseName}_bucket${fmtLabels({ ...snap.labels, le: String(le) })} ${num(acc)}`);
        }
        lines.push(`${baseName}_sum${fmtLabels(snap.labels)} ${num(snap.data.sum)}`);
        lines.push(`${baseName}_count${fmtLabels(snap.labels)} ${num(snap.data.count)}`);
      }
    }
    return lines.join("\n") + "\n";
  }
}

/* ───────────────────────── Global instance ───────────────────────── */

export const metrics = new Registry();

/* ───────────────────────── Formatting helpers ───────────────────────── */

function sanitizeName(n: string): string {
  // Prometheus metric name: [a-zA-Z_:][a-zA-Z0-9_:]*
  const s = String(n || "");
  const a = s.replace(/[^a-zA-Z0-9_:]/g, "_");
  if (!/^[a-zA-Z_:]/.test(a)) return "_" + a;
  return a;
}

function fmtLabels(lbls: Labels): string {
  const keys = Object.keys(lbls || {});
  if (!keys.length) return "";
  const parts: string[] = [];
  for (const k of keys.sort()) {
    const v = (lbls as any)[k];
    parts.push(`${sanitizeLabelName(k)}="${escapeLabelValue(String(v))}"`);
  }
  return "{" + parts.join(",") + "}";
}

function sanitizeLabelName(n: string): string {
  const a = n.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!/^[a-zA-Z_]/.test(a)) return "_" + a;
  return a;
}

function escapeLabelValue(v: string): string {
  return v.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}

function escapeHelp(h?: string): string {
  return (h || "").replace(/\\/g, "\\\\").replace(/\n/g, "\\n");
}

function num(x: number): string {
  if (!isFinite(x)) return "0";
  // Avoid scientific for small integers
  if (Math.abs(x) < 1e15 && Math.floor(x) === x) return String(x);
  return String(x);
}

/* ───────────────────────── Tiny self-test (optional) ───────────────────────── */

export function __selftest__(): string {
  const r = new Registry();
  const c = r.counter("events_total", "Count of events", ["type"]);
  c.inc({ type: "ok" }, 2);
  c.inc({ type: "err" }, 1);
  const g = r.gauge("inflight_jobs", "Jobs in flight", ["queue"]);
  g.set({ queue: "q1" }, 3);
  g.dec({ queue: "q1" }, 1);
  const h = r.histogram("latency_ms", "Latency in ms");
  h.observe(42);
  h.observe(100);
  const done = r.timer("db_ms", { op: "read" });
  // simulate
  for (let i=0;i<1e6;i++) {} // burn a little CPU
  const d = done();
  if (d < 0) return "timer_fail";
  const prom = r.toPrometheus();
  if (!prom.includes("events_total")) return "prom_fail";
  return "ok";
}