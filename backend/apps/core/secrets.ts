// core/precompute.ts
// Pure TS (no imports). Precompute engine for tiles/screeners/indices.
// - Task registry (id, input, compute -> artifact, output key template)
// - Incremental fetch via cursor (opaque string) with CAS-safe lock
// - Writes JSON artifacts + manifest to a pluggable TextStore
// - Invalidation by tags/datasets; TTL pruning
// - Handy helpers: returns, rolling stats, sparkline

/* ───────────── Types ───────────── */

export type Dict<T = any> = { [k: string]: T };

export type TextStore = {
  read(key: string): Promise<string | null>;
  write(key: string, val: string): Promise<void>;
  del(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
};

export type CursorIO = {
  read(dataset: string, provider: string): Promise<string>;
  write(dataset: string, provider: string, value: string, meta?: Dict): Promise<void>;
};

export type PrecomputeTask<I = any, R = any> = {
  id: string;                               // unique
  dataset: string;                          // logical dataset (e.g., "px-daily")
  provider: string;                         // "bloomberg" | "koyfin" | "hammer"
  tags?: string[];                          // e.g., ["today", "tile:market"]
  ttlMs?: number;                           // artifact freshness hint (for pruning)
  input: I;                                 // task-specific params (symbols, window…)
  // Fetch incrementally from source since cursor; opaque cursor value in/out:
  fetch: (input: I, cursor: string | undefined) => Promise<{ rows: any[]; nextCursor?: string }>;
  // Compute artifact from fetched rows + task input:
  compute: (rows: any[], input: I) => Promise<R> | R;
  // Where to write the artifact. You can template: ${id}, ${date}, ${hash}
  outputKey?: string;                       // default: precompute/${id}/${date}.json
};

export type RunOpts = {
  store?: TextStore;                        // where artifacts/manifests go
  cursors?: CursorIO;                       // checkpoint backend
  now?: () => Date;                         // for testing
  lockTtlMs?: number;                       // default 60_000
  meta?: Dict;                              // extra manifest fields
};

export type Manifest = {
  id: string;
  task: string;
  dataset: string;
  provider: string;
  tags?: string[];
  input: any;
  outputKey: string;
  rows: number;
  cursor?: string;
  started: string;
  finished: string;
  duration_ms: number;
  ok: boolean;
  error?: string;
  ttlMs?: number;
  meta?: Dict;
};

/* ───────────── Engine ───────────── */

export class PrecomputeEngine {
  private store: TextStore;
  private cursors?: CursorIO;
  private now: () => Date;
  private lockTtl: number;

  constructor(opts?: RunOpts) {
    this.store = opts?.store || MemoryTextStore();
    this.cursors = opts?.cursors;
    this.now = opts?.now || (() => new Date());
    this.lockTtl = opts?.lockTtlMs ?? 60_000;
  }

  setStore(s: TextStore) { this.store = s; }
  setCursors(c: CursorIO) { this.cursors = c; }

  /** Run one task end-to-end (lock, fetch, compute, write, checkpoint, manifest). */
  async run<I, R>(task: PrecomputeTask<I, R>, opts?: { meta?: Dict }): Promise<{ artifact: R; manifest: Manifest }> {
    const started = this.now();
    const lockKey = this.lockKey(task);
    await this.acquireLock(lockKey, this.lockTtl, `task:${task.id}`);

    let ok = false, errMsg = "", artifact!: R, outKey = "", nextCur: string | undefined;
    try {
      const cur = this.cursors ? (await this.cursors.read(task.dataset, task.provider)) || "" : "";
      const got = await task.fetch(task.input, cur || undefined);
      const rows = got?.rows || [];
      nextCur = got?.nextCursor;

      artifact = await task.compute(rows, task.input);
      outKey = this.resolveOutputKey(task, started, artifact);

      // write artifact + light index
      await this.store.write(outKey, JSON.stringify({ id: task.id, asOf: this.now().toISOString(), artifact }, null, 0) + "\n");
      await this.appendIndex(task, outKey);
      if (this.cursors && nextCur != null) {
        await this.cursors.write(task.dataset, task.provider, String(nextCur), { task: task.id });
      }
      ok = true;
    } catch (e: any) {
      errMsg = String(e?.message || e) || "error";
      ok = false;
    } finally {
      await this.releaseLock(lockKey);
    }

    const finished = this.now();
    const manifest: Manifest = {
      id: this.runId(task, started),
      task: task.id,
      dataset: task.dataset,
      provider: task.provider,
      tags: task.tags && task.tags.slice(),
      input: task.input,
      outputKey: outKey,
      rows: undefined as any, // filled after we read artifact size (optional)
      cursor: nextCur,
      started: started.toISOString(),
      finished: finished.toISOString(),
      duration_ms: Math.max(0, finished.getTime() - started.getTime()),
      ok,
      error: ok ? undefined : errMsg,
      ttlMs: task.ttlMs,
      meta: opts?.meta
    };

    // best-effort rows count if artifact has .rows or .length
    try {
      const parsed = typeof artifact === "object" && artifact
        ? (artifact as any)
        : { };
      const n = typeof parsed.rows?.length === "number" ? parsed.rows.length
        : (Array.isArray(parsed) ? parsed.length : 0);
      (manifest as any).rows = n;
    } catch { (manifest as any).rows = 0; }

    await this.writeManifest(task, manifest);

    if (!ok) throw new Error(`precompute.failed:${task.id}:${errMsg}`);
    return { artifact, manifest };
  }

  /** Remove artifacts by tag or dataset. */
  async invalidate(filter: { tag?: string; dataset?: string; taskIdPrefix?: string }): Promise<number> {
    const ix = await this.readIndex();
    const keep: string[] = [];
    const rm: string[] = [];
    for (const line of ix) {
      const rec = safeJSON<any>(line);
      if (!rec?.task || !rec?.key) continue;
      const matchTag = filter.tag ? (rec.tags || []).includes(filter.tag) : true;
      const matchDs  = filter.dataset ? rec.dataset === filter.dataset : true;
      const matchId  = filter.taskIdPrefix ? String(rec.task).startsWith(filter.taskIdPrefix) : true;
      if (matchTag && matchDs && matchId) rm.push(rec.key); else keep.push(line);
    }
    let n = 0;
    for (const k of rm) { await this.store.del(k); n++; }
    await this.store.write(this.indexKey(), keep.map(s => s + "\n").join(""));
    return n;
  }

  /** Prune artifacts older than TTLs in manifests (best effort). */
  async prune(): Promise<number> {
    const mkeys = await this.store.list("precompute/_manifests/");
    let removed = 0;
    for (const k of mkeys) {
      const txt = await this.store.read(k);
      if (!txt) continue;
      const man = safeJSON<Manifest>(txt);
      if (!man?.ttlMs || !man.outputKey) continue;
      const exp = Date.parse(man.finished) + man.ttlMs;
      if (Date.now() > exp) {
        try { await this.store.del(man.outputKey); removed++; } catch {}
      }
    }
    return removed;
  }

  /* ───────────── Internals ───────────── */

  private runId(task: PrecomputeTask<any, any>, t: Date): string {
    return `${task.id}:${t.getTime().toString(36)}`;
  }

  private resolveOutputKey(task: PrecomputeTask<any, any>, t: Date, artifact: any): string {
    const date = isoDate(t);
    const id = task.id;
    const hash = cheapHash(JSON.stringify({ id, date, artifactPreview: preview(artifact) }));
    const tpl = task.outputKey || "precompute/${id}/${date}/${hash}.json";
    return tpl
      .replace(/\$\{id\}/g, id)
      .replace(/\$\{date\}/g, date)
      .replace(/\$\{hash\}/g, hash);
  }

  private async appendIndex(task: PrecomputeTask<any, any>, key: string) {
    const rec = { task: task.id, dataset: task.dataset, provider: task.provider, tags: task.tags || [], key, ts: this.now().toISOString() };
    const line = JSON.stringify(rec);
    const idxKey = this.indexKey();
    const prev = (await this.store.read(idxKey)) || "";
    await this.store.write(idxKey, prev + line + "\n");
  }

  private indexKey() { return "precompute/_index.jsonl"; }

  private async readIndex(): Promise<string[]> {
    const txt = await this.store.read(this.indexKey());
    return txt ? txt.split(/\r?\n/).filter(Boolean) : [];
  }

  private manifestKey(task: PrecomputeTask<any, any>, manifestId: string) {
    return `precompute/_manifests/${sanitize(task.id)}/${sanitize(manifestId)}.json`;
  }

  private async writeManifest(task: PrecomputeTask<any, any>, m: Manifest) {
    await this.store.write(this.manifestKey(task, m.id), JSON.stringify(m) + "\n");
  }

  private lockKey(task: PrecomputeTask<any, any>) {
    return `precompute/_locks/${sanitize(task.provider)}/${sanitize(task.dataset)}/${sanitize(task.id)}.json`;
  }

  private async acquireLock(key: string, ttlMs: number, owner?: string): Promise<void> {
    const now = this.now();
    const raw = await this.store.read(key);
    if (raw) {
      const li = safeJSON<LockInfo>(raw);
      if (li?.token && !isExpired(li, now.getTime())) throw new Error("locked");
    }
    const lock: LockInfo = { token: genToken(), owner, since: now.toISOString(), ttlMs };
    await this.store.write(key, JSON.stringify(lock));
  }

  private async releaseLock(key: string): Promise<void> {
    // best effort: overwrite with expired lock
    await this.store.write(key, JSON.stringify({ token: "", since: new Date(0).toISOString(), ttlMs: 1 }));
  }
}

/* ───────────── Memory TextStore (default) ───────────── */

export function MemoryTextStore(): TextStore {
  const m = new Map<string, string>();
  return {
    async read(k) { return m.has(k) ? (m.get(k) as string) : null; },
    async write(k, v) { m.set(k, String(v)); },
    async del(k) { m.delete(k); },
    async list(prefix) {
      const out: string[] = [];
      for (const k of m.keys()) if (k.startsWith(prefix)) out.push(k);
      return out;
    }
  };
}

/* ───────────── Helpers & small analytics ───────────── */

type LockInfo = { token: string; owner?: string; since: string; ttlMs: number };

function isExpired(li: LockInfo, nowMs: number): boolean {
  const since = Date.parse(li.since || "");
  if (!Number.isFinite(since)) return true;
  return nowMs > since + (li.ttlMs || 0);
}

function sanitize(s: string) {
  return String(s || "").replace(/[^A-Za-z0-9._\-]/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "na";
}

function isoDate(d: Date): string {
  const z = (n: number) => (n < 10 ? "0" + n : "" + n);
  return `${d.getUTCFullYear()}-${z(d.getUTCMonth() + 1)}-${z(d.getUTCDate())}`;
}

function preview(x: any): any {
  if (x == null) return x;
  if (Array.isArray(x)) return x.length <= 5 ? x : x.slice(0, 5);
  if (typeof x === "object") {
    const o: any = {};
    let i = 0; for (const k in x) { o[k] = (x as any)[k]; if (++i >= 5) break; }
    return o;
  }
  return x;
}

function cheapHash(s: string): string {
  let x = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619) >>> 0; }
  return ("00000000" + x.toString(16)).slice(-8);
}

function safeJSON<T=any>(s: string): T { try { return JSON.parse(s) as T; } catch { return {} as any as T; } }
function genToken(): string { return "t_" + cheapHash(Date.now().toString(36) + Math.random().toString(36)); }

/* ───────────── Mini analytics helpers (for tiles) ───────────── */

export function dailyReturns(rows: { t?: string; c?: number }[]): { t: string; r: number }[] {
  const out: { t: string; r: number }[] = [];
  for (let i = 1; i < rows.length; i++) {
    const p = Number(rows[i-1].c), q = Number(rows[i].c);
    if (isFinite(p) && isFinite(q) && p !== 0) out.push({ t: toISO(rows[i].t), r: q/p - 1 });
  }
  return out;
}
export function rollingMean(xs: number[], w = 5): number[] {
  const out: number[] = new Array(xs.length).fill(NaN);
  let sum = 0;
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i];
    if (i >= w) sum -= xs[i - w];
    if (i >= w - 1) out[i] = sum / w;
  }
  return out;
}
export function sparkline(rows: { c: number }[], n = 20): number[] {
  if (!rows?.length) return [];
  const step = Math.max(1, Math.floor(rows.length / n));
  const out: number[] = [];
  for (let i = Math.max(0, rows.length - n*step); i < rows.length; i += step) out.push(rows[i].c);
  return out;
}
function toISO(x: any): string { try { return new Date(x).toISOString(); } catch { return String(x); } }

/* ───────────── Tiny self-test ───────────── */

export async function __selftest__(): Promise<string> {
  const eng = new PrecomputeEngine();
  const t: PrecomputeTask<{ sym: string }, { sym: string; n: number }> = {
    id: "tile:demo",
    dataset: "px-daily",
    provider: "sim",
    input: { sym: "SPY" },
    fetch: async (_inp, cur) => {
      const from = cur ? Number(cur) : 0;
      const rows = Array.from({ length: 5 }, (_, i) => ({ t: i + from, c: 100 + i + from }));
      return { rows, nextCursor: String(from + 5) };
    },
    compute: (rows, input) => ({ sym: input.sym, n: rows.length }),
    outputKey: "precompute/${id}/${date}/artifact-${hash}.json",
  };
  const r1 = await eng.run(t);
  const r2 = await eng.run(t);
  return r1.manifest.ok && r2.manifest.ok ? "ok" : "fail";
}