// core/master.ts
// Symbol Master / Crosswalk (pure TypeScript, ZERO imports).
// - Upsert instruments from any provider
// - Cross-link identifiers: ticker@mic, ISIN, FIGI, BBGID, CUSIP, SEDOL, UID
// - Resolve queries (exact IDs prioritized; ticker-only falls back to primary)
// - Search by ticker/name
// - Serialize/restore (JSONL) with pluggable text I/O
//
// No external deps. Safe for “no imports”.

/* ───────────── Types ───────────── */

export type Dict<T = any> = { [k: string]: T };

export type AssetClass =
  | "Equity" | "Bond" | "FX" | "Commodity" | "Index" | "Fund" | "Derivative" | "Other";

export type TickerAlias = {
  value: string;        // raw ticker
  mic?: string;         // primary MIC of the listing (e.g., XNAS)
  composite?: string;   // value@MIC
  active?: boolean;
};

export type ProviderIds = {
  provider: string;                 // e.g., "bloomberg" | "koyfin" | "hammer"
  ids: Partial<IdBundle>;           // provider-native IDs (bbgid/figi/...)
  asOf?: string;
};

export type Instrument = {
  uid: string;                      // internal master id
  assetClass?: AssetClass;
  name?: string;
  country?: string;
  currency?: string;
  primaryTicker?: string;           // ticker@MIC
  tickers?: TickerAlias[];
  isin?: string;
  figi?: string;
  bbgid?: string;
  cusip?: string;
  sedol?: string;
  meta?: Dict;                      // arbitrary attributes (sector/industry/etc.)
  providers?: ProviderIds[];
  active?: boolean;
  listed?: boolean;
  asOf?: string;
};

export type IdBundle = {
  uid?: string;
  ticker?: string;
  mic?: string;
  composite?: string; // ticker@mic (if provided, overrides ticker/mic)
  isin?: string;
  figi?: string;
  bbgid?: string;
  cusip?: string;
  sedol?: string;
};

export type ResolveResult =
  | { ok: true; uid: string; instrument: Instrument; reason: string; confidence: number }
  | { ok: false; reason: string };

/* ───────────── Storage (optional) ───────────── */

export type TextIO = {
  read(key: string): Promise<string | null>;
  write(key: string, val: string): Promise<void>;
};

/* ───────────── Master Index ───────────── */

export class Master {
  // Core store
  private rows = new Map<string, Instrument>();

  // Indexes
  private byUid = new Map<string, string>();           // uid -> uid (identity)
  private byTickerMic = new Map<string, string>();     // TICKER@MIC -> uid
  private byTicker = new Map<string, Set<string>>();   // TICKER -> set(uid)
  private byISIN = new Map<string, string>();
  private byFIGI = new Map<string, string>();
  private byBBGID = new Map<string, string>();
  private byCUSIP = new Map<string, string>();
  private bySEDOL = new Map<string, string>();

  constructor(initial?: Instrument[]) {
    if (initial?.length) for (const r of initial) this.add(r);
  }

  /* ───────────── Upsert / Link ───────────── */

  /** Add or merge an instrument. If uid absent, one is generated. Returns uid. */
  add(rec: Partial<Instrument>): string {
    const uid = rec.uid && rec.uid.trim() ? rec.uid : genUid(rec);
    const now = new Date().toISOString();
    const fresh: Instrument = {
      uid,
      assetClass: rec.assetClass || "Equity",
      name: rec.name || "",
      country: rec.country || rec.meta?.country || "",
      currency: rec.currency || rec.meta?.currency || "",
      primaryTicker: normalizeComposite(rec.primaryTicker || composite(rec.tickers?.[0]?.value, rec.tickers?.[0]?.mic)),
     
      isin: normId(rec.isin),
      figi: normId(rec.figi),
      bbgid: normId(rec.bbgid),
      cusip: normId(rec.cusip),
      sedol: normId(rec.sedol),
      meta: rec.meta ? shallowCopy(rec.meta) : undefined,
      providers: rec.providers ? rec.providers.map(p => ({ provider: String(p.provider||""), ids: shallowCopy(p.ids||{}), asOf: p.asOf })) : undefined,
      active: rec.active !== false,
      listed: rec.listed !== false,
      asOf: rec.asOf || now
    };

    const existing = this.rows.get(uid);
    const merged = existing ? mergeInstruments(existing, fresh) : fresh;
    this.rows.set(uid, merged);
    this.byUid.set(uid, uid);

    // index identifiers
    this.indexIds(merged);
    return uid;
  }

  /** Link extra identifiers (aliases) to an existing uid. */
  link(uid: string, ids: Partial<IdBundle>): void {
    const row = this.rows.get(uid);
    if (!row) throw new Error("master.link: unknown uid");
    // merge fields if new
    const patch: Partial<Instrument> = {};
    if (ids.isin && !row.isin) patch.isin = normId(ids.isin);
    if (ids.figi && !row.figi) patch.figi = normId(ids.figi);
    if (ids.bbgid && !row.bbgid) patch.bbgid = normId(ids.bbgid);
    if (ids.cusip && !row.cusip) patch.cusip = normId(ids.cusip);
    if (ids.sedol && !row.sedol) patch.sedol = normId(ids.sedol);

    const comp = normalizeComposite(ids.composite || composite(ids.ticker, ids.mic));
    if (comp) {
      // add or activate ticker alias
      const [tkr, mic] = splitComposite(comp);
      const aliases = row.tickers || [];
      if (!aliases.some(a => sameComp(a, comp))) aliases.push({ value: tkr, mic, composite: comp, active: true });
      patch.tickers = aliases;
      if (!row.primaryTicker) patch.primaryTicker = comp;
    }
    if (Object.keys(patch).length) {
      this.rows.set(uid, mergeInstruments(row, patch as Instrument));
    }
    // reindex
    this.indexIds(this.rows.get(uid)!);
  }

  /** Upsert from a provider row using a tiny mapping table. */
  upsertFromProvider(
    provider: string,
    row: Dict,
    mapping: Partial<Record<keyof IdBundle | "name" | "currency" | "mic" | "assetClass", string>>
  ): string {
    const idb: IdBundle = {
      uid: pick(row, mapping["uid"]),
      composite: normalizeComposite(pick(row, mapping["composite"]) || composite(pick(row, mapping["ticker"]), pick(row, mapping["mic"]))),
      isin: normId(pick(row, mapping["isin"])),
      figi: normId(pick(row, mapping["figi"])),
      bbgid: normId(pick(row, mapping["bbgid"])),
      cusip: normId(pick(row, mapping["cusip"])),
      sedol: normId(pick(row, mapping["sedol"])),
    };

    // try resolve first
    const hit = this.resolve(idb);
    const uid = hit.ok ? hit.uid : (idb.uid || genUid({ name: pick(row, mapping["name"]) || pick(row, mapping["ticker"]) }));

    const tick: TickerAlias | undefined = idb.composite
      ? toAlias(idb.composite)
      : undefined;

    const rec: Partial<Instrument> = {
      uid,
      name: pick(row, mapping["name"]) || "",
      currency: pick(row, mapping["currency"]) || "",
      assetClass: (pick(row, mapping["assetClass"]) as AssetClass) || "Equity",
      primaryTicker: tick ? tick.composite : undefined,
      tickers: tick ? [tick] : [],
      isin: idb.isin,
      figi: idb.figi,
      bbgid: idb.bbgid,
      cusip: idb.cusip,
      sedol: idb.sedol,
      providers: [{ provider, ids: compactIds(idb), asOf: new Date().toISOString() }]
    };

    return this.add(rec);
  }

  /* ───────────── Resolve & Search ───────────── */

  /**
   * Resolve an IdBundle or string:
   *  - Exact priority: uid > figi > bbgid > isin > cusip > sedol > composite(ticker@mic)
   *  - Ticker-only falls back to primary listing if unique; ambiguous otherwise.
   */
  resolve(q: IdBundle | string): ResolveResult {
    if (typeof q === "string") q = parseQuery(q);

    // 1) Unique IDs
   
    // 2) composite ticker
    const comp = normalizeComposite(q.composite || composite(q.ticker, q.mic));
    if (comp) {
      const u = this.byTickerMic.get(comp);
      if (u) return ok(u, "ticker@mic");
    }

    // 3) ticker only → pick primary if unique
    if (q.ticker) {
      const tk = normTicker(q.ticker);
      const set = this.byTicker.get(tk);
      if (set && set.size === 1) return ok(firstOf(set), "ticker_primary");
      if (set && set.size > 1) {
        // try pick by currency if provided
        if (q.mic) {
          const u = this.byTickerMic.get(tk + "@" + normMIC(q.mic));
          if (u) return ok(u, "ticker@mic");
        }
        // else ambiguous
        return { ok: false, reason: "ambiguous_ticker" };
      }
    }

    return { ok: false, reason: "not_found" };

    function ok(uid: string, reason: string): ResolveResult {
      const instrument = (this as Master).rows.get(uid)!;
      return { ok: true, uid, instrument, reason, confidence: score(reason) };
    }
    function score(reason: string): number {
      switch (reason) {
        case "uid": return 1.0;
        case "figi":
        case "bbgid":
        case "isin":
        case "cusip":
        case "sedol": return 0.99;
        case "ticker@mic": return 0.95;
        case "ticker_primary": return 0.80;
        default: return 0.5;
      }
    }
  }

  /** Simple search across ticker composites and names. */
  search(q: string, limit = 20): Instrument[] {
    const s = String(q || "").trim().toUpperCase();
    if (!s) return [];
    const out: Instrument[] = [];
    for (const ins of this.rows.values()) {
      if (out.length >= limit) break;
      const name = (ins.name || "").toUpperCase();
      const hitName = name.includes(s);
      const tickers = (ins.tickers || []).map(t => t.composite || (t.value + (t.mic ? "@" + t.mic : ""))).join(" ");
      const hitTick = tickers.toUpperCase().includes(s);
      if (hitName || hitTick) out.push(ins);
    }
    return out;
  }

  /* ───────────── Export / Import ───────────── */

  /** Dump JSONL lines; one instrument per line. */
  serializeJSONL(): string {
    let buf = "";
    for (const v of this.rows.values()) buf += JSON.stringify(v) + "\n";
    return buf;
  }

  /** Load JSONL lines (rebuilds indexes). */
  loadJSONL(text: string): void {
    this.rows.clear();
    this.clearIndexes();
    const lines = String(text || "").split(/\r?\n/).filter(Boolean);
    for (const ln of lines) {
      try {
        const obj = JSON.parse(ln) as Instrument;
        if (obj && obj.uid) this.rows.set(obj.uid, obj);
      } catch { /* ignore */ }
    }
    for (const ins of this.rows.values()) this.indexIds(ins);
  }

  /** Save to TextIO. */
  async save(io: TextIO, key = "master/instruments.jsonl"): Promise<void> {
    await io.write(key, this.serializeJSONL());
  }

  /** Restore from TextIO (if missing, no-op). */
  async restore(io: TextIO, key = "master/instruments.jsonl"): Promise<void> {
    const txt = await io.read(key);
    if (txt != null) this.loadJSONL(txt);
  }

  /* ───────────── Internals ───────────── */

  private clearIndexes() {
    this.byUid.clear();
    this.byTickerMic.clear();
    this.byTicker.clear();
    this.byISIN.clear();
    this.byFIGI.clear();
    this.byBBGID.clear();
    this.byCUSIP.clear();
    this.bySEDOL.clear();
  }

  private indexIds(ins: Instrument) {
    const uid = ins.uid;
    this.byUid.set(uid, uid);

    // primary + aliases
    const aliases = (ins.tickers || []).slice();
    if (ins.primaryTicker) {
      const prim = toAlias(ins.primaryTicker);
      if (!aliases.some(a => sameComp(a, prim.composite!))) aliases.unshift(prim);
    }
    for (const a of aliases) {
      if (!a?.value) continue;
      const comp = normalizeComposite(a.composite || composite(a.value, a.mic));
      if (!comp) continue;

      this.byTickerMic.set(comp, uid);
      const tk = normTicker(a.value);
      if (!this.byTicker.has(tk)) this.byTicker.set(tk, new Set());
      this.byTicker.get(tk)!.add(uid);
    }

    // unique IDs
    
  }
}

/* ───────────── Utilities ───────────── */

function shallowCopy<T extends Dict>(o: T): T {
  const out: any = Object.create(null);
  for (const k in (o||{})) out[k] = o[k];
  return out as T;
}

function mergeInstruments(a: Instrument, b: Instrument): Instrument {
  const out: Instrument = { ...a };

  out.assetClass = pickPref(a.assetClass, b.assetClass);
  out.name       = pickPref(a.name, b.name);
  out.country    = pickPref(a.country, b.country);
  out.currency   = pickPref(a.currency, b.currency);
  out.primaryTicker = pickPref(a.primaryTicker, normalizeComposite(b.primaryTicker));

  // merge tickers (dedupe by composite)
  const map: Record<string, TickerAlias> = Object.create(null);
  for (const t of a.tickers || []) map[normalizeComposite(t.composite || composite(t.value, t.mic))] = { ...t, composite: normalizeComposite(t.composite || composite(t.value, t.mic)) };
  for (const t of b.tickers || []) {
    const comp = normalizeComposite(t.composite || composite(t.value, t.mic));
    if (!comp) continue;
    if (!map[comp]) map[comp] = { ...t, composite: comp };
    else map[comp] = { ...map[comp], ...t, composite: comp, active: (map[comp].active ?? false) || (t.active ?? false) };
  }
  out.tickers = Object.values(map);

  // unique IDs prefer existing unless new provides a value
  out.isin  = a.isin  || b.isin  || undefined;
  out.figi  = a.figi  || b.figi  || undefined;
  out.bbgid = a.bbgid || b.bbgid || undefined;
  out.cusip = a.cusip || b.cusip || undefined;
  out.sedol = a.sedol || b.sedol || undefined;

  // meta/providers
  out.meta = { ...(a.meta||{}), ...(b.meta||{}) };
  const provs = [...(a.providers||[]), ...(b.providers||[])];
  out.providers = mergeProviders(provs);

  out.active = b.active ?? a.active ?? true;
  out.listed = b.listed ?? a.listed ?? true;
  out.asOf   = newer(a.asOf, b.asOf);

  return out;
}

function mergeProviders(xs: ProviderIds[]): ProviderIds[] {
  const key = (p: ProviderIds) => (p.provider || "").toLowerCase();
  const by: Record<string, ProviderIds> = Object.create(null);
  for (const p of xs) {
    const k = key(p);
    if (!k) continue;
    const prev = by[k];
    if (!prev) by[k] = { provider: p.provider, ids: shallowCopy(p.ids||{}), asOf: p.asOf };
    else {
      by[k] = {
        provider: prev.provider,
        ids: { ...(prev.ids||{}), ...(p.ids||{}) },
        asOf: newer(prev.asOf, p.asOf)
      };
    }
  }
  return Object.values(by);
}

function newer(a?: string, b?: string): string | undefined {
  if (!a) return b;
  if (!b) return a;
  return Date.parse(a) >= Date.parse(b) ? a : b;
}

function pickPref<A>(oldV?: A, newV?: A): A | undefined {
  return newV != null && newV !== "" ? newV : oldV;
}

function normId(x?: string): string | undefined {
  if (!x) return undefined;
  const s = String(x).replace(/\s+/g, "").toUpperCase();
  return s || undefined;
}

function normTicker(x?: string): string {
  return String(x || "").trim().toUpperCase();
}
function normMIC(x?: string): string {
  return String(x || "").trim().toUpperCase();
}
function normalizeComposite(x?: string): string {
  if (!x) return "";
  const [t, m] = splitComposite(x);
  if (!t) return "";
  return m ? `${normTicker(t)}@${normMIC(m)}` : normTicker(t);
}
function composite(ticker?: string, mic?: string): string {
  const t = normTicker(ticker);
  const m = normMIC(mic);
  return t ? (m ? `${t}@${m}` : t) : "";
}
function splitComposite(x: string): [string, string | undefined] {
  const s = String(x || "");
  const i = s.indexOf("@");
  if (i < 0) return [normTicker(s), undefined];
  return [normTicker(s.slice(0, i)), normMIC(s.slice(i + 1))];
}
function sameComp(a: TickerAlias, comp: string): boolean {
  const ac = normalizeComposite(a.composite || composite(a.value, a.mic));
  return ac === normalizeComposite(comp);
}
function toAlias(comp: string): TickerAlias {
  const [t, m] = splitComposite(comp);
  return { value: t, mic: m, composite: composite(t, m), active: true };
}

function pick(obj: Dict, key?: string): any {
  if (!key) return undefined;
  return obj[key];
}

function compactIds(ids: IdBundle): IdBundle {
  const out: IdBundle = {};
  if (ids.uid) out.uid = normId(ids.uid);
  if (ids.composite) out.composite = normalizeComposite(ids.composite);
  if (ids.isin) out.isin = normId(ids.isin);
  if (ids.figi) out.figi = normId(ids.figi);
  if (ids.bbgid) out.bbgid = normId(ids.bbgid);
  if (ids.cusip) out.cusip = normId(ids.cusip);
  if (ids.sedol) out.sedol = normId(ids.sedol);
  return out;
}

function genUid(seed?: Partial<Instrument>): string {
  const base = (seed?.name || seed?.primaryTicker || seed?.tickers?.[0]?.value || "ins") + ":" + Date.now().toString(36) + ":" + Math.random().toString(36).slice(2);
  return "ins_" + hash(base);
}

function firstOf<T>(set: Set<T>): T {
  // @ts-ignore
  for (const v of set) return v;
  // @ts-ignore
  return undefined as T;
}

/* ───────────── Query parsing ───────────── */

function parseQuery(s: string): IdBundle {
  const q = String(s || "").trim();
  if (!q) return {};
  // Recognize basic forms:
  //  - TICKER@MIC
  //  - TICKER MIC (space)
  //  - ISIN (2 letters + 9 alnum + 1 check)
  //  - FIGI (starts with BBG)
  //  - CUSIP (alnum len 9)
  //  - SEDOL (alnum len 6/7)
  const ISIN = /^[A-Z]{2}[A-Z0-9]{9}[0-9]$/i;
  const FIGI = /^BBG[BCDFGHJKMNPRSTVWXYZ0-9]{9,}$/i;
  const CUSIP = /^[A-Z0-9]{9}$/i;
  const SEDOL = /^[A-Z0-9]{6,7}$/i;

  if (q.includes("@")) {
    const [t, m] = splitComposite(q);
    return { composite: composite(t, m) };
  }
  const parts = q.split(/\s+/);
  if (parts.length === 2 && parts[1].length >= 3) {
    const t = normTicker(parts[0]);
    const m = normMIC(parts[1]);
    return { composite: composite(t, m) };
  }
  if (ISIN.test(q)) return { isin: normId(q) };
  if (FIGI.test(q)) return { figi: normId(q) };
  if (CUSIP.test(q)) return { cusip: normId(q) };
  if (SEDOL.test(q)) return { sedol: normId(q) };

  return { ticker: normTicker(q) };
}

/* ───────────── Hash (deterministic, non-crypto) ───────────── */

function hash(s: string): string {
  let x = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    x ^= s.charCodeAt(i);
    x = Math.imul(x, 16777619) >>> 0;
  }
  return ("00000000" + x.toString(16)).slice(-8) + ("00000000" + ((x ^ 0xa5a5a5a5) >>> 0).toString(16)).slice(-8);
}

/* ───────────── Helpers for tests ───────────── */

export function __selftest__(): string {
  const m = new Master();
  const a = m.add({
    name: "Apple Inc",
    primaryTicker: "AAPL@XNAS",
    isin: "US0378331005",
    figi: "BBG000B9XRY4"
  });
  m.link(a, { composite: "AAPL@XNGS" });

  const r1 = m.resolve("AAPL@XNAS");
  const r2 = m.resolve("US0378331005");
  const r3 = m.resolve("BBG000B9XRY4");
  const r4 = m.resolve("AAPL");

  if (!r1.ok || r1.uid !== a) return "fail_comp";
  if (!r2.ok || r2.uid !== a) return "fail_isin";
  if (!r3.ok || r3.uid !== a) return "fail_figi";
  if (!r4.ok) return "fail_ticker";
  return "ok";
}