// portfolio/positions.ts
// Import-free, Map-safe (no for..of on iterators) positions book.
//
// Compatible with older TS targets where Map.prototype.values() isn’t iterable.
// All Map iteration is done via Array.from(...) or map.forEach(...).

export type PosOptions = {
  base?: string;
  initialEquity?: number;
};

export type Trade = {
  symbol: string;
  qty: number;          // +buy, -sell
  price: number;
  ts?: number;
  tags?: string[];
};

export type Tick = {
  symbol: string;
  price: number;
  ts?: number;
};

export type PositionRow = {
  symbol: string;
  qty: number;
  avgCost: number;
  price: number;
  mktValue: number;
  unrealized: number;
  tags: string[];
  ts: number;
};

export type BookMetrics = {
  base: string;
  equity: number;
  net: number;
  gross: number;
  longMV: number;
  shortMV: number;
  longNames: number;
  shortNames: number;
  exposure: number;
  leverage: number;
  symbols: number;
  ts: number;
};

export type PositionsAPI = {
  trade(t: Trade): void;
  mark(t: Tick): void;
  setEquity(v: number): void;
  tag(symbol: string, add: string[]): void;
  clear(): void;
  remove(symbol: string): boolean;

  get(symbol: string): PositionRow | undefined;
  all(): PositionRow[];
  metrics(): BookMetrics;
  byTag(tag: string): PositionRow[];

  dump(): string;
  load(json: string): boolean;
};

export function createPositions(opts: PosOptions = {}): PositionsAPI {
  const base = String(opts.base || "USD");
  const book = new Map<string, PositionRow>();
  let equity = num(opts.initialEquity, 0);

  /* ----------------------------- Mutations ----------------------------- */

  function trade(t: Trade) {
    if (!t || !t.symbol || !isFinite(t.qty) || !isFinite(t.price)) return;
    const ts = t.ts ?? now();
    const sym = t.symbol;

    const cur =
      book.get(sym) ||
      { symbol: sym, qty: 0, avgCost: 0, price: t.price, mktValue: 0, unrealized: 0, tags: [], ts };

    if (Array.isArray(t.tags) && t.tags.length) {
      const S = new Set(cur.tags);
      for (let i = 0; i < t.tags.length; i++) S.add(String(t.tags[i]));
      cur.tags = Array.from(S);
    }

    const q0 = cur.qty;
    const q1 = q0 + t.qty;

    if (q0 === 0 || sameSign(q0, q1)) {
      if (q1 !== 0) {
        const notionalOld = Math.abs(q0) * cur.avgCost;
        const notionalNew = Math.abs(t.qty) * t.price;
        cur.avgCost = (notionalOld + notionalNew) / Math.max(1, Math.abs(q1));
      } else {
        cur.avgCost = 0;
      }
      cur.qty = round(q1);
    } else {
      cur.qty = round(q1);
      cur.avgCost = Math.abs(q1) > 0 ? t.price : 0;
    }

    cur.price = t.price;
    recompute(cur);
    cur.ts = ts;

    book.set(sym, cur);
    if (cur.qty === 0) book.delete(sym);
  }

  function mark(t: Tick) {
    if (!t || !t.symbol || !isFinite(t.price)) return;
    const cur = book.get(t.symbol);
    if (!cur) return;
    cur.price = t.price;
    cur.ts = t.ts ?? now();
    recompute(cur);
  }

  function setEquity(v: number) {
    equity = num(v, 0);
  }

  function tag(symbol: string, add: string[]) {
    const cur = book.get(symbol);
    if (!cur || !Array.isArray(add)) return;
    const S = new Set(cur.tags);
    for (let i = 0; i < add.length; i++) S.add(String(add[i]));
    cur.tags = Array.from(S);
    cur.ts = now();
  }

  function clear() { book.clear(); }
  function remove(symbol: string) { return book.delete(symbol); }

  /* ------------------------------- Queries ------------------------------ */

  function get(symbol: string): PositionRow | undefined {
    const cur = book.get(symbol);
    return cur ? cloneRow(cur) : undefined;
  }

  function all(): PositionRow[] {
    const out: PositionRow[] = [];
    // Avoid for..of on map.values()
    const vals = Array.from(book.values());
    for (let i = 0; i < vals.length; i++) out.push(cloneRow(vals[i]));
    out.sort((a, b) => (a.symbol < b.symbol ? -1 : a.symbol > b.symbol ? 1 : 0));
    return out;
  }

  function byTag(tag: string): PositionRow[] {
    const want = String(tag || "").toLowerCase();
    const res: PositionRow[] = [];
    const vals = Array.from(book.values());
    for (let i = 0; i < vals.length; i++) {
      const r = vals[i];
      let hit = false;
      for (let k = 0; k < r.tags.length; k++) {
        if (String(r.tags[k]).toLowerCase() === want) { hit = true; break; }
      }
      if (hit) res.push(cloneRow(r));
    }
    return res;
  }

  function metrics(): BookMetrics {
    let net = 0, longMV = 0, shortMV = 0;
    let longNames = 0, shortNames = 0;

    const vals = Array.from(book.values());
    for (let i = 0; i < vals.length; i++) {
      const r = vals[i];
      const mv = r.qty * r.price;
      net += mv;
      if (mv > 0) { longMV += mv; longNames++; }
      if (mv < 0) { shortMV += -mv; shortNames++; }
    }

    const gross = longMV + shortMV;
    const lev = equity > 0 ? gross / equity : 0;

    return {
      base,
      equity: round(equity),
      net: round(net),
      gross: round(gross),
      longMV: round(longMV),
      shortMV: round(shortMV),
      longNames,
      shortNames,
      exposure: round(gross),
      leverage: round(lev),
      symbols: book.size,
      ts: now(),
    };
  }

  /* ---------------------------- Persistence ----------------------------- */

  function dump(): string {
    try {
      return JSON.stringify({ base, equity, items: Array.from(book.values()) });
    } catch { return "{}"; }
  }

  function load(json: string): boolean {
    try {
      const o = JSON.parse(json || "{}");
      if (typeof o.equity === "number") equity = o.equity;
      book.clear();
      const items: any[] = Array.isArray(o.items) ? o.items : [];
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        const row: PositionRow = {
          symbol: String(it.symbol),
          qty: num(it.qty, 0),
          avgCost: num(it.avgCost, 0),
          price: num(it.price, 0),
          mktValue: 0,
          unrealized: 0,
          tags: Array.isArray(it.tags) ? it.tags.map(String) : [],
          ts: num(it.ts, now()),
        };
        recompute(row);
        if (row.symbol) book.set(row.symbol, row);
      }
      return true;
    } catch { return false; }
  }

  /* -------------------------------- Helpers ----------------------------- */

  function recompute(r: PositionRow) {
    r.mktValue = round(r.qty * r.price);
    r.unrealized = r.qty >= 0
      ? round((r.price - r.avgCost) * r.qty)
      : round((r.avgCost - r.price) * Math.abs(r.qty));
  }

  function cloneRow(r: PositionRow): PositionRow {
    return {
      symbol: r.symbol,
      qty: r.qty,
      avgCost: r.avgCost,
      price: r.price,
      mktValue: r.mktValue,
      unrealized: r.unrealized,
      tags: r.tags.slice(),
      ts: r.ts,
    };
  }

  function sameSign(a: number, b: number) {
    return (a === 0 || b === 0) ? a === b : (a > 0) === (b > 0);
  }

  function now() { return Date.now(); }
  function num(v: any, d: number) { const n = Number(v); return Number.isFinite(n) ? n : d; }
  function round(n: number) { return Math.round(n * 1e6) / 1e6; }

  /* --------------------------------- API -------------------------------- */

  return {
    trade,
    mark,
    setEquity,
    tag,
    clear,
    remove,
    get,
    all,
    metrics,
    byTag,
    dump,
    load,
  };
}