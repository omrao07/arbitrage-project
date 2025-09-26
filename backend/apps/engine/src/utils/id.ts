// utils/id.ts
// Works on TS targets < ES2020 (NO BigInt). Pure, dependency-free IDs + base62.

type Bytes = Uint8Array;

/* ------------------------------- RNG core ------------------------------- */
const RNG = {
  bytes(n: number): Bytes {
    const out = new Uint8Array(n);
    const g: any = (globalThis as any);
    const c = g?.crypto || g?.msCrypto;
    if (c && typeof c.getRandomValues === "function") {
      c.getRandomValues(out);
      return out;
    }
    for (let i = 0; i < n; i++) out[i] = Math.floor(Math.random() * 256);
    return out;
  },
  rnd53(): number {
    const b = RNG.bytes(7);
    const x =
      ((b[0] & 31) * 2 ** 48) +
      (b[1] * 2 ** 40) +
      (b[2] * 2 ** 32) +
      (b[3] * 2 ** 24) +
      (b[4] * 2 ** 16) +
      (b[5] * 2 ** 8) +
      b[6];
    return x / 0x20000000000000; // 2^53
  },
};

/* -------------------------------- UUID v4 ------------------------------- */
export function uuidv4(): string {
  const b = RNG.bytes(16);
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // variant 10
  const h = (n: number) => n.toString(16).padStart(2, "0");
  return (
    h(b[0]) + h(b[1]) + h(b[2]) + h(b[3]) + "-" +
    h(b[4]) + h(b[5]) + "-" +
    h(b[6]) + h(b[7]) + "-" +
    h(b[8]) + h(b[9]) + "-" +
    h(b[10]) + h(b[11]) + h(b[12]) + h(b[13]) + h(b[14]) + h(b[15])
  );
}

/* --------------------------------- ULID --------------------------------- */
const B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const B32_IDX: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  for (let i = 0; i < B32.length; i++) m[B32[i]] = i;
  return m;
})();

function ulidTime(ms: number): string {
  let t = Math.max(0, Math.floor(ms));
  const out = new Array(10);
  for (let i = 9; i >= 0; i--) { out[i] = B32[t % 32]; t = Math.floor(t / 32); }
  return out.join("");
}
function ulidRand(bytes: Bytes): string {
  // 80 bits -> 16 chars (Crockford base32)
  let s = "", acc = 0, bits = 0;
  for (let i = 0; i < bytes.length; i++) {
    acc = (acc << 8) | bytes[i]; bits += 8;
    while (bits >= 5) { bits -= 5; s += B32[(acc >>> bits) & 31]; }
  }
  if (bits > 0) s += B32[(acc << (5 - bits)) & 31];
  return s.slice(0, 16);
}

let __ulidTime = 0;
let __ulidLast = RNG.bytes(10);

export function ulid(ms?: number): string {
  const t = ms ?? Date.now();
  return ulidTime(t) + ulidRand(RNG.bytes(10));
}
export function ulidMonotonic(ms?: number): string {
  const t = ms ?? Date.now();
  let r = RNG.bytes(10);
  if (t === __ulidTime) {
    r = __ulidLast.slice();
    for (let i = r.length - 1; i >= 0; i--) { r[i] = (r[i] + 1) & 0xff; if (r[i] !== 0) break; }
  }
  __ulidTime = t; __ulidLast = r;
  return ulidTime(t) + ulidRand(r);
}
export function getTimeFromUlid(id: string): number | undefined {
  if (!id || id.length < 10) return undefined;
  let t = 0;
  for (let i = 0; i < 10; i++) {
    const v = B32_IDX[id[i].toUpperCase()]; if (v == null) return undefined;
    t = t * 32 + v;
  }
  return t;
}

/* ------------------------------ nanoid-like ------------------------------ */
const URL_ALPH = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
export function nanoid(len = 21): string {
  const L = Math.max(1, Math.floor(len));
  const b = RNG.bytes(L);
  let out = "";
  for (let i = 0; i < L; i++) out += URL_ALPH[b[i] % URL_ALPH.length];
  return out;
}

/* ---------------------------------- CUID --------------------------------- */
// Avoid BigInt: convert random bytes to base36 by chunking into 32-bit numbers.
function bytesToBase36(b: Uint8Array): string {
  let s = "";
  for (let i = 0; i < b.length; i += 4) {
    const v =
      (b[i] ?? 0) * 2 ** 24 +
      (b[i + 1] ?? 0) * 2 ** 16 +
      (b[i + 2] ?? 0) * 2 ** 8 +
      (b[i + 3] ?? 0);
    // each chunk ≤ 2^32-1 → up to 7 base36 chars; pad for stable width
    s += v.toString(36).padStart(7, "0");
  }
  return s.replace(/^0+/, "");
}
let __ctr = Math.floor(RNG.rnd53() * 36 ** 3);
export function cuid(len = 20): string {
  const now = Date.now().toString(36);
  __ctr = (__ctr + 1) % (36 ** 3);
  const ctr = __ctr.toString(36).padStart(3, "0");
  const rand = bytesToBase36(RNG.bytes(12));
  return ("c" + now + ctr + rand).slice(0, Math.max(8, len));
}

/* -------------------------- base62 + short uid --------------------------- */
// NOTE: number-only (safe up to 2^53-1). That covers Date.now() etc.
const B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const B62_IDX: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  for (let i = 0; i < B62.length; i++) m[B62[i]] = i;
  return m;
})();

export function toBase62(n: number): string {
  let x = Math.floor(Math.max(0, n));
  if (!isFinite(x) || x <= 0) return "0";
  let s = "";
  while (x > 0) {
    const d = x % 62;
    s = B62[d] + s;
    x = Math.floor(x / 62);
  }
  return s;
}
export function fromBase62(s: string): number {
  const str = (s ?? "").trim();
  let acc = 0;
  for (let i = 0; i < str.length; i++) {
    const v = B62_IDX[str[i]];
    if (v == null) throw new Error(`invalid base62 char '${str[i]}' at ${i}`);
    acc = acc * 62 + v;
    // clamp to MAX_SAFE_INTEGER to avoid overflow surprises
    if (acc > Number.MAX_SAFE_INTEGER) {
      throw new Error("fromBase62 overflow (> Number.MAX_SAFE_INTEGER)");
    }
  }
  return acc;
}

/** Short readable id with optional prefix (e.g., "ord_7gH2aQ"). */
export function uid(prefix?: string): string {
  const t = toBase62(Date.now()); // time component (safe as number)
  const r = nanoid(6);            // random tail
  return (prefix ? prefix + "_" : "") + (t + r).slice(0, 12);
}

/* --------------------------------- export -------------------------------- */
export const id = {
  uuidv4,
  ulid,
  ulidMonotonic,
  getTimeFromUlid,
  nanoid,
  cuid,
  uid,
  toBase62,
  fromBase62,
};

export default id;