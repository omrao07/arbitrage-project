// core/auth.ts
// Minimal, dependency-free auth/RBAC helpers (pure TS; Node built-ins only).
// - Token parsing (Bearer header or raw token)
// - Optional HMAC signing (u=<user>&r=<role>&exp=<iso>) with sig=<hex>
// - Role-based access control and capability checks
// - CSRF token mint/verify (HMAC)
// - Simple input guard/sanitizer for actions/routes

/* ───────────────────────── Types ───────────────────────── */

export type Role = "viewer" | "analyst" | "admin";
export type Capability =
  | "read:data"
  | "write:data"
  | "run:ingest"
  | "run:alerts"
  | "view:ops"
  | "admin:*";

export type Session = {
  user: string;
  role: Role;
  token: string;
  expiresAt?: string; // ISO
  meta?: Record<string, any>;
};

export type Policy = {
  roles: Record<Role, Capability[]>;
};

export type GuardResult =
  | { ok: true; session: Session }
  | { ok: false; status: 401 | 403; reason: string };

export type GuardInput = {
  authHeader?: string;          // e.g., "Bearer <token>"
  token?: string;               // raw token alternative
  secret?: string;              // HMAC secret (optional but recommended)
  minRole?: Role;               // require at least this role
  requireCaps?: Capability[];   // and these capabilities
  policy?: Policy;              // custom policy (falls back to default)
  method?: string;              // HTTP method (for CSRF guard)
  csrfToken?: string;           // provided CSRF token to verify on state-changing requests
};

/* ───────────────── Default Policy ───────────────── */

export const DefaultPolicy: Policy = {
  roles: {
    viewer:  ["read:data"],
    analyst: ["read:data","view:ops","run:alerts","run:ingest"],
    admin:   ["read:data","write:data","view:ops","run:alerts","run:ingest","admin:*"],
  }
};

/* ───────────────── Public API ───────────────── */

/** Parse and verify token; returns a Session or throws. */
export function authorize(input: { authHeader?: string; token?: string; secret?: string }): Session {
  const raw = pickToken(input.authHeader, input.token);
  if (!raw) throw err(401, "missing_token");

  // Signed format: "u:<user>|r:<role>|exp:<iso>|sig:<hex>"
  // Unsigned simple roles for dev: "adm_xxx" | "ana_xxx" | "view_xxx"
  if (raw.includes("|sig=")) {
    const { user, role, exp, sig, base } = parseSigned(raw);
    if (!inSet(role, ["viewer","analyst","admin"])) throw err(401, "bad_role");
    if (!input.secret) throw err(401, "missing_secret");
    if (!verifyHMAC(base, sig, input.secret)) throw err(401, "bad_signature");
    if (exp && isExpired(exp)) throw err(401, "token_expired");
    return { user, role, token: raw, expiresAt: exp };
  }

  // Dev tokens: prefix maps to role (NOT for production)
  if (raw.startsWith("adm_")) return { user: "admin", role: "admin", token: raw };
  if (raw.startsWith("ana_")) return { user: "analyst", role: "analyst", token: raw };
  return { user: "viewer", role: "viewer", token: raw };
}

/** Ensure role meets minimum requirement. */
export function requireRole(session: Session, min: Role): true {
  const order: Role[] = ["viewer","analyst","admin"];
  if (order.indexOf(session.role) < order.indexOf(min)) throw err(403, "forbidden_role");
  return true;
}

/** Capability check against policy. */
export function can(session: Session, cap: Capability, policy: Policy = DefaultPolicy): boolean {
  const caps = policy.roles[session.role] || [];
  return caps.includes(cap) || caps.includes("admin:*");
}

/** Route guard: combines auth, minRole, capabilities, and optional CSRF check. */
export function guard(inp: GuardInput): GuardResult {
  try {
    const sess = authorize({ authHeader: inp.authHeader, token: inp.token, secret: inp.secret });

    if (inp.minRole) requireRole(sess, inp.minRole);

    const pol = inp.policy || DefaultPolicy;
    if (inp.requireCaps && inp.requireCaps.length) {
      for (const c of inp.requireCaps) {
        if (!can(sess, c, pol)) throw err(403, "forbidden_cap");
      }
    }

    // CSRF for state-changing methods (POST/PUT/PATCH/DELETE)
    const m = (inp.method || "").toUpperCase();
    if (m && ["POST","PUT","PATCH","DELETE"].includes(m)) {
      if (!inp.secret) throw err(401, "missing_secret");
      if (!inp.csrfToken) throw err(401, "missing_csrf");
      if (!verifyCSRF(inp.csrfToken, sess, inp.secret)) throw err(401, "bad_csrf");
    }

    return { ok: true, session: sess };
  } catch (e: any) {
    const code = typeof e?.code === "number" ? e.code : 401;
    return { ok: false, status: code as 401|403, reason: e?.reason || "unauthorized" };
  }
}

/* ───────────────── CSRF tokens ───────────────── */

/**
 * Mint a CSRF token tied to session + timestamp.
 * Format: "cs:<user>|ts:<epoch>|sig:<hex>"
 */
export function mintCSRF(session: Session, secret: string, ttlSec = 3600): string {
  const ts = Math.floor(Date.now() / 1000);
  const base = `cs:${session.user}|ts:${ts}|ttl:${ttlSec}`;
  const sig = hmac(base, secret);
  return `${base}|sig:${sig}`;
}

export function verifyCSRF(token: string, session: Session, secret: string): boolean {
  try {
    const { base, ts, ttl, sig } = parseCSRF(token);
    // bind to user
    if (!base.startsWith(`cs:${session.user}|`)) return false;
    // time window
    const now = Math.floor(Date.now() / 1000);
    if (now > ts + ttl) return false;
    return verifyHMAC(base, sig, secret);
  } catch { return false; }
}

/* ───────────────── Sanitization helpers ───────────────── */

/** Strip prototype and only keep allowed keys (shallow). */
export function sanitize<T extends Record<string, any>>(obj: any, allowKeys: (keyof T)[]): T {
  const out: any = Object.create(null);
  if (!obj || typeof obj !== "object") return out;
  for (const k of allowKeys as string[]) {
    const v = (obj as any)[k];
    if (v === undefined) continue;
    out[k] = typeof v === "string" ? v.normalize().replace(/\u0000/g,"") : v;
  }
  return out as T;
}

/** Basic input guard for IDs (tickers, symbols, etc.). */
export function guardId(id: string, maxLen = 64): string {
  if (typeof id !== "string") throw err(400, "bad_id_type");
  const s = id.trim();
  if (!s) throw err(400, "empty_id");
  if (s.length > maxLen) throw err(400, "id_too_long");
  // whitelist common safe chars
  if (!/^[A-Za-z0-9._\-:/=+@]+$/.test(s)) throw err(400, "id_bad_chars");
  return s;
}

/* ───────────────── Token helpers ───────────────── */

function pickToken(h?: string, raw?: string): string {
  if (raw && raw.trim()) return raw.trim();
  if (!h) return "";
  const m = h.match(/^\s*Bearer\s+(.+)\s*$/i);
  return m ? m[1].trim() : h.trim();
}

function parseSigned(tok: string): { user: string; role: Role; exp?: string; sig: string; base: string } {
  // "u:<user>|r:<role>|exp:<iso>|sig:<hex>"
  const parts = tok.split("|");
  const map: Record<string,string> = {};
  for (const p of parts) {
    const i = p.indexOf(":");
    if (i>0) map[p.slice(0,i)] = p.slice(i+1);
  }
  const user = map["u"] || map["user"] || "user";
  const role = (map["r"] || "viewer") as Role;
  const exp  = map["exp"];
  const sig  = map["sig"] || "";
  const base = parts.filter(p => !p.startsWith("sig:")).join("|");
  return { user, role, exp, sig, base };
}

function isExpired(iso?: string): boolean {
  if (!iso) return false;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? (Date.now() > t) : false;
}

/* ───────────────── HMAC helpers ───────────────── */

declare const require: any; // avoid TS import; use Node built-in if available
let _crypto: any;
try { _crypto = require("crypto"); } catch { _crypto = undefined; }

function hmac(data: string, secret: string): string {
  if (_crypto?.createHmac) {
    return _crypto.createHmac("sha256", Buffer.from(secret, "utf8"))
      .update(Buffer.from(data, "utf8"))
      .digest("hex");
  }
  // Fallback (NOT cryptographically secure; dev only)
  return cheapHash(secret + "|" + data);
}

function verifyHMAC(data: string, sig: string, secret: string): boolean {
  const calc = hmac(data, secret);
  return constantTimeEq(calc, sig);
}

function constantTimeEq(a: string, b: string): boolean {
  const aa = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  if (aa.length !== bb.length) return false;
  let out = 0;
  for (let i = 0; i < aa.length; i++) out |= aa[i] ^ bb[i];
  return out === 0;
}

function cheapHash(s: string): string {
  // 32 hex chars pseudo-hash (do not use in prod)
  let x = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    x ^= s.charCodeAt(i);
    x = Math.imul(x, 16777619) >>> 0;
  }
  return padHex(x) + padHex((x ^ 0xa5a5a5a5) >>> 0) + padHex((x ^ 0x5a5a5a5a) >>> 0) + padHex((x ^ 0x3c3c3c3c) >>> 0);
}
function padHex(n: number): string { return ("00000000" + n.toString(16)).slice(-8); }

/* ───────────────── CSRF parsing ───────────────── */

function parseCSRF(tok: string): { base: string; ts: number; ttl: number; sig: string } {
  const parts = tok.split("|");
  const map: Record<string,string> = {};
  for (const p of parts) {
    const i = p.indexOf(":");
    if (i>0) map[p.slice(0,i)] = p.slice(i+1);
  }
  const ts  = Number(map["ts"] || 0);
  const ttl = Number(map["ttl"] || 0);
  const sig = map["sig"] || "";
  const base = parts.filter(p => !p.startsWith("sig:")).join("|");
  if (!ts || !ttl) throw new Error("bad_csrf");
  return { base, ts, ttl, sig };
}

/* ───────────────── Error helper ───────────────── */

function err(code: 400|401|403, reason: string) {
  const e: any = new Error(reason); e.code = code; e.reason = reason; return e;
}

function inSet<T extends string>(x: string, xs: readonly T[]): x is T {
  return (xs as readonly string[]).includes(x);
}