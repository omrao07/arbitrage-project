// src/server.ts
// Pure, self-contained HTTP server (no import/export statements).
// Includes tiny compose(), Router, and essential middlewares inline.
// If you’ve also created separate files (utils/router, middleware/*, routes/*),
// you can delete the inline bits below and wire those in instead.

// ----------------------------- Utilities ---------------------------------

function compose(stack) {
  return (req, res) => {
    let i = -1;
    const run = (idx) => {
      if (idx <= i) return;
      i = idx;
      const fn = stack[idx];
      if (!fn) return;
      return fn(req, res, () => run(idx + 1));
    };
    run(0);
  };
}

function toRegex(path) {
  const keys = [];
  const pat = path
    .replace(/\/+$/,"")
    .replace(/\/:([\w-]+)/g, (_m, k) => { keys.push(k); return "/([^/]+)"; })
    .replace(/\*/g, ".*");
  return { pattern: new RegExp("^" + pat + "/?$"), keys };
}

class Router {
  routes: any[];
  constructor() { this.routes = []; }
  get(p, h)    { this._add("GET", p, h); }
  post(p, h)   { this._add("POST", p, h); }
  put(p, h)    { this._add("PUT", p, h); }
  patch(p, h)  { this._add("PATCH", p, h); }
  delete(p, h) { this._add("DELETE", p, h); }
  any(p, h)    { this._add("*", p, h); }

  _add(method, path, handler) {
    const { pattern, keys } = toRegex(path);
    this.routes.push({ method, pattern, keys, handler });
  }

  handle() {
    return (req, res, next) => {
      const method = (req.method || "GET").toUpperCase();
      const url = new URL(req.url || "/", "http://localhost");
      req.path = url.pathname;
      req.query = Object.fromEntries(url.searchParams.entries());
      req.params = {};

      for (const r of this.routes) {
        if (r.method !== "*" && r.method !== method) continue;
        const m = r.pattern.exec(req.path);
        if (!m) continue;
        r.keys.forEach((k, i) => (req.params[k] = decodeURIComponent(m[i + 1] || "")));
        return r.handler(req, res);
      }
      return next();
    };
  }
}

function send(res, status, data, headers = {}) {
  const body = data === undefined ? "" : JSON.stringify(data);
  const h = {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body).toString(),
    ...headers,
  };
  res.writeHead(status, h);
  res.end(body);
}

const ok        = (res, data) => send(res, 200, data);
const created   = (res, data) => send(res, 201, data);
const noContent = (res)       => send(res, 204, "");

// ---------------------------- Middlewares --------------------------------

function errorBoundary() {
  return async (req, res, next) => {
    try { await next(); }
    catch (err) {
      const id = Math.random().toString(36).slice(2,10);
      try { console.error(`[error:${id}]`, err?.stack || err); } catch {}
      send(res, 500, { error: "Internal Server Error", id });
    }
  };
}

function logger() {
  return (req, res, next) => {
    const start = Date.now();
    const end = res.end;
    res.end = function (...args) {
      const ms = Date.now() - start;
      try { console.log(`[${new Date().toISOString()}] ${req.method} ${req.url} (${ms}ms)`); } catch {}
      return end.apply(this, args);
    };
    next();
  };
}

function cors(opts) {
  const o = {
    origin: opts?.origin || "*",
    methods: (opts?.methods || ["GET","POST","PUT","PATCH","DELETE","OPTIONS"]).join(","),
    headers: (opts?.headers || ["Content-Type","Authorization"]).join(","),
    credentials: !!opts?.credentials,
  };
  return (req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", o.origin);
    res.setHeader("Access-Control-Allow-Methods", o.methods);
    res.setHeader("Access-Control-Allow-Headers", o.headers);
    if (o.credentials) res.setHeader("Access-Control-Allow-Credentials", "true");
    if (req.method === "OPTIONS") { res.writeHead(204); return res.end(); }
    next();
  };
}

function json() {
  return (req, res, next) => {
    if (!["POST","PUT","PATCH"].includes(req.method)) return next();
    const ct = (req.headers["content-type"] || "").toString().toLowerCase();
    if (!ct.includes("application/json")) return next();

    let data = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1e6) { // ~1MB guard
        res.writeHead(413, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Payload too large" }));
        req.destroy();
      }
    });
    req.on("end", () => {
      try { req.body = data ? JSON.parse(data) : {}; next(); }
      catch { send(res, 400, { error: "Invalid JSON" }); }
    });
  };
}

// ------------------------------ Routes -----------------------------------

function registerRoutes(router) {
  // Health
  router.get("/health", (_req, res) => ok(res, { ok: true, ts: new Date().toISOString() }));
  router.get("/ready",  (_req, res) => ok(res, { ready: true }));
  router.get("/",       (_req, res) => ok(res, { name: "apps/api", version: 1 }));

  // Demo auth
  router.post("/api/v1/auth/login", (req, res) => {
    const email = req.body?.email;
    if (!email) return send(res, 400, { error: "email required" });
    const token = Buffer.from(email + "|" + Date.now()).toString("base64");
    ok(res, { token, user: { email } });
  });
  router.post("/api/v1/auth/logout", (_req, res) => ok(res, { ok: true }));
  router.get("/api/v1/auth/me", (req, res) => {
    const token = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
    if (!token) return send(res, 401, { error: "missing token" });
    try {
      const email = Buffer.from(token, "base64").toString("utf8").split("|")[0];
      ok(res, { email });
    } catch { send(res, 401, { error: "invalid token" }); }
  });

  // Demo users (in-memory)
  const users = new Map();
  const makeUser = (email, name) => {
    const now = new Date().toISOString();
    return { id: Math.random().toString(36).slice(2,10), email, name, createdAt: now, updatedAt: now };
  };

  router.get("/api/v1/users", (_req, res) => ok(res, Array.from(users.values())));

  router.post("/api/v1/users", (req, res) => {
    const email = req.body?.email;
    if (!email) return send(res, 400, { error: "email required" });

    // duplicate check without .values()
    const pairs = Array.from(users);
    for (let i = 0; i < pairs.length; i++) {
      if (pairs[i][1].email === email) return send(res, 400, { error: "email exists" });
    }

    const u = makeUser(email, req.body?.name);
    users.set(u.id, u);
    created(res, u);
  });

  router.get("/api/v1/users/:id", (req, res) => {
    const u = users.get(req.params.id);
    if (!u) return send(res, 404, { error: "user not found" });
    ok(res, u);
  });

  router.delete("/api/v1/users/:id", (req, res) => {
    const okDel = users.delete(req.params.id);
    if (!okDel) return send(res, 404, { error: "user not found" });
    noContent(res);
  });

  // 404 fallback
  router.any("*", (_req, res) => send(res, 404, { error: "Not found" }));
}

// ---------------------------- Env + Server -------------------------------

function loadEnv(defaults) {
  const out = { ...defaults };
  try {
    for (const k in defaults) {
      if (process.env[k] && process.env[k].length > 0) out[k] = process.env[k];
    }
  } catch {}
  return out;
}

const env = loadEnv({ PORT: "3001", NODE_ENV: "development" });

// Node http without import syntax
const http = typeof require === "function" ? require("http") : null;
if (!http) {
  throw new Error("Node 'http' module not available in this runtime.");
}

const router = new Router();
registerRoutes(router);

const app = compose([
  errorBoundary(),
  logger(),

  json(),
  router.handle(),
]);

const server = http.createServer((req, res) => app(req, res));
server.listen(Number(env.PORT), () => {
  try { console.log(`[api] http://localhost:${env.PORT} (${env.NODE_ENV})`); } catch {}
});