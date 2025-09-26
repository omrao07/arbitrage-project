// data/loaders/api.ts
// Minimal, import-free HTTP client with JSON helpers, retries, backoff, timeout,
// and an optional token-bucket rate limiter. Works with Node >=18 (global fetch)
// and falls back to http/https.request if fetch is unavailable.
//
// Usage:
//   const api = createApiClient({ baseUrl: "https://api.example.com", timeoutMs: 8000 });
//   const res = await api.getJson("/v1/quotes", { symbol: "AAPL" });
//   if (res.ok) console.log(res.data);
//
// Exposed methods:
//   - request(method, path, { query, headers, body, timeoutMs, retries, retryOn })
//   - get(path, opts), post(path, opts), put(...), del(...)
//   - getJson(path, query?), postJson(path, body?), putJson(...), delJson(...)
//   - setHeader(k,v), removeHeader(k), setRateLimit({rate, perMs, burst})

export type ApiOptions = {
  baseUrl?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;     // default 10000
  retries?: number;       // default 2
  backoffMs?: number;     // default 250
  jitter?: boolean;       // default true
  rate?: number;          // tokens per window (default unlimited)
  perMs?: number;         // window length in ms
  burst?: number;         // bucket size
  retryOn?: (code: number | null, err?: unknown) => boolean; // default: 5xx + network
};

export type ApiRequest = {
  method: string;
  url: string;
  status: number;
  headers: Record<string, string>;
  body?: any;
};

export type ApiResponse<T = unknown> = {
  ok: boolean;
  status: number;
  data?: T;
  text?: string;
  headers: Record<string, string>;
  req: ApiRequest;
  error?: string;
};

export function createApiClient(opts: ApiOptions = {}) {
  const state = {
    baseUrl: (opts.baseUrl || "").replace(/\/+$/, ""),
    headers: Object.assign({}, opts.headers || {}),
    timeoutMs: nOr(opts.timeoutMs, 10000),
    retries: intOr(opts.retries, 2),
    backoffMs: intOr(opts.backoffMs, 250),
    jitter: opts.jitter !== false,
    retryOn: opts.retryOn || defaultRetryOn,
    bucket: createBucket(opts.rate, opts.perMs, opts.burst),
  };

  function setHeader(k: string, v: string) { state.headers[k.toLowerCase()] = v; }
  function removeHeader(k: string) { delete state.headers[k.toLowerCase()]; }

  function setRateLimit(r?: { rate?: number; perMs?: number; burst?: number }) {
    state.bucket = createBucket(r?.rate, r?.perMs, r?.burst);
  }

  async function request<T = unknown>(method: string, path: string, opt?: {
    query?: Record<string, any>,
    headers?: Record<string, string>,
    body?: any,
    timeoutMs?: number,
    retries?: number,
    retryOn?: (code: number | null, err?: unknown) => boolean,
  }): Promise<ApiResponse<T>> {
    const q = opt?.query ? "?" + toQuery(opt.query) : "";
    const url = joinUrl(state.baseUrl, path) + q;
    const hdrs = mergeHeaders(state.headers, opt?.headers);
    const timeoutMs = nOr(opt?.timeoutMs, state.timeoutMs);
    const retries = intOr(opt?.retries, state.retries);
    const retryOn = opt?.retryOn || state.retryOn;

    // Rate limit
    await state.bucket.take();

    let attempt = 0;
    let lastErr: any = null;

    while (attempt <= retries) {
      attempt++;
      try {
        const res = await doRequest(method, url, hdrs, opt?.body, timeoutMs);
        if (retryOn(res.status, undefined) && attempt <= retries) {
          await delay(backoffDelay(state.backoffMs, attempt, state.jitter));
          continue;
        }
        return res as ApiResponse<T>;
      } catch (err) {
        lastErr = err;
        if (retryOn(null, err) && attempt <= retries) {
          await delay(backoffDelay(state.backoffMs, attempt, state.jitter));
          continue;
        }
        return {
          ok: false,
          status: 0,
          headers: {},
          req: { method, url, status: 0, headers: hdrs, body: redactBody(opt?.body) },
          error: String(err?.message || err || "network error"),
        };
      }
    }

    return {
      ok: false,
      status: 0,
      headers: {},
      req: { method, url, status: 0, headers: hdrs, body: redactBody(opt?.body) },
      error: String(lastErr?.message || lastErr || "network error"),
    };
  }

  // Shorthand request helpers
  const get  = (p: string, o?: any) => request("GET", p, o);
  const post = (p: string, o?: any) => request("POST", p, o);
  const put  = (p: string, o?: any) => request("PUT", p, o);
  const del  = (p: string, o?: any) => request("DELETE", p, o);

  // JSON convenience methods
  async function getJson<T = unknown>(p: string, query?: Record<string, any>): Promise<ApiResponse<T>> {
    const r = await get(p, { query, headers: { accept: "application/json" } });
    return parseJson<T>(r);
  }

  async function postJson<T = unknown>(p: string, body?: any): Promise<ApiResponse<T>> {
    const r = await post(p, {
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    return parseJson<T>(r);
  }

  async function putJson<T = unknown>(p: string, body?: any): Promise<ApiResponse<T>> {
    const r = await put(p, {
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    return parseJson<T>(r);
  }

  async function delJson<T = unknown>(p: string, query?: Record<string, any>): Promise<ApiResponse<T>> {
    const r = await del(p, { query, headers: { accept: "application/json" } });
    return parseJson<T>(r);
  }

  return {
    // core
    request, get, post, put, del,
    // json helpers
    getJson, postJson, putJson, delJson,
    // config
    setHeader, removeHeader, setRateLimit,
  };
}

/* ------------------------------- Internals ------------------------------- */

async function doRequest(method: string, url: string, headers: Record<string, string>, body: any, timeoutMs: number): Promise<ApiResponse> {
  // Prefer global fetch (Node >=18). Fallback to http/https if not present.
  if (typeof (globalThis as any).fetch === "function") {
    const controller = typeof (globalThis as any).AbortController === "function" ? new (globalThis as any).AbortController() : null;
    const timer = controller ? setTimeout(() => { try { controller.abort(); } catch {} }, timeoutMs) : null;
    try {
      const resp = await (globalThis as any).fetch(url, {
        method,
        headers,
        body,
        signal: controller ? controller.signal : undefined,
      });
      const text = await resp.text();
      const hdrs = lowerHeaders(resp.headers);
      return {
        ok: resp.ok,
        status: resp.status,
        text,
        headers: hdrs,
        req: { method, url, status: resp.status, headers, body: redactBody(body) },
        error: resp.ok ? undefined : `${resp.status} ${resp.statusText}`,
      };
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  // Fallback: http/https.request
  const isHttps = url.startsWith("https:");
  const mod = isHttps
    ? (typeof require === "function" ? require("https") : null)
    : (typeof require === "function" ? require("http") : null);

  if (!mod) throw new Error("No HTTP module available");

  return new Promise<ApiResponse>((resolve, reject) => {
    try {
      const u = new (globalThis as any).URL(url);
      const options: any = {
        method,
        protocol: u.protocol,
        hostname: u.hostname,
        port: u.port || (isHttps ? 443 : 80),
        path: u.pathname + (u.search || ""),
        headers,
      };

      const req = mod.request(options, (res: any) => {
        const chunks: any[] = [];
        res.on("data", (d: any) => chunks.push(d));
        res.on("end", () => {
          const buf = Buffer.concat(chunks);
          const text = buf.toString("utf8");
          const hdrs = lowerHeaders(res.headers || {});
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            text,
            headers: hdrs,
            req: { method, url, status: res.statusCode, headers, body: redactBody(body) },
            error: res.statusCode >= 200 && res.statusCode < 300 ? undefined : `${res.statusCode} ${res.statusMessage || ""}`.trim(),
          });
        });
      });

      // timeout
      req.setTimeout(timeoutMs, () => {
        try { req.destroy(new Error("timeout")); } catch {}
      });

      req.on("error", (err: any) => reject(err));

      if (body != null) {
        if (typeof body === "string" || Buffer.isBuffer(body)) {
          req.write(body);
        } else {
          const asStr = JSON.stringify(body);
          req.write(asStr);
        }
      }
      req.end();
    } catch (err) {
      reject(err);
    }
  });
}

function parseJson<T = unknown>(resp: ApiResponse): ApiResponse<T> {
  if (!resp) return { ok: false, status: 0, headers: {}, req: { method: "", url: "", status: 0, headers: {} }, error: "empty response" };
  if (!resp.text) return { ...(resp as any), data: undefined };
  try {
    const data = JSON.parse(resp.text);
    return { ...(resp as any), data };
  } catch {
    // not JSON, return original text
    return resp as any;
  }
}

function toQuery(params: Record<string, any>): string {
  const sp: string[] = [];
  for (const k of Object.keys(params)) {
    const v = (params as any)[k];
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) {
      for (const it of v) sp.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(it)));
    } else {
      sp.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(v)));
    }
  }
  return sp.join("&");
}

function joinUrl(base: string, path: string): string {
  if (!base) return path;
  if (!path) return base;
  if (path.startsWith("http")) return path;
  const a = base.endsWith("/") ? base.slice(0, -1) : base;
  const b = path.startsWith("/") ? path : "/" + path;
  return a + b;
}

function mergeHeaders(a: Record<string, string>, b?: Record<string, string>) {
  const out: Record<string, string> = {};
  for (const k in a) out[k.toLowerCase()] = a[k];
  if (b) for (const k in b) out[k.toLowerCase()] = b[k];
  return out;
}

function lowerHeaders(h: any): Record<string, string> {
  const out: Record<string, string> = {};
  // fetch Headers
  if (h && typeof h.forEach === "function") {
    h.forEach((v: string, k: string) => out[String(k).toLowerCase()] = String(v));
    return out;
  }
  // Node res.headers is a plain object
  for (const k in (h || {})) out[String(k).toLowerCase()] = String((h as any)[k]);
  return out;
}

function defaultRetryOn(code: number | null, err?: unknown): boolean {
  if (code === null) return true;               // network error -> retry
  if (code >= 500 && code < 600) return true;   // 5xx
  if (code === 429) return true;                // rate limited
  return false;
}

function backoffDelay(baseMs: number, attempt: number, jitter: boolean): number {
  // exponential backoff: base * 2^(attempt-1), capped + jitter
  const cap = 30_000;
  let d = Math.min(cap, baseMs * Math.pow(2, Math.max(0, attempt - 1)));
  if (jitter) d += Math.floor(Math.random() * Math.min(d, 500));
  return d;
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function nOr(v: any, d: number) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function intOr(v: any, d: number) { const n = Number(v); return Number.isFinite(n) ? (n | 0) : d; }

function redactBody(body: any) {
  // keep size reasonable in logs; redact obvious secrets
  if (body == null) return undefined;
  try {
    const s = typeof body === "string" ? body : JSON.stringify(body);
    if (s.length > 2048) return s.slice(0, 2048) + "...";
    return s.replace(/"password"\s*:\s*"[^"]+"/gi, '"password":"***"')
            .replace(/"secret"\s*:\s*"[^"]+"/gi, '"secret":"***"')
            .replace(/"token"\s*:\s*"[^"]+"/gi, '"token":"***"');
  } catch { return undefined; }
}

/* -------------------------- Token Bucket limiter ------------------------- */

function createBucket(rate?: number, perMs?: number, burst?: number) {
  // If rate not specified, unlimited
  if (!rate || rate <= 0) {
    return { take: async () => {} };
  }
  const windowMs = Math.max(1, perMs ?? 1000);      // default 1s window
  const capacity = Math.max(rate, burst ?? rate);   // default burst = rate
  let tokens = capacity;
  let last = Date.now();

  async function take() {
    refill();
    if (tokens >= 1) {
      tokens -= 1;
      return;
    }
    // wait until a token becomes available
    while (tokens < 1) {
      const wait = Math.max(5, Math.ceil(((1 - tokens) / rate) * windowMs));
      await delay(wait);
      refill();
    }
    tokens -= 1;
  }

  function refill() {
    const now = Date.now();
    const elapsed = now - last;
    if (elapsed <= 0) return;
    const add = (elapsed / windowMs) * rate!;
    tokens = Math.min(capacity, tokens + add);
    last = now;
  }

  return { take };
}