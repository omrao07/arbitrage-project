// frontend/lib/api.js
// Tiny, production‑ready API client used by your dashboard components.
// - Unified fetch with timeouts, retries, JSON parsing, and error surfacing
// - Helpers for strategies (list/patch/start/stop/presets), risk, PnL
// - SSE + WebSocket utilities for live feeds (tape/strategy status)
//
// Usage:
//   import * as api from "@/lib/api";
//   const pnl = await api.getPnL();
//   const risk = await api.getRisk();
//   await api.patchStrategy("short_interest_alpha", { enabled: true });

const DEFAULT_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== "undefined" ? window.location.origin : "http://localhost:3000");

const DEFAULT_HEADERS = {
  "Content-Type": "application/json",
};

// Optional bearer token (Bolt/Next: configure in env if needed)
const TOKEN =
  (typeof window !== "undefined" && window.localStorage?.getItem?.("token")) ||
  process.env.NEXT_PUBLIC_API_TOKEN ||
  "";

// ---- Core fetch wrapper ------------------------------------------------------

/**
 * @param {RequestInfo} url
 * @param {RequestInit & {timeoutMs?: number, retries?: number}} opts
 */
export async function http(url, opts = {}) {
  const {
    timeoutMs = 12_000,
    retries = 0,
    headers = {},
    ...rest
  } = opts;

  const finalUrl = String(url).startsWith("http") ? String(url) : `${DEFAULT_BASE}${url}`;
  const h = { ...DEFAULT_HEADERS, ...headers };
  if (TOKEN && !h.Authorization) h.Authorization = `Bearer ${TOKEN}`;

  let attempt = 0;
  let lastErr;

  while (attempt <= retries) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort("timeout"), timeoutMs);

    try {
      const res = await fetch(finalUrl, { ...rest, headers: h, signal: ctrl.signal, cache: "no-store" });
      clearTimeout(t);

      const isJSON = (res.headers.get("content-type") || "").includes("application/json");
      const body = isJSON ? await res.json().catch(() => ({})) : await res.text();

      if (!res.ok) {
        const msg = isJSON ? body?.error || body?.message || `HTTP ${res.status}` : `HTTP ${res.status}`;
        throw new HttpError(msg, res.status, body);
      }
      return body?.data ?? body;
    } catch (e) {
      clearTimeout(t);
      lastErr = e;
      const retryable =
        (e?.name === "AbortError") ||
        (e instanceof TypeError) ||
        (e instanceof HttpError && e.status >= 500);

      if (attempt < retries && retryable) {
        const backoff = Math.min(2000, 250 * Math.pow(2, attempt));
        await sleep(backoff);
        attempt++;
        continue;
      }
      throw e instanceof Error ? e : new Error(String(e));
    }
  }
  throw lastErr || new Error("Unknown error");
}

export class HttpError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   * @param {any} body
   */
  constructor(message, status, body) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.body = body;
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ---- REST endpoints ----------------------------------------------------------

// P&L
export function getPnL() {
  return http("/api/pnl", { method: "GET", retries: 1 });
}

// Risk payload
export function getRisk() {
  return http("/api/risk", { method: "GET", retries: 1 });
}

// Strategy catalog/state
export function getStrategies() {
  return http("/api/strategies", { method: "GET", retries: 1 });
}

/**
 * @param {string} name
 * @param {{ enabled?:boolean, weight?:number, region?:string, mode?:"paper"|"live", params?:object }} body
 */
export function patchStrategy(name, body) {
  return http(`/api/strategy/${encodeURIComponent(name)}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export function startStrategies(names /**: string[] | undefined */) {
  return http("/api/strategies/start", {
    method: "POST",
    body: JSON.stringify(names?.length ? { names } : {}),
  });
}
export function stopStrategies(names /**: string[] | undefined */) {
  return http("/api/strategies/stop", {
    method: "POST",
    body: JSON.stringify(names?.length ? { names } : {}),
  });
}

export function savePreset(name, items /**: any[] */) {
  return http("/api/strategies/presets/save", {
    method: "POST",
    body: JSON.stringify({ name, items }),
  });
}
export function applyPreset(name) {
  return http("/api/strategies/presets/apply", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

// ---- Streaming utilities -----------------------------------------------------

/**
 * Server‑Sent Events helper.
 * @param {string} path e.g. "/api/tape?symbols=AAPL,BTCUSD"
 * @param {(msg:any)=>void} onMessage
 * @param {(err?:any)=>void} [onError]
 * @returns {() => void} unsubscribe
 */
export function sse(path, onMessage, onError) {
  const url = path.startsWith("http") ? path : `${DEFAULT_BASE}${path}`;
  const es = new EventSource(url, { withCredentials: !!TOKEN }); // if auth cookie is used
  es.onmessage = (e) => {
    try {
      const m = JSON.parse(e.data);
      Array.isArray(m) ? m.forEach(onMessage) : onMessage(m);
    } catch {
      // ignore malformed chunk
    }
  };
  es.onerror = (e) => {
    onError?.(e);
  };
  return () => {
    try {
      es.close();
    } catch {}
  };
}

/**
 * WebSocket helper.
 * @param {string} url e.g. "wss://host/ws/tape"
 * @param {{ onOpen?():void, onClose?(e:CloseEvent):void, onError?(e:Event):void, onMessage?(m:any):void, protocols?:string|string[] }} handlers
 * @returns {{ send:(m:any)=>void, close:()=>void }}
 */
export function ws(url, handlers = {}) {
  const sock = new WebSocket(url, handlers.protocols);
  sock.onopen = () => handlers.onOpen?.();
  sock.onclose = (e) => handlers.onClose?.(e);
  sock.onerror = (e) => handlers.onError?.(e);
  sock.onmessage = (e) => {
    try {
      const m = JSON.parse(e.data);
      handlers.onMessage?.(m);
    } catch {
      // ignore malformed
    }
  };
  return {
    send(m) {
      try {
        sock.send(typeof m === "string" ? m : JSON.stringify(m));
      } catch {}
    },
    close() {
      try {
        sock.close(1000);
      } catch {}
    },
  };
}

// Convenience wrappers
export function openTapeSSE(symbols = []) {
  const qs = symbols.length ? `?symbols=${encodeURIComponent(symbols.join(","))}` : "";
  return sse(`/api/tape${qs}`, () => {});
}
export function openStrategyStatusWS() {
  const url =
    process.env.NEXT_PUBLIC_WS_STRATEGIES ||
    (DEFAULT_BASE.replace(/^http/, "ws") + "/ws/strategies");
  return ws(url, {});
}

// ---- JSONL / chunked streaming (optional) -----------------------------------

/**
 * Stream and parse NDJSON/JSONL endpoint (fetch).
 * @param {string} path
 * @param {(obj:any)=>void} onItem
 */
export async function streamJsonLines(path, onItem) {
  const res = await fetch(path.startsWith("http") ? path : `${DEFAULT_BASE}${path}`);
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        onItem(JSON.parse(line));
      } catch {
        // ignore bad line
      }
    }
  }
}

export default {
  http,
  getPnL,
  getRisk,
  getStrategies,
  patchStrategy,
  startStrategies,
  stopStrategies,
  savePreset,
  applyPreset,
  sse,
  ws,
  openTapeSSE,
  openStrategyStatusWS,
  streamJsonLines,
};