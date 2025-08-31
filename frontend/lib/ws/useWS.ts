// frontend/hooks/usews.ts
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type WSStatus =
  | "idle"
  | "connecting"
  | "open"
  | "closing"
  | "closed"
  | "error"
  | "paused"; // paused due to offline/hidden

export type UseWSOptions<TIn = any, TOut = any> = {
  /** ws:// or wss:// endpoint */
  url: string;

  /** Protocols parameter for WebSocket */
  protocols?: string | string[];

  /** Automatically reconnect after close/error (default true) */
  autoReconnect?: boolean;

  /** Max reconnect attempts (default Infinity) */
  maxRetries?: number;

  /** Base backoff ms (default 500) */
  backoffBaseMs?: number;

  /** Max backoff ms (default 10_000) */
  backoffMaxMs?: number;

  /** Heartbeat ping every ms (default 30_000; 0 disables) */
  heartbeatMs?: number;

  /** Message parser (default: JSON.parse with fallback to raw string) */
  parse?: (ev: MessageEvent) => TIn;

  /** Serializer for sendJson (default: JSON.stringify) */
  serialize?: (obj: TOut) => string;

  /** If true, queue messages while disconnected and flush on open (default true) */
  queueWhileDisconnected?: boolean;

  /** If true, pause socket when tab hidden or navigator offline (default true) */
  pauseWhenHiddenOrOffline?: boolean;

  /** Optional initial topics to subscribe after open */
  subscribeTopics?: string[];

  /** Custom subscribe/unsubscribe frames */
  toSubscribeFrame?: (topic: string) => string | ArrayBufferLike | Blob | ArrayBufferView;
  toUnsubscribeFrame?: (topic: string) => string | ArrayBufferLike | Blob | ArrayBufferView;

  /** Called on first open and every re-open */
  onOpen?: (ev: Event) => void;

  /** Called on message (before state update) */
  onMessage?: (data: TIn, ev: MessageEvent) => void;

  /** Called on error */
  onError?: (ev: Event) => void;

  /** Called on close */
  onClose?: (ev: CloseEvent) => void;
};

export type UseWSReturn<TIn = any, TOut = any> = {
  /** Current state */
  status: WSStatus;
  /** True when socket is OPEN */
  ready: boolean;
  /** Last parsed message (stateful) */
  last?: TIn;
  /** All messages (bounded) */
  history: TIn[];
  /** Send raw data */
  send: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => boolean;
  /** Send object as JSON using options.serialize */
  sendJson: (obj: TOut) => boolean;
  /** Close gracefully */
  close: (code?: number, reason?: string) => void;
  /** Force reconnect now (resets backoff) */
  reconnect: () => void;
  /** Subscribe to a topic (uses frames if provided) */
  subscribe: (topic: string) => void;
  /** Unsubscribe from a topic */
  unsubscribe: (topic: string) => void;
  /** Currently subscribed topics */
  topics: Set<string>;
};

/**
 * useWS — resilient WebSocket hook with:
 *  - auto-reconnect (backoff + jitter)
 *  - heartbeat ping
 *  - offline/hidden pause
 *  - message queue during disconnect
 *  - simple pub/sub helpers
 */
export function useWS<TIn = any, TOut = any>(opts: UseWSOptions<TIn, TOut>): UseWSReturn<TIn, TOut> {
  const {
    url,
    protocols,
    autoReconnect = true,
    maxRetries = Infinity,
    backoffBaseMs = 500,
    backoffMaxMs = 10_000,
    heartbeatMs = 30_000,
    parse = (ev: MessageEvent) => {
      try { return JSON.parse(ev.data); } catch { return ev.data as any; }
    },
    serialize = (obj: any) => JSON.stringify(obj),
    queueWhileDisconnected = true,
    pauseWhenHiddenOrOffline = true,
    subscribeTopics = [],
    // in usews.ts defaults
toSubscribeFrame = (t: string) => JSON.stringify({ op: "sub", topic: t }),
toUnsubscribeFrame = (t: string) => JSON.stringify({ op: "unsub", topic: t }),
    onOpen,
    onMessage,
    onError,
    onClose,
  } = opts;

  const wsRef = useRef<WebSocket | null>(null);
  const statusRef = useRef<WSStatus>("idle");
  const [status, setStatus] = useState<WSStatus>("idle");
  const [ready, setReady] = useState(false);

  const [last, setLast] = useState<TIn | undefined>(undefined);
  const [history, setHistory] = useState<TIn[]>([]);
  const historyLimit = 500; // keep memory bounded

  const retriesRef = useRef(0);
  const retryTimerRef = useRef<number | null>(null);
  const heartbeatRef = useRef<number | null>(null);
  const pausedRef = useRef(false);

  const sendQueueRef = useRef<(string | ArrayBufferLike | Blob | ArrayBufferView)[]>([]);
  const topicsRef = useRef<Set<string>>(new Set(subscribeTopics));

  const setState = (s: WSStatus) => {
    statusRef.current = s;
    setStatus(s);
    setReady(s === "open");
  };

  const clearRetryTimer = () => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  };

  const clearHeartbeat = () => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  };

  const close = useCallback((code?: number, reason?: string) => {
    clearRetryTimer();
    clearHeartbeat();
    try {
      wsRef.current?.close(code, reason);
    } catch {
      // ignore
    } finally {
      wsRef.current = null;
      setState("closed");
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (!autoReconnect) return;
    if (retriesRef.current >= maxRetries) return;

    const attempt = ++retriesRef.current;
    const backoff = Math.min(backoffMaxMs, backoffBaseMs * 2 ** (attempt - 1));
    const jitter = Math.floor(Math.random() * Math.min(1000, backoff / 5));
    const delay = backoff + jitter;

    clearRetryTimer();
    retryTimerRef.current = window.setTimeout(() => {
      connect(true);
    }, delay) as unknown as number;
  }, [autoReconnect, maxRetries, backoffBaseMs, backoffMaxMs]);

  const flushQueue = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const q = sendQueueRef.current;
    while (q.length) {
      ws.send(q.shift()!);
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearHeartbeat();
    if (!heartbeatMs) return;
    heartbeatRef.current = window.setInterval(() => {
      // simple ping; customize if your server expects certain frame
      // simple ping; customize if your server expects a certain frame
const frame = JSON.stringify({ op: "ping", t: Date.now() });
      try {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(frame);
        }
      } catch {
        // ignore
      }
    }, heartbeatMs) as unknown as number;
  }, [heartbeatMs, serialize]);

  const resubscribeAll = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    topicsRef.current.forEach((t) => {
      try { ws.send(toSubscribeFrame(t)); } catch { /* ignore */ }
    });
  }, [toSubscribeFrame]);

  const connect = useCallback((isRetry = false) => {
    if (pausedRef.current) {
      setState("paused");
      return;
    }
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return; // already active
    }

    try {
      setState("connecting");
      const ws = new WebSocket(url, protocols);
      wsRef.current = ws;

      ws.onopen = (ev) => {
        retriesRef.current = 0;
        setState("open");
        onOpen?.(ev);
        startHeartbeat();
        flushQueue();
        resubscribeAll();
      };

      ws.onmessage = (ev) => {
        let data: TIn;
        try { data = parse(ev); } catch (e) { /* bad parse; skip */ return; }
        onMessage?.(data, ev);
        setLast(data);
        setHistory((prev) => {
          const next = prev.length >= historyLimit ? prev.slice(-historyLimit + 1) : prev.slice();
          next.push(data);
          return next;
        });
      };

      ws.onerror = (ev) => {
        setState("error");
        onError?.(ev);
      };

      ws.onclose = (ev) => {
        clearHeartbeat();
        setState("closed");
        onClose?.(ev);
        if (!pausedRef.current) scheduleReconnect();
      };
    } catch (e) {
      setState("error");
      scheduleReconnect();
    }
  }, [url, protocols, parse, onOpen, onMessage, onError, onClose, startHeartbeat, flushQueue, resubscribeAll, scheduleReconnect]);

  const reconnect = useCallback(() => {
    retriesRef.current = 0;
    clearRetryTimer();
    close();
    connect();
  }, [close, connect]);

  const send = useCallback((data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(data); return true; } catch { return false; }
    }
    if (queueWhileDisconnected) {
      sendQueueRef.current.push(data);
      return true;
    }
    return false;
  }, [queueWhileDisconnected]);

  const sendJson = useCallback((obj: any) => {
    try { return send(serialize(obj)); } catch { return false; }
  }, [send, serialize]);

  const subscribe = useCallback((topic: string) => {
    topicsRef.current.add(topic);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(toSubscribeFrame(topic)); } catch { /* ignore */ }
    }
  }, [toSubscribeFrame]);

  const unsubscribe = useCallback((topic: string) => {
    topicsRef.current.delete(topic);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(toUnsubscribeFrame(topic)); } catch { /* ignore */ }
    }
  }, [toUnsubscribeFrame]);

  // visibility/offline pause / resume
  useEffect(() => {
    if (!pauseWhenHiddenOrOffline) {
      connect();
      return;
    }

    const update = () => {
      const hidden = typeof document !== "undefined" && document.visibilityState === "hidden";
      const offline = typeof navigator !== "undefined" && navigator.onLine === false;
      const shouldPause = hidden || offline;

      if (shouldPause && !pausedRef.current) {
        pausedRef.current = true;
        close(1000, "pause");
        setState("paused");
      } else if (!shouldPause && pausedRef.current) {
        pausedRef.current = false;
        connect();
      }
    };

    update();
    const vis = () => update();
    const online = () => update();
    const offline = () => update();

    document.addEventListener("visibilitychange", vis);
    window.addEventListener("online", online);
    window.addEventListener("offline", offline);

    return () => {
      document.removeEventListener("visibilitychange", vis);
      window.removeEventListener("online", online);
      window.removeEventListener("offline", offline);
    };
  }, [pauseWhenHiddenOrOffline, connect, close]);

  // initial connect + cleanup
  useEffect(() => {
    connect();
    return () => {
      clearRetryTimer();
      clearHeartbeat();
      try { wsRef.current?.close(1000, "unmount"); } catch {}
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, protocols]);

  const topics = useMemo(() => new Set(topicsRef.current), [status, ready]);

  return {
    status,
    ready,
    last,
    history,
    send,
    sendJson,
    close,
    reconnect,
    subscribe,
    unsubscribe,
    topics,
  };
}