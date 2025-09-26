// core/bus.ts
// Tiny, import-free event bus for the engine.
// - Topics (strings), wildcard "*" listener, once-listeners
// - Async-safe emit (errors isolated), queue drain order preserved
// - waitFor(topic?, predicate?, timeoutMs?) -> Promise
// - Namespaced channels via createChannel(prefix)
// - Introspect: size(), listeners(topic), clear()

export type BusEvent<T = unknown> = {
  topic: string;        // e.g., "data/tick", "risk/update"
  payload?: T;          // any data
  ts: string;           // ISO timestamp
  id: string;           // random id
};

export type Listener<T = unknown> = (e: BusEvent<T>) => void | Promise<void>;

export type Unsubscribe = () => void;

export type WaitOptions<T = unknown> = {
  topic?: string;                             // if omitted, matches any
  predicate?: (e: BusEvent<T>) => boolean;    // optional matcher
  timeoutMs?: number;                          // optional timeout
};

export function createBus() {
  const map = new Map<string, Set<Listener>>();
  const WILDCARD = "*";

  let devLog = false;
  function setDebug(v: boolean) { devLog = !!v; }

  function on<T = unknown>(topic: string, fn: Listener<T>): Unsubscribe {
    const t = topic || "";
    let set = map.get(t);
    if (!set) { set = new Set(); map.set(t, set); }
    set.add(fn as Listener);
    return () => { try { set!.delete(fn as Listener); if (set!.size === 0) map.delete(t); } catch {} };
  }

  function once<T = unknown>(topic: string, fn: Listener<T>): Unsubscribe {
    const wrap: Listener<T> = (e) => {
      try { fn(e); } finally { off(topic, wrap as Listener); }
    };
    return on(topic, wrap);
  }

  function off(topic: string, fn: Listener) {
    const set = map.get(topic || "");
    if (set) {
      set.delete(fn);
      if (set.size === 0) map.delete(topic || "");
    }
  }

  function clear(topic?: string) {
    if (topic == null) { map.clear(); return; }
    map.delete(topic || "");
  }

  function listeners(topic?: string): number {
    if (!topic) {
        let total = 0;  
    }
    return map.get(topic || "")?.size || 0;
  }

  function size(): number { return map.size; }

  function id(): string { return Math.random().toString(36).slice(2, 10); }
  function iso(): string { try { return new Date().toISOString(); } catch { return "" + Date.now(); } }

  async function emit<T = unknown>(topic: string, payload?: T): Promise<void> {
    const evt: BusEvent<T> = { topic, payload, ts: iso(), id: id() };
    if (devLog) { try { console.debug?.("[bus:emit]", topic, evt); } catch {} }

    // snapshot listeners to avoid mutation during iteration
    const list: Listener[] = [];
    const s1 = map.get(topic || "");
   ; 
    const s2 = map.get(WILDCARD);
   

    // dispatch sequentially; isolate listener errors
    for (const fn of list) {
      try {
        const r = fn(evt as BusEvent);
        if (r && typeof (r as any).then === "function") {
          await (r as Promise<void>);
        }
      } catch (err) {
        if (devLog) { try { console.warn?.("[bus:listener:error]", topic, err); } catch {} }
      }
    }
  }

  /**
   * Wait for the next event that matches (topic?, predicate?).
   * If topic is omitted, listens on wildcard.
   */
  function waitFor<T = unknown>(opts?: WaitOptions<T>): Promise<BusEvent<T>> {
    const t = opts?.topic ?? WILDCARD;
    const pred = opts?.predicate ?? (() => true);
    const timeoutMs = opts?.timeoutMs ?? 0;

    return new Promise<BusEvent<T>>((resolve, reject) => {
      let timer: any;
      const offFn = on<T>(t, (e) => {
        try {
          if (opts?.topic && e.topic !== opts.topic) return; // when using wildcard
          if (!pred(e)) return;
          offFn();
          if (timer) clearTimeout(timer);
          resolve(e);
        } catch (err) {
          offFn();
          if (timer) clearTimeout(timer);
          reject(err);
        }
      });

      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          offFn();
          reject(new Error("bus.waitFor: timed out"));
        }, timeoutMs);
      }
    });
  }

  /**
   * Subscribe to all topics via wildcard "*".
   */
  function onAny<T = unknown>(fn: Listener<T>): Unsubscribe {
    return on<T>(WILDCARD, fn);
  }

  /**
   * Create a namespaced bus facade (prefix/).
   */
  function createChannel(prefix: string) {
    const p = prefix.endsWith("/") ? prefix : prefix + "/";
    return {
      on<T = unknown>(key: string, fn: Listener<T>) { return on<T>(p + key, fn); },
      once<T = unknown>(key: string, fn: Listener<T>) { return once<T>(p + key, fn); },
      emit<T = unknown>(key: string, payload?: T) { return emit<T>(p + key, payload); },
      waitFor<T = unknown>(keyOrPred?: string | ((e: BusEvent<T>) => boolean), timeoutMs?: number) {
        if (typeof keyOrPred === "string") {
          return waitFor<T>({ topic: p + keyOrPred, timeoutMs });
        }
        return waitFor<T>({ topic: undefined, predicate: (e) => e.topic.startsWith(p) && (!!keyOrPred ? (keyOrPred as any)(e) : true), timeoutMs });
      },
    };
  }

  return {
    // publish/subscribe
    on,
    once,
    onAny,
    off,
    emit,
    waitFor,
    // maintenance
    clear,
    listeners,
    size,
    setDebug,
    // namespacing
    createChannel,
  };
}

/* ------------------------------------------------------------------ */
/* Example usage (remove or keep as reference):

const bus = createBus();
const off = bus.on("data/tick", e => console.log("tick:", e.payload));
bus.emit("data/tick", { t: Date.now() });
off();

bus.waitFor({ topic: "risk/update", timeoutMs: 3000 }).then(console.log).catch(console.error);

const riskCh = bus.createChannel("risk");
riskCh.on("breach", e => console.log("risk breach", e.payload));
riskCh.emit("breach", { limit: "drawdown", value: 0.23 });

*/