/**
 * actions/dispatcher.ts
 * Tiny, import-free event bus for settings/actions.
 *
 * - No external deps. Works in browser and Node.
 * - Subscribe, unsubscribe, one-time listeners, waitFor (with timeout).
 * - Safe emit (isolates listener errors), optional dev logging.
 *
 * Usage:
 *   onSettingsEvent(e => console.log(e));
 *   emitSettingsEvent({ type: "profile/saved", payload: { name: "Ada" } });
 *   const off = onSettingsEventOnce(e => console.log("first only", e));
 *   await waitForSettingsEvent(ev => ev.type === "layouts/loaded", 5000);
 */

/* ------------------------------- Event Type ------------------------------- */

export type SettingsEvent = {
  type: string;          // e.g., "profile/saved", "connections/updated"
  payload?: unknown;     // any data
  ts?: string;           // ISO timestamp (auto-filled)
  id?: string;           // optional correlation id
};

/* -------------------------------- Internals -------------------------------- */

type Listener = (e: SettingsEvent) => void;

const listeners = new Set<Listener>();
let DEV_LOG = false;

/** Enable/disable console logging of emitted events (for debugging). */
export function setSettingsDispatcherDebug(v: boolean) {
  DEV_LOG = !!v;
}

/* --------------------------------- API ----------------------------------- */

/** Subscribe to all settings events. Returns an unsubscribe function. */
export function onSettingsEvent(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Subscribe to the next matching event only (auto-unsubscribe). */
export function onSettingsEventOnce(fn: Listener): () => void {
  const wrap: Listener = (e) => {
    try { fn(e); } finally { listeners.delete(wrap); }
  };
  listeners.add(wrap);
  return () => listeners.delete(wrap);
}

/** Emit a settings event to all subscribers. */
export function emitSettingsEvent(e: Omit<SettingsEvent, "ts">): void {
  const evt: SettingsEvent = { ts: isoNow(), ...e };
  if (DEV_LOG && typeof console !== "undefined") {
    try { console.debug?.("[settings:event]", evt.type, evt); } catch {}
  }
  // Copy to avoid mutation during iteration
  const copy = Array.from(listeners);
  for (const fn of copy) {
    try { fn(evt); } catch (err) {
      // Swallow listener errors to avoid breaking other subscribers
      if (DEV_LOG && typeof console !== "undefined") {
        try { console.warn?.("[settings:event:error]", evt.type, err); } catch {}
      }
    }
  }
}

/** Remove all listeners (use with care). */
export function clearSettingsEventListeners(): void {
  listeners.clear();
}

/**
 * Wait for the first event that satisfies the predicate.
 * Resolves with the event or rejects on timeout (ms, optional).
 */
export function waitForSettingsEvent(
  predicate: (e: SettingsEvent) => boolean,
  timeoutMs?: number
): Promise<SettingsEvent> {
  return new Promise((resolve, reject) => {
    let timer: any;
    const off = onSettingsEvent((e) => {
      if (!predicate(e)) return;
      off();
      if (timer) clearTimeout(timer);
      resolve(e);
    });
    if (Number.isFinite(timeoutMs) && (timeoutMs as number) > 0) {
      timer = setTimeout(() => {
        off();
        reject(new Error("waitForSettingsEvent: timed out"));
      }, timeoutMs);
    }
  });
}

/* -------------------------------- Utils ---------------------------------- */

function isoNow(): string {
  try { return new Date().toISOString(); } catch { return String(Date.now()); }
}

/* ---------------------------- Optional Channels --------------------------- */
/**
 * If you want lightweight namespacing, you can create scoped dispatchers:
 *
 *   const { on, emit } = createChannel("flags");
 *   on(e => console.log("flags:", e));
 *   emit({ type: "flags/updated", payload: {...} });
 */
export function createChannel(scope: string) {
  const prefix = scope.endsWith("/") ? scope : scope + "/";
  return {
    on(fn: Listener) {
      return onSettingsEvent((e) => {
        if (e.type.startsWith(prefix)) fn(e);
      });
    },
    once(fn: Listener) {
      return onSettingsEventOnce((e) => {
        if (e.type.startsWith(prefix)) fn(e);
      });
    },
    emit(e: Omit<SettingsEvent, "ts">) {
      emitSettingsEvent({ ...e, type: e.type.startsWith(prefix) ? e.type : prefix + e.type });
    },
    waitFor(pred: (e: SettingsEvent) => boolean, timeoutMs?: number) {
      return waitForSettingsEvent((e) => e.type.startsWith(prefix) && pred(e), timeoutMs);
    },
  };
}