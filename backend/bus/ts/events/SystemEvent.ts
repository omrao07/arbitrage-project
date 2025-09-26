// events/system_event.ts

/** ========= Shared primitives ========= */
export type Env = "dev" | "staging" | "prod" | string;
export type HealthStatus = "UP" | "DEGRADED" | "DOWN";
export type AlertLevel = "INFO" | "WARN" | "ERROR" | "CRITICAL";
export type LogLevel = "DEBUG" | "INFO" | "WARN" | "ERROR";

export interface BaseEvent {
  ts: string;            // ISO8601 timestamp
  env?: Env;             // deployment environment
  service?: string;      // logical service name (e.g., "pricing", "ui", "executor")
  host?: string;         // hostname or pod
  source?: string;       // producer node/id
  seq?: number;          // monotonic sequence if available
}

/** ========= Concrete event shapes ========= */
export interface HeartbeatEvent extends BaseEvent {
  type: "heartbeat";
  uptimeSec?: number;
  pid?: number;
  version?: string;      // app version
}

export interface HealthEvent extends BaseEvent {
  type: "health";
  status: HealthStatus;
  checks?: Record<string, { status: HealthStatus; detail?: string }>;
}

export interface MetricEvent extends BaseEvent {
  type: "metric";
  name: string;          // e.g., "latency_ms", "orders_per_sec"
  value: number;         // numeric value
  unit?: string;         // "ms", "count", etc.
  labels?: Record<string, string>; // dimensions: {route:"/trade", method:"POST"}
}

export interface AlertEvent extends BaseEvent {
  type: "alert";
  level: AlertLevel;
  message: string;
  code?: string;         // e.g., "ORDERS_LAGGING"
  context?: Record<string, any>;
}

export interface AuditEvent extends BaseEvent {
  type: "audit";
  actor: string;         // user/service who did the action
  action: string;        // e.g., "ORDER_CANCEL", "LOGIN", "POLICY_UPDATE"
  target?: string;       // resource id
  result?: "SUCCESS" | "FAILURE";
  details?: Record<string, any>;
}

export interface ConfigEvent extends BaseEvent {
  type: "config";
  changeId: string;      // unique id for change
  path: string;          // config path/key
  oldValue?: any;
  newValue?: any;
  reason?: string;
}

export interface DeployEvent extends BaseEvent {
  type: "deploy";
  version: string;       // app version
  gitSha?: string;
  image?: string;        // container image tag
  strategy?: "Rolling" | "Recreate" | "BlueGreen" | "Canary" | string;
  status?: "STARTED" | "SUCCEEDED" | "FAILED";
  notes?: string;
}

export interface LogEvent extends BaseEvent {
  type: "log";
  level: LogLevel;
  message: string;
  logger?: string;       // logger name/category
  stack?: string;        // optional stack trace
  fields?: Record<string, any>;
}

/** ========= Union ========= */
export type SystemEvent =
  | HeartbeatEvent
  | HealthEvent
  | MetricEvent
  | AlertEvent
  | AuditEvent
  | ConfigEvent
  | DeployEvent
  | LogEvent;

/** ========= Type guards ========= */
export const isHeartbeat = (e: SystemEvent): e is HeartbeatEvent => e.type === "heartbeat";
export const isHealth    = (e: SystemEvent): e is HealthEvent    => e.type === "health";
export const isMetric    = (e: SystemEvent): e is MetricEvent    => e.type === "metric";
export const isAlert     = (e: SystemEvent): e is AlertEvent     => e.type === "alert";
export const isAudit     = (e: SystemEvent): e is AuditEvent     => e.type === "audit";
export const isConfig    = (e: SystemEvent): e is ConfigEvent    => e.type === "config";
export const isDeploy    = (e: SystemEvent): e is DeployEvent    => e.type === "deploy";
export const isLog       = (e: SystemEvent): e is LogEvent       => e.type === "log";

/** ========= Validation (minimal; no deps) ========= */
function isIso(s: unknown): s is string {
  return typeof s === "string" && !Number.isNaN(Date.parse(s));
}
function isFiniteNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}
function req(cond: boolean, msg: string, acc: string[]) { if (!cond) acc.push(msg); }

export function validateSystemEvent(e: unknown): string[] {
  const errs: string[] = [];
  const x = e as Partial<SystemEvent>;
  req(!!x, "event required", errs);
  if (!x) return ["event null/undefined"];
  req(typeof x.type === "string", "type must be string", errs);
  req(isIso(x.ts), "ts must be ISO8601", errs);

  switch (x.type) {
    case "heartbeat":
      // no required numeric fields
      break;
    case "health": {
      const h = x as HealthEvent;
      req(["UP", "DEGRADED", "DOWN"].includes(h.status as any), "health.status invalid", errs);
      break;
    }
    case "metric": {
      const m = x as MetricEvent;
      req(typeof m.name === "string" && m.name.length > 0, "metric.name required", errs);
      req(isFiniteNum(m.value), "metric.value must be number", errs);
      break;
    }
    case "alert": {
      const a = x as AlertEvent;
      req(["INFO", "WARN", "ERROR", "CRITICAL"].includes(a.level as any), "alert.level invalid", errs);
      req(typeof a.message === "string" && a.message.length > 0, "alert.message required", errs);
      break;
    }
    case "audit": {
      const a = x as AuditEvent;
      req(typeof a.actor === "string" && a.actor.length > 0, "audit.actor required", errs);
      req(typeof a.action === "string" && a.action.length > 0, "audit.action required", errs);
      break;
    }
    case "config": {
      const c = x as ConfigEvent;
      req(typeof c.changeId === "string" && c.changeId.length > 0, "config.changeId required", errs);
      req(typeof c.path === "string" && c.path.length > 0, "config.path required", errs);
      break;
    }
    case "deploy": {
      const d = x as DeployEvent;
      req(typeof d.version === "string" && d.version.length > 0, "deploy.version required", errs);
      break;
    }
    case "log": {
      const l = x as LogEvent;
      req(["DEBUG", "INFO", "WARN", "ERROR"].includes(l.level as any), "log.level invalid", errs);
      req(typeof l.message === "string" && l.message.length > 0, "log.message required", errs);
      break;
    }
    default:
      errs.push("unsupported type");
  }
  return errs;
}

/** ========= Constructors ========= */
export const nowIso = () => new Date().toISOString();

export function makeHeartbeat(p: Omit<HeartbeatEvent, "type" | "ts"> & { ts?: string }): HeartbeatEvent {
  return { type: "heartbeat", ts: p.ts ?? nowIso(), ...p };
}
export function makeHealth(p: Omit<HealthEvent, "type" | "ts"> & { ts?: string }): HealthEvent {
  return { type: "health", ts: p.ts ?? nowIso(), ...p };
}
export function makeMetric(p: Omit<MetricEvent, "type" | "ts"> & { ts?: string }): MetricEvent {
  return { type: "metric", ts: p.ts ?? nowIso(), ...p };
}
export function makeAlert(p: Omit<AlertEvent, "type" | "ts"> & { ts?: string }): AlertEvent {
  return { type: "alert", ts: p.ts ?? nowIso(), ...p };
}
export function makeAudit(p: Omit<AuditEvent, "type" | "ts"> & { ts?: string }): AuditEvent {
  return { type: "audit", ts: p.ts ?? nowIso(), ...p };
}
export function makeConfig(p: Omit<ConfigEvent, "type" | "ts"> & { ts?: string }): ConfigEvent {
  return { type: "config", ts: p.ts ?? nowIso(), ...p };
}
export function makeDeploy(p: Omit<DeployEvent, "type" | "ts"> & { ts?: string }): DeployEvent {
  return { type: "deploy", ts: p.ts ?? nowIso(), ...p };
}
export function makeLog(p: Omit<LogEvent, "type" | "ts"> & { ts?: string }): LogEvent {
  return { type: "log", ts: p.ts ?? nowIso(), ...p };
}

/** ========= Serialization ========= */
export function toJSON(e: SystemEvent): string {
  return JSON.stringify(e);
}
export function fromJSON(json: string): SystemEvent {
  const obj = JSON.parse(json);
  const errs = validateSystemEvent(obj);
  if (errs.length) throw new Error("Invalid SystemEvent: " + errs.join("; "));
  return obj as SystemEvent;
}

/** ========= Default headers for bus ========= */
export function defaultHeaders() {
  return {
    "content-type": "application/json",
    "x-event-type": "system",
  };
}