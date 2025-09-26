// ts/config.ts
// Central config loader for Bus + infra

import * as dotenv from "dotenv";
dotenv.config(); // loads from .env if present

export type Env = "development" | "staging" | "production" | "test";

export interface Config {
  env: Env;
  logLevel: "debug" | "info" | "warn" | "error";

  bus: {
    backend: "kafka" | "nats" | "redis";
  };

  kafka: {
    brokers: string[];
    clientId: string;
    groupId: string;
    compression: "none" | "gzip" | "snappy" | "lz4" | "zstd";
  };

  nats: {
    servers: string[];
    clientName: string;
    stream?: string;
  };

  redis: {
    url: string;
    username?: string;
    password?: string;
    db: number;
    groupId: string;
    consumerId: string;
  };
}

function env<T>(key: string, def?: T): T {
  const v = process.env[key];
  if (v === undefined || v === "") {
    if (def !== undefined) return def;
    throw new Error(`Missing required env var: ${key}`);
  }
  return v as unknown as T;
}

export const config: Config = Object.freeze({
  env: (process.env.NODE_ENV as Env) ?? "development",
  logLevel: (process.env.LOG_LEVEL as Config["logLevel"]) ?? "info",

  bus: {
    backend: (process.env.BUS_BACKEND as Config["bus"]["backend"]) ?? "nats",
  },

  kafka: {
    brokers: env("KAFKA_BROKERS", "localhost:9092").split(","),
    clientId: env("KAFKA_CLIENT_ID", "bus-client"),
    groupId: env("KAFKA_GROUP_ID", "bus-group"),
    compression: (process.env.KAFKA_COMPRESSION as Config["kafka"]["compression"]) ?? "none",
  },

  nats: {
    servers: env("NATS_SERVERS", "nats://127.0.0.1:4222").split(","),
    clientName: env("NATS_CLIENT_NAME", "bus-client"),
    stream: process.env.NATS_STREAM,
  },

  redis: {
    url: env("REDIS_URL", "redis://127.0.0.1:6379"),
    username: process.env.REDIS_USERNAME,
    password: process.env.REDIS_PASSWORD,
    db: Number(process.env.REDIS_DB ?? 0),
    groupId: env("REDIS_GROUP_ID", "bus-group"),
    consumerId: env("REDIS_CONSUMER", `c-${Math.floor(Math.random() * 1e6)}`),
  },
});

export default config;