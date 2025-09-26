// utils/serializer.ts
import * as zlib from "zlib";

export type SerializationFormat = "json" | "string" | "buffer";

export interface SerializeOptions {
  compress?: boolean;            // gzip compression
  format?: SerializationFormat;  // default "json"
}

export interface DeserializeOptions {
  compressed?: boolean;          // if input is gzip compressed
  format?: SerializationFormat;  // default "json"
}

/**
 * Serialize data into Buffer or string for transport.
 */
export function serialize(
  data: any,
  opts: SerializeOptions = {}
): Buffer | string {
  const format = opts.format ?? "json";
  let payload: Buffer;

  if (format === "json") {
    const json = JSON.stringify(data);
    payload = Buffer.from(json, "utf-8");
  } else if (format === "string") {
    payload = Buffer.from(String(data), "utf-8");
  } else if (format === "buffer") {
    if (Buffer.isBuffer(data)) {
      payload = data;
    } else {
      throw new Error("Expected Buffer when using format=buffer");
    }
  } else {
    throw new Error(`Unsupported serialization format: ${format}`);
  }

  if (opts.compress) {
    return zlib.gzipSync(payload);
  }

  return payload;
}

/**
 * Deserialize data back into JS object/string/Buffer.
 */
export function deserialize(
  input: Buffer | string,
  opts: DeserializeOptions = {}
): any {
  const format = opts.format ?? "json";
  let payload: Buffer;

  if (typeof input === "string") {
    payload = Buffer.from(input, "utf-8");
  } else {
    payload = input;
  }

  if (opts.compressed) {
    payload = zlib.gunzipSync(payload);
  }

  if (format === "json") {
    return JSON.parse(payload.toString("utf-8"));
  } else if (format === "string") {
    return payload.toString("utf-8");
  } else if (format === "buffer") {
    return payload;
  } else {
    throw new Error(`Unsupported deserialization format: ${format}`);
  }
}

/* ---------------- Example usage ---------------- */
if (require.main === module) {
  const obj = { id: 42, msg: "hello" };

  const encoded = serialize(obj, { compress: true, format: "json" });
  console.log("encoded length:", encoded.length);

  const decoded = deserialize(encoded, { compressed: true, format: "json" });
  console.log("decoded object:", decoded);
}