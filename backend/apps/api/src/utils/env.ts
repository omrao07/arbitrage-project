// utils/env.ts
// Simple environment loader (pure Node, no imports)

export function loadEnv(defaults) {
  const out = { ...defaults };

  try {
    for (const key in defaults) {
      if (process.env[key] && process.env[key].length > 0) {
        out[key] = process.env[key];
      }
    }
  } catch {
    // ignore if process.env not available
  }

  return out;
}