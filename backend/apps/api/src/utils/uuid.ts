// utils/uuid.ts
// Lightweight UUID v4-like generator (pure Node, no imports)

export function uuid() {
  function rnd(len) {
    let out = "";
    for (let i = 0; i < len; i++) {
      out += Math.floor(Math.random() * 16).toString(16);
    }
    return out;
  }

  // format: 8-4-4-4-12
  return (
    rnd(8) +
    "-" +
    rnd(4) +
    "-4" + rnd(3) + // version 4
    "-" +
    ((8 + Math.random() * 4) | 0).toString(16) + rnd(3) + // variant
    "-" +
    rnd(12)
  );
}