// middleware/error.ts
// Global error boundary (no imports)

export function errorBoundary() {
  return async (req, res, next) => {
    try {
      await next();
    } catch (err) {
      const id = Math.random().toString(36).slice(2, 10);
      const msg = err && err.message ? err.message : "Unknown error";
      const stack = err && err.stack ? err.stack : String(err);

      // Log to console
      try {
        console.error(`[error:${id}] ${msg}\n${stack}`);
      } catch {}

      // Safe JSON response
      const payload = JSON.stringify({ error: "Internal Server Error", id });
      res.writeHead(500, {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": Buffer.byteLength(payload).toString(),
      });
      res.end(payload);
    }
  };
}