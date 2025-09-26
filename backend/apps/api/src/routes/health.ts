// routes/health.ts
// Health + readiness endpoints (pure Node, no imports)

export function healthRoutes(router) {
  // GET /health
  router.get("/health", (_req, res) => {
    const payload = JSON.stringify({ ok: true, ts: new Date().toISOString() });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // GET /ready
  router.get("/ready", (_req, res) => {
    const payload = JSON.stringify({ ready: true });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // GET /
  router.get("/", (_req, res) => {
    const payload = JSON.stringify({ name: "apps/api", version: 1 });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });
}