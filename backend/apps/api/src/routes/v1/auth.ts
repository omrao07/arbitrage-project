// routes/v1/auth.ts
// Demo auth endpoints (no imports, no deps)

export function v1AuthRoutes(router) {
  // POST /api/v1/auth/login
  router.post("/api/v1/auth/login", (req, res) => {
    const body = req.body || {};
    const email = body.email;
    if (!email) {
      const payload = JSON.stringify({ error: "email required" });
      res.writeHead(400, { "Content-Type": "application/json" });
      return res.end(payload);
    }

    // simple demo token
    const token = Buffer.from(email + "|" + Date.now()).toString("base64");
    const payload = JSON.stringify({ token, user: { email } });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // POST /api/v1/auth/logout
  router.post("/api/v1/auth/logout", (_req, res) => {
    const payload = JSON.stringify({ ok: true });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // GET /api/v1/auth/me
  router.get("/api/v1/auth/me", (req, res) => {
    const header = req.headers["authorization"] || "";
    const token = header.replace(/^Bearer\s+/i, "");
    if (!token) {
      const payload = JSON.stringify({ error: "missing token" });
      res.writeHead(401, { "Content-Type": "application/json" });
      return res.end(payload);
    }

    try {
      const decoded = Buffer.from(token, "base64").toString("utf8");
      const email = decoded.split("|")[0];
      const payload = JSON.stringify({ email });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(payload);
    } catch {
      const payload = JSON.stringify({ error: "invalid token" });
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(payload);
    }
  });
}