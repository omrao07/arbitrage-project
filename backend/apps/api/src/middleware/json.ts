// middleware/json.ts
// Minimal JSON body parser middleware (no imports)

export function json() {
  return (req, res, next) => {
    // Only parse JSON for POST, PUT, PATCH
    if (!["POST", "PUT", "PATCH"].includes(req.method)) {
      return next();
    }

    const ct = (req.headers["content-type"] || "").toString().toLowerCase();
    if (!ct.includes("application/json")) {
      return next();
    }

    let data = "";
    req.setEncoding("utf8");

    req.on("data", (chunk) => {
      data += chunk;
      // Basic guard: avoid huge payloads
      if (data.length > 1e6) {
        res.writeHead(413, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Payload too large" }));
        req.destroy();
      }
    });

    req.on("end", () => {
      try {
        req.body = data ? JSON.parse(data) : {};
        next();
      } catch {
        const payload = JSON.stringify({ error: "Invalid JSON" });
        res.writeHead(400, {
          "Content-Type": "application/json; charset=utf-8",
          "Content-Length": Buffer.byteLength(payload).toString(),
        });
        res.end(payload);
      }
    });
  };
}