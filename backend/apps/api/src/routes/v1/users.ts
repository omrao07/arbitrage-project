// routes/v1/users.ts
// User CRUD routes (pure Node, no imports)

const users = new Map(); // in-memory store

function makeUser(email, name) {
  const now = new Date().toISOString();
  return {
    id: Math.random().toString(36).slice(2, 10),
    email,
    name,
    createdAt: now,
    updatedAt: now,
  };
}

export function v1UserRoutes(router) {
  // GET /api/v1/users
  router.get("/api/v1/users", (_req, res) => {
    const payload = JSON.stringify(Array.from(users.values()));
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // POST /api/v1/users
  router.post("/api/v1/users", (req, res) => {
    const body = req.body || {};
    const email = body.email;

    if (!email) {
      const payload = JSON.stringify({ error: "email required" });
      res.writeHead(400, { "Content-Type": "application/json" });
      return res.end(payload);
    }

    // check duplicate email without .values()
    const allUsers = Array.from(users); // gives [ [id, user], [id, user] ... ]
    for (let i = 0; i < allUsers.length; i++) {
      const u = allUsers[i][1]; // index 1 is the user object
      if (u.email === email) {
        const payload = JSON.stringify({ error: "email exists" });
        res.writeHead(400, { "Content-Type": "application/json" });
        return res.end(payload);
      }
    }

    const user = makeUser(email, body.name);
    users.set(user.id, user);

    const payload = JSON.stringify(user);
    res.writeHead(201, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // GET /api/v1/users/:id
  router.get("/api/v1/users/:id", (req, res) => {
    const u = users.get(req.params.id);
    if (!u) {
      const payload = JSON.stringify({ error: "user not found" });
      res.writeHead(404, { "Content-Type": "application/json" });
      return res.end(payload);
    }

    const payload = JSON.stringify(u);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(payload);
  });

  // DELETE /api/v1/users/:id
  router.delete("/api/v1/users/:id", (req, res) => {
    const ok = users.delete(req.params.id);
    if (!ok) {
      const payload = JSON.stringify({ error: "user not found" });
      res.writeHead(404, { "Content-Type": "application/json" });
      return res.end(payload);
    }

    res.writeHead(204);
    res.end();
  });
}