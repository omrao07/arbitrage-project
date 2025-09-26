// routes/index.ts
// Central route registration (pure Node, no imports)

import { healthRoutes } from "./health.js";
import { v1AuthRoutes } from "./v1/auth.js";
import { v1UserRoutes } from "./v1/users.js";

export function registerRoutes(router) {
  // Core health endpoints
  healthRoutes(router);

  // Auth routes
  v1AuthRoutes(router);

  // User routes
  v1UserRoutes(router);
}