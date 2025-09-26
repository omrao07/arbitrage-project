/**
 * middleware/cors.ts
 * Tiny CORS middleware for our custom router.
 *
 * - Adds CORS headers
 * - Handles OPTIONS preflight requests
 */

export function cors(opts?: {
  origin?: string;
  methods?: string[];
  headers?: string[];
  credentials?: boolean;
}){
  const o = {
    origin: opts?.origin || "*",
    methods: (opts?.methods || ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]).join(","),
    headers: (opts?.headers || ["Content-Type", "Authorization"]).join(","),
    credentials: !!opts?.credentials,
  };

  return (req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", o.origin);
    res.setHeader("Access-Control-Allow-Methods", o.methods);
    res.setHeader("Access-Control-Allow-Headers", o.headers);
    if (o.credentials) {
      res.setHeader("Access-Control-Allow-Credentials", "true");
    }

    // Short-circuit OPTIONS requests
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      return res.end();
    }

    return next();
  };
}