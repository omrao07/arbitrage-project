// utils/router.ts
// Minimal router + middleware pipeline (pure Node, no imports)

export function compose(stack) {
  return (req, res) => {
    let i = -1;
    const run = (idx) => {
      if (idx <= i) return;
      i = idx;
      const fn = stack[idx];
      if (!fn) return;
      return fn(req, res, () => run(idx + 1));
    };
    run(0);
  };
}

function toRegex(path) {
  const keys = [];
  const pat = path
    .replace(/\/+$/, "")
    .replace(/\/:([\w-]+)/g, (_m, k) => {
      keys.push(k);
      return "/([^/]+)";
    })
    .replace(/\*/g, ".*");
  return { pattern: new RegExp("^" + pat + "/?$"), keys };
}

export class Router {
  routes: any[];
  constructor() {
    this.routes = [];
  }

  get(path, h) { this.add("GET", path, h); }
  post(path, h) { this.add("POST", path, h); }
  put(path, h) { this.add("PUT", path, h); }
  patch(path, h) { this.add("PATCH", path, h); }
  delete(path, h) { this.add("DELETE", path, h); }
  any(path, h) { this.add("*", path, h); }

  handle() {
    return (req, res, next) => {
      const method = (req.method || "GET").toUpperCase();
      const url = new URL(req.url, "http://localhost");
      req.path = url.pathname;
      req.query = Object.fromEntries(url.searchParams.entries());
      req.params = {};

      for (const r of this.routes) {
        if (r.method !== "*" && r.method !== method) continue;
        const m = r.pattern.exec(req.path);
        if (!m) continue;
        req.params = {};
        r.keys.forEach((k, i) => (req.params[k] = decodeURIComponent(m[i + 1] || "")));
        return r.handler(req, res);
      }

      return next();
    };
  }

  add(method, path, handler) {
    const { pattern, keys } = toRegex(path);
    this.routes.push({ method, pattern, keys, handler });
  }
}