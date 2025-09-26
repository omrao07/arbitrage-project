// middleware/logger.ts
// Simple request logger (no imports)

export function logger() {
  return (req, res, next) => {
    const start = Date.now();

    // Patch res.end to log when response finishes
    const originalEnd = res.end;
    res.end = function (...args) {
      const ms = Date.now() - start;
      try {
        console.log(
          `[${new Date().toISOString()}] ${req.method} ${req.url} (${ms}ms)`
        );
      } catch {}
      return originalEnd.apply(res, args);
    };

    next();
  };
}