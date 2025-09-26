// apps/terminal/app/(workspaces)/analytics/_data/runRegression.server.ts
// Server-only helper to request an OLS regression from your backend.
//
// Example usage (server action or server component):
//   const model = await runRegression({
//     target: "RET",
//     factors: ["MKT", "SMB", "HML"],
//     intercept: true,
//     window: 252
//   });

import "server-only";

/** request shape */
export type RunRegressionReq = {
  target: string;              // dependent variable, e.g. "RET"
  factors: string[];           // independent vars, e.g. ["MKT","SMB"]
  intercept?: boolean;         // include α, default true
  window?: number;             // lookback days
  portfolioId?: string;        // optional: run regression on portfolio returns
};

/** response shape */
export type RegressionModel = {
  target: string;
  factors: string[];
  intercept: boolean;
  n: number;
  alpha: number;
  betas: Record<string, number>;
  tstats: Record<string, number>;
  r2: number;
  adjR2: number;
  stderr: number;
  residuals: number[];
  yhat: number[];
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.API_BASE_URL ||
  "";

/** build endpoint path */
function endpoint(): string {
  if (BACKEND_URL) {
    return `${BACKEND_URL.replace(/\/+$/, "")}/analytics/regression`;
  }
  return "/api/analytics/regression"; // local Next route fallback
}

export async function runRegression(req: RunRegressionReq): Promise<RegressionModel> {
  const payload: RunRegressionReq = {
    intercept: true,
    window: 252,
    ...req,
  };

  if (!payload.target || !payload.factors?.length) {
    throw new Error("runRegression: must provide `target` and at least one `factor`.");
  }

  const url = endpoint();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`runRegression: ${res.status} ${res.statusText} ${text}`);
  }

  const raw = await res.json();

  // normalize: support both flat + wrapped shapes
  if (raw?.model) return normalize(raw.model);
  return normalize(raw);
}

function normalize(raw: any): RegressionModel {
  const {
    target = "unknown",
    factors = [],
    intercept = true,
    n = 0,
    alpha = 0,
    betas = {},
    tstats = {},
    r2 = 0,
    adjR2 = 0,
    stderr = 0,
    residuals = [],
    yhat = [],
  } = raw || {};

  return {
    target: String(target),
    factors: Array.isArray(factors) ? factors.map(String) : [],
    intercept: Boolean(intercept),
    n: Number(n),
    alpha: Number(alpha),
    betas: Object.fromEntries(Object.entries(betas || {}).map(([k, v]) => [k, Number(v)])),
    tstats: Object.fromEntries(Object.entries(tstats || {}).map(([k, v]) => [k, Number(v)])),
    r2: Number(r2),
    adjR2: Number(adjR2),
    stderr: Number(stderr),
    residuals: Array.isArray(residuals) ? residuals.map(Number) : [],
    yhat: Array.isArray(yhat) ? yhat.map(Number) : [],
  };
}