"use client";

import React, { useMemo, useState } from "react";

/* ===========================
   Types
=========================== */
type SeriesMap = Record<string, number[]>;

type FactorRow = {
  name: string;
  value: number;
  benchmark?: number;
};

type RegressionModel = {
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

/* ===========================
   Demo helpers (replace with API later)
=========================== */
function makeDemoSeries(tickers: string[], len = 252): SeriesMap {
  const out: SeriesMap = {};
  tickers.forEach((t, idx) => {
    const arr: number[] = [];
    let x = 0;
    for (let i = 0; i < len; i++) {
      const r =
        (Math.sin((i + idx) * 0.11) +
          Math.cos((i - idx) * 0.07) * 0.6 +
          (Math.random() - 0.5) * 0.8) /
        100;
      x += r;
      arr.push(r); // returns, not prices
    }
    out[t] = arr;
  });
  return out;
}

/* ===========================
   CorrelationMatrix (pure TSX, no deps)
=========================== */
function pearson(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return NaN;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, m = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; m++;
  }
  if (m < 2) return NaN;
  const cov = sxy - (sx * sy) / m;
  const vx = sxx - (sx * sx) / m;
  const vy = syy - (sy * sy) / m;
  const denom = Math.sqrt(vx * vy);
  if (!Number.isFinite(denom) || denom === 0) return NaN;
  return Math.max(-1, Math.min(1, cov / denom));
}
function buildCorr(series: SeriesMap): { keys: string[]; values: number[][] } {
  const keys = Object.keys(series);
  const m = keys.length;
  const values: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    values[i][i] = 1;
    for (let j = i + 1; j < m; j++) {
      const r = pearson(series[keys[i]], series[keys[j]]);
      values[i][j] = r; values[j][i] = r;
    }
  }
  return { keys, values };
}
function colorFor(rho: number): string {
  if (!Number.isFinite(rho)) return "#2a2a2a";
  const t = (rho + 1) / 2; // 0..1
  const r = Math.round(255 * t);
  const b = Math.round(255 * (1 - t));
  const g = Math.round(255 * (0.6 - Math.abs(rho) * 0.6));
  return `rgb(${r},${g},${b})`;
}
function Legend() {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="opacity-70">-1</span>
      <div className="h-3 w-32 rounded overflow-hidden flex">
        {Array.from({ length: 32 }).map((_, i) => {
          const t = i / 31;
          const rho = t * 2 - 1;
          return <div key={i} style={{ width: "3.125%", background: colorFor(rho) }} />;
        })}
      </div>
      <span className="opacity-70">+1</span>
    </div>
  );
}
function CorrelationMatrix({
  series,
  decimals = 2,
  height = 480,
  onCellClick,
}: {
  series: SeriesMap;
  decimals?: number;
  height?: number;
  onCellClick?: (row: string, col: string, rho: number) => void;
}) {
  const { keys, values } = useMemo(() => buildCorr(series), [series]);
  const size = keys.length;
  const cell = Math.max(24, Math.min(64, Math.floor((height - 80) / Math.max(1, size))));
  const gridW = cell * size;

  return (
    <div className="w-full" style={{ color: "#ddd" }}>
      <div className="flex items-end justify-between mb-2">
        <div className="text-sm opacity-80">Correlation Matrix (ρ)</div>
        <Legend />
      </div>
      <div className="overflow-auto rounded border border-[#333]" style={{ maxHeight: height }}>
        <div style={{ width: gridW + 140 }} className="relative">
          <div className="sticky top-0 z-10 pl-[140px] bg-[#0b0b0b]">
            <div className="grid" style={{ gridTemplateColumns: `repeat(${size}, ${cell}px)` }}>
              {keys.map((k) => (
                <div key={`top-${k}`} className="text-xs text-center p-1 truncate">
                  {k}
                </div>
              ))}
            </div>
          </div>
          <div className="flex">
            <div className="sticky left-0 z-10 w-[140px] bg-[#0b0b0b]">
              {keys.map((k) => (
                <div
                  key={`left-${k}`}
                  style={{ height: cell }}
                  className="text-xs flex items-center pl-2 border-b border-[#222] truncate"
                >
                  {k}
                </div>
              ))}
            </div>
            <div>
              {values.map((row, i) => (
                <div
                  key={`row-${i}`}
                  className="grid"
                  style={{ gridTemplateColumns: `repeat(${size}, ${cell}px)` }}
                >
                  {row.map((rho, j) => (
                    <button
                      key={`cell-${i}-${j}`}
                      title={`${keys[i]} × ${keys[j]} — ρ=${Number(rho).toFixed(decimals)}`}
                      onClick={() => onCellClick?.(keys[i], keys[j], rho)}
                      className="border border-[#222] focus:outline-none"
                      style={{
                        height: cell,
                        background: colorFor(rho),
                        cursor: onCellClick ? "pointer" : "default",
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ===========================
   FactorsPanel (pure TSX)
=========================== */
function FactorsPanel({
  factors,
  decimals = 2,
  onClickFactor,
}: {
  factors: FactorRow[];
  decimals?: number;
  onClickFactor?: (f: FactorRow) => void;
}) {
  if (!factors?.length) {
    return (
      <div className="bg-[#0b0b0b] text-gray-400 p-4 rounded border border-[#222]">
        No factors available
      </div>
    );
  }
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.value)), 1);

  return (
    <div className="bg-[#0b0b0b] p-4 rounded-lg shadow-lg">
      <h2 className="text-sm text-gray-300 mb-3 font-semibold">Factor Exposures</h2>
      <div className="space-y-2">
        {factors.map((f, idx) => {
          const pct = (f.value / maxAbs) * 50; // -50..+50
          const color = f.value > 0 ? "#16a34a" : f.value < 0 ? "#dc2626" : "#666";
          return (
            <div
              key={idx}
              className="flex items-center gap-2 cursor-pointer"
              onClick={() => onClickFactor?.(f)}
              title={`${f.name}: ${f.value.toFixed(decimals)}`}
            >
              <span className="text-xs w-32 truncate text-gray-300">{f.name}</span>
              <div className="relative flex-1 h-4 bg-[#1a1a1a] rounded overflow-hidden border border-[#333]">
                <div
                  className="absolute top-0 bottom-0"
                  style={{
                    left: "50%",
                    width: `${Math.abs(pct)}%`,
                    background: color,
                    transform: f.value < 0 ? "translateX(-100%)" : "none",
                  }}
                />
                {f.benchmark !== undefined && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-yellow-400"
                    style={{ left: `${50 + (f.benchmark / maxAbs) * 50}%` }}
                  />
                )}
              </div>
              <span className="text-xs w-12 text-right" style={{ color }}>
                {f.value.toFixed(decimals)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ===========================
   RegressionBuilder (pure TSX OLS)
=========================== */
// minimal matrix helpers
type Matrix = number[][];
function transpose(A: Matrix): Matrix {
  const m = A.length, n = A[0].length;
  const T: Matrix = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j];
  return T;
}
function matmul(A: Matrix, B: Matrix): Matrix {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C: Matrix = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      for (let j = 0; j < n; j++) C[i][j] += a * B[p][j];
    }
  }
  return C;
}
function matvec(A: Matrix, x: number[]): number[] {
  const y = new Array(A.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    for (let j = 0; j < A[0].length; j++) s += A[i][j] * x[j];
    y[i] = s;
  }
  return y;
}
function invert(M: Matrix): Matrix {
  const n = M.length;
  const A: Matrix = M.map((r) => r.slice());
  const I: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) throw new Error("Singular matrix");
    [A[c], A[p]] = [A[p], A[c]];
    [I[c], I[p]] = [I[p], I[c]];
    const d = A[c][c];
    for (let j = 0; j < n; j++) { A[c][j] /= d; I[c][j] /= d; }
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = A[r][c];
      for (let j = 0; j < n; j++) { A[r][j] -= f * A[c][j]; I[r][j] -= f * I[c][j]; }
    }
  }
  return I;
}
function mean(a: number[]): number {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length;
}
function ols(y: number[], X: Matrix, colNames: string[]): RegressionModel {
  const n = y.length;
  const k = X[0].length;
  const XT = transpose(X);
  const XTX = matmul(XT, X);
  const XTXi = invert(XTX);
  const XTy = matvec(XT, y).map((v) => [v]);
  const B = matmul(XTXi, XTy).map((r) => r[0]); // coefficients
  const yhat = matvec(X, B);
  const resid = y.map((v, i) => v - yhat[i]);
  const ssTot = y.reduce((s, v) => s + (v - mean(y)) ** 2, 0);
  const ssRes = resid.reduce((s, e) => s + e * e, 0);
  const r2 = 1 - ssRes / ssTot;
  const adjR2 = 1 - (1 - r2) * ((n - 1) / (n - k));
  const sigma2 = ssRes / (n - k);
  const varB = XTXi.map((row, i) => row[i] * sigma2);
  const stderrB = varB.map((v) => Math.sqrt(v));
  const tstats = B.map((b, i) => b / (stderrB[i] || 1e-12));

  const coefMap: Record<string, number> = {};
  const tMap: Record<string, number> = {};
  let alpha = 0;

  colNames.forEach((name, idx) => {
    if (name === "__intercept__") alpha = B[idx];
    else { coefMap[name] = B[idx]; tMap[name] = tstats[idx]; }
  });

  return {
    target: "",
    factors: colNames.filter((n) => n !== "__intercept__"),
    intercept: colNames.includes("__intercept__"),
    n,
    alpha,
    betas: coefMap,
    tstats: tMap,
    r2,
    adjR2,
    stderr: Math.sqrt(sigma2),
    residuals: resid,
    yhat,
  };
}
function align(series: SeriesMap, names: string[]): Record<string, number[]> {
  const lens = names.map((n) => series[n]?.length ?? 0);
  const L = Math.min(...lens);
  const out: Record<string, number[]> = {};
  names.forEach((n) => (out[n] = (series[n] || []).slice(-L)));
  return out;
}
function RegressionBuilder({
  series,
  defaultTarget,
  defaultFactors,
  defaultIntercept = true,
  decimals = 4,
  onFit,
}: {
  series: SeriesMap;
  defaultTarget?: string;
  defaultFactors?: string[];
  defaultIntercept?: boolean;
  decimals?: number;
  onFit?: (m: RegressionModel) => void;
}) {
  const allKeys = Object.keys(series);
  const [target, setTarget] = useState<string>(defaultTarget ?? allKeys[0] ?? "");
  const [factors, setFactors] = useState<string[]>(
    defaultFactors ?? allKeys.filter((k) => k !== target).slice(0, 3)
  );
  const [useIntercept, setUseIntercept] = useState<boolean>(defaultIntercept);

  const model = useMemo<RegressionModel | null>(() => {
    if (!target || factors.length === 0) return null;
    const fields = [target, ...factors];
    const aligned = align(series, fields);
    const y = aligned[target];
    const Xcols = factors.map((f) => aligned[f]);
    const n = y.length;
    const k = Xcols.length + (useIntercept ? 1 : 0);
    const X: Matrix = Array.from({ length: n }, () => Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      let c = 0;
      if (useIntercept) X[i][c++] = 1;
      for (let j = 0; j < Xcols.length; j++) X[i][c++] = Xcols[j][i];
    }
    try {
      const m = ols(y, X, [...(useIntercept ? ["__intercept__"] : []), ...factors]);
      const withLabels: RegressionModel = { ...m, target };
      onFit?.(withLabels);
      return withLabels;
    } catch {
      return null;
    }
  }, [series, target, factors.join("|"), useIntercept, onFit]);

  return (
    <div className="bg-[#0b0b0b] p-4 rounded-lg border border-[#222] text-gray-200">
      <h2 className="text-sm font-semibold mb-3">Regression Builder (OLS)</h2>

      <div className="flex flex-wrap items-end gap-3 mb-4">
        <div>
          <label className="text-xs block mb-1 opacity-80">Target</label>
          <select
            className="bg-[#121212] border border-[#333] rounded px-2 py-1"
            value={target}
            onChange={(e) => {
              const t = e.target.value;
              setTarget(t);
              if (factors.includes(t)) setFactors(factors.filter((x) => x !== t));
            }}
          >
            {allKeys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>

        <div className="min-w-[260px]">
          <label className="text-xs block mb-1 opacity-80">Factors</label>
          <div className="bg-[#121212] border border-[#333] rounded p-2 max-h-40 overflow-auto">
            {allKeys
              .filter((k) => k !== target)
              .map((opt) => {
                const checked = factors.includes(opt);
                return (
                  <label key={opt} className="flex items-center gap-2 text-xs py-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        if (e.target.checked) setFactors([...factors, opt]);
                        else setFactors(factors.filter((x) => x !== opt));
                      }}
                    />
                    <span className="truncate">{opt}</span>
                  </label>
                );
              })}
          </div>
        </div>

        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={useIntercept}
            onChange={(e) => setUseIntercept(e.target.checked)}
          />
          Include intercept
        </label>
      </div>

      {model ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <Stat label="n" value={model.n} />
            <Stat label="R²" value={model.r2.toFixed(4)} />
            <Stat label="Adj R²" value={model.adjR2.toFixed(4)} />
            <Stat label="StdErr" value={model.stderr.toFixed(4)} />
          </div>

          <table className="w-full text-sm border border-[#2a2a2a]">
            <thead className="bg-[#111]">
              <tr>
                <th className="text-left p-2 border-b border-[#2a2a2a]">Coefficient</th>
                <th className="text-right p-2 border-b border-[#2a2a2a]">Beta</th>
                <th className="text-right p-2 border-b border-[#2a2a2a]">t-stat</th>
              </tr>
            </thead>
            <tbody>
              {model.intercept && (
                <tr>
                  <td className="p-2 border-b border-[#1f1f1f] text-gray-400">Intercept</td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">
                    {model.alpha.toFixed(4)}
                  </td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">—</td>
                </tr>
              )}
              {model.factors.map((f) => (
                <tr key={f}>
                  <td className="p-2 border-b border-[#1f1f1f]">{f}</td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">
                    {model.betas[f].toFixed(4)}
                  </td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">
                    {model.tstats[f].toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mt-3">
            <div className="text-xs mb-1 opacity-80">Residuals (last 100)</div>
            <div className="flex gap-[2px] h-12 items-end">
              {model.residuals.slice(-100).map((r, i) => {
                const h = Math.min(48, Math.abs(r) * 2000);
                const color = r >= 0 ? "#16a34a" : "#dc2626";
                return <div key={i} style={{ width: 2, height: Math.max(2, h), background: color }} />;
              })}
            </div>
          </div>
        </>
      ) : (
        <div className="text-sm text-gray-400">Select a target and at least one factor to fit.</div>
      )}
    </div>
  );
}
function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-[#111] border border-[#222] rounded p-2">
      <div className="text-[11px] opacity-70">{label}</div>
      <div className="text-sm">{value}</div>
    </div>
  );
}

/* ===========================
   PAGE: compose everything
=========================== */
export default function AnalyticsWorkspacePage() {
  const [tickers, setTickers] = useState<string[]>([
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "GOOG",
    "NVDA",
  ]);
  const [target, setTarget] = useState<string>("AAPL");
  const [factorSet, setFactorSet] = useState<string[]>(["MKT", "SMB", "HML", "MOM"]);
  const [len, setLen] = useState<number>(252);

  const series = useMemo(() => makeDemoSeries(tickers, len), [tickers.join(","), len]);
  const factorSeries = useMemo(() => makeDemoSeries(factorSet, len), [factorSet.join("|"), len]);

  const combinedSeries: SeriesMap = useMemo(
    () => ({ ...series, ...factorSeries }),
    [series, factorSeries]
  );

  const [model, setModel] = useState<RegressionModel | null>(null);
  const factorExposures: FactorRow[] = useMemo(() => {
    if (!model) return [];
    return model.factors.map((f) => ({
      name: f,
      value: model.betas[f],
      benchmark: f === "MKT" ? 1 : undefined,
    }));
  }, [model]);

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* toolbar */}
      <div className="bg-[#0b0b0b] border border-[#222] rounded-lg p-3 flex flex-wrap gap-3 items-end">
        <div className="flex flex-col">
          <label className="text-xs text-gray-400 mb-1">Tickers (comma-separated)</label>
          <input
            className="bg-[#121212] border border-[#333] rounded px-2 py-1 min-w-[360px]"
            value={tickers.join(",")}
            onChange={(e) =>
              setTickers(
                e.target.value
                  .split(",")
                  .map((s) => s.trim().toUpperCase())
                  .filter(Boolean)
              )
            }
            placeholder="AAPL, MSFT, AMZN"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-xs text-gray-400 mb-1">Target</label>
          <select
            className="bg-[#121212] border border-[#333] rounded px-2 py-1"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-xs text-gray-400 mb-1">Factors (comma-separated)</label>
          <input
            className="bg-[#121212] border border-[#333] rounded px-2 py-1 min-w-[260px]"
            value={factorSet.join(",")}
            onChange={(e) =>
              setFactorSet(
                e.target.value
                  .split(",")
                  .map((s) => s.trim().toUpperCase())
                  .filter(Boolean)
              )
            }
            placeholder="MKT, SMB, HML, MOM"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-xs text-gray-400 mb-1">Lookback (days)</label>
          <input
            type="number"
            min={30}
            max={2000}
            className="bg-[#121212] border border-[#333] rounded px-2 py-1 w-28"
            value={len}
            onChange={(e) => setLen(Math.max(30, Math.min(2000, Number(e.target.value))))}
          />
        </div>
      </div>

      {/* top row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1">
          <FactorsPanel factors={factorExposures} />
        </div>
        <div className="lg:col-span-2">
          <CorrelationMatrix series={series} height={520} />
        </div>
      </div>

      {/* bottom row */}
      <div className="bg-[#0b0b0b] border border-[#222] rounded-lg p-3">
        <RegressionBuilder
          series={combinedSeries}
          defaultTarget={target}
          defaultFactors={factorSet}
          defaultIntercept={true}
          decimals={4}
          onFit={(m) => setModel(m)}
        />
      </div>
    </div>
  );
}