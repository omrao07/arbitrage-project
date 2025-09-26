"use client";

import React, { useMemo, useState } from "react";

/** ---------- shared types ---------- */
export type SeriesMap = Record<string, number[]>;

export type RegressionModel = {
  target: string;
  factors: string[];
  intercept: boolean;
  n: number;
  betas: Record<string, number>;    // factor -> beta
  alpha: number;                    // intercept
  r2: number;
  adjR2: number;
  tstats: Record<string, number>;   // factor -> t
  stderr: number;                   // residual std error
  residuals: number[];              // y - y_hat
  yhat: number[];                   // fitted
};

export type RegressionBuilderProps = {
  series: SeriesMap;                        // e.g., { AAPL:[...], MKT:[...], SMB:[...] }
  defaultTarget?: string;
  defaultFactors?: string[];
  defaultIntercept?: boolean;
  decimals?: number;
  onFit?: (model: RegressionModel) => void; // called whenever model fits
};

/** ---------- small numeric helpers (no deps) ---------- */

function dropNa(a: number[]): number[] {
  return a.filter(x => Number.isFinite(x));
}

/** zscore for standardization (optional if you add a toggle later) */
function mean(a: number[]): number {
  let s = 0; const n = a.length;
  for (let i = 0; i < n; i++) s += a[i];
  return s / n;
}
function variance(a: number[], m = mean(a)): number {
  let s2 = 0; const n = a.length;
  for (let i = 0; i < n; i++) { const d = a[i] - m; s2 += d * d; }
  return s2 / (n - 1);
}

/** matrix ops */
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
  const m = A.length, n = A[0].length;
  const y = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i][j] * x[j];
    y[i] = s;
  }
  return y;
}
/** Gauss-Jordan inverse (for small k <= ~20) */
function invert(M: Matrix): Matrix {
  const n = M.length;
  const A: Matrix = M.map(row => row.slice());
  const I: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );
  for (let col = 0; col < n; col++) {
    // find pivot
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
    if (Math.abs(A[pivot][col]) < 1e-12) throw new Error("Matrix not invertible (singular).");
    // swap
    [A[col], A[pivot]] = [A[pivot], A[col]];
    [I[col], I[pivot]] = [I[pivot], I[col]];
    // normalize row
    const d = A[col][col];
    for (let j = 0; j < n; j++) { A[col][j] /= d; I[col][j] /= d; }
    // eliminate others
    for (let r = 0; r < n; r++) if (r !== col) {
      const f = A[r][col];
      for (let j = 0; j < n; j++) {
        A[r][j] -= f * A[col][j];
        I[r][j] -= f * I[col][j];
      }
    }
  }
  return I;
}

/** OLS via normal equations: beta = (X'X)^-1 X'y */
function ols(y: number[], X: Matrix, colNames: string[]): RegressionModel {
  const n = y.length;
  const k = X[0].length;

  const XT = transpose(X);
  const XTX = matmul(XT, X);
  const XTXi = invert(XTX);
  const XTy = matvec(XT, y).map(v => [v]); // as column
  const B = matmul(XTXi, XTy).map(r => r[0]); // coefficients (k)

  // fitted values & residuals
  const yhat = matvec(X, B);
  const resid = y.map((v, i) => v - yhat[i]);

  // stats
  const ssTot = y.reduce((s, v) => s + (v - mean(y)) ** 2, 0);
  const ssRes = resid.reduce((s, e) => s + e * e, 0);
  const r2 = 1 - ssRes / ssTot;
  const adjR2 = 1 - (1 - r2) * ((n - 1) / (n - k));

  // variance of residuals
  const sigma2 = ssRes / (n - k); // MSE
  // Var(beta) = sigma^2 * (X'X)^-1  -> stdErr = sqrt(diag())
  const varB = XTXi.map((row, i) => row[i] * sigma2);
  const stderrB = varB.map(v => Math.sqrt(v));
  const tstats = B.map((b, i) => b / (stderrB[i] || 1e-12));

  const coefMap: Record<string, number> = {};
  const tMap: Record<string, number> = {};

  colNames.forEach((name, idx) => {
    if (name === "__intercept__") return;
    coefMap[name] = B[idx];
    tMap[name] = tstats[idx];
  });

  const alphaIdx = colNames.indexOf("__intercept__");
  const alpha = alphaIdx >= 0 ? B[alphaIdx] : 0;

  return {
    target: "",
    factors: colNames.filter(n => n !== "__intercept__"),
    intercept: alphaIdx >= 0,
    n,
    betas: coefMap,
    alpha,
    r2,
    adjR2,
    tstats: tMap,
    stderr: Math.sqrt(sigma2),
    residuals: resid,
    yhat,
  };
}

/** Align arrays by minimum common length. */
function align(series: SeriesMap, names: string[]): Record<string, number[]> {
  const lens = names.map(n => series[n]?.length ?? 0);
  const L = Math.min(...lens);
  const out: Record<string, number[]> = {};
  names.forEach(n => out[n] = (series[n] || []).slice(-L));
  return out;
}

/** ---------- component ---------- */
export default function RegressionBuilder({
  series,
  defaultTarget,
  defaultFactors,
  defaultIntercept = true,
  decimals = 4,
  onFit,
}: RegressionBuilderProps) {
  const allKeys = Object.keys(series);
  const [target, setTarget] = useState<string>(defaultTarget ?? allKeys[0] ?? "");
  const [selected, setSelected] = useState<string[]>(
    defaultFactors ?? allKeys.filter(k => k !== target).slice(0, 3)
  );
  const [useIntercept, setUseIntercept] = useState<boolean>(defaultIntercept);

  const model = useMemo<RegressionModel | null>(() => {
    if (!target || selected.length === 0) return null;

    const fields = [target, ...selected];
    const aligned = align(series, fields);
    const y = aligned[target];
    const Xcols = selected.map(f => aligned[f]);

    // build design matrix
    const n = y.length;
    const k = Xcols.length + (useIntercept ? 1 : 0);
    const X: Matrix = Array.from({ length: n }, () => Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      let c = 0;
      if (useIntercept) X[i][c++] = 1;
      for (let j = 0; j < Xcols.length; j++) X[i][c++] = Xcols[j][i];
    }

    try {
      const m = ols(y, X, [
        ...(useIntercept ? ["__intercept__"] : []),
        ...selected,
      ]);
      const withLabels: RegressionModel = { ...m, target };
      onFit?.(withLabels);
      return withLabels;
    } catch (e) {
      console.error("OLS error:", e);
      return null;
    }
  }, [series, target, selected.join("|"), useIntercept, onFit]);

  return (
    <div className="bg-[#0b0b0b] p-4 rounded-lg border border-[#222] text-gray-200">
      <h2 className="text-sm font-semibold mb-3">Regression Builder (OLS)</h2>

      {/* controls */}
      <div className="flex flex-wrap items-end gap-3 mb-4">
        <div>
          <label className="text-xs block mb-1 opacity-80">Target</label>
          <select
            className="bg-[#121212] border border-[#333] rounded px-2 py-1"
            value={target}
            onChange={(e) => {
              const t = e.target.value;
              setTarget(t);
              if (selected.includes(t)) {
                setSelected(selected.filter(x => x !== t));
              }
            }}
          >
            {allKeys.map(k => <option key={k} value={k}>{k}</option>)}
          </select>
        </div>

        <div className="min-w-[260px]">
          <label className="text-xs block mb-1 opacity-80">Factors</label>
          <MultiSelect
            options={allKeys.filter(k => k !== target)}
            value={selected}
            onChange={setSelected}
          />
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

      {/* results */}
      {model ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <Stat label="n" value={model.n} />
            <Stat label="R²" value={model.r2.toFixed(4)} />
            <Stat label="Adj R²" value={model.adjR2.toFixed(4)} />
            <Stat label="StdErr" value={model.stderr.toFixed(decimals)} />
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
                    {model.alpha.toFixed(decimals)}
                  </td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">—</td>
                </tr>
              )}
              {model.factors.map((f) => (
                <tr key={f}>
                  <td className="p-2 border-b border-[#1f1f1f]">{f}</td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">
                    {model.betas[f].toFixed(decimals)}
                  </td>
                  <td className="p-2 text-right border-b border-[#1f1f1f]">
                    {model.tstats[f].toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* tiny residuals sparkline (pure CSS) */}
          <div className="mt-3">
            <div className="text-xs mb-1 opacity-80">Residuals (last 100)</div>
            <div className="flex gap-[2px] h-12 items-end">
              {model.residuals.slice(-100).map((r, i) => {
                const h = Math.min(48, Math.abs(r) * 2000); // scale
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

/** ---------- small UI bits ---------- */

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-[#111] border border-[#222] rounded p-2">
      <div className="text-[11px] opacity-70">{label}</div>
      <div className="text-sm">{value}</div>
    </div>
  );
}

function MultiSelect({
  options,
  value,
  onChange,
}: {
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
}) {
  return (
    <div className="bg-[#121212] border border-[#333] rounded p-2 max-h-40 overflow-auto">
      {options.map((opt) => {
        const checked = value.includes(opt);
        return (
          <label key={opt} className="flex items-center gap-2 text-xs py-1 cursor-pointer">
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => {
                if (e.target.checked) onChange([...value, opt]);
                else onChange(value.filter((x) => x !== opt));
              }}
            />
            <span className="truncate">{opt}</span>
          </label>
        );
      })}
    </div>
  );
}