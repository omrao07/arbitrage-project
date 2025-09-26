"use client";

import React, { useMemo, useState } from "react";

/* ================================
 * Types
 * ================================ */

export type Multiples = {
  /** trailing/forward multiples for peers you want to compare against */
  evEbitda?: number; // peer average
  pe?: number;
  ps?: number;
};

export type Snapshot = {
  ticker: string;
  name?: string;
  sharesOut: number;       // fully diluted shares
  price: number;           // current price
  netDebt?: number;        // (Debt - Cash), can be negative
  fcf0?: number;           // last FY free cash flow
  revenue0?: number;       // optional, for PS cross-check
  ebitda0?: number;        // optional, for EV/EBITDA cross-check
  eps0?: number;           // optional, for P/E cross-check
};

export type DCFParams = {
  years?: number;          // explicit forecast years
  wacc?: number;           // discount rate (e.g., 0.09)
  growth?: number;         // base FCF CAGR in explicit period
  fadeTo?: number;         // growth that we fade toward by year N (optional)
  exitMode?: "perpetuity" | "exitMultiple";
  exitG?: number;          // Gordon growth in perpetuity mode
  exitEVx?: number;        // EV/EBITDA exit multiple if exitMultiple
  ebitdaMargin?: number;   // to infer EBITDA from revenue or FCF if needed
};

export interface ValuationProps {
  snap: Snapshot;
  peers?: Multiples;       // comparable multiples (peer averages/median)
  title?: string;
}

/* ================================
 * Local UI (no deps)
 * ================================ */

const Card: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`rounded-2xl border border-neutral-200/70 bg-white shadow ${className ?? ""}`}>{children}</div>
);
const CardHeader: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3 ${className ?? ""}`}>
    {children}
  </div>
);
const CardTitle: React.FC<React.PropsWithChildren> = ({ children }) => (
  <h2 className="text-lg font-semibold">{children}</h2>
);
const CardContent: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`px-4 py-3 ${className ?? ""}`}>{children}</div>
);
const Badge: React.FC<React.PropsWithChildren<{ tone?: "neutral" | "green" | "red" | "indigo" }>> = ({
  children,
  tone = "neutral",
}) => {
  const tones: Record<string, string> = {
    neutral: "bg-neutral-100 text-neutral-800",
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    indigo: "bg-indigo-100 text-indigo-800",
  };
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
};

/* ================================
 * Math helpers
 * ================================ */

const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const fmtB = (x: number, d = 2) => (Number.isFinite(x) ? (x / 1e9).toFixed(d) + "B" : "–");
const fmtPct = (x: number, d = 1) => (Number.isFinite(x) ? `${(x * 100).toFixed(d)}%` : "–");
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

function arr(n: number, fn: (i: number) => number): number[] {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = fn(i);
  return a;
}

function npv(rate: number, cashflows: number[]): number {
  return cashflows.reduce((acc, cf, t) => acc + cf / Math.pow(1 + rate, t + 1), 0);
}

/** simple linear fade from g0 -> gN across N years */
function fadedGrowth(years: number, g0: number, gN?: number): number[] {
  if (gN === undefined || gN === null) return arr(years, () => g0);
  return arr(years, (i) => g0 + (i / Math.max(1, years - 1)) * (gN - g0));
}

function projectFCF(fcf0: number, gs: number[]): number[] {
  let f = fcf0;
  return gs.map((g) => (f = f * (1 + g)));
}

/* ================================
 * Component
 * ================================ */

const ValuationPanel: React.FC<ValuationProps> = ({ snap, peers, title = "Valuation" }) => {
  // --- controls (defaults are conservative-ish) ---
  const [years, setYears] = useState<number>(5);
  const [wacc, setWacc] = useState<number>(0.09);
  const [growth, setGrowth] = useState<number>(0.08);
  const [fadeTo, setFadeTo] = useState<number>(0.03);
  const [exitMode, setExitMode] = useState<"perpetuity" | "exitMultiple">("perpetuity");
  const [exitG, setExitG] = useState<number>(0.025);
  const [exitEVx, setExitEVx] = useState<number>(12);
  const [ebitdaMargin, setEbitdaMargin] = useState<number>(0.28);

  // --- core DCF ---
  const dcf = useMemo(() => {
    const y = clamp(years, 3, 10);
    const gs = fadedGrowth(y, growth, fadeTo);
    const fcf0 = snap.fcf0 ?? Math.max(0, (snap.ebitda0 ?? 0) * 0.6); // fallback
    const fcf = projectFCF(fcf0, gs); // years 1..N

    // terminal
    let terminalEV = 0;
    if (exitMode === "perpetuity") {
      const last = fcf[fcf.length - 1];
      // Gordon: EV_T = FCF_{T+1} / (WACC - g)
      const tv = (last * (1 + exitG)) / Math.max(1e-9, wacc - exitG);
      terminalEV = tv / Math.pow(1 + wacc, y);
    } else {
      // Exit multiple on EBITDA in year N
      const rev0 = snap.revenue0 ?? (fcf0 / 0.1); // rough fallback
      const revN = rev0 * Math.pow(1 + growth, y);
      const ebitdaN = revN * ebitdaMargin;
      const EV_T = ebitdaN * exitEVx;
      terminalEV = EV_T / Math.pow(1 + wacc, y);
    }

    const pv = npv(wacc, fcf) + terminalEV;
    const netDebt = snap.netDebt ?? 0;
    const equityValue = pv - netDebt;
    const fair = equityValue / Math.max(1e-9, snap.sharesOut);
    return {
      fcf,
      terminalEV,
      enterpriseValuePV: pv,
      equityValue,
      fair,
    };
  }, [years, wacc, growth, fadeTo, exitMode, exitG, exitEVx, ebitdaMargin, snap]);

  // --- multiples cross-check ---
  const multiples = useMemo(() => {
    const out: { label: string; fair?: number }[] = [];
    const sh = Math.max(1e-9, snap.sharesOut);
    const netDebt = snap.netDebt ?? 0;

    if (peers?.pe && Number.isFinite(snap.eps0)) {
      out.push({ label: "P/E", fair: peers.pe! * (snap.eps0 as number) });
    }
    if (peers?.ps && Number.isFinite(snap.revenue0)) {
      const mktCap = peers.ps! * (snap.revenue0 as number);
      out.push({ label: "P/S", fair: mktCap / sh });
    }
    if (peers?.evEbitda) {
      const ebitda = snap.ebitda0 ?? (snap.revenue0 ?? 0) * ebitdaMargin;
      const EV = peers.evEbitda * ebitda;
      const eq = (EV - netDebt) / sh;
      out.push({ label: "EV/EBITDA", fair: eq });
    }
    return out;
  }, [peers, snap, ebitdaMargin]);

  // --- blended fair value ---
  const blendFair = useMemo(() => {
    const vals = [dcf.fair, ...multiples.map((m) => m.fair!).filter((x) => Number.isFinite(x))];
    if (vals.length === 0) return NaN;
    // median is more robust than mean
    const s = [...vals].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
  }, [dcf.fair, multiples]);

  const upside = useMemo(() => (Number.isFinite(blendFair) ? (blendFair - snap.price) / snap.price : NaN), [blendFair, snap.price]);

  // --- sensitivity grid (WACC x Exit growth) ---
  const sensGrid = useMemo(() => {
    const waccs = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02].map((x) => clamp(x, 0.04, 0.20));
    const gs = [exitG - 0.01, exitG - 0.005, exitG, exitG + 0.005, exitG + 0.01].map((x) => clamp(x, 0.0, 0.05));

    const fcf0 = snap.fcf0 ?? Math.max(0, (snap.ebitda0 ?? 0) * 0.6);
    const fcf = projectFCF(fcf0, fadedGrowth(years, growth, fadeTo));
    const last = fcf[fcf.length - 1];
    const sh = Math.max(1e-9, snap.sharesOut);
    const nd = snap.netDebt ?? 0;

    const table = gs.map((g) =>
      waccs.map((r) => {
        const tv = (last * (1 + g)) / Math.max(1e-9, r - g);
        const pv = npv(r, fcf) + tv / Math.pow(1 + r, years);
        const eq = (pv - nd) / sh;
        return eq;
      })
    );

    return { waccs, gs, table };
  }, [wacc, exitG, snap, years, growth, fadeTo]);

  // --- helpers ---
  const tone = (x: number) => (x >= 0 ? "green" : "red");

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <Badge tone="indigo">{snap.ticker}</Badge>
          <span className="text-neutral-600">{snap.name ?? ""}</span>
          <span className="mx-2 text-neutral-400">•</span>
          <span className="text-neutral-600">Price</span>
          <span className="font-mono">${fmt(snap.price, 2)}</span>
          <span className="mx-2 text-neutral-400">•</span>
          <span className="text-neutral-600">Shares</span>
          <span className="font-mono">{fmt(snap.sharesOut / 1e6, 1)}M</span>
        </div>
      </CardHeader>

      <CardContent className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {/* Controls */}
        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="mb-2 text-xs uppercase text-neutral-500">DCF Controls</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <label className="flex items-center justify-between gap-3">
              <span>Years</span>
              <input
                type="number"
                min={3}
                max={10}
                value={years}
                onChange={(e) => setYears(Number(e.target.value))}
                className="h-8 w-24 rounded-md border border-neutral-300 px-2"
              />
            </label>
            <label className="flex items-center justify-between gap-3">
              <span>WACC</span>
              <input
                type="number"
                step="0.001"
                value={wacc}
                onChange={(e) => setWacc(Number(e.target.value))}
                className="h-8 w-24 rounded-md border border-neutral-300 px-2"
              />
            </label>
            <label className="flex items-center justify-between gap-3">
              <span>CAGR (yrs 1-N)</span>
              <input
                type="number"
                step="0.001"
                value={growth}
                onChange={(e) => setGrowth(Number(e.target.value))}
                className="h-8 w-24 rounded-md border border-neutral-300 px-2"
              />
            </label>
            <label className="flex items-center justify-between gap-3">
              <span>Fade to</span>
              <input
                type="number"
                step="0.001"
                value={fadeTo}
                onChange={(e) => setFadeTo(Number(e.target.value))}
                className="h-8 w-24 rounded-md border border-neutral-300 px-2"
              />
            </label>

            <label className="col-span-2 flex items-center justify-between gap-3">
              <span>Exit mode</span>
              <select
                value={exitMode}
                onChange={(e) => setExitMode(e.target.value as "perpetuity" | "exitMultiple")}
                className="h-8 w-40 rounded-md border border-neutral-300 px-2"
              >
                <option value="perpetuity">Perpetuity (Gordon)</option>
                <option value="exitMultiple">Exit Multiple</option>
              </select>
            </label>

            {exitMode === "perpetuity" ? (
              <label className="col-span-2 flex items-center justify-between gap-3">
                <span>Exit growth (g)</span>
                <input
                  type="number"
                  step="0.001"
                  value={exitG}
                  onChange={(e) => setExitG(Number(e.target.value))}
                  className="h-8 w-24 rounded-md border border-neutral-300 px-2"
                />
              </label>
            ) : (
              <>
                <label className="flex items-center justify-between gap-3">
                  <span>Exit EV/EBITDA</span>
                  <input
                    type="number"
                    step="0.1"
                    value={exitEVx}
                    onChange={(e) => setExitEVx(Number(e.target.value))}
                    className="h-8 w-24 rounded-md border border-neutral-300 px-2"
                  />
                </label>
                <label className="flex items-center justify-between gap-3">
                  <span>EBITDA margin</span>
                  <input
                    type="number"
                    step="0.001"
                    value={ebitdaMargin}
                    onChange={(e) => setEbitdaMargin(Number(e.target.value))}
                    className="h-8 w-24 rounded-md border border-neutral-300 px-2"
                  />
                </label>
              </>
            )}
          </div>
        </div>

        {/* DCF Output */}
        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="mb-2 text-xs uppercase text-neutral-500">DCF Output</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex items-center justify-between">
              <span>PV (Enterprise)</span>
              <span className="font-mono">{fmtB(dcf.enterpriseValuePV)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Terminal (PV)</span>
              <span className="font-mono">{fmtB(dcf.terminalEV)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Equity Value</span>
              <span className="font-mono">{fmtB(dcf.equityValue)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Fair Value / Share</span>
              <span className="font-mono">${fmt(dcf.fair, 2)}</span>
            </div>
            <div className="col-span-2 mt-2 rounded-md border p-2">
              <div className="mb-1 text-xs uppercase text-neutral-500">FCF Forecast</div>
              <div className="flex flex-wrap gap-2 font-mono text-xs">
                {dcf.fcf.map((v, i) => (
                  <span key={i} className="rounded bg-neutral-50 px-2 py-0.5">
                    Y{i + 1}: {fmtB(v, 2)}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Summary & Multiples */}
        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="mb-2 text-xs uppercase text-neutral-500">Summary</div>
          <div className="flex items-center justify-between text-sm">
            <span>Blended Fair Value</span>
            <span className="font-mono">${fmt(blendFair, 2)}</span>
          </div>
          <div className="mt-1 flex items-center justify-between text-sm">
            <span>Upside vs Price</span>
            <span className={`font-mono ${Number.isFinite(upside) && upside >= 0 ? "text-green-600" : "text-red-600"}`}>
              {fmtPct(upside)}
            </span>
          </div>

          <div className="mt-3 mb-1 text-xs uppercase text-neutral-500">Multiples Cross-Check</div>
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr className="border-b bg-neutral-50 text-left text-xs font-semibold uppercase tracking-wide text-neutral-600">
                <th className="px-2 py-1">Method</th>
                <th className="px-2 py-1 text-right">Fair Px</th>
                <th className="px-2 py-1 text-right">Spread</th>
              </tr>
            </thead>
            <tbody>
              {multiples.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-2 py-3 text-center text-neutral-500">
                    Add peer multiples or EPS/Revenue to see cross-checks.
                  </td>
                </tr>
              ) : (
                multiples.map((m) => {
                  const spread = Number.isFinite(m.fair!) ? (m.fair! - snap.price) / snap.price : NaN;
                  return (
                    <tr key={m.label} className="border-b last:border-0">
                      <td className="px-2 py-1">{m.label}</td>
                      <td className="px-2 py-1 text-right font-mono">${fmt(m.fair ?? NaN, 2)}</td>
                      <td className={`px-2 py-1 text-right font-mono ${Number.isFinite(spread) ? (spread >= 0 ? "text-green-600" : "text-red-600") : ""}`}>
                        {fmtPct(spread)}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </CardContent>

      {/* Sensitivity grid */}
      <CardContent>
        <div className="mb-2 text-xs uppercase text-neutral-500">Perpetuity Sensitivity (Fair / Share)</div>
        <div className="overflow-auto">
          <table className="min-w-[520px] border-collapse text-sm">
            <thead>
              <tr>
                <th className="px-2 py-1 text-left">g ↓ / WACC →</th>
                {sensGrid.waccs.map((r, i) => (
                  <th key={i} className="px-2 py-1 text-right font-mono">{fmtPct(r)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sensGrid.gs.map((g, ri) => (
                <tr key={ri}>
                  <td className="px-2 py-1 font-mono">{fmtPct(g)}</td>
                  {sensGrid.table[ri].map((fv, ci) => {
                    const spr = (fv - snap.price) / snap.price;
                    return (
                      <td key={ci} className={`px-2 py-1 text-right font-mono ${tone(spr) === "green" ? "text-green-600" : "text-red-600"}`}>
                        ${fmt(fv, 2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

export default ValuationPanel;