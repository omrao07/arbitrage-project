"use client";

import React from "react";

export default function DashboardPage() {
  // Lazy require so no static imports at the top
  const PNLSparkline = require("../analytics/_components/PNLSparkline").default;
  const PositionsSnapshot = require("../portfolio/_components/PositionsSnapshot").default;
  const RiskLights = require("../risk/_components/RiskLights").default;
  const TopMovers = require("../market/_components/TopMovers").default;

  return (
    <div className="min-h-screen bg-[#0b0b0b] text-gray-100 px-6 py-6">
      <h1 className="text-xl font-semibold mb-6">Trading Dashboard</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Portfolio snapshot */}
        <div className="lg:col-span-2">
          <PositionsSnapshot
            positions={[
              {
                id: "1",
                symbol: "AAPL",
                name: "Apple",
                assetClass: "Equity",
                region: "US",
                quantity: 100,
                avgPrice: 175,
                marketPrice: 182,
                dayPnl: 120,
                dayPct: 0.66,
                currency: "USD",
              },
              {
                id: "2",
                symbol: "BTC-USD",
                name: "Bitcoin",
                assetClass: "Crypto",
                region: "Global",
                quantity: 0.5,
                avgPrice: 40000,
                marketPrice: 45000,
                dayPnl: 500,
                dayPct: 1.1,
                currency: "USD",
              },
            ]}
            baseCurrency="USD"
            title="Portfolio Snapshot"
          />
        </div>

        {/* Risk lights */}
        <div>
          <RiskLights
            title="Risk Lights"
            items={[
              {
                id: "var",
                label: "1d VaR",
                value: 1.25,
                unit: "%",
                thresholds: { ok: 1.5, warn: 2.5, direction: "higher" },
              },
              {
                id: "lev",
                label: "Leverage",
                value: 1.8,
                unit: "x",
                thresholds: { ok: 2.0, warn: 3.0, direction: "higher" },
              },
              {
                id: "dd",
                label: "MTD Drawdown",
                value: -2.4,
                unit: "%",
                thresholds: { ok: 0, warn: -3, direction: "lower" },
                format: (v: number) => `${v.toFixed(2)}%`,
              },
            ]}
          />
        </div>
      </div>

      {/* Movers + PnL sparkline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        <div className="lg:col-span-2">
          <TopMovers autoFetch endpoint="/api/market/top-movers" universe="US" />
        </div>

        <div className="bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg p-4">
          <h2 className="text-sm font-semibold mb-2">PnL Trend</h2>
          <PNLSparkline
            data={[-1.2, -0.4, 0.6, 1.4, 0.8, 2.0]}
            width={260}
            height={80}
            baseline={0}
            fill
            smooth
            showLastValue
            format={(v: number) =>
              v >= 0 ? `+$${v.toFixed(2)}k` : `-$${Math.abs(v).toFixed(2)}k`
            }
          />
        </div>
      </div>
    </div>
  );
}