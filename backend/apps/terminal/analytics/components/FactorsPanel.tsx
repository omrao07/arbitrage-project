"use client";

import React from "react";

export type Factor = {
  name: string;         // "Market Beta"
  value: number;        // +0.85, -0.22 etc.
  benchmark?: number;   // optional comparison line
};

export type FactorsPanelProps = {
  factors: Factor[];
  decimals?: number;           // default 2
  onClickFactor?: (f: Factor) => void;
};

export default function FactorsPanel({
  factors,
  decimals = 2,
  onClickFactor,
}: FactorsPanelProps) {
  if (!factors || factors.length === 0) {
    return (
      <div className="bg-[#0b0b0b] text-gray-400 p-4 rounded border border-[#222]">
        No factors available
      </div>
    );
  }

  const maxAbs = Math.max(...factors.map(f => Math.abs(f.value)), 1);

  return (
    <div className="bg-[#0b0b0b] p-4 rounded-lg shadow-lg">
      <h2 className="text-sm text-gray-300 mb-3 font-semibold">
        Factor Exposures
      </h2>
      <div className="space-y-2">
        {factors.map((f, idx) => {
          const pct = (f.value / maxAbs) * 50; // -50..+50 range
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
                    style={{
                      left: `${50 + (f.benchmark / maxAbs) * 50}%`,
                    }}
                  />
                )}
              </div>
              <span
                className="text-xs w-12 text-right"
                style={{ color }}
              >
                {f.value.toFixed(decimals)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}