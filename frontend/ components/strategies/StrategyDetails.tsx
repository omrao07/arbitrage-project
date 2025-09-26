"use client";

import React from "react";

export type Strategy = {
  id: string;
  name: string;
  family: string;
  region: string;
  type: string;
  risk: string;
  description?: string;
  pnlYTD?: number;
  inception?: string;
  manager?: string;
};

type Props = {
  strategy: Strategy;
  onBack?: () => void;
};

const StrategyDetails: React.FC<Props> = ({ strategy, onBack }) => {
  return (
    <div className="w-full max-w-3xl rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">Strategy Details</h2>
        {onBack && (
          <button
            onClick={onBack}
            className="rounded-md border border-neutral-300 bg-neutral-50 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100"
          >
            ← Back
          </button>
        )}
      </div>

      {/* Body */}
      <div className="px-6 py-4 space-y-4">
        <div>
          <h3 className="text-xl font-bold">{strategy.name}</h3>
          <p className="text-sm text-neutral-600">
            {strategy.family} • {strategy.region} • {strategy.type}
          </p>
        </div>

        {strategy.description && (
          <p className="text-neutral-700 leading-relaxed">{strategy.description}</p>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className="rounded-lg border border-neutral-200 p-3">
            <div className="text-xs uppercase text-neutral-500">Risk</div>
            <div className="text-sm font-medium">{strategy.risk}</div>
          </div>

          <div className="rounded-lg border border-neutral-200 p-3">
            <div className="text-xs uppercase text-neutral-500">PnL YTD</div>
            <div className="text-sm font-mono">
              {strategy.pnlYTD !== undefined ? `${strategy.pnlYTD.toFixed(2)}%` : "–"}
            </div>
          </div>

          {strategy.inception && (
            <div className="rounded-lg border border-neutral-200 p-3">
              <div className="text-xs uppercase text-neutral-500">Inception</div>
              <div className="text-sm">{strategy.inception}</div>
            </div>
          )}

          {strategy.manager && (
            <div className="rounded-lg border border-neutral-200 p-3">
              <div className="text-xs uppercase text-neutral-500">Manager</div>
              <div className="text-sm">{strategy.manager}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StrategyDetails;