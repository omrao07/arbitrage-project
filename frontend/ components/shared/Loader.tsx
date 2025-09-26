"use client";

import React from "react";

type LeaderProps = {
  rank: number;
  name: string;
  value: number | string;
  subtitle?: string;
  highlight?: boolean;
};

const Leader: React.FC<LeaderProps> = ({ rank, name, value, subtitle, highlight }) => {
  return (
    <div
      className={`flex items-center justify-between rounded-xl border px-4 py-3 shadow-sm transition
        ${highlight ? "border-yellow-400 bg-yellow-50" : "border-neutral-200 bg-white"}
      `}
    >
      {/* Rank */}
      <div
        className={`mr-4 flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold
          ${highlight ? "bg-yellow-400 text-white" : "bg-neutral-200 text-neutral-700"}
        `}
      >
        {rank}
      </div>

      {/* Name & subtitle */}
      <div className="flex-1">
        <div className="text-sm font-semibold">{name}</div>
        {subtitle && <div className="text-xs text-neutral-500">{subtitle}</div>}
      </div>

      {/* Value */}
      <div className="ml-4 font-mono text-sm font-medium text-neutral-900">{value}</div>
    </div>
  );
};

export default Leader;