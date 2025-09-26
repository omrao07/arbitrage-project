// overview.tsx
import React from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

export interface StatCard {
  id: string;
  label: string;
  value: string | number;
  delta?: number; // % change
  unit?: string;
  href?: string;
}

export interface OverviewProps {
  title?: string;
  stats: StatCard[];
  /** Optional custom render for a card */
  renderCard?: (s: StatCard) => React.ReactNode;
}

const Overview: React.FC<OverviewProps> = ({ title = "Overview", stats, renderCard }) => {
  return (
    <div className="w-full rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-zinc-100">{title}</h2>
        <span className="text-xs text-zinc-500">Updated {new Date().toLocaleTimeString()}</span>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((s) =>
          renderCard ? (
            renderCard(s)
          ) : (
            <a
              key={s.id}
              href={s.href}
              className="flex flex-col gap-2 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 hover:bg-zinc-800/50 transition focus:outline-none focus:ring-2 focus:ring-amber-400/50"
            >
              <div className="text-sm text-zinc-400">{s.label}</div>
              <div className="flex items-end gap-2">
                <div className="text-2xl font-semibold text-zinc-100 tabular-nums">
                  {s.value}
                  {s.unit && <span className="ml-1 text-base text-zinc-400">{s.unit}</span>}
                </div>
                {typeof s.delta === "number" && (
                  <div
                    className={`flex items-center gap-1 text-sm font-medium ${
                      s.delta >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {s.delta >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                    {s.delta > 0 ? "+" : ""}
                    {s.delta.toFixed(2)}%
                  </div>
                )}
              </div>
            </a>
          )
        )}
      </div>
    </div>
  );
};

export default Overview;