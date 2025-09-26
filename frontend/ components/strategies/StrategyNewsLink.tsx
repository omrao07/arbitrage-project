"use client";

import React from "react";

export type StrategyNews = {
  id: string;
  title: string;
  url: string;
  source?: string;
  date?: string; // ISO string or YYYY-MM-DD
};

type Props = {
  strategyName: string;
  news: StrategyNews[];
};

const StrategyNewsLink: React.FC<Props> = ({ strategyName, news }) => {
  return (
    <div className="w-full rounded-xl border border-neutral-200 bg-white shadow">
      <div className="border-b px-4 py-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">News – {strategyName}</h2>
        <span className="text-xs text-neutral-500">{news.length} articles</span>
      </div>

      {news.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-neutral-500">
          No related news found.
        </div>
      ) : (
        <ul className="divide-y">
          {news.map((n) => (
            <li key={n.id} className="px-4 py-3 hover:bg-neutral-50">
              <a
                href={n.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block"
              >
                <div className="font-medium text-sm text-blue-600 hover:underline">
                  {n.title}
                </div>
                <div className="mt-0.5 flex items-center gap-2 text-xs text-neutral-500">
                  {n.source && <span>{n.source}</span>}
                  {n.date && (
                    <>
                      <span>•</span>
                      <span>
                        {new Date(n.date).toLocaleDateString(undefined, {
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })}
                      </span>
                    </>
                  )}
                </div>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default StrategyNewsLink;