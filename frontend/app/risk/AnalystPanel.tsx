'use client';
import React from 'react';

type Props = {
  className?: string;
};

export default function AnalystPanel({ className = '' }: Props) {
  return (
    <div
      className={`rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4 ${className}`}
    >
      <h2 className="text-sm font-semibold mb-2">Analyst Panel</h2>
      <p className="text-xs text-neutral-500">
        This is a placeholder for analyst insights, reports, or commentary.
      </p>
    </div>
  );
}