"use client";

import React, { useEffect } from "react";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  // Log to console (or send to logging service)
  useEffect(() => {
    console.error("Dashboard error:", error);
  }, [error]);

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-6">
      <h2 className="text-lg font-semibold text-red-400 mb-2">
        ⚠️ Dashboard crashed
      </h2>
      <p className="text-sm text-gray-400 mb-4 max-w-lg">
        {error.message || "An unexpected error occurred while loading the dashboard."}
      </p>
      {error.digest && (
        <p className="text-[11px] text-gray-500 mb-4">
          Error ID: <code>{error.digest}</code>
        </p>
      )}
      <button
        onClick={() => reset()}
        className="bg-red-600/80 hover:bg-red-600 text-white text-sm px-4 py-2 rounded"
      >
        Reload dashboard
      </button>
    </div>
  );
}