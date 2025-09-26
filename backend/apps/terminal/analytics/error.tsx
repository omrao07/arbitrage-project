"use client";

import { useEffect } from "react";

export default function AnalyticsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  // optional: log to your monitoring (Sentry, Logtail, etc.)
  useEffect(() => {
    console.error("Analytics route error:", error);
  }, [error]);

  return (
    <div className="min-h-[400px] flex flex-col items-center justify-center bg-[#0b0b0b] text-gray-200 p-6 rounded-lg border border-[#222]">
      <h2 className="text-lg font-semibold text-red-400 mb-2">
        Analytics Module Error
      </h2>
      <p className="text-sm text-gray-400 mb-4 max-w-xl text-center">
        Something went wrong while loading analytics.  
        {error?.message && (
          <span className="block mt-1 text-gray-500">
            ({error.message})
          </span>
        )}
      </p>

      <div className="flex gap-3">
        <button
          onClick={() => reset()}
          className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded shadow"
        >
          Try again
        </button>
        <button
          onClick={() => window.location.reload()}
          className="bg-gray-700 hover:bg-gray-600 text-gray-100 px-4 py-2 rounded"
        >
          Reload page
        </button>
      </div>

      {error?.digest && (
        <p className="mt-4 text-[11px] text-gray-500">
          Error ID: {error.digest}
        </p>
      )}
    </div>
  );
}