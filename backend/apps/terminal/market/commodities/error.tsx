"use client";

import React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function CommoditiesError({
  error,
  reset,
}: {
  error?: Error & { digest?: string };
  reset?: () => void;
}) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="max-w-md rounded-xl border border-neutral-800 bg-neutral-900 p-6 shadow-lg">
        <div className="flex items-center gap-3">
          <AlertTriangle className="h-6 w-6 text-rose-500" />
          <h1 className="text-lg font-semibold">Commodities Module Error</h1>
        </div>

        <p className="mt-3 text-sm text-neutral-400">
          Something went wrong while loading commodities data.  
          {error?.message && (
            <>
              <br />
              <span className="text-neutral-300">Details:</span> {error.message}
            </>
          )}
        </p>

        <div className="mt-5 flex gap-3">
          {reset && (
            <button
              onClick={reset}
              className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
            >
              <RefreshCw className="h-4 w-4" />
              Retry
            </button>
          )}
          <button
            onClick={() => window.location.reload()}
            className="rounded-lg border border-neutral-700 px-4 py-2 text-sm font-medium text-neutral-300 hover:bg-neutral-800"
          >
            Reload
          </button>
        </div>
      </div>
    </div>
  );
}