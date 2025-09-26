"use client";

import React from "react";
import { Loader2 } from "lucide-react";

export default function ExecutionLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 text-neutral-100">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
        <p className="text-sm text-neutral-400">Executing order… please wait</p>
      </div>
    </div>
  );
}