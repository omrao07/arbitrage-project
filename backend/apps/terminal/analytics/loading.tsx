"use client";

export default function AnalyticsLoading() {
  return (
    <div className="min-h-[400px] flex flex-col gap-6 p-6 bg-[#0b0b0b] rounded-lg border border-[#222] animate-pulse">
      {/* header skeleton */}
      <div className="h-6 w-48 bg-[#1a1a1a] rounded" />

      {/* stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="h-16 bg-[#1a1a1a] rounded" />
        <div className="h-16 bg-[#1a1a1a] rounded" />
        <div className="h-16 bg-[#1a1a1a] rounded" />
        <div className="h-16 bg-[#1a1a1a] rounded" />
      </div>

      {/* chart skeleton */}
      <div className="h-64 bg-[#1a1a1a] rounded" />

      {/* table skeleton */}
      <div className="space-y-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-6 bg-[#1a1a1a] rounded" />
        ))}
      </div>
    </div>
  );
}