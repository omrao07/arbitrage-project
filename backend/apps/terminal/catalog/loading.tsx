"use client";

export default function CatalogLoading() {
  return (
    <div className="flex flex-col gap-6 p-6 bg-[#0b0b0b] rounded-lg border border-[#222] animate-pulse">
      {/* filters skeleton */}
      <div className="flex flex-wrap gap-4">
        <div className="h-10 w-40 bg-[#1a1a1a] rounded" />
        <div className="h-10 w-40 bg-[#1a1a1a] rounded" />
        <div className="h-10 w-60 bg-[#1a1a1a] rounded flex-1" />
        <div className="h-10 w-28 bg-[#1a1a1a] rounded" />
      </div>

      {/* results list skeleton */}
      <div className="flex flex-col gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="h-16 bg-[#1a1a1a] rounded border border-[#222]"
          />
        ))}
      </div>

      {/* pagination skeleton */}
      <div className="flex items-center justify-between border-t border-[#222] pt-4">
        <div className="h-4 w-24 bg-[#1a1a1a] rounded" />
        <div className="flex gap-2">
          <div className="h-8 w-16 bg-[#1a1a1a] rounded" />
          <div className="h-8 w-16 bg-[#1a1a1a] rounded" />
          <div className="h-8 w-16 bg-[#1a1a1a] rounded" />
        </div>
      </div>
    </div>
  );
}