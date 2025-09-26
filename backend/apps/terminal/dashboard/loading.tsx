export default function DashboardLoading() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center px-6">
      {/* animated spinner */}
      <div className="w-10 h-10 border-2 border-gray-700 border-t-emerald-500 rounded-full animate-spin mb-4" />

      {/* text */}
      <p className="text-sm text-gray-400">Loading dashboard data…</p>

      {/* skeleton shimmer for panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full max-w-5xl mt-6 animate-pulse">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="h-32 bg-[#111] rounded-lg border border-[#1f1f1f]"
          />
        ))}
      </div>
    </div>
  );
}