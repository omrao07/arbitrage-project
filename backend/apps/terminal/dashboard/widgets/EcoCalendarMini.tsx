"use client";

import React, { useEffect, useMemo, useState } from "react";

/** Event shape expected from the API */
export type EcoEvent = {
  id: string;
  time: string;        // ISO timestamp in UTC
  country: string;     // e.g., "US", "IN", "EU"
  name: string;        // e.g., "Nonfarm Payrolls"
  importance?: 1 | 2 | 3; // 1=low, 2=med, 3=high
  actual?: string | number | null;
  consensus?: string | number | null;
  previous?: string | number | null;
  unit?: string;       // e.g., "%", "k"
  source?: string;     // optional url
};

export type EcoCalendarMiniProps = {
  /** start date (local) in YYYY-MM-DD; default = today */
  startDate?: string;
  /** number of days forward from startDate; default = 3 */
  days?: number;
  /** limit rows shown (after filters); default = 12 */
  limit?: number;
  /** filter to countries, e.g., ["US","IN"]; default = all */
  countries?: string[];
  /** show only >= this importance; default = 1 (all) */
  minImportance?: 1 | 2 | 3;
  /** API endpoint to fetch events from */
  endpoint?: string;
  /** called when a row is clicked */
  onSelect?: (e: EcoEvent) => void;
  /** optional title override */
  title?: string;
  /** set true to show event times in UTC (default = local TZ) */
  utc?: boolean;
};

export default function EcoCalendarMini({
  startDate,
  days = 3,
  limit = 12,
  countries,
  minImportance = 1,
  endpoint = "/api/eco/calendar",
  onSelect,
  title = "Economic Calendar",
  utc = false,
}: EcoCalendarMiniProps) {
  const [events, setEvents] = useState<EcoEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // compute window
  const windowDates = useMemo(() => {
    const tzNow = new Date();
    const base = startDate ? new Date(startDate + "T00:00:00") : tzNow;
    const s = new Date(Date.UTC(base.getFullYear(), base.getMonth(), base.getDate()));
    const e = new Date(s);
    e.setUTCDate(e.getUTCDate() + Math.max(1, days));
    return {
      startISO: s.toISOString().slice(0, 10),
      endISO: e.toISOString().slice(0, 10),
    };
  }, [startDate, days]);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      setLoading(true);
      setErr(null);
      try {
        const url = new URL(endpoint, typeof window !== "undefined" ? window.location.origin : "http://localhost");
        url.searchParams.set("start", windowDates.startISO);
        url.searchParams.set("end", windowDates.endISO);
        if (countries && countries.length) url.searchParams.set("countries", countries.join(","));
        if (minImportance) url.searchParams.set("minImportance", String(minImportance));
        const res = await fetch(url.toString(), { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const data = await res.json();
        const arr: EcoEvent[] = Array.isArray(data) ? data : data.items ?? [];
        if (!cancelled) setEvents(normalizeEvents(arr));
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Failed to load calendar");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [endpoint, windowDates.startISO, windowDates.endISO, minImportance, countries?.join("|")]);

  const filtered = useMemo(() => {
    return events
      .filter((ev) => (countries && countries.length ? countries.includes(ev.country) : true))
      .filter((ev) => (minImportance ? (ev.importance ?? 1) >= minImportance : true))
      .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime())
      .slice(0, limit);
  }, [events, countries, minImportance, limit]);

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-3 py-2 border-b border-[#222] flex items-center justify-between">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-400">
          {windowDates.startISO} → {windowDates.endISO}
        </div>
      </div>

      {/* body */}
      {loading ? (
        <SkeletonRows />
      ) : err ? (
        <div className="p-3 text-xs text-red-400">{err}</div>
      ) : filtered.length === 0 ? (
        <div className="p-3 text-xs text-gray-400">No events in this window.</div>
      ) : (
        <ul className="divide-y divide-[#1f1f1f]">
          {filtered.map((ev) => {
            const when = formatTime(ev.time, utc);
            return (
              <li key={ev.id}>
                <button
                  className="w-full text-left px-3 py-2 hover:bg-[#111] focus:outline-none"
                  onClick={() => onSelect?.(ev)}
                  title={ev.name}
                >
                  <div className="flex items-start gap-3">
                    <div className="flex flex-col items-center w-12 shrink-0">
                      <Badge tone={toneForImportance(ev.importance)}>{ev.country}</Badge>
                      <div className="text-[10px] text-gray-500 mt-1">{when}</div>
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-gray-200 truncate">{ev.name}</div>
                      <div className="mt-1 grid grid-cols-3 gap-2 text-[11px] text-gray-400">
                        <Field label="Actual" value={formatVal(ev.actual, ev.unit)} />
                        <Field label="Consensus" value={formatVal(ev.consensus, ev.unit)} />
                        <Field label="Previous" value={formatVal(ev.previous, ev.unit)} />
                      </div>
                    </div>
                    {ev.importance ? (
                      <span
                        className={`text-[10px] px-2 py-[2px] rounded border ${
                          ev.importance >= 3
                            ? "border-red-600 text-red-300"
                            : ev.importance === 2
                            ? "border-amber-600 text-amber-300"
                            : "border-gray-600 text-gray-300"
                        }`}
                        title={`Importance: ${ev.importance}`}
                      >
                        {Array(ev.importance).fill("●").join("")}
                      </span>
                    ) : null}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/* ---------------- helpers ---------------- */

function normalizeEvents(arr: EcoEvent[]): EcoEvent[] {
  return arr
    .filter((x) => x && x.id && x.time && x.name && x.country)
    .map((x) => ({
      ...x,
      id: String(x.id),
      time: new Date(x.time).toISOString(),
      country: String(x.country).toUpperCase(),
      importance: (x.importance as any) ?? 1,
    }));
}

function formatVal(v: any, unit?: string): string {
  if (v === null || v === undefined || v === "") return "—";
  const s = typeof v === "number" ? String(v) : String(v);
  return unit ? `${s}${unit}` : s;
}

function formatTime(isoUTC: string, utc: boolean): string {
  const d = new Date(isoUTC);
  if (utc) {
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mm = String(d.getUTCMinutes()).padStart(2, "0");
    return `${hh}:${mm} UTC`;
  }
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

function toneForImportance(i?: number) {
  if (!i || i <= 1) return "gray";
  if (i === 2) return "amber";
  return "red";
}

function Badge({ children, tone = "gray" }: { children: React.ReactNode; tone?: "gray" | "amber" | "red" }) {
  const map: Record<string, string> = {
    gray: "bg-gray-700/30 text-gray-200 border border-gray-700/60",
    amber: "bg-amber-700/30 text-amber-200 border border-amber-700/60",
    red: "bg-red-700/30 text-red-200 border border-red-700/60",
  };
  return <span className={`text-[10px] px-2 py-[1px] rounded ${map[tone]}`}>{children}</span>;
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="truncate">
      <span className="text-[10px] text-gray-500 mr-1">{label}</span>
      <span className="text-[11px] text-gray-300">{value}</span>
    </div>
  );
}

function SkeletonRows() {
  return (
    <ul className="divide-y divide-[#1f1f1f] animate-pulse">
      {Array.from({ length: 6 }).map((_, i) => (
        <li key={i} className="px-3 py-2">
          <div className="flex items-center gap-3">
            <div className="w-12 h-5 bg-[#1a1a1a] rounded" />
            <div className="flex-1 h-4 bg-[#1a1a1a] rounded" />
            <div className="w-12 h-4 bg-[#1a1a1a] rounded" />
          </div>
        </li>
      ))}
    </ul>
  );
}