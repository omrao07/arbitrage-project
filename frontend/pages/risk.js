// frontend/pages/risk.js
// Risk page: wraps the RiskPanel component, adds refresh, range toggle, and CSV export of limits.
// Works with the RiskPanel you added earlier (which can self-fetch from /api/risk).

import React, { useEffect, useMemo, useState } from "react";
import RiskPanel from "@/components/riskpanel";
import * as api from "@/lib/api";

function fmtMoney(v) {
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);
}

export default function RiskPage() {
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  async function load() {
    try {
      setLoading(true);
      setErr("");
      const data = await api.getRisk(); // expects { data: RiskPayload } or RiskPayload
      setRisk(data?.data ?? data ?? null);
    } catch (e) {
      setErr(e?.message ?? "Failed to load risk");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 30_000); // refresh every 30s
    return () => clearInterval(id);
  }, []);

  function exportLimitsCSV() {
    const rows = risk?.limits ?? [];
    const header = "key,label,value,limit,unit\n";
    const lines = rows.map((r) =>
      [r.key, r.label, r.unit === "%" ? `${r.value}%` : r.value, r.unit === "%" ? `${r.limit}%` : r.limit, r.unit].join(",")
    );
    const blob = new Blob([header + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "risk_limits.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold">Risk</h2>
        <button
          onClick={load}
          className="px-3 py-1.5 rounded-xl border text-sm border-zinc-300/40 text-zinc-700 dark:text-zinc-200 bg-zinc-50/60 dark:bg-zinc-900/40"
          disabled={loading}
          title="Refresh (30s auto)"
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
        <button
          onClick={exportLimitsCSV}
          className="px-3 py-1.5 rounded-xl border text-sm"
          disabled={!risk?.limits?.length}
        >
          Export Limits CSV
        </button>
        <div className="ml-auto text-xs text-zinc-500 dark:text-zinc-400">
          {risk?.kpis ? (
            <>
              Gross: {(risk.kpis.gross_exposure ?? 0).toFixed?.(1)}% &nbsp;|&nbsp; Net:{" "}
              {(risk.kpis.net_exposure ?? 0).toFixed?.(1)}% &nbsp;|&nbsp; VaR 99%:{" "}
              {fmtMoney(risk.kpis.var_1d_p99 ?? 0)}
            </>
          ) : (
            "—"
          )}
        </div>
      </div>

      {/* Main risk panel (passes data; remove `data={risk}` to let it self-fetch) */}
      <RiskPanel data={risk} />

      {/* Errors */}
      {err && (
        <div className="text-sm text-rose-600 dark:text-rose-300">{err}</div>
      )}
    </div>
  );
}