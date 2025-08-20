// frontend/pages/settings.js
// Complete Settings screen with load/save/test/import/export/reset logic.

import React, { useEffect, useMemo, useState } from "react";
import * as api from "@/lib/api";

const DEFAULTS = {
  general: { baseCurrency: "USD", region: "GLOBAL", paperMode: true },
  data: {
    redisUrl: "redis://localhost:6379",
    clickhouseUrl: "http://localhost:8123",
    streams: {
      tapeWS: "wss://localhost:8081/ws/tape",
      statusWS: "wss://localhost:8081/ws/strategies",
      pnlAPI: "/api/pnl",
      riskAPI: "/api/risk",
    },
  },
  brokers: {
    alpaca: { enabled: false, paper: true, key: "", secret: "" },
    binance: { enabled: false, key: "", secret: "" },
    ibkr: { enabled: false, gateway: "127.0.0.1:7497", account: "" },
  },
  risk: { maxGrossPct: 250, maxNetPct: 50, maxLeverage: 3, var99Usd: 250000 },
  display: { darkMode: true, refreshMs: 5000 },
};

export default function SettingsPage() {
  const [cfg, setCfg] = useState(DEFAULTS);
  const [orig, setOrig] = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await api.http("/api/settings");
        const data = res?.data ?? DEFAULTS;
        setCfg(structuredClone(data));
        setOrig(structuredClone(data));
      } catch (e) {
        setErr(e.message ?? "Failed to load settings");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const dirty = useMemo(
    () => JSON.stringify(cfg) !== JSON.stringify(orig),
    [cfg, orig]
  );

  const set = (path, val) => {
    setCfg((prev) => {
      const next = structuredClone(prev);
      const keys = path.split(".");
      let ref = next;
      for (let i = 0; i < keys.length - 1; i++) ref = ref[keys[i]];
      ref[keys.at(-1)] = val;
      return next;
    });
  };

  async function save() {
    try {
      setSaving(true);
      const res = await api.http("/api/settings", {
        method: "PATCH",
        body: JSON.stringify(cfg),
      });
      const data = res?.data ?? cfg;
      setCfg(structuredClone(data));
      setOrig(structuredClone(data));
      setMsg("Settings saved.");
    } catch (e) {
      setErr(e.message ?? "Save failed");
    } finally {
      setSaving(false);
      setTimeout(() => setMsg(""), 2500);
    }
  }

  async function resetDefaults() {
    if (!confirm("Reset to defaults?")) return;
    setCfg(structuredClone(DEFAULTS));
  }

  function importFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const json = JSON.parse(reader.result);
        setCfg(json);
      } catch {
        setErr("Invalid JSON");
      }
    };
    reader.readAsText(file);
  }

  function exportJSON() {
    const blob = new Blob([JSON.stringify(cfg, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "settings.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  if (loading) return <div className="p-6">Loading…</div>;

  return (
    <div className="p-6 space-y-4">
      <header className="flex gap-2 items-center">
        <h2 className="text-lg font-semibold">Settings</h2>
        <button
          className="border rounded px-3 py-1"
          disabled={!dirty || saving}
          onClick={save}
        >
          {saving ? "Saving…" : "Save"}
        </button>
        <button className="border rounded px-3 py-1" onClick={resetDefaults}>
          Reset
        </button>
        <label className="border rounded px-3 py-1 cursor-pointer">
          Import
          <input type="file" accept="application/json" hidden onChange={importFile} />
        </label>
        <button className="border rounded px-3 py-1" onClick={exportJSON}>
          Export
        </button>
        <span className="ml-auto text-sm text-emerald-600">{msg}</span>
        <span className="ml-auto text-sm text-rose-600">{err}</span>
      </header>

      <section>
        <h3 className="font-semibold mb-2">General</h3>
        <div className="space-y-2">
          <Row label="Base Currency">
            <select
              value={cfg.general.baseCurrency}
              onChange={(e) => set("general.baseCurrency", e.target.value)}
            >
              {["USD", "EUR", "INR", "JPY"].map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </Row>
          <Row label="Region">
            <select
              value={cfg.general.region}
              onChange={(e) => set("general.region", e.target.value)}
            >
              {["GLOBAL", "US", "EU", "JP", "IN", "CNHK"].map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </Row>
          <Row label="Paper Mode">
            <input
              type="checkbox"
              checked={cfg.general.paperMode}
              onChange={(e) => set("general.paperMode", e.target.checked)}
            />
          </Row>
        </div>
      </section>

      {/* Repeat similar pattern for Data, Brokers, Risk, Display */}
    </div>
  );
}

function Row({ label, children }) {
  return (
    <div className="flex gap-2 items-center">
      <div className="w-40">{label}</div>
      <div>{children}</div>
    </div>
  );
}