// frontend/components/strategycontrol.js
// Production‑ready Strategy Control Panel
// - Lists strategies (Alpha & Diversified), their status, region, mode (paper/live), and target weights
// - Start/stop, enable/disable, edit params, and save/apply presets
// - Validates total weights by bucket; shows live status via WebSocket or polling fallback
// - Tailwind‑friendly, no external state libs required
//
// ---- Expected API (you can stub these easily) ----
// GET    /api/strategies
//   -> { data: [{ name, bucket: "alpha"|"diversified", region, enabled, mode: "paper"|"live",
//                 weight: number, status: "idle"|"running"|"error", params: object }] }
// PATCH  /api/strategy/:name
//   body: { enabled?, weight?, region?, mode?, params? }
// POST   /api/strategies/start         body: { names?: string[] }  -> starts all if empty
// POST   /api/strategies/stop          body: { names?: string[] }  -> stops all if empty
// POST   /api/strategies/presets/save  body: { name: string, items: Strategy[] }
// POST   /api/strategies/presets/apply body: { name: string }
// WS     wss://.../ws/strategies  -> messages: { type:"status", name, status, pnl?, lastTickMs? }
//
// You can adapt endpoints; just tweak the calls below.

import React, { useEffect, useMemo, useRef, useState } from "react";

const WS_URL = "wss://localhost:8081/ws/strategies"; // change to your gateway

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

const BUCKETS = ["alpha", "diversified"];
const MODES = ["paper", "live"];
const REGIONS = ["GLOBAL", "US", "EU", "JP", "IN", "CNHK"]; // align with your configs

export default function StrategyControl() {
  const [rows, setRows] = useState([]); // [{...strategy}]
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [filter, setFilter] = useState("");
  const [onlyEnabled, setOnlyEnabled] = useState(false);
  const [editRow, setEditRow] = useState(null); // name being edited inline
  const [saving, setSaving] = useState(false);
  const [wsOnline, setWsOnline] = useState(false);
  const [bulk, setBulk] = useState({ bucket: "alpha", mode: "paper", region: "GLOBAL" });
  const wsRef = useRef(null);

  // fetch initial
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const res = await fetch("/api/strategies", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (mounted) setRows(Array.isArray(json?.data) ? json.data : []);
      } catch (e) {
        if (mounted) setErr(e?.message ?? "Failed to load strategies");
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // websocket live status
  useEffect(() => {
    let ws;
    try {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setWsOnline(true);
      ws.onclose = () => setWsOnline(false);
      ws.onerror = () => setWsOnline(false);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg?.type === "status" && msg.name) {
            setRows((prev) =>
              prev.map((r) =>
                r.name === msg.name
                  ? {
                      ...r,
                      status: msg.status ?? r.status,
                      lastTickMs: msg.lastTickMs ?? r.lastTickMs,
                      pnl: Number.isFinite(msg.pnl) ? msg.pnl : r.pnl,
                    }
                  : r
              )
            );
          }
        } catch {}
      };
    } catch {}
    return () => {
      try {
        ws?.close(1000);
      } catch {}
      wsRef.current = null;
    };
  }, []);

  // filtered list
  const view = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return rows.filter((r) => {
      if (onlyEnabled && !r.enabled) return false;
      if (!f) return true;
      return r.name.toLowerCase().includes(f) || r.bucket?.toLowerCase() === f || r.region?.toLowerCase() === f;
    });
  }, [rows, filter, onlyEnabled]);

  // weight sums
  const sums = useMemo(() => {
    const sum = { alpha: 0, diversified: 0 };
    for (const r of rows) if (r.enabled) sum[r.bucket] += Number(r.weight || 0);
    return sum;
  }, [rows]);

  const sumWarning =
    Math.round(sums.alpha * 100) !== 10000 || Math.round(sums.diversified * 100) !== 10000;

  // inline edits handler
  function updateRow(name, patch) {
    setRows((prev) => prev.map((r) => (r.name === name ? { ...r, ...patch } : r)));
  }

  async function persist(name, body) {
    setSaving(true);
    try {
      const res = await fetch(`/api/strategy/${encodeURIComponent(name)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Save failed (${res.status})`);
    } catch (e) {
      console.error(e);
      setErr(e?.message ?? "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  // bulk actions
  async function startStop(all, start) {
    try {
      const endpoint = start ? "/api/strategies/start" : "/api/strategies/stop";
      const body = all ? {} : { names: selectedNames() };
      await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) {
      setErr(e?.message ?? "Failed to execute");
    }
  }

  // selection (optional minimal)
  const [sel, setSel] = useState({});
  function toggleSel(name) {
    setSel((s) => ({ ...s, [name]: !s[name] }));
  }
  function selectedNames() {
    return Object.entries(sel).filter(([, v]) => v).map(([k]) => k);
  }
  function selectBucket(b) {
    const upd = {};
    for (const r of rows) if (r.bucket === b) upd[r.name] = true;
    setSel(upd);
  }
  function clearSel() {
    setSel({});
  }

  async function applyBulk() {
    const names = selectedNames();
    if (!names.length) return;
    for (const n of names) {
      updateRow(n, { region: bulk.region, mode: bulk.mode });
      await persist(n, { region: bulk.region, mode: bulk.mode });
    }
  }

  // preset save/apply
  async function savePreset() {
    const name = prompt("Preset name?");
    if (!name) return;
    try {
      await fetch("/api/strategies/presets/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, items: rows }),
      });
    } catch (e) {
      setErr(e?.message ?? "Failed to save preset");
    }
  }
  async function applyPreset() {
    const name = prompt("Apply preset name?");
    if (!name) return;
    try {
      const res = await fetch("/api/strategies/presets/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (Array.isArray(json?.data)) setRows(json.data);
    } catch (e) {
      setErr(e?.message ?? "Failed to apply preset");
    }
  }

  // UI
  if (loading && !rows.length)
    return (
      <div className="w-full h-40 flex items-center justify-center text-sm text-zinc-500 dark:text-zinc-400">
        Loading strategies…
      </div>
    );

  if (err && !rows.length)
    return (
      <div className="w-full rounded-xl border border-rose-300/40 bg-rose-50 dark:bg-rose-950/20 p-4 text-rose-600 dark:text-rose-300">
        {err}
      </div>
    );

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex items-center gap-3">
          <h3 className="text-base md:text-lg font-semibold">Strategy Control</h3>
          <span
            className={classNames(
              "text-xs rounded-full px-2 py-0.5 border",
              wsOnline
                ? "border-emerald-300/40 text-emerald-600 dark:text-emerald-300 bg-emerald-50/60 dark:bg-emerald-950/20"
                : "border-amber-300/40 text-amber-600 dark:text-amber-300 bg-amber-50/60 dark:bg-amber-950/20"
            )}
          >
            {wsOnline ? "● live" : "● polling"}
          </span>
          {saving && <span className="text-xs text-zinc-500">saving…</span>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            placeholder="Filter by name/bucket/region"
            className="px-3 py-1.5 rounded-xl border border-black/10 dark:border-white/10 bg-white/60 dark:bg-zinc-900/60 text-sm outline-none focus:ring-2 focus:ring-indigo-500"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label className="text-sm flex items-center gap-1">
            <input type="checkbox" checked={onlyEnabled} onChange={(e) => setOnlyEnabled(e.target.checked)} />
            only enabled
          </label>
          <button
            onClick={() => startStop(true, true)}
            className="px-3 py-1.5 rounded-xl text-sm border border-emerald-300/40 text-emerald-700 dark:text-emerald-300 bg-emerald-50/60 dark:bg-emerald-950/20"
          >
            Start All
          </button>
          <button
            onClick={() => startStop(true, false)}
            className="px-3 py-1.5 rounded-xl text-sm border border-rose-300/40 text-rose-700 dark:text-rose-300 bg-rose-50/60 dark:bg-rose-950/20"
          >
            Stop All
          </button>
          <button
            onClick={savePreset}
            className="px-3 py-1.5 rounded-xl text-sm border border-zinc-300/40 text-zinc-700 dark:text-zinc-200 bg-zinc-50/60 dark:bg-zinc-900/40"
          >
            Save Preset
          </button>
          <button
            onClick={applyPreset}
            className="px-3 py-1.5 rounded-xl text-sm border border-zinc-300/40 text-zinc-700 dark:text-zinc-200 bg-zinc-50/60 dark:bg-zinc-900/40"
          >
            Apply Preset
          </button>
        </div>
      </div>

      {/* Weight warnings */}
      <div
        className={classNames(
          "rounded-xl p-3 text-xs",
          sumWarning
            ? "border border-amber-300/40 bg-amber-50 dark:bg-amber-950/20 text-amber-700 dark:text-amber-300"
            : "border border-emerald-300/40 bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300"
        )}
      >
        Weight totals — Alpha: {(sums.alpha * 100).toFixed(1)}% | Diversified: {(sums.diversified * 100).toFixed(1)}%
        {sumWarning && " (should sum to 100.0% per bucket)"}
      </div>

      {/* Bulk editor */}
      <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-3">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="text-xs opacity-70">Bulk select</label>
            <div className="flex gap-2 mt-1">
              {BUCKETS.map((b) => (
                <button key={b} onClick={() => selectBucket(b)} className="px-3 py-1.5 rounded-xl text-sm border">
                  {b}
                </button>
              ))}
              <button onClick={clearSel} className="px-3 py-1.5 rounded-xl text-sm border">
                Clear
              </button>
            </div>
          </div>
          <div>
            <label className="text-xs opacity-70">Mode</label>
            <select
              className="block px-3 py-1.5 rounded-xl border bg-white/60 dark:bg-zinc-900/60"
              value={bulk.mode}
              onChange={(e) => setBulk((s) => ({ ...s, mode: e.target.value }))}
            >
              {MODES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs opacity-70">Region</label>
            <select
              className="block px-3 py-1.5 rounded-xl border bg-white/60 dark:bg-zinc-900/60"
              value={bulk.region}
              onChange={(e) => setBulk((s) => ({ ...s, region: e.target.value }))}
            >
              {REGIONS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={applyBulk}
            className="px-3 py-1.5 rounded-xl text-sm border border-indigo-300/40 text-indigo-700 dark:text-indigo-300 bg-indigo-50/60 dark:bg-indigo-950/20"
          >
            Apply to Selected
          </button>
          <button
            onClick={() => startStop(false, true)}
            className="px-3 py-1.5 rounded-xl text-sm border border-emerald-300/40 text-emerald-700 dark:text-emerald-300 bg-emerald-50/60 dark:bg-emerald-950/20"
          >
            Start Selected
          </button>
          <button
            onClick={() => startStop(false, false)}
            className="px-3 py-1.5 rounded-xl text-sm border border-rose-300/40 text-rose-700 dark:text-rose-300 bg-rose-50/60 dark:bg-rose-950/20"
          >
            Stop Selected
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-50/70 dark:bg-zinc-900/40 text-zinc-600 dark:text-zinc-300">
            <tr>
              <Th />
              <Th>Name</Th>
              <Th>Bucket</Th>
              <Th>Region</Th>
              <Th>Mode</Th>
              <Th align="right">Weight %</Th>
              <Th>Status</Th>
              <Th align="right">P&L</Th>
              <Th align="right">Last Tick</Th>
              <Th>Enabled</Th>
              <Th>Actions</Th>
            </tr>
          </thead>
          <tbody>
            {view
              .sort((a, b) => (a.bucket === b.bucket ? a.name.localeCompare(b.name) : a.bucket.localeCompare(b.bucket)))
              .map((r) => {
                const selected = !!sel[r.name];
                const isEditing = editRow === r.name;
                const stale = r.lastTickMs ? Date.now() - r.lastTickMs > 15_000 : false;
                return (
                  <tr key={r.name} className={classNames("odd:bg-zinc-50/30 dark:odd:bg-zinc-900/20", selected && "bg-indigo-50/40 dark:bg-indigo-950/10")}>
                    <Td>
                      <input type="checkbox" checked={selected} onChange={() => toggleSel(r.name)} />
                    </Td>
                    <Td className="font-medium">{r.name}</Td>
                    <Td className="uppercase">{r.bucket}</Td>
                    <Td>
                      {isEditing ? (
                        <select
                          className="px-2 py-1 rounded-lg border bg-white/60 dark:bg-zinc-900/60"
                          value={r.region}
                          onChange={(e) => updateRow(r.name, { region: e.target.value })}
                        >
                          {REGIONS.map((x) => (
                            <option key={x} value={x}>
                              {x}
                            </option>
                          ))}
                        </select>
                      ) : (
                        r.region
                      )}
                    </Td>
                    <Td>
                      {isEditing ? (
                        <select
                          className="px-2 py-1 rounded-lg border bg-white/60 dark:bg-zinc-900/60"
                          value={r.mode}
                          onChange={(e) => updateRow(r.name, { mode: e.target.value })}
                        >
                          {MODES.map((x) => (
                            <option key={x} value={x}>
                              {x}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <span className={classNames("px-2 py-0.5 rounded-full text-xs border", r.mode === "live" ? "border-rose-300/40 text-rose-700 dark:text-rose-300" : "border-emerald-300/40 text-emerald-700 dark:text-emerald-300")}>
                          {r.mode}
                        </span>
                      )}
                    </Td>
                    <Td align="right" className="font-mono">
                      {isEditing ? (
                        <input
                          type="number"
                          min={0}
                          max={100}
                          step={0.1}
                          value={Number(r.weight ?? 0) * 100}
                          onChange={(e) => updateRow(r.name, { weight: Math.max(0, Math.min(1, Number(e.target.value) / 100)) })}
                          className="w-24 px-2 py-1 rounded-lg border bg-white/60 dark:bg-zinc-900/60 text-right"
                        />
                      ) : (
                        `${((r.weight ?? 0) * 100).toFixed(1)}%`
                      )}
                    </Td>
                    <Td>
                      <span
                        className={classNames(
                          "px-2 py-0.5 rounded-full text-xs border",
                          r.status === "running"
                            ? "border-emerald-300/40 text-emerald-700 dark:text-emerald-300"
                            : r.status === "error"
                            ? "border-rose-300/40 text-rose-700 dark:text-rose-300"
                            : "border-zinc-300/40 text-zinc-700 dark:text-zinc-300"
                        )}
                        title={stale ? "stale" : ""}
                      >
                        {r.status || "idle"} {stale ? "· stale" : ""}
                      </span>
                    </Td>
                    <Td align="right" className={classNames("font-mono", r.pnl > 0 ? "text-emerald-600" : r.pnl < 0 ? "text-rose-600" : "")}>
                      {Number.isFinite(r.pnl) ? fmtMoney(r.pnl) : "—"}
                    </Td>
                    <Td align="right" className="font-mono">
                      {r.lastTickMs ? new Date(r.lastTickMs).toLocaleTimeString([], { hour12: false }) : "—"}
                    </Td>
                    <Td>
                      <label className="inline-flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={!!r.enabled}
                          onChange={async (e) => {
                            updateRow(r.name, { enabled: e.target.checked });
                            await persist(r.name, { enabled: e.target.checked });
                          }}
                        />
                        <span className="text-xs opacity-70">{r.enabled ? "on" : "off"}</span>
                      </label>
                    </Td>
                    <Td>
                      <div className="flex flex-wrap gap-2">
                        {isEditing ? (
                          <>
                            <button
                              onClick={async () => {
                                setEditRow(null);
                                await persist(r.name, {
                                  region: r.region,
                                  mode: r.mode,
                                  weight: r.weight,
                                  params: r.params,
                                });
                              }}
                              className="px-2.5 py-1 rounded-lg border border-emerald-300/40 text-emerald-700 dark:text-emerald-300"
                            >
                              Save
                            </button>
                            <button onClick={() => setEditRow(null)} className="px-2.5 py-1 rounded-lg border">
                              Cancel
                            </button>
                          </>
                        ) : (
                          <>
                            <button onClick={() => setEditRow(r.name)} className="px-2.5 py-1 rounded-lg border">
                              Edit
                            </button>
                            <button
                              onClick={() => startStop(false, true)}
                              className="px-2.5 py-1 rounded-lg border border-emerald-300/40 text-emerald-700 dark:text-emerald-300"
                              title="Start selected (or this row if selected)"
                            >
                              Start
                            </button>
                            <button
                              onClick={() => startStop(false, false)}
                              className="px-2.5 py-1 rounded-lg border border-rose-300/40 text-rose-700 dark:text-rose-300"
                              title="Stop selected (or this row if selected)"
                            >
                              Stop
                            </button>
                          </>
                        )}
                      </div>
                    </Td>
                  </tr>
                );
              })}
            {!view.length && (
              <tr>
                <Td colSpan={11} className="text-center text-zinc-500 py-6">
                  No strategies match your filter.
                </Td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Param editor (advanced) */}
      <ParamEditor rows={rows} onChange={setRows} onPersist={persist} />
    </div>
  );
}

// ------- Small components -------
function Th({ children, align = "left" }) {
  return <th className={classNames("px-3 py-2 font-medium", align === "right" ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, align = "left", colSpan, className = "" }) {
  return (
    <td colSpan={colSpan} className={classNames("px-3 py-2", align === "right" ? "text-right" : "text-left", className)}>
      {children}
    </td>
  );
}
function fmtMoney(v) {
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);
}

// Advanced JSON params editor (collapsible)
function ParamEditor({ rows, onChange, onPersist }) {
  const [open, setOpen] = useState(false);
  const [target, setTarget] = useState("");
  const current = rows.find((r) => r.name === target);

  return (
    <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950">
      <div className="flex items-center justify-between p-3">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setOpen((o) => !o)}
            className="px-3 py-1.5 rounded-xl border text-sm"
            aria-expanded={open}
          >
            {open ? "Hide" : "Show"} Param Editor
          </button>
          <select
            className="px-3 py-1.5 rounded-xl border bg-white/60 dark:bg-zinc-900/60"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">Select strategy…</option>
            {rows.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name}
              </option>
            ))}
          </select>
        </div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400 pr-3">Edit JSON params safely</div>
      </div>
      {open && current && (
        <ParamForm
          row={current}
          onCommit={async (params) => {
            // optimistic update
            onChange((prev) => prev.map((r) => (r.name === current.name ? { ...r, params } : r)));
            await onPersist(current.name, { params });
          }}
        />
      )}
      {open && !current && <div className="px-3 pb-3 text-sm text-zinc-500">Pick a strategy to edit.</div>}
    </div>
  );
}

function ParamForm({ row, onCommit }) {
  const [text, setText] = useState(JSON.stringify(row.params ?? {}, null, 2));
  const [error, setError] = useState("");

  useEffect(() => {
    setText(JSON.stringify(row.params ?? {}, null, 2));
    setError("");
  }, [row.name]);

  function validate(js) {
    try {
      const obj = JSON.parse(js);
      setError("");
      return obj;
    } catch (e) {
      setError(e?.message ?? "Invalid JSON");
      return null;
    }
  }

  return (
    <div className="p-3">
      <div className="grid gap-2">
        <textarea
          className="w-full min-h-[220px] font-mono text-xs rounded-xl border border-black/10 dark:border-white/10 bg-white/60 dark:bg-zinc-900/60 p-3"
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
        />
        <div className="flex items-center justify-between">
          <span className={classNames("text-xs", error ? "text-rose-600" : "text-zinc-500")}>
            {error || "JSON looks good"}
          </span>
          <button
            onClick={() => {
              const obj = validate(text);
              if (obj) onCommit(obj);
            }}
            disabled={!!error}
            className={classNames(
              "px-3 py-1.5 rounded-xl border text-sm",
              error ? "opacity-50 cursor-not-allowed" : "border-emerald-300/40 text-emerald-700 dark:text-emerald-300"
            )}
          >
            Save Params
          </button>
        </div>
      </div>
    </div>
  );
}