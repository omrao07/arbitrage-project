"use client";

import React, { useMemo, useState } from "react";

export type Strategy = {
  id?: string;
  name: string;
  family: string;
  region: string;
  type: string;
  risk: "Low" | "Medium" | "High" | string;
  description?: string;
  pnlYTD?: number;
  inception?: string; // YYYY-MM-DD
  manager?: string;
  mode?: "Manual" | "Automated";
  // optional risk limits
  maxGross?: number;   // e.g., 2.0 = 200%
  maxNet?: number;     // e.g., 1.0 = 100%
  maxDD?: number;      // e.g., 0.2  = 20%
};

type Props = {
  /** When provided, form is in "edit" mode and prefilled */
  initial?: Partial<Strategy>;
  /** Dropdown enumerations (optional; sensible defaults used if omitted) */
  families?: string[];
  regions?: string[];
  types?: string[];
  risks?: string[];
  /** Called with normalized Strategy on submit */
  onSubmit: (s: Strategy) => void;
  /** Called on cancel/back */
  onCancel?: () => void;
  /** Show delete button (edit mode) */
  onDelete?: (id: string) => void;
};

const FALLBACKS = {
  families: ["Equity L/S", "Stat Arb", "Macro Futures", "Options Vol", "Credit"],
  regions: ["Global", "US", "Europe", "Asia", "EM"],
  types: ["Directional", "Relative Value", "Arbitrage", "Carry", "Event"],
  risks: ["Low", "Medium", "High"],
};

export default function StrategyForm({
  initial,
  families = FALLBACKS.families,
  regions = FALLBACKS.regions,
  types = FALLBACKS.types,
  risks = FALLBACKS.risks,
  onSubmit,
  onCancel,
  onDelete,
}: Props) {
  const [form, setForm] = useState<Strategy>({
    id: initial?.id,
    name: initial?.name ?? "",
    family: initial?.family ?? families[0],
    region: initial?.region ?? regions[0],
    type: initial?.type ?? types[0],
    risk: (initial?.risk as Strategy["risk"]) ?? (risks[1] as any) ?? "Medium",
    description: initial?.description ?? "",
    pnlYTD: initial?.pnlYTD,
    inception: initial?.inception ?? "",
    manager: initial?.manager ?? "",
    mode: initial?.mode ?? "Automated",
    maxGross: initial?.maxGross ?? 2.0,
    maxNet: initial?.maxNet ?? 1.0,
    maxDD: initial?.maxDD ?? 0.2,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const isEdit = Boolean(initial?.id);

  const set = <K extends keyof Strategy>(key: K, value: Strategy[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const validate = (): boolean => {
    const e: Record<string, string> = {};
    if (!form.name?.trim()) e.name = "Name is required.";
    if (!form.family) e.family = "Family is required.";
    if (!form.region) e.region = "Region is required.";
    if (!form.type) e.type = "Type is required.";
    if (!form.risk) e.risk = "Risk is required.";

    // numeric sanity
    const np = form.pnlYTD;
    if (np !== undefined && np !== null && !Number.isFinite(np)) e.pnlYTD = "PnL YTD must be a number.";
    if (form.maxGross !== undefined && (form.maxGross as number) <= 0) e.maxGross = "Max gross > 0";
    if (form.maxNet !== undefined && (form.maxNet as number) < 0) e.maxNet = "Max net ≥ 0";
    if (form.maxDD !== undefined && ((form.maxDD as number) <= 0 || (form.maxDD as number) >= 1))
      e.maxDD = "Max drawdown as fraction (0–1).";

    // date format (simple check)
    if (form.inception && !/^\d{4}-\d{2}-\d{2}$/.test(form.inception)) e.inception = "Use YYYY-MM-DD.";

    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    // normalize numbers
    const normalized: Strategy = {
      ...form,
      pnlYTD: form.pnlYTD === undefined || form.pnlYTD === null ? undefined : Number(form.pnlYTD),
      maxGross: form.maxGross !== undefined ? Number(form.maxGross) : undefined,
      maxNet: form.maxNet !== undefined ? Number(form.maxNet) : undefined,
      maxDD: form.maxDD !== undefined ? Number(form.maxDD) : undefined,
    };
    onSubmit(normalized);
  };

  const danger = useMemo(
    () =>
      (form.maxGross ?? 0) > 3 ||
      (form.maxNet ?? 0) > 1.5 ||
      (form.maxDD ?? 0) > 0.35,
    [form.maxGross, form.maxNet, form.maxDD]
  );

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{isEdit ? "Edit Strategy" : "New Strategy"}</h2>
        <div className="flex items-center gap-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded-md border border-neutral-300 bg-neutral-50 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100"
            >
              Cancel
            </button>
          )}
          <button
            type="submit"
            className="rounded-md bg-neutral-900 px-3 py-1.5 text-sm text-white hover:bg-black"
          >
            {isEdit ? "Save Changes" : "Create Strategy"}
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 gap-4 p-4 md:grid-cols-2">
        {/* Left column */}
        <div className="space-y-3">
          <Field label="Name" error={errors.name}>
            <input
              value={form.name}
              onChange={(e) => set("name", e.target.value)}
              placeholder="e.g., Equity L/S – Momentum"
              className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </Field>

          <Field label="Family" error={errors.family}>
            <select
              value={form.family}
              onChange={(e) => set("family", e.target.value)}
              className="w-full rounded-md border border-neutral-300 px-2 py-2 text-sm focus:border-neutral-500"
            >
              {families.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </Field>

          <Field label="Region" error={errors.region}>
            <select
              value={form.region}
              onChange={(e) => set("region", e.target.value)}
              className="w-full rounded-md border border-neutral-300 px-2 py-2 text-sm focus:border-neutral-500"
            >
              {regions.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </Field>

          <Field label="Type" error={errors.type}>
            <select
              value={form.type}
              onChange={(e) => set("type", e.target.value)}
              className="w-full rounded-md border border-neutral-300 px-2 py-2 text-sm focus:border-neutral-500"
            >
              {types.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </Field>

          <Field label="Risk" error={errors.risk}>
            <select
              value={form.risk}
              onChange={(e) => set("risk", e.target.value as Strategy["risk"])}
              className="w-full rounded-md border border-neutral-300 px-2 py-2 text-sm focus:border-neutral-500"
            >
              {risks.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </Field>

          <Field label="Mode">
            <div className="flex gap-3">
              {["Automated", "Manual"].map((m) => (
                <label key={m} className="inline-flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    checked={form.mode === m}
                    onChange={() => set("mode", m as Strategy["mode"])}
                  />
                  {m}
                </label>
              ))}
            </div>
          </Field>
        </div>

        {/* Right column */}
        <div className="space-y-3">
          <Field label="PnL YTD (%)" error={errors.pnlYTD}>
            <input
              type="number"
              step="0.01"
              value={form.pnlYTD ?? ""}
              onChange={(e) => set("pnlYTD", e.target.value === "" ? undefined : Number(e.target.value))}
              placeholder="e.g., 7.25"
              className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </Field>

          <Field label="Inception (YYYY-MM-DD)" error={errors.inception}>
            <input
              type="date"
              value={form.inception ?? ""}
              onChange={(e) => set("inception", e.target.value)}
              className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </Field>

          <Field label="Manager">
            <input
              value={form.manager ?? ""}
              onChange={(e) => set("manager", e.target.value)}
              placeholder="e.g., Jane Doe"
              className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </Field>

          <Field label="Description">
            <textarea
              value={form.description ?? ""}
              onChange={(e) => set("description", e.target.value)}
              rows={4}
              placeholder="Short summary of the signal, data, and risk approach…"
              className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </Field>
        </div>

        {/* Risk limits box (full width) */}
        <div className={`md:col-span-2 rounded-lg border p-3 ${danger ? "border-red-300 bg-red-50/40" : "border-neutral-200"}`}>
          <div className="text-xs uppercase text-neutral-500 mb-2">Risk Limits</div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <NumField
              label="Max Gross (× equity)"
              value={form.maxGross}
              onChange={(v) => set("maxGross", v)}
              min={0.1}
              step={0.1}
              error={errors.maxGross}
            />
            <NumField
              label="Max Net (× equity)"
              value={form.maxNet}
              onChange={(v) => set("maxNet", v)}
              min={0}
              step={0.1}
              error={errors.maxNet}
            />
            <NumField
              label="Max Drawdown (fraction)"
              value={form.maxDD}
              onChange={(v) => set("maxDD", v)}
              min={0.01}
              max={0.99}
              step={0.01}
              error={errors.maxDD}
            />
          </div>
          {danger && (
            <div className="mt-2 text-xs text-red-700">
              Heads up: these limits look aggressive. Make sure your global risk policies allow it.
            </div>
          )}
        </div>
      </div>

      {/* Footer actions */}
      <div className="flex items-center justify-between border-t px-4 py-3">
        {isEdit && onDelete && form.id && (
          <button
            type="button"
            onClick={() => onDelete(form.id!)}
            className="rounded-md border border-red-300 bg-red-50 px-3 py-1.5 text-sm text-red-700 hover:bg-red-100"
          >
            Delete Strategy
          </button>
        )}
        <div className="text-xs text-neutral-500">
          Fields with unusual values will be highlighted; numbers are validated on submit.
        </div>
      </div>
    </form>
  );
}

/* ---------- Small helpers ---------- */

const Field: React.FC<React.PropsWithChildren<{ label: string; error?: string }>> = ({ label, error, children }) => (
  <div>
    <div className="mb-1 text-xs uppercase text-neutral-500">{label}</div>
    {children}
    {error && <div className="mt-1 text-xs text-red-600">{error}</div>}
  </div>
);

function NumField({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  error,
}: {
  label: string;
  value?: number;
  onChange: (v: number | undefined) => void;
  min?: number;
  max?: number;
  step?: number;
  error?: string;
}) {
  return (
    <div>
      <div className="mb-1 text-xs uppercase text-neutral-500">{label}</div>
      <input
        type="number"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
      />
      {error && <div className="mt-1 text-xs text-red-600">{error}</div>}
    </div>
  );
}