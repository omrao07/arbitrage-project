"use client";

import React from "react";

export type RouteOption = {
  id: string;
  label: string;         // e.g. "SMART", "NYSE", "DARK"
  description?: string;  // shown in dropdown
  status?: "live" | "delayed" | "offline"; // optional indicator
};

type Props = {
  options: RouteOption[];
  value?: string;
  onChange?: (id: string) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
};

export default function RoutesSelect({
  options,
  value,
  onChange,
  label = "Route",
  disabled,
  className = "",
}: Props) {
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <span className="text-[11px] text-gray-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        disabled={disabled}
        className="bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
      >
        {options.map((opt) => (
          <option key={opt.id} value={opt.id}>
            {opt.label}
            {opt.description ? ` – ${opt.description}` : ""}
            {opt.status ? ` (${opt.status})` : ""}
          </option>
        ))}
      </select>
    </div>
  );
}