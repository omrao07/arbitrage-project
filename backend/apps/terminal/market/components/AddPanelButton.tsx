"use client";

import React from "react";

type PanelButtonProps = {
  label: string;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "danger";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  full?: boolean;
};

export default function PanelButton({
  label,
  onClick,
  variant = "primary",
  size = "md",
  disabled = false,
  full = false,
}: PanelButtonProps) {
  const base =
    "inline-flex items-center justify-center rounded-lg font-medium transition focus:outline-none focus:ring-2 focus:ring-offset-1";

  const variants: Record<typeof variant, string> = {
    primary: "bg-emerald-600 hover:bg-emerald-500 text-white focus:ring-emerald-500",
    secondary: "bg-neutral-800 hover:bg-neutral-700 text-neutral-200 focus:ring-neutral-500",
    danger: "bg-rose-600 hover:bg-rose-500 text-white focus:ring-rose-500",
  };

  const sizes: Record<typeof size, string> = {
    sm: "px-2.5 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-5 py-3 text-base",
  };

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`${base} ${variants[variant]} ${sizes[size]} ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${full ? "w-full" : ""}`}
    >
      {label}
    </button>
  );
}