import * as React from "react";

export type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  variant?: "default" | "secondary" | "destructive" | "outline";
};

const VARIANT: Record<NonNullable<BadgeProps["variant"]>, string> = {
  default: "bg-black text-white border-transparent",
  secondary: "bg-gray-100 text-gray-900 border-transparent",
  destructive: "bg-red-600 text-white border-transparent",
  outline: "bg-transparent text-gray-900 border",
};

export function Badge({ className = "", variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${VARIANT[variant]} ${className}`}
      {...props}
    />
  );
}