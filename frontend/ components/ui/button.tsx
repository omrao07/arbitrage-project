import * as React from "react";

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "secondary" | "ghost" | "destructive";
  size?: "sm" | "md" | "lg" | "icon" | "xs";
};

const VARIANT: Record<NonNullable<ButtonProps["variant"]>, string> = {
  default: "bg-black text-white hover:bg-black/90",
  secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200",
  ghost: "bg-transparent hover:bg-gray-100 text-gray-900",
  destructive: "bg-red-600 text-white hover:bg-red-700",
};

const SIZE: Record<NonNullable<ButtonProps["size"]>, string> = {
  xs: "h-7 px-2 text-xs",
  sm: "h-8 px-3 text-sm",
  md: "h-10 px-4",
  lg: "h-12 px-5 text-base",
  icon: "h-9 w-9 p-0",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "default", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={`inline-flex items-center justify-center rounded-xl border border-transparent transition focus:outline-none focus:ring-2 focus:ring-black/30 ${VARIANT[variant]} ${SIZE[size]} ${className}`}
      {...props}
    />
  )
);
Button.displayName = "Button";