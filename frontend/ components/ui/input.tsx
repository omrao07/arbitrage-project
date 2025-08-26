import * as React from "react";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className = "", ...props }, ref) => (
    <input
      ref={ref}
      className={`h-10 w-full rounded-xl border px-3 text-sm outline-none focus:ring-2 focus:ring-black/30 ${className}`}
      {...props}
    />
  )
);
Input.displayName = "Input";