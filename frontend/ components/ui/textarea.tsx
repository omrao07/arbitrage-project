import * as React from "react";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className = "", rows = 3, ...props }, ref) => (
    <textarea
      ref={ref}
      rows={rows}
      className={`w-full rounded-xl border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-black/30 ${className}`}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";