import * as React from "react";

export function ScrollArea({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`overflow-auto [scrollbar-width:thin] [scrollbar-color:#c7c7c7_transparent] ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}