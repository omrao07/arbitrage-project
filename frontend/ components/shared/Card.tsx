"use client";

import React from "react";

export type CardProps = {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;     // stuff like buttons, badges
  footer?: React.ReactNode;    // optional footer row
  padded?: boolean;            // toggle default padding
  className?: string;
  children?: React.ReactNode;
};

const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  right,
  footer,
  padded = true,
  className = "",
  children,
}) => {
  return (
    <div
      className={`w-full rounded-2xl border border-neutral-200 bg-white shadow ${className}`}
    >
      {(title || right) && (
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div>
            {title && <h3 className="text-lg font-semibold">{title}</h3>}
            {subtitle && (
              <p className="mt-0.5 text-xs text-neutral-500">{subtitle}</p>
            )}
          </div>
          {right && <div className="ml-2">{right}</div>}
        </div>
      )}

      {children && (
        <div className={padded ? "p-4" : ""}>
          {children}
        </div>
      )}

      {footer && (
        <div className="border-t px-4 py-2 text-sm text-neutral-600">
          {footer}
        </div>
      )}
    </div>
  );
};

export default Card;