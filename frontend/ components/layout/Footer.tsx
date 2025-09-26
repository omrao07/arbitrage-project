"use client";

import React from "react";

type FooterProps = {
  brand?: string;
  links?: { label: string; href: string }[];
  year?: number;
};

const Footer: React.FC<FooterProps> = ({ brand = "My Project", links = [], year }) => {
  const currentYear = year ?? new Date().getFullYear();

  return (
    <footer className="w-full border-t border-neutral-200 bg-white py-6 text-sm text-neutral-600">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-4 md:flex-row">
        {/* Left: Brand / copyright */}
        <div className="text-center md:text-left">
          © {currentYear} {brand}. All rights reserved.
        </div>

        {/* Right: Links */}
        {links.length > 0 && (
          <div className="flex flex-wrap items-center justify-center gap-4 md:justify-end">
            {links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="transition hover:text-neutral-900"
              >
                {link.label}
              </a>
            ))}
          </div>
        )}
      </div>
    </footer>
  );
};

export default Footer;