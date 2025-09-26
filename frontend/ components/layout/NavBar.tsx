"use client";

import React, { useState } from "react";
import { Menu, X } from "lucide-react";

type NavLink = { label: string; href: string };

type NavbarProps = {
  brand?: string;
  links?: NavLink[];
  right?: React.ReactNode;
};

const Navbar: React.FC<NavbarProps> = ({ brand = "My Project", links = [], right }) => {
  const [open, setOpen] = useState(false);

  return (
    <nav className="w-full border-b border-neutral-200 bg-white shadow-sm">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:py-4">
        {/* Brand */}
        <div className="text-lg font-bold text-neutral-900">{brand}</div>

        {/* Desktop links */}
        <div className="hidden gap-6 md:flex">
          {links.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm text-neutral-600 transition hover:text-neutral-900"
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* Right slot */}
        <div className="hidden md:block">{right}</div>

        {/* Mobile toggle */}
        <button
          onClick={() => setOpen((o) => !o)}
          className="md:hidden"
          aria-label="Toggle Menu"
        >
          {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="border-t border-neutral-200 bg-white px-4 py-3 md:hidden">
          <div className="flex flex-col gap-3">
            {links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-sm text-neutral-600 transition hover:text-neutral-900"
                onClick={() => setOpen(false)}
              >
                {link.label}
              </a>
            ))}
            {right && <div className="mt-2">{right}</div>}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;