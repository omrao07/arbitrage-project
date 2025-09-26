"use client";

import React, { useState } from "react";
import { Home, BarChart, Layers, Settings, Menu } from "lucide-react";

export type SidebarLink = {
  label: string;
  href: string;
  icon?: React.ReactNode;
};

export type SidebarProps = {
  brand?: string;
  links?: SidebarLink[];
  footer?: React.ReactNode;
};

const Sidebar: React.FC<SidebarProps> = ({
  brand = "My Project",
  links = [
    { label: "Dashboard", href: "/", icon: <Home className="h-4 w-4" /> },
    { label: "Strategies", href: "/strategies", icon: <Layers className="h-4 w-4" /> },
    { label: "Analytics", href: "/analytics", icon: <BarChart className="h-4 w-4" /> },
    { label: "Settings", href: "/settings", icon: <Settings className="h-4 w-4" /> },
  ],
  footer,
}) => {
  const [open, setOpen] = useState(true);

  return (
    <div className="flex h-screen">
      {/* Collapsible sidebar */}
      <div
        className={`${
          open ? "w-56" : "w-16"
        } flex flex-col border-r border-neutral-200 bg-white transition-all duration-300`}
      >
        {/* Brand */}
        <div className="flex items-center justify-between px-4 py-3">
          {open && <span className="text-lg font-bold text-neutral-900">{brand}</span>}
          <button
            onClick={() => setOpen((o) => !o)}
            className="rounded-md p-1 hover:bg-neutral-100"
          >
            <Menu className="h-5 w-5 text-neutral-600" />
          </button>
        </div>

        {/* Links */}
        <nav className="flex-1 space-y-1 px-2">
          {links.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
            >
              {link.icon}
              {open && <span>{link.label}</span>}
            </a>
          ))}
        </nav>

        {/* Footer */}
        {footer && (
          <div className="border-t border-neutral-200 px-3 py-2 text-sm text-neutral-600">
            {footer}
          </div>
        )}
      </div>

      {/* Content placeholder */}
      <div className="flex-1 bg-neutral-50 p-6 overflow-y-auto">
        {/* page content goes here */}
      </div>
    </div>
  );
};

export default Sidebar;