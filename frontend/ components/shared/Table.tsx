"use client";

import React from "react";

export type Column<T> = {
  key: keyof T | string;
  header: string;
  render?: (row: T) => React.ReactNode;
  align?: "left" | "center" | "right";
  width?: string | number;
};

export type TableProps<T> = {
  columns: Column<T>[];
  data: T[];
  emptyMessage?: string;
  striped?: boolean;
  className?: string;
};

function Table<T>({ columns, data, emptyMessage = "No data available", striped = true, className = "" }: TableProps<T>) {
  return (
    <div className={`overflow-x-auto rounded-xl border border-neutral-200 bg-white shadow ${className}`}>
      <table className="min-w-full border-collapse text-sm">
        <thead className="bg-neutral-50">
          <tr>
            {columns.map((col, i) => (
              <th
                key={i}
                className={`px-4 py-2 text-left text-xs font-semibold text-neutral-600 ${
                  col.align === "center" ? "text-center" : col.align === "right" ? "text-right" : "text-left"
                }`}
                style={{ width: col.width }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-4 py-6 text-center text-neutral-500">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row, ri) => (
              <tr
                key={ri}
                className={`transition ${
                  striped ? (ri % 2 === 0 ? "bg-white" : "bg-neutral-50/50") : "bg-white"
                } hover:bg-neutral-100/60`}
              >
                {columns.map((col, ci) => (
                  <td
                    key={ci}
                    className={`px-4 py-2 ${
                      col.align === "center" ? "text-center" : col.align === "right" ? "text-right" : "text-left"
                    }`}
                  >
                    {col.render ? col.render(row) : (row as any)[col.key]}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default Table;