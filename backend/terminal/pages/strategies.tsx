// strategies.tsx
import React, { useState } from "react";

export type Strategy = {
  id: string;
  name: string;
  desk?: string;
  tags?: string[];
  status: "live" | "paused" | "paper";
  capital?: number;
  targetWeight?: number;
  gross?: number;
  net?: number;
  sharpe?: number;
  vol?: number;
  pnlYTD?: number;
  pnlD?: number;
  positions?: number;
};

interface StrategiesProps {
  strategies: Strategy[];
  onChange?: (next: Strategy[]) => void;
}

const Strategies: React.FC<StrategiesProps> = ({ strategies, onChange }) => {
  const [rows, setRows] = useState<Strategy[]>(strategies);

  // Toggle strategy status between live <-> paused
  const toggleStatus = (id: string) => {
    const next = rows.map((s) =>
      s.id === id
        ? { ...s, status: s.status === "live" ? "paused" : "live" }
        : s
    );
    
  };

  // Export strategies as CSV
  const exportCSV = () => {
    const head = [
      "id",
      "name",
      "desk",
      "tags",
      "status",
      "capital",
      "targetWeight",
      "gross",
      "net",
      "sharpe",
      "vol",
      "pnlYTD",
      "pnlD",
      "positions",
    ].join(",");

    const rowsCSV = rows.map((s) =>
      [
        s.id,
        `"${(s.name || "").replace(/"/g, '""')}"`, // escape quotes
        s.desk || "",
        (s.tags || []).join("|"),
        s.status,
        s.capital ?? "",
        s.targetWeight ?? "",
        s.gross ?? "",
        s.net ?? "",
        s.sharpe ?? "",
        s.vol ?? "",
        s.pnlYTD ?? "",
        s.pnlD ?? "",
        s.positions ?? "",
      ].join(",")
    );

    const csv = [head, ...rowsCSV].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "strategies.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="p-4">
      <div className="flex justify-between mb-2">
        <h2 className="text-xl font-semibold">Strategies</h2>
       
      </div>

      <table className="w-full border border-gray-700">
        <thead>
          <tr className="bg-gray-800 text-gray-200">
            <th className="p-2">ID</th>
            <th>Name</th>
            <th>Desk</th>
            <th>Tags</th>
            <th>Status</th>
            <th>Capital</th>
            <th>Weight %</th>
            <th>Gross</th>
            <th>Net</th>
            <th>Sharpe</th>
            <th>Vol</th>
            <th>PnL YTD</th>
            <th>PnL D</th>
            <th>Positions</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.id} className="border-t border-gray-700 text-sm">
              <td className="p-2">{s.id}</td>
              <td>{s.name}</td>
              <td>{s.desk}</td>
              <td>{(s.tags || []).join(", ")}</td>
              <td>{s.status}</td>
              <td>{s.capital?.toLocaleString()}</td>
              <td>{s.targetWeight}</td>
              <td>{s.gross}</td>
              <td>{s.net}</td>
              <td>{s.sharpe}</td>
              <td>{s.vol}</td>
              <td>{s.pnlYTD}</td>
              <td>{s.pnlD}</td>
              <td>{s.positions}</td>
              <td>
               
                  size="sm"
                  variant="outline"
                
                
                  {s.status === "live" ? "Pause" : "Go Live"}
                
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Strategies;