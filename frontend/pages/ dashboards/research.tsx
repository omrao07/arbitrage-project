// frontend/components/Research.tsx
import React, { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

interface Note {
  id: string;
  title: string;
  content: string;
  created_at: string;
}

interface ChartData {
  timestamp: string;
  value: number;
}

export default function Research() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    async function fetchData() {
      try {
        const [notesRes, chartRes] = await Promise.all([
          fetch("/api/research/notes"),
          fetch("/api/research/chart"),
        ]);
        setNotes(await notesRes.json());
        setChartData(await chartRes.json());
      } catch (err) {
        console.error("Research fetch error:", err);
      }
    }
    fetchData();
  }, []);

  const runQuery = async () => {
    try {
      const res = await fetch(`/api/research/query?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      setChartData(data);
    } catch (err) {
      console.error("Query error:", err);
    }
  };

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <h2 className="text-xl font-semibold mb-4">Research Dashboard</h2>

      {/* Query input */}
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          placeholder="Enter ticker or keyword (e.g. AAPL, 'inflation')"
          className="flex-1 border rounded-lg px-3 py-2 text-sm dark:bg-gray-800"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          onClick={runQuery}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700"
        >
          Run Query
        </button>
      </div>

      {/* Chart area */}
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Chart Preview</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <XAxis dataKey="timestamp" hide />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#0A84FF" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Notes list */}
      <div>
        <h3 className="text-lg font-medium mb-2">Research Notes</h3>
        <div className="space-y-3">
          {notes.map((n) => (
            <div key={n.id} className="p-3 rounded-lg border dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
              <div className="flex justify-between items-center">
                <h4 className="font-semibold">{n.title}</h4>
                <span className="text-xs opacity-70">{new Date(n.created_at).toLocaleDateString()}</span>
              </div>
              <p className="text-sm mt-1">{n.content}</p>
            </div>
          ))}
          {notes.length === 0 && (
            <p className="text-sm opacity-60">No notes available.</p>
          )}
        </div>
      </div>
    </div>
  );
}