// frontend/components/Fno.tsx
import React, { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, AreaChart, Area } from "recharts";

interface OptionData {
  strike: number;
  call_oi: number;
  put_oi: number;
}

interface FutureData {
  timestamp: string;
  price: number;
  volume: number;
}

export default function Fno() {
  const [futures, setFutures] = useState<FutureData[]>([]);
  const [options, setOptions] = useState<OptionData[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const [futRes, optRes] = await Promise.all([
          fetch("/api/fno/futures"),
          fetch("/api/fno/options"),
        ]);
        setFutures(await futRes.json());
        setOptions(await optRes.json());
      } catch (err) {
        console.error("FNO fetch error:", err);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <h2 className="text-xl font-semibold mb-4">Futures & Options Dashboard</h2>

      {/* Futures Chart */}
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Futures Price & Volume</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={futures}>
            <XAxis dataKey="timestamp" hide />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="price" stroke="#0A84FF" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Options Chain */}
      <div>
        <h3 className="text-lg font-medium mb-2">Options Open Interest (OI)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={options}>
            <XAxis dataKey="strike" />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="call_oi" stroke="#34C759" fill="#34C759" fillOpacity={0.3} />
            <Area type="monotone" dataKey="put_oi" stroke="#FF3B30" fill="#FF3B30" fillOpacity={0.3} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}