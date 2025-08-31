// frontend/components/AnalystPanel.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
} from "recharts";

type ScreenerRow = {
  ticker: string;
  name: string;
  price: number;
  chg1d: number;     // % change
  vol: number;       // volume
  pe?: number;
  sector?: string;
};

type NewsItem = {
  id: string;
  ts: number;        // epoch ms
  source: string;    // e.g., "Yahoo", "Moneycontrol"
  headline: string;
  url?: string;
  sentiment?: number; // -1..+1
  ticker?: string;
};

type Note = {
  id: string;
  title: string;
  content: string;
  created_at: string;
  ticker?: string;
};

type Task = {
  id: string;
  text: string;
  done: boolean;
  created_at: string;
};

type SentPt = { t: string; score: number };

export default function AnalystPanel() {
  const [screener, setScreener] = useState<ScreenerRow[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [sentSeries, setSentSeries] = useState<SentPt[]>([]);
  const [filter, setFilter] = useState<string>("");

  const [newNoteTitle, setNewNoteTitle] = useState("");
  const [newNoteBody, setNewNoteBody] = useState("");

  const [newTask, setNewTask] = useState("");
  const [query, setQuery] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const [sc, nw, nt, tk, st] = await Promise.all([
          fetch("/api/analyst/screener").then(r => r.json()),
          fetch("/api/analyst/news").then(r => r.json()),
          fetch("/api/analyst/notes").then(r => r.json()),
          fetch("/api/analyst/tasks").then(r => r.json()),
          fetch("/api/analyst/sentiment").then(r => r.json()),
        ]);
        setScreener(sc);
        setNews(nw);
        setNotes(nt);
        setTasks(tk);
        setSentSeries(st);
      } catch (e) {
        console.error("AnalystPanel fetch error:", e);
      }
    })();
  }, []);

  const filteredScreener = useMemo(() => {
    const f = filter.trim().toLowerCase();
    if (!f) return screener;
    return screener.filter(
      (r) =>
        r.ticker.toLowerCase().includes(f) ||
        (r.name || "").toLowerCase().includes(f) ||
        (r.sector || "").toLowerCase().includes(f)
    );
  }, [filter, screener]);

  const runQuery = async () => {
    try {
      const res = await fetch(`/api/analyst/query?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      // For demo: if query returns a list of tickers, refilter screener/news
      if (Array.isArray(data?.tickers) && data.tickers.length) {
        const set = new Set<string>(data.tickers.map((t: string) => t.toUpperCase()));
        setFilter(""); // reset text filter
        setScreener((prev) => prev.filter((r) => set.has(r.ticker.toUpperCase())));
        setNews((prev) => prev.filter((n) => !n.ticker || set.has((n.ticker || "").toUpperCase())));
      }
      if (Array.isArray(data?.sentiment)) setSentSeries(data.sentiment);
    } catch (e) {
      console.error("Query error:", e);
    }
  };

  const createNote = async () => {
    if (!newNoteTitle.trim() || !newNoteBody.trim()) return;
    try {
      const res = await fetch("/api/analyst/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newNoteTitle, content: newNoteBody }),
      });
      const note = await res.json();
      setNotes((x) => [note, ...x]);
      setNewNoteTitle(""); setNewNoteBody("");
    } catch (e) {
      console.error("Create note error:", e);
    }
  };

  const addTask = async () => {
    if (!newTask.trim()) return;
    try {
      const res = await fetch("/api/analyst/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newTask }),
      });
      const task = await res.json();
      setTasks((x) => [task, ...x]);
      setNewTask("");
    } catch (e) {
      console.error("Add task error:", e);
    }
  };

  const toggleTask = async (id: string) => {
    try {
      await fetch(`/api/analyst/tasks/${id}/toggle`, { method: "POST" });
      setTasks((x) => x.map(t => t.id === id ? { ...t, done: !t.done } : t));
    } catch (e) {
      console.error("Toggle task error:", e);
    }
  };

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Analyst Panel</h2>
          <p className="text-sm opacity-70">Screen, read, note, and act—fast.</p>
        </div>
        <div className="flex gap-2">
          <input
            className="border rounded-lg px-3 py-2 text-sm dark:bg-gray-800"
            placeholder="Filter: ticker / sector / name"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <input
            className="border rounded-lg px-3 py-2 text-sm dark:bg-gray-800"
            placeholder="Query (e.g., 'AI semi supply chain')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button onClick={runQuery} className="px-3 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700">
            Run
          </button>
        </div>
      </header>

      {/* Grid: Screener + Sentiment / News + Notes/Tasks */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* Screener */}
        <section className="xl:col-span-2">
          <h3 className="text-lg font-medium mb-2">Screener</h3>
          <div className="overflow-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800">
                <tr>
                  <Th>Ticker</Th><Th>Name</Th><Th right>Price</Th><Th right>1D</Th><Th right>Vol</Th><Th right>P/E</Th><Th>Sector</Th>
                </tr>
              </thead>
              <tbody>
                {filteredScreener.map((r) => (
                  <tr key={r.ticker} className="border-t border-gray-100 dark:border-gray-800">
                    <Td>{r.ticker}</Td>
                    <Td>{r.name}</Td>
                    <Td right>{fmtNum(r.price, 2)}</Td>
                    <Td right className={r.chg1d >= 0 ? "text-green-600" : "text-red-500"}>
                      {fmtPct(r.chg1d)}
                    </Td>
                    <Td right>{fmtInt(r.vol)}</Td>
                    <Td right>{r.pe ? r.pe.toFixed(1) : "—"}</Td>
                    <Td>{r.sector || "—"}</Td>
                  </tr>
                ))}
                {filteredScreener.length === 0 && (
                  <tr><Td colSpan={7} className="opacity-60">No rows</Td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Sentiment mini + News */}
        <section className="xl:col-span-1 space-y-4">
          <div className="rounded-xl border dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800">
            <div className="text-sm mb-2">Sentiment (composite)</div>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={sentSeries}>
                <XAxis dataKey="t" hide />
                <YAxis domain={[-1, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="score" stroke="#0A84FF" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl border dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800">
            <div className="text-sm mb-2">Top News</div>
            <div className="space-y-2 max-h-64 overflow-auto">
              {news.map((n) => (
                <a
                  key={n.id}
                  href={n.url || "#"}
                  target="_blank"
                  rel="noreferrer"
                  className="block p-2 rounded-lg border dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <div className="flex justify-between">
                    <div className="font-medium">{n.headline}</div>
                    <div className="text-xs opacity-70">{n.source}</div>
                  </div>
                  <div className="text-xs opacity-70">
                    {new Date(n.ts).toLocaleTimeString()} {n.ticker ? `• ${n.ticker}` : ""}
                    {typeof n.sentiment === "number" && (
                      <span className={`ml-2 ${n.sentiment >= 0 ? "text-green-600" : "text-red-500"}`}>
                        {n.sentiment >= 0 ? "↑" : "↓"} {Math.abs(n.sentiment).toFixed(2)}
                      </span>
                    )}
                  </div>
                </a>
              ))}
              {news.length === 0 && <div className="text-sm opacity-60">No headlines.</div>}
            </div>
          </div>
        </section>

        {/* Notes & Tasks */}
        <section className="xl:col-span-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-xl border dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-medium">Notes</h3>
              <button
                onClick={createNote}
                className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-700"
              >
                Save
              </button>
            </div>
            <div className="flex gap-2 mb-2">
              <input
                className="flex-1 border rounded-lg px-3 py-2 text-sm dark:bg-gray-800"
                placeholder="Title"
                value={newNoteTitle}
                onChange={(e) => setNewNoteTitle(e.target.value)}
              />
            </div>
            <textarea
              className="w-full border rounded-lg px-3 py-2 text-sm h-24 dark:bg-gray-800"
              placeholder="Write your note..."
              value={newNoteBody}
              onChange={(e) => setNewNoteBody(e.target.value)}
            />
            <div className="mt-3 space-y-2 max-h-48 overflow-auto">
              {notes.map((n) => (
                <div key={n.id} className="p-2 rounded-lg border dark:border-gray-700">
                  <div className="flex justify-between">
                    <div className="font-semibold">{n.title}</div>
                    <div className="text-xs opacity-70">{new Date(n.created_at).toLocaleDateString()}</div>
                  </div>
                  <div className="text-sm">{n.content}</div>
                </div>
              ))}
              {notes.length === 0 && <div className="text-sm opacity-60">No notes yet.</div>}
            </div>
          </div>

          <div className="rounded-xl border dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-800">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-medium">Tasks</h3>
              <button
                onClick={addTask}
                className="px-3 py-1.5 rounded-lg bg-green-600 text-white text-sm hover:bg-green-700"
              >
                Add
              </button>
            </div>
            <div className="flex gap-2 mb-2">
              <input
                className="flex-1 border rounded-lg px-3 py-2 text-sm dark:bg-gray-800"
                placeholder="New task..."
                value={newTask}
                onChange={(e) => setNewTask(e.target.value)}
              />
            </div>
            <div className="space-y-2 max-h-64 overflow-auto">
              {tasks.map((t) => (
                <label key={t.id} className="flex items-center gap-2 p-2 rounded-lg border dark:border-gray-700">
                  <input
                    type="checkbox"
                    checked={t.done}
                    onChange={() => toggleTask(t.id)}
                  />
                  <span className={t.done ? "line-through opacity-60" : ""}>{t.text}</span>
                  <span className="ml-auto text-xs opacity-70">{new Date(t.created_at).toLocaleString()}</span>
                </label>
              ))}
              {tasks.length === 0 && <div className="text-sm opacity-60">No tasks.</div>}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/* ----------------------- Small presentational helpers ---------------------- */
function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={`p-2 ${right ? "text-right" : "text-left"}`}>{children}</th>;
}
function Td({ children, right, colSpan, className }: { children?: React.ReactNode; right?: boolean; colSpan?: number; className?: string }) {
  return <td colSpan={colSpan} className={`p-2 ${right ? "text-right" : "text-left"} ${className || ""}`}>{children}</td>;
}

/* --------------------------------- Formatters ------------------------------ */
function fmtNum(x: number, d = 2) {
  try { return x.toLocaleString(undefined, { maximumFractionDigits: d }); } catch { return String(x); }
}
function fmtInt(x: number) {
  try { return x.toLocaleString(undefined, { maximumFractionDigits: 0 }); } catch { return String(x); }
}
function fmtPct(x: number) {
  return `${(x * 100).toFixed(2)}%`;
}