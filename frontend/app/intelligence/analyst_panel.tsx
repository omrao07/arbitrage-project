'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

type KPI = { label: string; value: string | number };
type Row = { [key: string]: string | number | null };

type AnalystResult = {
  kpis?: KPI[];
  columns?: string[];
  rows?: Row[];
  notes?: string;
};

type Props = {
  endpoint?: string; // e.g. '/api/analyst/analyze'
  title?: string;
  className?: string;
  defaultUniverse?: string[];
  defaultPreset?: string;
};

function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function AnalystPanel({
  endpoint = '/api/analyst/analyze',
  title = 'Analyst Panel',
  className = '',
  defaultUniverse = ['AAPL', 'MSFT', 'NVDA'],
  defaultPreset = 'Momentum & Risk',
}: Props) {
  const [prompt, setPrompt] = useState<string>('Find long/short opportunities with improving momentum and falling downside risk.');
  const [universe, setUniverse] = useState<string>(defaultUniverse.join(','));
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [preset, setPreset] = useState<string>(defaultPreset);
  const [riskLimit, setRiskLimit] = useState<number>(2.0);
  const [minMktCap, setMinMktCap] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalystResult | null>(null);
  const [streamLog, setStreamLog] = useState<string>('');
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // autoscroll for log
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [streamLog, loading]);

  const canRun = useMemo(() => {
    return !loading && prompt.trim().length > 0;
  }, [prompt, loading]);

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  async function run() {
    if (!canRun) return;
    setError(null);
    setLoading(true);
    setStreamLog('');
    setResult(null);

    const payload = {
      id: uid(),
      prompt: prompt.trim(),
      universe: parseUniverse(universe),
      range: { from: dateFrom || null, to: dateTo || null },
      preset,
      constraints: { riskLimit, minMktCap },
    };

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream, application/json, text/plain',
        },
        body: JSON.stringify(payload),
        signal,
      });

      const ctype = res.headers.get('content-type') || '';

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        // Streaming (SSE or chunked text)
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
        let finalParsed: AnalystResult | null = null;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          if (ctype.includes('text/event-stream')) {
            const lines = (acc + chunk).split('\n');
            acc = lines.pop() || '';
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;
              const data = line.slice(5).trim();
              if (!data) continue;
              if (data === '[DONE]') continue;
              // Log for transparency
              setStreamLog((s) => s + data + '\n');

              // If server emits JSON fragments like {"kpis":[...]} or {"row":{...}}
              try {
                const obj = JSON.parse(data);
                finalParsed = mergePartial(finalParsed, obj);
                setResult({ ...(finalParsed || {}) });
              } catch {
                // not JSON; just append to notes
                setResult((prev) => {
                  const base: AnalystResult = prev || {};
                  const notes = (base.notes || '') + (base.notes ? '\n' : '') + data;
                  return { ...base, notes };
                });
              }
            }
          } else {
            // plain chunked text
            setStreamLog((s) => s + chunk);
          }
        }
      } else if (ctype.includes('application/json')) {
        const json = await res.json();
        setResult(normalizeResult(json));
      } else {
        const txt = await res.text();
        setResult({ notes: txt });
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        setError(e?.message || 'Request failed');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function onPresetClick(p: string) {
    setPreset(p);
    if (p === 'Momentum & Risk') {
      setPrompt('Rank symbols by momentum (3m/6m) with falling realized vol & drawdown improving. Surface top 10 longs and 10 shorts.');
    } else if (p === 'Event Driven') {
      setPrompt('Surface symbols with positive earnings surprise, rising revisions and bullish news sentiment in last 7 days.');
    } else if (p === 'Mean Revert') {
      setPrompt('Find overextended moves (z-score > 2) into strong liquidity; suggest fade entries with tight stops.');
    } else if (p === 'Pairs/StatArb') {
      setPrompt('Identify cointegrated pairs with temporary spread divergence; propose long/short legs and target spreads.');
    }
  }

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-red-400 text-red-600 hover:bg-red-50">
              Stop
            </button>
          ) : (
            <button
              onClick={run}
              disabled={!canRun}
              className={`text-xs px-3 py-1 rounded border ${canRun ? 'border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700' : 'border-neutral-300 bg-neutral-200 text-neutral-500 cursor-not-allowed'}`}
            >
              Run
            </button>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-4">
            <label className="block text-xs text-neutral-500 mb-1">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Describe the analysis you want…"
            />
          </div>

          <div>
            <label className="block text-xs text-neutral-500 mb-1">Preset</label>
            <select
              value={preset}
              onChange={(e) => onPresetClick(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"
            >
              <option>Momentum & Risk</option>
              <option>Event Driven</option>
              <option>Mean Revert</option>
              <option>Pairs/StatArb</option>
            </select>
          </div>

          <div className="md:col-span-2">
            <label className="block text-xs text-neutral-500 mb-1">Universe (comma-separated)</label>
            <input
              value={universe}
              onChange={(e) => setUniverse(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
              placeholder="AAPL, MSFT, NVDA"
            />
          </div>

          <div>
            <label className="block text-xs text-neutral-500 mb-1">Date From</label>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"
            />
          </div>

          <div>
            <label className="block text-xs text-neutral-500 mb-1">Date To</label>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"
            />
          </div>

          <NumberField
            label="Max Risk (σ)"
            value={riskLimit}
            setValue={setRiskLimit}
            min={0}
            step={0.1}
          />

          <NumberField
            label="Min MktCap ($B)"
            value={minMktCap}
            setValue={setMinMktCap}
            min={0}
            step={1}
          />
        </div>

        {/* Preset quick actions */}
        <div className="flex flex-wrap gap-2 text-xs">
          <PresetButton text="Momentum & Risk" onClick={() => onPresetClick('Momentum & Risk')} />
          <PresetButton text="Event Driven" onClick={() => onPresetClick('Event Driven')} />
          <PresetButton text="Mean Revert" onClick={() => onPresetClick('Mean Revert')} />
          <PresetButton text="Pairs/StatArb" onClick={() => onPresetClick('Pairs/StatArb')} />
        </div>
      </div>

      {/* Body */}
      <div className="grid grid-rows-[auto_minmax(0,1fr)] gap-3 px-3 pb-3">
        {/* KPIs */}
        {result?.kpis && result.kpis.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {result.kpis.map((k, i) => (
              <div key={i} className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-3 py-2">
                <div className="text-[11px] text-neutral-500">{k.label}</div>
                <div className="text-sm font-semibold">{String(k.value)}</div>
              </div>
            ))}
          </div>
        )}

        {/* Table & Log */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 min-h-0">
          <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
            <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Results</div>
            <div className="overflow-auto max-h-[50vh]">
              <ResultsTable result={result} />
            </div>
          </div>

          <div className="min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 flex flex-col">
            <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Run Log</div>
            <div ref={listRef} className="flex-1 overflow-auto p-3">
              {streamLog ? (
                <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">{streamLog}</pre>
              ) : (
                <div className="text-[11px] text-neutral-500">No streaming output yet.</div>
              )}
            </div>
          </div>
        </div>

        {/* Notes / Error */}
        {result?.notes && (
          <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-3 py-2">
            <div className="text-xs font-medium mb-1">Notes</div>
            <div className="text-sm whitespace-pre-wrap">{result.notes}</div>
          </div>
        )}
        {error && (
          <div className="rounded-lg border border-red-300 bg-red-50 text-red-700 px-3 py-2 text-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- subcomponents & helpers ---------------- */

function PresetButton({ text, onClick }: { text: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="rounded-full border border-neutral-300 dark:border-neutral-700 px-2 py-1 hover:bg-neutral-50 dark:hover:bg-neutral-800"
    >
      {text}
    </button>
  );
}

function NumberField({
  label,
  value,
  setValue,
  min = 0,
  step = 1,
}: {
  label: string;
  value: number;
  setValue: (v: number) => void;
  min?: number;
  step?: number;
}) {
  return (
    <div>
      <label className="block text-xs text-neutral-500 mb-1">{label}</label>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        min={min}
        step={step}
        onChange={(e) => setValue(parseFloat(e.target.value || '0'))}
        className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
      />
    </div>
  );
}

function ResultsTable({ result }: { result: AnalystResult | null }) {
  const cols: string[] = useMemo(() => {
    if (!result?.rows || result.rows.length === 0) return [];
    if (result.columns && result.columns.length > 0) return result.columns;
    // infer columns from union of keys
    const keys = new Set<string>();
    result.rows.forEach((r) => Object.keys(r).forEach((k) => keys.add(k)));
    return Array.from(keys);
  }, [result]);

  if (!result || !result.rows || result.rows.length === 0) {
    return (
      <div className="p-3 text-xs text-neutral-500">
        No results yet. Click <span className="font-medium">Run</span> to execute your analysis.
      </div>
    );
  }

  return (
    <table className="min-w-full text-sm">
      <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
        <tr>
          {cols.map((c) => (
            <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {result.rows.map((r, i) => (
          <tr key={i} className="odd:bg-white even:bg-neutral-50 dark:odd:bg-neutral-900 dark:even:bg-neutral-950">
            {cols.map((c) => (
              <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800 align-top">
                {formatCell(r[c])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatCell(v: any): string {
  if (v === null || typeof v === 'undefined') return '';
  if (typeof v === 'number') {
    // If it looks like a small pct, format as pct; else 2 decimals
    if (Math.abs(v) > 0 && Math.abs(v) < 0.5) return `${(v * 100).toFixed(2)}%`;
    return String(Number.isInteger(v) ? v : v.toFixed(4));
  }
  return String(v);
}

function parseUniverse(s: string): string[] {
  return s
    .split(',')
    .map((x) => x.trim().toUpperCase())
    .filter((x) => x.length > 0);
}

/** Merge partial streaming objects into a single AnalystResult */
function mergePartial(base: AnalystResult | null, patch: any): AnalystResult {
  const out: AnalystResult = { ...(base || {}) };

  if (patch == null) return out;

  // rows may arrive incrementally
  if (Array.isArray(patch.rows)) {
    const prev = out.rows || [];
    out.rows = [...prev, ...patch.rows];
  } else if (patch.row && typeof patch.row === 'object') {
    const prev = out.rows || [];
    out.rows = [...prev, patch.row as Row];
  }

  // columns may be declared once
  if (Array.isArray(patch.columns)) {
    out.columns = patch.columns as string[];
  }

  // KPIs can replace or extend
  if (Array.isArray(patch.kpis)) {
    out.kpis = patch.kpis as KPI[];
  } else if (patch.kpi && patch.kpi.label) {
    const prev = out.kpis || [];
    out.kpis = [...prev, patch.kpi as KPI];
  }

  if (typeof patch.notes === 'string') {
    out.notes = (out.notes ? out.notes + '\n' : '') + patch.notes;
  }

  return out;
}

function normalizeResult(anyJson: any): AnalystResult {
  const r: AnalystResult = {};
  if (!anyJson || typeof anyJson !== 'object') return { notes: String(anyJson ?? '') };

  if (Array.isArray(anyJson.kpis)) r.kpis = anyJson.kpis;
  if (Array.isArray(anyJson.columns)) r.columns = anyJson.columns;
  if (Array.isArray(anyJson.rows)) r.rows = anyJson.rows;
  if (typeof anyJson.notes === 'string') r.notes = anyJson.notes;

  // allow { data: { kpis, rows, columns } }
  if (!r.rows && anyJson.data && typeof anyJson.data === 'object') {
    const d = anyJson.data;
    if (Array.isArray(d.kpis)) r.kpis = d.kpis;
    if (Array.isArray(d.columns)) r.columns = d.columns;
    if (Array.isArray(d.rows)) r.rows = d.rows;
  }

  return r;
}