'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ---------------- Types ---------------- */

type KPI = { label: string; value: string | number };
type Row = { [key: string]: string | number | null };

type FactorKey =
  | 'value'         // e.g., book/price, earnings yield
  | 'quality'       // ROE, gross margin, accruals
  | 'momentum'      // 3m/6m/12m return
  | 'low_vol'       // historical vol
  | 'size'          // log mkt cap
  | 'growth'        // revenue/eps growth
  | 'liquidity'     // ADV/turnover
  | 'sentiment';    // news/analyst revisions

type Transform = 'raw' | 'zscore' | 'rank' | 'winsorize';

type FactorSpec = {
  key: FactorKey;
  weight: number;          // can be negative for short bias
  transform: Transform;    // zscore/rank/raw/winsorize
  horizon?: '1M' | '3M' | '6M' | '12M'; // for momentum/growth
  invert?: boolean;        // e.g., lower is better
};

type BacktestResult = {
  kpis?: KPI[];
  columns?: string[];      // table columns
  rows?: Row[];            // securities with factor scores/exposures
  series?: { t: number; v: number }[]; // cumulative perf of composite factor
  corr?: number[][];       // factor x factor correlation matrix
  factors?: FactorKey[];   // order for corr matrix
  notes?: string;
};

type Props = {
  endpoint?: string; // e.g. '/api/factors/explore'
  title?: string;
  className?: string;
  defaultUniverse?: string[];
};

/* ---------------- Utils ---------------- */

function uid() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }
function ts() { return Date.now(); }

function clamp(n: number, a: number, b: number) { return Math.max(a, Math.min(b, n)); }

function pct(x: number, dp = 2) { return `${(x * 100).toFixed(dp)}%`; }

function fmt(n: unknown): string {
  if (n == null) return '';
  if (typeof n === 'number') {
    if (Math.abs(n) < 1 && Math.abs(n) > 0) return n.toFixed(4);
    if (Math.abs(n) >= 1000) return n.toFixed(0);
    return n.toFixed(3);
  }
  return String(n);
}

function parseUniverse(s: string): string[] {
  return s.split(',').map(x => x.trim().toUpperCase()).filter(Boolean);
}

/* ---------------- Component ---------------- */

export default function FactorExplorer({
  endpoint = '/api/factors/explore',
  title = 'Factor Explorer',
  className = '',
  defaultUniverse = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META'],
}: Props) {
  // Controls
  const [universe, setUniverse] = useState(defaultUniverse.join(','));
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [rebalance, setRebalance] = useState<'D' | 'W' | 'M'>('M');
  const [longShort, setLongShort] = useState<boolean>(true);
  const [leverage, setLeverage] = useState<number>(1.0);

  const [factors, setFactors] = useState<FactorSpec[]>([
    { key: 'value', weight: 0.4, transform: 'zscore', invert: true },
    { key: 'quality', weight: 0.2, transform: 'zscore' },
    { key: 'momentum', weight: 0.3, transform: 'zscore', horizon: '6M' },
    { key: 'low_vol', weight: 0.1, transform: 'zscore', invert: true },
  ]);

  // Run state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Results
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [log, setLog] = useState<string>('');

  const canRun = useMemo(() => !loading && factors.some(f => Math.abs(f.weight) > 0.0001), [loading, factors]);

  // Auto-scroll log
  const logRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [log, loading]);

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  function setFactor(idx: number, patch: Partial<FactorSpec>) {
    setFactors(prev => {
      const copy = [...prev];
      copy[idx] = { ...copy[idx], ...patch };
      return copy;
    });
  }

  function addFactor() {
    setFactors(prev => [...prev, { key: 'sentiment', weight: 0.2, transform: 'zscore' }]);
  }

  function removeFactor(i: number) {
    setFactors(prev => prev.filter((_, idx) => idx !== i));
  }

  async function run() {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setLog('');
    setResult(null);

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    const payload = {
      id: uid(),
      universe: parseUniverse(universe),
      range: { from: dateFrom || null, to: dateTo || null },
      settings: { rebalance, longShort, leverage: clamp(leverage, 0, 5) },
      factors, // send as-is; backend can validate
    };

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream, application/json, text/plain' },
        body: JSON.stringify(payload),
        signal,
      });

      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
        let accResult: BacktestResult = {};
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
              setLog(s => s + data + '\n');
              try {
                const patch = JSON.parse(data);
                accResult = mergeResult(accResult, normalizeResult(patch));
                setResult({ ...accResult });
              } catch {
                // not JSON, append to notes
                setResult(prev => {
                  const base = prev || {};
                  const notes = ((base.notes as string) || '') + (base.notes ? '\n' : '') + data;
                  return { ...base, notes };
                });
              }
            }
          } else {
            setLog(s => s + chunk);
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
      if (e?.name !== 'AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-red-400 text-red-600 hover:bg-red-50">Stop</button>
          ) : (
            <button onClick={run} disabled={!canRun} className={`text-xs px-3 py-1 rounded border ${canRun ? 'border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700' : 'border-neutral-300 bg-neutral-200 text-neutral-500 cursor-not-allowed'}`}>Run</button>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 space-y-3 border-b border-neutral-200 dark:border-neutral-800">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-2">
            <label className="block text-xs text-neutral-500 mb-1">Universe (comma-separated)</label>
            <input value={universe} onChange={e => setUniverse(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm" placeholder="AAPL, MSFT, NVDA" />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Date From</label>
            <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm" />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Date To</label>
            <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm" />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Rebalance</label>
            <select value={rebalance} onChange={e => setRebalance(e.target.value as any)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
              <option value="D">Daily</option>
              <option value="W">Weekly</option>
              <option value="M">Monthly</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Long/Short</label>
            <select value={longShort ? 'LS' : 'L'} onChange={e => setLongShort(e.target.value === 'LS')} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
              <option value="LS">Long/Short</option>
              <option value="L">Long Only</option>
            </select>
          </div>
          <NumberField label="Leverage (x)" value={leverage} setValue={setLeverage} min={0} step={0.1} />
        </div>

        {/* Factor list */}
        <div className="space-y-2">
          <div className="text-xs font-medium">Factors</div>
          {factors.map((f, i) => (
            <div key={i} className="grid grid-cols-1 md:grid-cols-6 gap-2 items-end">
              <div>
                <label className="block text-[11px] text-neutral-500 mb-1">Key</label>
                <select value={f.key} onChange={e => setFactor(i, { key: e.target.value as FactorKey })} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
                  <option value="value">Value</option>
                  <option value="quality">Quality</option>
                  <option value="momentum">Momentum</option>
                  <option value="low_vol">Low Vol</option>
                  <option value="size">Size</option>
                  <option value="growth">Growth</option>
                  <option value="liquidity">Liquidity</option>
                  <option value="sentiment">Sentiment</option>
                </select>
              </div>
              <NumberField label="Weight" value={f.weight} setValue={(v) => setFactor(i, { weight: v })} step={0.1} />
              <div>
                <label className="block text-[11px] text-neutral-500 mb-1">Transform</label>
                <select value={f.transform} onChange={e => setFactor(i, { transform: e.target.value as Transform })} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
                  <option value="zscore">Z-Score</option>
                  <option value="rank">Rank</option>
                  <option value="winsorize">Winsorize</option>
                  <option value="raw">Raw</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-neutral-500 mb-1">Horizon</label>
                <select value={f.horizon || '6M'} onChange={e => setFactor(i, { horizon: e.target.value as any })} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
                  <option value="1M">1M</option>
                  <option value="3M">3M</option>
                  <option value="6M">6M</option>
                  <option value="12M">12M</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-neutral-500 mb-1">Invert</label>
                <select value={f.invert ? 'yes' : 'no'} onChange={e => setFactor(i, { invert: e.target.value === 'yes' })} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
              <div className="flex gap-2">
                <button onClick={() => removeFactor(i)} className="text-xs px-2 py-2 rounded border border-rose-300 text-rose-700 hover:bg-rose-50">Remove</button>
              </div>
            </div>
          ))}
          <button onClick={addFactor} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">
            + Add factor
          </button>
        </div>
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* Left: KPIs + Chart */}
        <div className="space-y-3 lg:col-span-1">
          <Card title="KPIs">
            {result?.kpis?.length ? (
              <div className="grid grid-cols-2 gap-2">
                {result.kpis.map((k, i) => (
                  <div key={i} className="rounded border border-neutral-200 dark:border-neutral-800 px-3 py-2">
                    <div className="text-[11px] text-neutral-500">{k.label}</div>
                    <div className="text-sm font-semibold">{String(k.value)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <Empty label="No KPIs yet" />
            )}
          </Card>

          <Card title="Composite Performance">
            {result?.series && result.series.length > 1 ? (
              <LineChart series={result.series} />
            ) : (
              <Empty label="No series yet" />
            )}
          </Card>

          <Card title="Factor Correlation">
            {result?.corr && result.corr.length ? (
              <CorrHeatmap matrix={result.corr} labels={result.factors || guessLabelsFrom(factors)} />
            ) : (
              <Empty label="No correlation yet" />
            )}
          </Card>
        </div>

        {/* Right: Table + Log */}
        <div className="lg:col-span-2 min-h-0 grid grid-rows-[minmax(0,1fr)_180px] gap-3">
          <div className="min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
            <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Exposures / Scores</div>
            <div className="overflow-auto max-h-[60vh]">
              <ResultsTable result={result} />
            </div>
          </div>
          <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
            <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Run Log</div>
            <div ref={logRef} className="flex-1 overflow-auto p-3">
              {error ? (
                <div className="text-xs text-rose-600">{error}</div>
              ) : (
                <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">
                  {log || 'No streaming output yet.'}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- Subcomponents ---------------- */

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function Empty({ label }: { label: string }) {
  return <div className="text-xs text-neutral-500">{label}</div>;
}

function NumberField({
  label, value, setValue, min = 0, step = 1,
}: { label: string; value: number; setValue: (v: number) => void; min?: number; step?: number }) {
  return (
    <div>
      <label className="block text-xs text-neutral-500 mb-1">{label}</label>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        min={min}
        step={step}
        onChange={e => setValue(parseFloat(e.target.value || '0'))}
        className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
      />
    </div>
  );
}

function guessLabelsFrom(factors: FactorSpec[]): FactorKey[] {
  const seen: FactorKey[] = [];
  for (const f of factors) {
    if (!seen.includes(f.key)) seen.push(f.key);
  }
  return seen;
}

function ResultsTable({ result }: { result: BacktestResult | null }) {
  const cols: string[] = useMemo(() => {
    if (!result?.rows || result.rows.length === 0) return [];
    if (result.columns?.length) return result.columns;
    const keys = new Set<string>();
    result.rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)));
    return Array.from(keys);
  }, [result]);

  if (!result?.rows?.length) {
    return <div className="p-3 text-xs text-neutral-500">No results yet. Click <span className="font-medium">Run</span>.</div>;
  }

  return (
    <table className="min-w-full text-sm">
      <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
        <tr>
          {cols.map(c => (
            <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {result.rows.map((r, i) => (
          <tr key={i} className="odd:bg-white even:bg-neutral-50 dark:odd:bg-neutral-900 dark:even:bg-neutral-950">
            {cols.map(c => (
              <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800 align-top">
                {fmt(r[c])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* --- Simple SVG Line Chart (cumulative returns) --- */
function LineChart({ series, width = 320, height = 120 }: { series: { t: number; v: number }[]; width?: number; height?: number }) {
  const xs = series.map((p, i) => i);
  const ys = series.map(p => p.v);
  const xmin = 0, xmax = Math.max(1, xs[xs.length - 1] ?? 1);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const spanY = ymax - ymin || 1;
  const stepX = width / (xmax - xmin || 1);

  const path = series.map((p, i) => {
    const x = i * stepX;
    const y = height - ((p.v - ymin) / spanY) * height;
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');

  const tone = (ys[ys.length - 1] ?? 0) >= (ys[0] ?? 0) ? '#059669' : '#dc2626';

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="block">
      <path d={path} fill="none" stroke={tone} strokeWidth={1.5} />
    </svg>
  );
}

/* --- Correlation Heatmap (factor x factor) --- */
function CorrHeatmap({ matrix, labels }: { matrix: number[][]; labels: (string | number)[] }) {
  const n = Math.min(matrix.length, labels.length);
  if (!n) return <Empty label="No data" />;

  // convert [-1,1] -> color
  function color(v: number): string {
    const x = clamp((v + 1) / 2, 0, 1); // 0..1
    // blue (neg) -> white -> red (pos)
    const r = Math.round(255 * x);
    const g = Math.round(255 * (1 - Math.abs(x - 0.5) * 2)); // fade mid
    const b = Math.round(255 * (1 - x));
    return `rgb(${r},${g},${b})`;
  }

  return (
    <div className="text-xs">
      <div className="grid" style={{ gridTemplateColumns: `80px repeat(${n}, 20px)` }}>
        <div />
        {labels.slice(0, n).map((l, i) => (
          <div key={i} className="text-[10px] text-center truncate">{String(l)}</div>
        ))}
        {matrix.slice(0, n).map((row, i) => (
          <React.Fragment key={i}>
            <div className="pr-2 text-right text-[10px] truncate">{String(labels[i])}</div>
            {row.slice(0, n).map((v, j) => (
              <div key={j} title={v.toFixed(2)} className="w-[20px] h-[20px] border border-neutral-200 dark:border-neutral-800" style={{ background: color(Number(v || 0)) }} />
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/* ---------------- Result merge / normalize ---------------- */

function mergeResult(base: BacktestResult, patch: BacktestResult): BacktestResult {
  const out: BacktestResult = { ...(base || {}) };
  if (patch.kpis) out.kpis = patch.kpis;
  if (patch.columns) out.columns = patch.columns;
  if (patch.rows) out.rows = [...(out.rows || []), ...patch.rows];
  if (patch.series) out.series = patch.series;
  if (patch.corr) out.corr = patch.corr;
  if (patch.factors) out.factors = patch.factors;
  if (patch.notes) out.notes = ((out.notes || '') + (out.notes ? '\n' : '') + patch.notes);
  return out;
}

function normalizeResult(j: any): BacktestResult {
  if (!j || typeof j !== 'object') return { notes: String(j ?? '') };
  const d = j.data && typeof j.data === 'object' ? j.data : j;
  const r: BacktestResult = {};
  if (Array.isArray(d.kpis)) r.kpis = d.kpis;
  if (Array.isArray(d.columns)) r.columns = d.columns;
  if (Array.isArray(d.rows)) r.rows = d.rows;
  if (Array.isArray(d.series)) r.series = d.series;
  if (Array.isArray(d.corr)) r.corr = d.corr;
  if (Array.isArray(d.factors)) r.factors = d.factors;
  if (typeof d.notes === 'string') r.notes = d.notes;
  return r;
}