'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ===================== Types ===================== */

type KPI = { label: string; value: string | number };
type Row = {
  counterparty: string;
  le?: string;                // legal entity
  netting_set?: string;
  currency?: string;
  ee?: number;                // Expected Exposure
  epe?: number;               // Expected Positive Exposure
  ene?: number;               // Expected Negative Exposure
  pfe95?: number;
  pfe99?: number;
  im?: number;                // Initial Margin
  vm?: number;                // Variation Margin
  csa_threshold?: number;
  mta?: number;               // Minimum Transfer Amount
  limit?: number;
  usage_pct?: number;         // limit usage (0..1)
  wwr?: boolean;              // wrong-way risk flag
  products?: string;          // e.g., IRS/FRA/FXO
  rating?: string;            // internal/external rating
  breaches_30d?: number;
  notes?: string;
};

type SeriesPoint = { t: number; ee?: number; pfe95?: number; pfe99?: number };

type GraphEdge = { from: string; to: string; weight: number }; // exposure matrix simplified

type Payload = {
  kpis?: KPI[];
  columns?: string[];
  rows?: Row[];
  row?: Row;
  series?: { counterparty: string; points: SeriesPoint[] };
  graph?: { nodes: string[]; edges: GraphEdge[] };
  notes?: string;
};

type Props = {
  endpoint?: string;          // e.g. '/api/risk/counterparty'
  title?: string;
  className?: string;
  defaultCurrency?: string;
};

/* ===================== Utils ===================== */

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const pct = (x: number | undefined, dp = 1) =>
  typeof x === 'number' ? `${(x * 100).toFixed(dp)}%` : '';

function fmtNum(x: any): string {
  if (x == null) return '';
  if (typeof x === 'number') {
    const abs = Math.abs(x);
    if (abs >= 1e9) return (x / 1e9).toFixed(2) + 'B';
    if (abs >= 1e6) return (x / 1e6).toFixed(2) + 'M';
    if (abs >= 1e3) return (x / 1e3).toFixed(2) + 'k';
    return x.toFixed(2);
  }
  return String(x);
}

/* ===================== Component ===================== */

export default function CounterpartyExposure({
  endpoint = '/api/risk/counterparty',
  title = 'Counterparty Exposure',
  className = '',
  defaultCurrency = 'USD',
}: Props) {
  // filters / controls
  const [asOf, setAsOf] = useState<string>(''); // YYYY-MM-DD
  const [entity, setEntity] = useState<string>('Consolidated');
  const [currency, setCurrency] = useState<string>(defaultCurrency);
  const [products, setProducts] = useState<string>('ALL'); // ALL/IRS/FX/Equity/Commodity/Other
  const [scenario, setScenario] = useState<string>('Base'); // Base/Stress1/Stress2/WWR
  const [query, setQuery] = useState<string>('');
  const [limitOnly, setLimitOnly] = useState<boolean>(false);

  // results
  const [kpis, setKpis] = useState<KPI[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [columns, setColumns] = useState<string[] | null>(null);
  const [series, setSeries] = useState<Record<string, SeriesPoint[]>>({});
  const [graph, setGraph] = useState<{ nodes: string[]; edges: GraphEdge[] } | null>(null);
  const [notes, setNotes] = useState<string>('');

  // run state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  // selection
  const [selectedCP, setSelectedCP] = useState<string | null>(null);
  const selectedSeries = selectedCP ? series[selectedCP] || [] : [];

  // autoscroll log
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [log, loading]);

  // derived & filtered
  const filteredRows = useMemo(() => {
    let out = [...rows];
    if (query.trim()) {
      const q = query.toLowerCase();
      out = out.filter((r) =>
        [r.counterparty, r.netting_set, r.products, r.rating]
          .filter(Boolean)
          .some((s) => String(s).toLowerCase().includes(q)),
      );
    }
    if (limitOnly) {
      out = out.filter((r) => (r.usage_pct ?? 0) >= 0.8 || (r.breaches_30d ?? 0) > 0);
    }
    // sort by PFE99 desc as default
    out.sort((a, b) => (b.pfe99 ?? 0) - (a.pfe99 ?? 0));
    return out;
  }, [rows, query, limitOnly]);

  function stop() {
    abortRef.current?.abort();
    setLoading(false);
  }

  async function run() {
    setLoading(true);
    setError(null);
    setLog('');
    setKpis([]);
    setRows([]);
    setColumns(null);
    setSeries({});
    setGraph(null);
    setNotes('');
    setSelectedCP(null);

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    const payload = {
      id: uid(),
      asOf: asOf || null,
      entity,
      currency,
      products,
      scenario,
    };

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
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
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
              setLog((s) => s + data + '\n');
              try {
                const patch: Payload = JSON.parse(data);
                applyPatch(patch);
              } catch {
                setNotes((n) => (n ? n + '\n' : '') + data);
              }
            }
          } else {
            setLog((s) => s + chunk);
          }
        }
      } else if (ctype.includes('application/json')) {
        const json: Payload = await res.json();
        applyPatch(json);
      } else {
        const txt = await res.text();
        setNotes(txt);
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

  function applyPatch(p: Payload) {
    if (Array.isArray(p.kpis)) setKpis(p.kpis);
    if (Array.isArray(p.columns)) setColumns(p.columns);

    if (Array.isArray(p.rows)) {
      setRows((prev) => mergeRows(prev, p.rows!));
    }
    if (p.row) {
      setRows((prev) => mergeRows(prev, [p.row!]));
    }
    if (p.series && p.series.counterparty) {
      const cp = p.series.counterparty;
      setSeries((prev) => ({ ...prev, [cp]: p.series!.points || [] }));
    }
    if (p.graph) {
      setGraph(p.graph);
    }
    if (typeof p.notes === 'string') {
      setNotes((n) => (n ? n + '\n' : '') + p.notes);
    }
  }

  function mergeRows(prev: Row[], incoming: Row[]) {
    const map = new Map<string, Row>();
    for (const r of prev) map.set(keyFor(r), r);
    for (const r of incoming) map.set(keyFor(r), { ...map.get(keyFor(r)), ...r });
    return Array.from(map.values());
  }
  function keyFor(r: Row) {
    return `${r.counterparty}::${r.netting_set || ''}::${r.currency || ''}`;
  }

  function exportCSV() {
    const cols =
      columns ||
      [
        'counterparty','le','netting_set','currency','products','rating',
        'ee','epe','ene','pfe95','pfe99','im','vm','csa_threshold','mta',
        'limit','usage_pct','wwr','breaches_30d',
      ];
    const header = cols.join(',');
    const lines = filteredRows.map((r) =>
      cols
        .map((c) => {
          const v: any = (r as any)[c];
          if (typeof v === 'number' && c === 'usage_pct') return (v * 100).toFixed(2) + '%';
          if (typeof v === 'number') return v;
          if (typeof v === 'boolean') return v ? 'true' : 'false';
          return `"${String(v ?? '')?.replace(/"/g, '""')}"`;
        })
        .join(','),
    );
    const csv = [header, ...lines].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'counterparty_exposure.csv'; a.click();
    URL.revokeObjectURL(url);
  }

  const topN = useMemo(() => filteredRows.slice(0, 8), [filteredRows]);

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export CSV</button>
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
          ) : (
            <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 border-b border-neutral-200 dark:border-neutral-800 grid grid-cols-1 md:grid-cols-6 gap-3">
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">As of</label>
          <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm" />
        </div>
        <Select label="Entity" value={entity} setValue={setEntity} options={['Consolidated','Broker','Bank','HedgeFund']} />
        <Select label="Currency" value={currency} setValue={setCurrency} options={['USD','EUR','GBP','INR','JPY']} />
        <Select label="Products" value={products} setValue={setProducts} options={['ALL','IRS','FX','Equity','Commodity','Crypto']} />
        <Select label="Scenario" value={scenario} setValue={setScenario} options={['Base','Stress1','Stress2','WWR']} />
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Search</label>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="counterparty, netting set, rating…" className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm" />
        </div>
        <div className="flex items-end gap-2 md:col-span-2">
          <label className="inline-flex items-center gap-2 text-sm">
            <input type="checkbox" checked={limitOnly} onChange={(e) => setLimitOnly(e.target.checked)} className="accent-indigo-600" />
            Show at-risk only (usage≥80% or recent breach)
          </label>
        </div>
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        <div className="space-y-3">
          <Card title="KPIs">
            {kpis.length ? (
              <div className="grid grid-cols-2 gap-2">
                {kpis.map((k, i) => (
                  <div key={i} className="rounded border border-neutral-200 dark:border-neutral-800 px-3 py-2">
                    <div className="text-[11px] text-neutral-500">{k.label}</div>
                    <div className="text-sm font-semibold">{String(k.value)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <Empty label="No KPIs yet." />
            )}
          </Card>

          <Card title="Top Exposures (PFE99)">
            {topN.length ? <TopBar rows={topN} onSelect={setSelectedCP} selected={selectedCP} /> : <Empty label="No data" />}
          </Card>

          <Card title="Network (simplified)">
            {graph ? <MiniNetwork graph={graph} onSelect={setSelectedCP} selected={selectedCP} /> : <Empty label="No graph" />}
          </Card>
        </div>

        {/* Table */}
        <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Counterparties ({filteredRows.length})</div>
          <div className="overflow-auto">
            <Table rows={filteredRows} columns={columns} onSelect={setSelectedCP} selected={selectedCP} />
          </div>
        </div>
      </div>

      {/* Detail + Log */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 border-t border-neutral-200 dark:border-neutral-800">
        <Card title={`Detail ${selectedCP ? `· ${selectedCP}` : ''}`}>
          {selectedCP ? (
            <div className="space-y-3">
              <LinesChart series={selectedSeries} />
              <div className="text-[11px] text-neutral-500">
                EE / PFE over time. Select a row to update. (Your API can stream `("series" ("counterparty":"CP","points":[...]&rbrace;))
              </div>
            </div>
          ) : (
            <Empty label="Select a counterparty to see time series." />
          )}
        </Card>
        <Card title="Notes">
          <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || 'No notes.'}</pre>
        </Card>
        <Card title="Run Log">
          <div ref={logRef} className="max-h-[26vh] overflow-auto">
            {error ? <div className="text-xs text-rose-600">{error}</div> : <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">{log || 'No streaming output yet.'}</pre>}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ===================== Subcomponents ===================== */

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

function Select({
  label, value, setValue, options,
}: { label: string; value: string; setValue: (v: string) => void; options: string[] }) {
  return (
    <div>
      <label className="block text-[11px] text-neutral-500 mb-1">{label}</label>
      <select value={value} onChange={(e) => setValue(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

/* ---- Table ---- */

function Table({
  rows, columns, onSelect, selected,
}: { rows: Row[]; columns: string[] | null; onSelect: (cp: string) => void; selected: string | null }) {
  const cols =
    columns ||
    ['counterparty','netting_set','currency','products','rating','ee','pfe95','pfe99','im','vm','limit','usage_pct','wwr','breaches_30d'];

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
        {rows.map((r, i) => {
          const isSel = selected === r.counterparty;
          const usage = typeof r.usage_pct === 'number' ? r.usage_pct : undefined;
          return (
            <tr
              key={i}
              onClick={() => onSelect(r.counterparty)}
              className={`${isSel ? 'bg-indigo-50 dark:bg-indigo-900/30' : i % 2 ? 'bg-white dark:bg-neutral-900' : 'bg-neutral-50 dark:bg-neutral-950'} cursor-pointer`}
            >
              {cols.map((c) => (
                <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800 align-top">
                  {renderCell(c, r, usage)}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function renderCell(col: string, r: Row, usage?: number) {
  const v = (r as any)[col];
  if (col === 'usage_pct') {
    return (
      <div className="min-w-[90px]">
        <div className="flex items-center justify-between text-[11px] text-neutral-500">
          <span>{pct(usage)}</span><span>{fmtNum(r.limit)}</span>
        </div>
        <div className="h-2 w-full rounded bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
          <div
            className={`h-2 ${usage && usage >= 0.8 ? 'bg-rose-500' : 'bg-emerald-500'}`}
            style={{ width: `${clamp((usage || 0) * 100, 0, 100)}%` }}
          />
        </div>
      </div>
    );
  }
  if (typeof v === 'number') return fmtNum(v);
  if (typeof v === 'boolean') return v ? '⚠️' : '';
  return String(v ?? '');
}

/* ---- Top bar chart (SVG) ---- */

function TopBar({ rows, onSelect, selected }: { rows: Row[]; onSelect: (cp: string) => void; selected: string | null }) {
  const max = Math.max(...rows.map((r) => r.pfe99 || 0), 1);
  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <button
          key={r.counterparty}
          onClick={() => onSelect(r.counterparty)}
          className={`w-full text-left ${selected === r.counterparty ? 'opacity-100' : 'opacity-90 hover:opacity-100'}`}
          title={`${r.counterparty} • PFE99 ${fmtNum(r.pfe99)} • Usage ${pct(r.usage_pct)}`}
        >
          <div className="text-xs mb-1 flex items-center justify-between">
            <span className="truncate">{r.counterparty}</span>
            <span className="text-neutral-500">{fmtNum(r.pfe99)}</span>
          </div>
          <div className="h-3 w-full rounded bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
            <div className="h-3 bg-indigo-600" style={{ width: `${((r.pfe99 || 0) / max) * 100}%` }} />
          </div>
        </button>
      ))}
    </div>
  );
}

/* ---- Tiny line chart (EE/PFE) ---- */

function LinesChart({ series }: { series: SeriesPoint[] }) {
  if (!series?.length) return <Empty label="No series" />;
  const width = 360, height = 120;
  const xs = series.map((_, i) => i);
  const minY = Math.min(...series.flatMap((p) => [p.ee ?? 0, p.pfe95 ?? 0, p.pfe99 ?? 0]));
  const maxY = Math.max(...series.flatMap((p) => [p.ee ?? 0, p.pfe95 ?? 0, p.pfe99 ?? 0]));
  const spanY = maxY - minY || 1;
  const stepX = width / Math.max(1, xs[xs.length - 1] ?? 1);

  function pathFor(key: keyof SeriesPoint) {
    const pts = series.map((p, i) => {
      const yv = (p[key] as number | undefined) ?? null;
      if (yv == null) return null;
      const x = i * stepX;
      const y = height - ((yv - minY) / spanY) * height;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    }).filter(Boolean) as string[];
    return pts.join(' ');
  }

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="block">
      <path d={pathFor('ee')} fill="none" stroke="#2563eb" strokeWidth={1.5} />
      <path d={pathFor('pfe95')} fill="none" stroke="#f59e0b" strokeWidth={1.2} />
      <path d={pathFor('pfe99')} fill="none" stroke="#dc2626" strokeWidth={1.2} />
    </svg>
  );
}

/* ---- Mini network graph ---- */

function MiniNetwork({
  graph, onSelect, selected,
}: { graph: { nodes: string[]; edges: GraphEdge[] }; onSelect: (cp: string) => void; selected: string | null }) {
  const width = 360, height = 220;
  const n = graph.nodes.length;
  const radius = Math.min(width, height) / 2 - 20;
  const centerX = width / 2;
  const centerY = height / 2;

  const positions = useMemo(() => {
    const pos: Record<string, { x: number; y: number }> = {};
    graph.nodes.forEach((node, i) => {
      const angle = (i / n) * Math.PI * 2;
      pos[node] = { x: centerX + radius * Math.cos(angle), y: centerY + radius * Math.sin(angle) };
    });
    return pos;
  }, [graph.nodes, n, centerX, centerY, radius]);

  const maxW = Math.max(...graph.edges.map((e) => e.weight), 1);

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="block">
      {graph.edges.map((e, idx) => {
        const a = positions[e.from], b = positions[e.to];
        if (!a || !b) return null;
        const w = 0.5 + (e.weight / maxW) * 3.0;
        return <line key={idx} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#9ca3af" strokeWidth={w} opacity={0.7} />;
      })}
      {graph.nodes.map((node) => {
        const p = positions[node];
        const isSel = selected === node;
        return (
          <g key={node} onClick={() => onSelect(node)} style={{ cursor: 'pointer' }}>
            <circle cx={p.x} cy={p.y} r={isSel ? 8 : 6} fill={isSel ? '#4f46e5' : '#374151'} />
            <text x={p.x} y={p.y - 10} fontSize="10" textAnchor="middle" fill="#6b7280">{node}</text>
          </g>
        );
      })}
    </svg>
  );
}