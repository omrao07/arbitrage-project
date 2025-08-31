'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */

type ActionType =
  | 'DIVIDEND' | 'SPLIT' | 'BONUS' | 'RIGHTS' | 'MERGER'
  | 'SPINOFF' | 'BUYBACK' | 'AGM' | 'EGM' | 'BOARD'
  | 'EARNINGS' | 'OTHER';

type CorpAction = {
  id: string;
  symbol: string;
  exchange?: string;
  country?: string;
  type: ActionType;
  exDate?: string;      // YYYY-MM-DD
  recordDate?: string;
  payDate?: string;
  announceDate?: string;
  ratio?: string;       // e.g., "1:2" or "3-for-1"
  amount?: number;      // dividend per share
  currency?: string;
  notes?: string;
  source?: string;      // provider
};

type Patch = {
  columns?: string[];
  row?: CorpAction;
  rows?: CorpAction[];
  notes?: string;
  kpis?: { label: string; value: string | number }[];
};

type Props = {
  endpoint?: string;           // e.g. '/api/corp-actions'
  title?: string;
  className?: string;
};

/* ================= Utils ================= */

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const fmtMoney = (x: number | undefined, c = 'USD') =>
  typeof x === 'number'
    ? new Intl.NumberFormat(undefined, { style: 'currency', currency: c, maximumFractionDigits: 4 }).format(x)
    : '';
const fmtDate = (s?: string) => (s ? new Date(s + 'T00:00:00Z').toLocaleDateString() : '');
const toISO = (s: string) => s; // assuming YYYY-MM-DD incoming

/* Export helpers */
function download(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function toCSV(rows: CorpAction[], columns: string[]): string {
  const header = columns.join(',');
  const lines = rows.map((r) =>
    columns.map((c) => {
      const v: any = (r as any)[c];
      if (typeof v === 'number') return v;
      return `"${String(v ?? '').replace(/"/g, '""')}"`;
    }).join(','),
  );
  return [header, ...lines].join('\n');
}

function toICS(rows: CorpAction[]): string {
  // Minimal ICS for exDate (or announceDate/payDate fallback)
  const wrap = (x: string) => x.replace(/(\r\n|\n|\r)/gm, '\\n');
  const dtstamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  const ev = rows.map((r) => {
    const when = r.exDate || r.announceDate || r.payDate || r.recordDate;
    if (!when) return '';
    const dt = when.replace(/-/g, '') + 'T090000Z';
    const title = `${r.symbol} • ${r.type}${r.amount ? ` ${r.currency || ''} ${r.amount}` : r.ratio ? ` ${r.ratio}` : ''}`.trim();
    const desc =
      `Symbol: ${r.symbol}\\nType: ${r.type}\\nEx: ${r.exDate || ''}\\nRecord: ${r.recordDate || ''}\\nPay: ${r.payDate || ''}\\nNotes: ${wrap(r.notes || '')}`;
    return [
      'BEGIN:VEVENT',
      `UID:${r.id || uid()}`,
      `DTSTAMP:${dtstamp}`,
      `DTSTART:${dt}`,
      `SUMMARY:${wrap(title)}`,
      `DESCRIPTION:${desc}`,
      'END:VEVENT',
    ].join('\n');
  }).filter(Boolean).join('\n');
  return `BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Your App//Corporate Actions//EN\n${ev}\nEND:VCALENDAR\n`;
}

/* Small viz */
function dotColor(t: ActionType) {
  switch (t) {
    case 'DIVIDEND': return '#10b981';
    case 'SPLIT': return '#6366f1';
    case 'BONUS': return '#8b5cf6';
    case 'RIGHTS': return '#f59e0b';
    case 'MERGER': return '#ef4444';
    case 'SPINOFF': return '#22c55e';
    case 'BUYBACK': return '#06b6d4';
    case 'AGM':
    case 'EGM':
    case 'BOARD': return '#64748b';
    case 'EARNINGS': return '#e11d48';
    default: return '#9ca3af';
  }
}

/* ================= Component ================= */

export default function CorporateActions({
  endpoint = '/api/corp-actions',
  title = 'Corporate Actions',
  className = '',
}: Props) {
  // Filters
  const [symbols, setSymbols] = useState<string>('AAPL, MSFT, RELIANCE.NS');
  const [types, setTypes] = useState<ActionType | 'ALL'>('ALL');
  const [from, setFrom] = useState<string>(''); // YYYY-MM-DD
  const [to, setTo] = useState<string>('');     // YYYY-MM-DD
  const [country, setCountry] = useState<string>('ALL');
  const [query, setQuery] = useState<string>('');
  const [onlyUpcoming, setOnlyUpcoming] = useState<boolean>(true);

  // Data/state
  const [kpis, setKpis] = useState<{ label: string; value: string | number }[]>([]);
  const [rows, setRows] = useState<CorpAction[]>([]);
  const [columns, setColumns] = useState<string[] | null>(null);
  const [notes, setNotes] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  const [selected, setSelected] = useState<CorpAction | null>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [log, loading]);

  // Derived
  const symbolList = useMemo(
    () => symbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean),
    [symbols]
  );

  const filtered = useMemo(() => {
    let out = rows.slice();
    if (types !== 'ALL') out = out.filter(r => r.type === types);
    if (country !== 'ALL') out = out.filter(r => (r.country || '').toUpperCase() === country.toUpperCase());
    if (onlyUpcoming) {
      const today = new Date().toISOString().slice(0, 10);
      out = out.filter(r => (r.exDate || r.announceDate || r.payDate || '9999-12-31') >= today);
    }
    if (symbolList.length) out = out.filter(r => symbolList.includes(r.symbol.toUpperCase()));
    if (from) out = out.filter(r => [r.exDate, r.announceDate, r.payDate, r.recordDate].some(d => (d || '') >= from));
    if (to) out = out.filter(r => [r.exDate, r.announceDate, r.payDate, r.recordDate].some(d => (d || '') <= to));
    const q = query.trim().toLowerCase();
    if (q) {
      out = out.filter(r =>
        [r.symbol, r.type, r.notes, r.exchange, r.country, r.source]
          .filter(Boolean)
          .some(s => String(s).toLowerCase().includes(q)),
      );
    }
    out.sort((a, b) => (a.exDate || a.announceDate || a.payDate || '').
      localeCompare(b.exDate || b.announceDate || b.payDate || ''));
    return out;
  }, [rows, types, country, onlyUpcoming, symbolList, from, to, query]);

  const timeline = useMemo(() => {
    // group by date; each date shows badges
    const bucket = new Map<string, CorpAction[]>();
    for (const r of filtered) {
      const d = r.exDate || r.announceDate || r.payDate || r.recordDate || 'TBD';
      if (!bucket.has(d)) bucket.set(d, []);
      bucket.get(d)!.push(r);
    }
    return Array.from(bucket.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [filtered]);

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  async function run() {
    setLoading(true);
    setError(null);
    setLog('');
    setKpis([]); setRows([]); setColumns(null); setNotes(''); setSelected(null);

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;
    const payload = {
      id: uid(),
      symbols: symbolList,
      types: types === 'ALL' ? null : types,
      from: from || null,
      to: to || null,
      country: country === 'ALL' ? null : country,
      onlyUpcoming,
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

      if (res.body && ctype.includes('text/event-stream')) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = (acc + chunk).split('\n'); acc = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim(); if (!data) continue;
            if (data === '[DONE]') continue;
            setLog(s => s + data + '\n');
            try { applyPatch(JSON.parse(data)); }
            catch { setNotes(n => (n ? n + '\n' : '') + data); }
          }
        }
      } else if (ctype.includes('application/json')) {
        applyPatch(await res.json());
      } else {
        setNotes(await res.text());
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function applyPatch(p: Patch) {
    if (Array.isArray(p.kpis)) setKpis(p.kpis);
    if (Array.isArray(p.columns)) setColumns(p.columns);
    if (Array.isArray(p.rows)) setRows(prev => mergeRows(prev, p.rows!));
    if (p.row) setRows(prev => mergeRows(prev, [p.row!]));
    if (typeof p.notes === 'string') setNotes(n => (n ? n + '\n' : '') + p.notes);
  }

  function mergeRows(prev: CorpAction[], incoming: CorpAction[]) {
    const map = new Map<string, CorpAction>();
    for (const r of prev) map.set(r.id || `${r.symbol}:${r.type}:${r.exDate || r.announceDate || r.payDate || ''}`, r);
    for (const r of incoming) {
      const k = r.id || `${r.symbol}:${r.type}:${r.exDate || r.announceDate || r.payDate || ''}`;
      map.set(k, { ...(map.get(k) || {}), ...r });
    }
    return Array.from(map.values());
  }

  function exportCSV() {
    const cols =
      columns ||
      ['symbol','type','exDate','recordDate','payDate','announceDate','amount','currency','ratio','exchange','country','source','notes'];
    download('corporate_actions.csv', toCSV(filtered, cols), 'text/csv');
  }
  function exportICS() {
    download('corporate_actions.ics', toICS(filtered), 'text/calendar');
  }

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export CSV</button>
          <button onClick={exportICS} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export ICS</button>
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
          ) : (
            <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>
          )}
        </div>
      </div>

      {/* controls */}
      <div className="p-3 grid grid-cols-1 md:grid-cols-8 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <Field label="Symbols (CSV)">
          <input value={symbols} onChange={e=>setSymbols(e.target.value)} placeholder="AAPL, MSFT, RELIANCE.NS"
                 className="w-full rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"/>
        </Field>
        <Field label="Type">
          <select value={types} onChange={e=>setTypes(e.target.value as any)}
                  className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option>
            {(['DIVIDEND','SPLIT','BONUS','RIGHTS','MERGER','SPINOFF','BUYBACK','AGM','EGM','BOARD','EARNINGS','OTHER'] as ActionType[]).map(t=>
              <option key={t} value={t}>{t}</option>
            )}
          </select>
        </Field>
        <Field label="From"><input type="date" value={from} onChange={e=>setFrom(e.target.value)}
                                   className="w-full rounded border px-2 py-2 text-sm"/></Field>
        <Field label="To"><input type="date" value={to} onChange={e=>setTo(e.target.value)}
                                 className="w-full rounded border px-2 py-2 text-sm"/></Field>
        <Field label="Country">
          <select value={country} onChange={e=>setCountry(e.target.value)}
                  className="w-full rounded border px-2 py-2 text-sm">
            <option>ALL</option><option>US</option><option>IN</option><option>EU</option><option>GB</option><option>JP</option>
          </select>
        </Field>
        <Field label="Search">
          <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="symbol/type/exchange…"
                 className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
        <div className="flex items-end">
          <label className="inline-flex items-center gap-2 text-sm">
            <input type="checkbox" className="accent-indigo-600" checked={onlyUpcoming} onChange={e=>setOnlyUpcoming(e.target.checked)}/>
            Upcoming only
          </label>
        </div>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* left column: KPIs + timeline */}
        <div className="space-y-3">
          <Card title="KPIs">
            {kpis.length ? (
              <div className="grid grid-cols-2 gap-2">
                {kpis.map((k,i)=>(
                  <div key={i} className="rounded border border-neutral-200 dark:border-neutral-800 px-3 py-2">
                    <div className="text-[11px] text-neutral-500">{k.label}</div>
                    <div className="text-sm font-semibold">{String(k.value)}</div>
                  </div>
                ))}
              </div>
            ) : <Empty label="No KPIs yet"/>}
          </Card>

          <Card title="Timeline">
            {timeline.length ? (
              <div className="space-y-3 max-h-[50vh] overflow-auto pr-1">
                {timeline.map(([d, list]) => (
                  <div key={d} className="border-l-2 pl-3">
                    <div className="text-xs font-semibold mb-1">{fmtDate(d)}</div>
                    <div className="space-y-1">
                      {list.map(item => (
                        <button
                          key={`${item.symbol}-${item.type}-${item.exDate||item.announceDate||item.payDate}`}
                          onClick={()=>setSelected(item)}
                          className="w-full text-left rounded border border-neutral-200 dark:border-neutral-800 px-2 py-1 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                          title={item.notes || ''}
                        >
                          <div className="flex items-center gap-2">
                            <span className="inline-block h-2 w-2 rounded-full" style={{ background: dotColor(item.type) }} />
                            <span className="text-xs font-medium">{item.symbol}</span>
                            <span className="text-[11px] text-neutral-500">{item.type}</span>
                            {item.amount!=null && <span className="text-[11px] text-emerald-700">{fmtMoney(item.amount, item.currency || 'USD')}</span>}
                            {item.ratio && <span className="text-[11px] text-indigo-600">{item.ratio}</span>}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : <Empty label="No items"/>}
          </Card>
        </div>

        {/* table */}
        <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Corporate Actions ({filtered.length})</div>
          <div className="overflow-auto">
            <Table rows={filtered} columns={columns} onSelect={setSelected}/>
          </div>
        </div>
      </div>

      {/* details + log */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 border-t border-neutral-200 dark:border-neutral-800">
        <Card title={`Detail${selected ? ` · ${selected.symbol}` : ''}`}>
          {selected ? (
            <div className="text-sm space-y-1">
              <Row label="Symbol" value={selected.symbol}/>
              <Row label="Type" value={selected.type}/>
              <Row label="Ex-Date" value={fmtDate(selected.exDate)}/>
              <Row label="Record" value={fmtDate(selected.recordDate)}/>
              <Row label="Pay" value={fmtDate(selected.payDate)}/>
              <Row label="Announced" value={fmtDate(selected.announceDate)}/>
              <Row label="Amount" value={fmtMoney(selected.amount, selected.currency || 'USD')}/>
              <Row label="Ratio" value={selected.ratio || ''}/>
              <Row label="Exchange" value={selected.exchange || ''}/>
              <Row label="Country" value={selected.country || ''}/>
              <Row label="Source" value={selected.source || ''}/>
              <Row label="Notes" value={selected.notes || ''}/>
            </div>
          ) : <Empty label="Select a row to see details."/>}
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

/* =============== Subcomponents =============== */

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}
function Empty({ label }: { label: string }) { return <div className="text-xs text-neutral-500">{label}</div>; }
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="block text-[11px] text-neutral-500 mb-1">{label}</label>{children}</div>;
}
function Row({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-2"><div className="text-[11px] text-neutral-500">{label}</div><div className="text-sm">{value}</div></div>;
}

function Table({ rows, columns, onSelect }:{ rows:CorpAction[]; columns:string[]|null; onSelect:(r:CorpAction)=>void }) {
  const cols = columns?.length ? columns : ['symbol','type','exDate','recordDate','payDate','announceDate','amount','currency','ratio','exchange','country','source'];
  return (
    <table className="min-w-full text-sm">
      <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
        <tr>{cols.map(c => <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r,i)=>(
          <tr key={r.id || `${r.symbol}-${r.type}-${r.exDate||r.announceDate||r.payDate||i}`}
              onClick={()=>onSelect(r)}
              className={`${i%2?'bg-neutral-50 dark:bg-neutral-950':'bg-white dark:bg-neutral-900'} cursor-pointer`}>
            {cols.map(c=>{
              const v:any = (r as any)[c];
              let text = '';
              if (['exDate','recordDate','payDate','announceDate'].includes(c)) text = fmtDate(v);
              else if (c==='amount') text = fmtMoney(v, r.currency || 'USD');
              else text = String(v ?? '');
              return <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{text}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}