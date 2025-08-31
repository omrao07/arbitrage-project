'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */

type Importance = 'low' | 'medium' | 'high';
type EventType =
  | 'CPI' | 'PPI' | 'GDP' | 'PMI' | 'JOBS' | 'RETAIL' | 'CB' | 'RATE' | 'INFLATION'
  | 'AUCTION' | 'TRADE' | 'CONF' | 'SENTIMENT' | 'HOUSING' | 'OTHER';

type EconEvent = {
  id: string;
  time: string;         // ISO timestamp e.g., 2025-08-30T12:30:00Z
  country: string;      // e.g., US, IN, EU, GB, JP
  region?: string;      // e.g., Americas, EMEA, APAC
  ticker?: string;      // calendar code, e.g., "US CPI"
  type: EventType;
  title: string;        // "CPI YoY (Aug)"
  importance: Importance;
  actual?: number | string;
  forecast?: number | string;
  previous?: number | string;
  unit?: string;        // %, bps, bn, k
  source?: string;      // provider (e.g., "EconDB", "TradingEconomics")
  notes?: string;
  currency?: string;    // reporting currency if monetary
};

type Patch = {
  kpis?: { label: string; value: string | number }[];
  rows?: EconEvent[];
  row?:  EconEvent;
  columns?: string[];
  notes?: string;
};

type Props = {
  endpoint?: string; // e.g. '/api/econ/calendar'
  title?: string;
  className?: string;
};

/* ================= Utils ================= */

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const fmtTs = (iso?: string) => (iso ? new Date(iso).toLocaleString() : '');
const fmtDate = (iso?: string) => (iso ? new Date(iso).toLocaleDateString() : '');
const fmtTime = (iso?: string) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '');

function download(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function toCSV(rows: EconEvent[], columns: string[]): string {
  const header = columns.join(',');
  const lines = rows.map(r =>
    columns.map(c => {
      const v: any = (r as any)[c];
      if (typeof v === 'number') return v;
      return `"${String(v ?? '').replace(/"/g, '""')}"`;
    }).join(',')
  );
  return [header, ...lines].join('\n');
}

function toICS(rows: EconEvent[]): string {
  // Minimal ICS converting each event's time to DTSTART
  const wrap = (x: string) => x.replace(/(\r\n|\n|\r)/gm, '\\n');
  const dtstamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  const ev = rows.map(r => {
    if (!r.time) return '';
    const dt = r.time.replace(/[-:]/g, '').replace('.000Z', 'Z');
    const title = `${r.country} · ${r.title} (${r.type})`;
    const desc = [
      `Country: ${r.country}`,
      `Type: ${r.type}`,
      `Importance: ${r.importance}`,
      `Actual: ${r.actual ?? ''}`,
      `Forecast: ${r.forecast ?? ''}`,
      `Previous: ${r.previous ?? ''}`,
      `Notes: ${r.notes ?? ''}`,
      `Source: ${r.source ?? ''}`,
    ].join('\\n');
    return [
      'BEGIN:VEVENT',
      `UID:${r.id || uid()}`,
      `DTSTAMP:${dtstamp}`,
      `DTSTART:${dt}`,
      `SUMMARY:${wrap(title)}`,
      `DESCRIPTION:${wrap(desc)}`,
      'END:VEVENT',
    ].join('\n');
  }).filter(Boolean).join('\n');
  return `BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Your App//Econ Calendar//EN\n${ev}\nEND:VCALENDAR\n`;
}

function dotColor(imp: Importance) {
  if (imp === 'high') return '#dc2626';
  if (imp === 'medium') return '#f59e0b';
  return '#10b981';
}

/* ================= Component ================= */

export default function EconCalender({
  endpoint = '/api/econ/calendar',
  title = 'Economic Calendar',
  className = '',
}: Props) {
  // Filters
  const [from, setFrom] = useState<string>(''); // YYYY-MM-DD
  const [to, setTo] = useState<string>('');     // YYYY-MM-DD
  const [countries, setCountries] = useState<string>('US, IN, EU, GB, JP');
  const [regions, setRegions] = useState<string>('ALL'); // ALL | Americas, EMEA, APAC
  const [importance, setImportance] = useState<'ALL' | Importance>('ALL');
  const [types, setTypes] = useState<'ALL' | EventType>('ALL');
  const [query, setQuery] = useState<string>('');

  // Data/state
  const [kpis, setKpis] = useState<{ label: string; value: string | number }[]>([]);
  const [rows, setRows] = useState<EconEvent[]>([]);
  const [columns, setColumns] = useState<string[] | null>(null);
  const [notes, setNotes] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const [selected, setSelected] = useState<EconEvent | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' }); }, [log, loading]);

  // Derived
  const countryList = useMemo(
    () => countries.split(',').map(s => s.trim().toUpperCase()).filter(Boolean),
    [countries]
  );

  const filtered = useMemo(() => {
    let out = rows.slice();
    if (importance !== 'ALL') out = out.filter(r => r.importance === importance);
    if (types !== 'ALL') out = out.filter(r => r.type === types);
    if (regions !== 'ALL') out = out.filter(r => (r.region || '').toUpperCase() === regions.toUpperCase());
    if (countryList.length) out = out.filter(r => countryList.includes((r.country || '').toUpperCase()));
    if (from) out = out.filter(r => (r.time || '').slice(0, 10) >= from);
    if (to) out = out.filter(r => (r.time || '').slice(0, 10) <= to);
    const q = query.trim().toLowerCase();
    if (q) {
      out = out.filter(r => [r.title, r.ticker, r.type, r.notes, r.source]
        .filter(Boolean)
        .some(s => String(s).toLowerCase().includes(q)));
    }
    out.sort((a, b) => (a.time || '').localeCompare(b.time || ''));
    return out;
  }, [rows, importance, types, regions, countryList, from, to, query]);

  const days = useMemo(() => {
    // Group by YYYY-MM-DD
    const bucket = new Map<string, EconEvent[]>();
    for (const ev of filtered) {
      const d = (ev.time || '').slice(0, 10) || 'TBD';
      (bucket.get(d) || bucket.set(d, []).get(d)!).push(ev);
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
      from: from || null,
      to: to || null,
      countries: countryList,
      regions: regions === 'ALL' ? null : regions,
      importance: importance === 'ALL' ? null : importance,
      type: types === 'ALL' ? null : types,
      query: query || null,
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

  function mergeRows(prev: EconEvent[], incoming: EconEvent[]) {
    const map = new Map<string, EconEvent>();
    for (const r of prev) map.set(r.id, r);
    for (const r of incoming) map.set(r.id, { ...(map.get(r.id) || {}), ...r });
    return Array.from(map.values());
  }

  function exportCSV() {
    const cols =
      columns?.length
        ? columns
        : ['time','country','region','type','title','importance','actual','forecast','previous','unit','source','notes'];
    download('econ_calendar.csv', toCSV(filtered, cols), 'text/csv');
  }
  function exportICS() {
    download('econ_calendar.ics', toICS(filtered), 'text/calendar');
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
        <Field label="From"><input type="date" value={from} onChange={e=>setFrom(e.target.value)} className="w-full rounded border px-2 py-2 text-sm"/></Field>
        <Field label="To"><input type="date" value={to} onChange={e=>setTo(e.target.value)} className="w-full rounded border px-2 py-2 text-sm"/></Field>
        <Field label="Countries (CSV)">
          <input value={countries} onChange={e=>setCountries(e.target.value)} placeholder="US, IN, EU, GB, JP" className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
        <Field label="Region">
          <select value={regions} onChange={e=>setRegions(e.target.value)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option><option>Americas</option><option>EMEA</option><option>APAC</option>
          </select>
        </Field>
        <Field label="Importance">
          <select value={importance} onChange={e=>setImportance(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
          </select>
        </Field>
        <Field label="Type">
          <select value={types} onChange={e=>setTypes(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option>
            {(['CPI','PPI','GDP','PMI','JOBS','RETAIL','CB','RATE','INFLATION','AUCTION','TRADE','CONF','SENTIMENT','HOUSING','OTHER'] as EventType[]).map(t=>
              <option key={t} value={t}>{t}</option>
            )}
          </select>
        </Field>
        <Field label="Search">
          <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="title/ticker/source…" className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* left: KPIs + timeline */}
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
            ) : <Empty label="No KPIs yet" />}
          </Card>

          <Card title="Timeline">
            {days.length ? (
              <div className="space-y-3 max-h-[50vh] overflow-auto pr-1">
                {days.map(([d, list])=>(
                  <div key={d} className="border-l-2 pl-3">
                    <div className="text-xs font-semibold mb-1">{fmtDate(d)}</div>
                    <div className="space-y-1">
                      {list.map(ev=>(
                        <button
                          key={ev.id}
                          onClick={()=>setSelected(ev)}
                          className="w-full text-left rounded border border-neutral-200 dark:border-neutral-800 px-2 py-1 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                          title={ev.notes || ''}
                        >
                          <div className="flex items-center gap-2">
                            <span className="inline-block h-2 w-2 rounded-full" style={{ background: dotColor(ev.importance) }} />
                            <span className="text-xs font-medium">{ev.country}</span>
                            <span className="text-[11px] text-neutral-500">{fmtTime(ev.time)}</span>
                            <span className="text-xs truncate">{ev.title}</span>
                            {ev.forecast!=null && <span className="text-[11px] text-indigo-600 ml-auto">F: {String(ev.forecast)}{ev.unit || ''}</span>}
                            {ev.actual!=null && <span className="text-[11px] text-emerald-700">A: {String(ev.actual)}{ev.unit || ''}</span>}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : <Empty label="No items" />}
          </Card>
        </div>

        {/* table */}
        <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Events ({filtered.length})</div>
          <div className="overflow-auto">
            <Table rows={filtered} columns={columns} onSelect={setSelected} />
          </div>
        </div>
      </div>

      {/* details + log */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 border-t border-neutral-200 dark:border-neutral-800">
        <Card title={`Detail${selected ? ` · ${selected.country}` : ''}`}>
          {selected ? (
            <div className="text-sm space-y-1">
              <Row label="When" value={fmtTs(selected.time)} />
              <Row label="Country" value={selected.country} />
              <Row label="Region" value={selected.region || ''} />
              <Row label="Type" value={selected.type} />
              <Row label="Title" value={selected.title} />
              <Row label="Importance" value={selected.importance} />
              <Row label="Actual" value={`${selected.actual ?? ''} ${selected.unit ?? ''}`} />
              <Row label="Forecast" value={`${selected.forecast ?? ''} ${selected.unit ?? ''}`} />
              <Row label="Previous" value={`${selected.previous ?? ''} ${selected.unit ?? ''}`} />
              <Row label="Source" value={selected.source || ''} />
              <Row label="Notes" value={selected.notes || ''} />
            </div>
          ) : <Empty label="Select an event to see details."/>}
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
function Row({ label, value }: { label: string; value: string | number }) {
  return <div className="flex justify-between gap-2"><div className="text-[11px] text-neutral-500">{label}</div><div className="text-sm">{String(value)}</div></div>;
}

function Table({ rows, columns, onSelect }:{ rows:EconEvent[]; columns:string[]|null; onSelect:(r:EconEvent)=>void }) {
  const cols = columns?.length
    ? columns
    : ['time','country','region','type','title','importance','actual','forecast','previous','unit','source'];
  return (
    <table className="min-w-full text-sm">
      <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
        <tr>{cols.map(c => <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r,i)=>(
          <tr key={r.id} onClick={()=>onSelect(r)}
              className={`${i%2?'bg-neutral-50 dark:bg-neutral-950':'bg-white dark:bg-neutral-900'} cursor-pointer`}>
            {cols.map(c=>{
              let v:any = (r as any)[c];
              if (c === 'time') v = fmtTs(r.time);
              return <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{String(v ?? '')}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}