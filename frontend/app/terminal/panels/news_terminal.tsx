'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type Source = 'YAHOO' | 'MONEYCNTL' | 'REUTERS' | 'TWITTER' | 'OTHER';
type Importance = 'low' | 'medium' | 'high';

type NewsItem = {
  id: string;                 // stable id from backend; fallback: hash(url+ts)
  ts: string;                 // ISO timestamp
  title: string;
  url: string;
  source: Source | string;    // e.g., 'YAHOO', 'MONEYCNTL'
  tickers?: string[];         // ['AAPL','RELIANCE.NS']
  symbols?: string[];         // alias of tickers
  summary?: string;
  body?: string;
  image?: string;             // thumbnail url (optional)
  importance?: Importance;    // low/medium/high (optional)
  sentiment?: number;         // [-1..+1]
  language?: string;          // 'en', 'hi'
  country?: string;           // 'US', 'IN'
  tags?: string[];            // topic labels
  meta?: Record<string, any>; // anything else
};

type Patch = {
  rows?: NewsItem[];   // batch
  row?: NewsItem;      // single
  kpis?: { label: string; value: string | number }[];
  notes?: string;
};

type Props = {
  endpoint?: string;   // '/api/news/terminal'
  title?: string;
  className?: string;
};

/* ================= Utils ================= */
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const relTime = (iso?: string) => iso ? new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }).format(
  Math.round((new Date(iso).getTime() - Date.now()) / 60000), 'minute'
) : '';
const fmtTs = (iso?: string) => iso ? new Date(iso).toLocaleString() : '';
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const scoreToColor = (s:number|undefined) => {
  if (typeof s !== 'number') return '#6b7280';
  if (s >= 0.25) return '#059669';
  if (s <= -0.25) return '#dc2626';
  return '#f59e0b';
};
function download(filename:string, text:string, mime='text/plain'){
  const blob = new Blob([text], { type:mime }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}
function toCSV(rows: NewsItem[]) {
  const cols = ['ts','source','title','url','tickers','sentiment','importance','language','country','tags'];
  const head = cols.join(',');
  const lines = rows.map(r => {
    const get = (k:string) => {
      const v:any = (r as any)[k];
      if (Array.isArray(v)) return `"${v.join(' ').replace(/"/g,'""')}"`;
      return `"${String(v ?? '').replace(/"/g,'""')}"`;
    };
    return cols.map(get).join(',');
  });
  return [head, ...lines].join('\n');
}

/* ================= Component ================= */
export default function NewsTerminal({
  endpoint = '/api/news/terminal',
  title = 'News Terminal',
  className = '',
}: Props){
  // controls
  const [symbols, setSymbols] = useState('AAPL, TSLA, RELIANCE.NS');
  const [source, setSource]   = useState<'ALL' | 'YAHOO' | 'MONEYCNTL' | 'REUTERS' | 'TWITTER' | 'OTHER'>('ALL');
  const [query, setQuery]     = useState('');
  const [from, setFrom]       = useState('');
  const [to, setTo]           = useState('');
  const [minSent, setMinSent] = useState<number>(-1);
  const [language, setLanguage]=useState<'ALL'|'en'|'hi'>('ALL');
  const [country, setCountry] = useState<'ALL'|'US'|'IN'|'EU'|'GB'|'JP'>('ALL');
  const [sortBy, setSortBy]   = useState<'time'|'sentiment'|'source'>('time');

  // state
  const [rows, setRows] = useState<NewsItem[]>([]);
  const [kpis, setKpis] = useState<{label:string; value:string|number}[]>([]);
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string|null>(null);
  const [log, setLog] = useState('');
  const [selected, setSelected] = useState<NewsItem | null>(null);
  const [unread, setUnread] = useState<number>(0);

  const abortRef = useRef<AbortController|null>(null);
  const logRef = useRef<HTMLDivElement|null>(null);
  useEffect(()=>{ logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [log, loading]);

  const symbolList = useMemo(
    () => symbols.split(',').map(s=>s.trim().toUpperCase()).filter(Boolean),
    [symbols]
  );

  // derived filter/sort
  const filtered = useMemo(()=> {
    let out = rows.slice();
    if (source !== 'ALL') out = out.filter(r => (r.source || '').toUpperCase() === source);
    if (language !== 'ALL') out = out.filter(r => (r.language || '').toLowerCase() === language);
    if (country !== 'ALL') out = out.filter(r => (r.country || '').toUpperCase() === country);
    if (from) out = out.filter(r => (r.ts || '').slice(0,10) >= from);
    if (to) out = out.filter(r => (r.ts || '').slice(0,10) <= to);
    if (symbolList.length){
      out = out.filter(r => {
        const set = new Set([...(r.tickers||[]), ...(r.symbols||[])]
          .map(s => String(s).toUpperCase()));
        return symbolList.some(s => set.has(s));
      });
    }
    out = out.filter(r => (r.sentiment ?? -1) >= minSent);
    const q = query.trim().toLowerCase();
    if (q){
      out = out.filter(r =>
        [r.title, r.summary, r.body, r.source, ...(r.tags||[])]
          .filter(Boolean)
          .some(s => String(s).toLowerCase().includes(q))
      );
    }
    if (sortBy === 'time') out.sort((a,b)=> (b.ts||'').localeCompare(a.ts||''));
    else if (sortBy === 'sentiment') out.sort((a,b)=> (b.sentiment||0)-(a.sentiment||0));
    else if (sortBy === 'source') out.sort((a,b)=> String(a.source).localeCompare(String(b.source)));
    return out;
  }, [rows, source, language, country, from, to, symbolList, minSent, query, sortBy]);

  // auto KPIs
  useEffect(()=>{
    const N = filtered.length;
    const bySrc = new Map<string, number>();
    for (const r of filtered) bySrc.set(String(r.source||'OTHER'), (bySrc.get(String(r.source||'OTHER'))||0)+1);
    const avgS = N ? (filtered.reduce((s,r)=>s+(r.sentiment||0),0)/N) : 0;
    const topSrc = [...bySrc.entries()].sort((a,b)=>b[1]-a[1])[0]?.[0] || '—';
    setKpis([
      { label:'Items', value: N },
      { label:'Avg Sentiment', value: avgS.toFixed(2) },
      { label:'Top Source', value: topSrc },
    ]);
  }, [filtered]);

  /* ============== Streaming / Fetching ============== */
  function stop(){ abortRef.current?.abort(); abortRef.current=null; setLoading(false); }
  async function run(){
    setLoading(true); setError(null); setLog(''); setNotes('');
    setRows([]); setSelected(null); setUnread(0);
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    const payload = {
      id: uid(),
      symbols: symbolList,
      source: source==='ALL'? null : source,
      from: from || null,
      to: to || null,
      minSentiment: minSent,
      language: language==='ALL'? null : language,
      country: country==='ALL'? null : country,
      sort: sortBy,
    };

    try{
      const res = await fetch(endpoint, {
        method:'POST',
        headers:{ 'Content-Type':'application/json', 'Accept':'text/event-stream, application/json, text/plain' },
        body: JSON.stringify(payload),
        signal,
      });
      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && ctype.includes('text/event-stream')){
        const reader = res.body.getReader(); const dec = new TextDecoder('utf-8'); let acc='';
        while(true){
          const {value, done} = await reader.read(); if (done) break;
          const chunk = dec.decode(value, {stream:true});
          const lines = (acc + chunk).split('\n'); acc = lines.pop() || '';
          for (const line of lines){
            if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim(); if (!data) continue;
            if (data === '[DONE]') continue;
            setLog(s=>s+data+'\n');
            try { applyPatch(JSON.parse(data)); }
            catch { setNotes(n => (n?n+'\n':'') + data); }
          }
        }
      } else if (ctype.includes('application/json')){
        applyPatch(await res.json());
      } else {
        setNotes(await res.text());
      }
    }catch(e:any){
      if (e?.name!=='AbortError') setError(e?.message || 'Request failed');
    }finally{
      setLoading(false);
      abortRef.current = null;
    }
  }

  function applyPatch(p:Patch){
    if (Array.isArray(p.kpis)) {
      setKpis(prev=>{
        const map=new Map(prev.map(k=>[k.label,k.value]));
        for(const k of p.kpis!) map.set(k.label,k.value);
        return Array.from(map.entries()).map(([label,value])=>({label, value}));
      });
    }
    if (Array.isArray(p.rows)) setRows(prev => mergeRows(prev, p.rows!));
    if (p.row) setRows(prev => mergeRows(prev, [p.row!]));
    if (typeof p.notes === 'string') setNotes(n => (n?n+'\n':'') + p.notes);
    setUnread(u=>u+((p.rows?.length||0) + (p.row ? 1 : 0)));
  }

  function mergeRows(prev: NewsItem[], incoming: NewsItem[]){
    const norm = (s:string)=> s.trim().replace(/^https?:\/\//,'').replace(/\/$/,'');
    const key = (r:NewsItem)=> r.id || `${norm(r.url||'')}_${r.ts||''}`;
    const map = new Map<string, NewsItem>(prev.map(r=>[key(r), r]));
    for (const r of incoming){
      const k = key(r);
      const old = map.get(k);
      map.set(k, { ...(old||{}), ...r });
    }
    // keep most recent first
    return Array.from(map.values()).sort((a,b)=> (b.ts||'').localeCompare(a.ts||''));
  }

  /* ============== Export ============== */
  function exportCSV(){ download('news.csv', toCSV(filtered), 'text/csv'); }
  function exportJSON(){ download('news.json', JSON.stringify(filtered, null, 2), 'application/json'); }

  /* ============== TTS (optional) ============== */
  function speakSelected(){
    if (!selected) return;
    // guard for SSR and browser support
    if (typeof window === 'undefined' || !(window as any).speechSynthesis) return;
    const synth = (window as any).speechSynthesis as SpeechSynthesis;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(`${selected.source}. ${selected.title}. ${selected.summary || ''}`);
    u.lang = (selected.language === 'hi') ? 'hi-IN' : 'en-US';
    synth.speak(u);
  }

  /* ================= UI ================= */
  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}{unread? <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-indigo-600 text-white">{unread} new</span>:null}</div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export CSV</button>
          <button onClick={exportJSON} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export JSON</button>
          {loading
            ? <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
            : <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>}
        </div>
      </div>

      {/* controls */}
      <div className="p-3 grid grid-cols-1 md:grid-cols-10 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <Field label="Symbols (CSV)">
          <input value={symbols} onChange={e=>setSymbols(e.target.value)} placeholder="AAPL, TSLA, RELIANCE.NS"
                 className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
        <Field label="Source">
          <select value={source} onChange={e=>setSource(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option>
            <option value="YAHOO">Yahoo</option>
            <option value="MONEYCNTL">Moneycontrol</option>
            <option value="REUTERS">Reuters</option>
            <option value="TWITTER">Twitter</option>
            <option value="OTHER">Other</option>
          </select>
        </Field>
        <Field label="From"><input type="date" value={from} onChange={e=>setFrom(e.target.value)} className="w-full rounded border px-2 py-2 text-sm"/></Field>
        <Field label="To"><input type="date" value={to} onChange={e=>setTo(e.target.value)} className="w-full rounded border px-2 py-2 text-sm"/></Field>
        <Field label="Min sentiment">
          <input type="number" step="0.05" min={-1} max={1} value={minSent}
                 onChange={e=>setMinSent(clamp(Number(e.target.value)||-1,-1,1))}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Field label="Language">
          <select value={language} onChange={e=>setLanguage(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option><option value="en">English</option><option value="hi">Hindi</option>
          </select>
        </Field>
        <Field label="Country">
          <select value={country} onChange={e=>setCountry(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option><option value="US">US</option><option value="IN">IN</option><option value="EU">EU</option><option value="GB">GB</option><option value="JP">JP</option>
          </select>
        </Field>
        <Field label="Sort">
          <select value={sortBy} onChange={e=>setSortBy(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="time">Time</option><option value="sentiment">Sentiment</option><option value="source">Source</option>
          </select>
        </Field>
        <Field label="Search">
          <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="title/summary/tags…" className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* left: KPIs + notes */}
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
          <Card title="Notes">
            <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || 'No notes.'}</pre>
            {error && <div className="text-xs text-rose-600 mt-2">{error}</div>}
          </Card>
        </div>

        {/* middle: feed */}
        <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Feed ({filtered.length})</div>
          <div className="overflow-auto">
            {filtered.length ? (
              <ul className="divide-y divide-neutral-200 dark:divide-neutral-800">
                {filtered.map((r,i)=>(
                  <li key={r.id || `${r.url}-${i}`} className={`hover:bg-neutral-50 dark:hover:bg-neutral-900 cursor-pointer`}
                      onClick={()=>{ setSelected(r); setUnread(u=> Math.max(0, u-1)); }}>
                    <div className="p-3 grid grid-cols-12 gap-2 items-center">
                      {/* sentiment bar */}
                      <div className="col-span-1 flex flex-col items-center">
                        <div className="h-10 w-1.5 rounded" style={{ background: scoreToColor(r.sentiment) }} title={`Sentiment: ${r.sentiment ?? '—'}`}/>
                        <div className="text-[10px] mt-1 text-neutral-500">{(r.sentiment!=null? (r.sentiment>0?'+':'')+r.sentiment.toFixed(2) : '')}</div>
                      </div>
                      {/* title + meta */}
                      <div className="col-span-9">
                        <div className="flex items-center gap-2 text-xs">
                          <Badge>{String(r.source || '').toUpperCase()}</Badge>
                          {r.importance && <Badge tone={r.importance==='high'?1:r.importance==='low'?-1:0}>{r.importance}</Badge>}
                          {r.language && <Badge>{r.language}</Badge>}
                          {r.country && <Badge>{r.country}</Badge>}
                          {r.tickers?.slice(0,4).map(t=><Badge key={t}>{t}</Badge>)}
                          {r.tags?.slice(0,3).map(t=><Badge key={t} variant="soft">{t}</Badge>)}
                          <span className="ml-auto text-[11px] text-neutral-500">{relTime(r.ts)}</span>
                        </div>
                        <div className="text-sm font-medium mt-0.5">{r.title}</div>
                        <div className="text-[12px] text-neutral-600 dark:text-neutral-300 line-clamp-2">{r.summary || r.body || ''}</div>
                      </div>
                      {/* thumb + actions */}
                      <div className="col-span-2 flex flex-col items-end gap-2">
                        {r.image ? <img src={r.image} alt="" className="h-12 w-16 object-cover rounded border border-neutral-200 dark:border-neutral-800"/> : <div className="h-12 w-16 rounded bg-neutral-100 dark:bg-neutral-800" />}
                        <div className="flex gap-2">
                          <a href={r.url} target="_blank" rel="noreferrer" className="text-[11px] underline text-indigo-600">Open</a>
                          <button className="text-[11px] underline" onClick={(e)=>{e.stopPropagation(); setSelected(r); speakSelected();}}>Speak</button>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            ) : <div className="p-6 text-xs text-neutral-500">No results. Click <span className="font-medium">Run</span> to fetch or adjust filters.</div>}
          </div>
        </div>
      </div>

      {/* detail + log */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 border-t border-neutral-200 dark:border-neutral-800">
        <Card title={`Detail${selected ? ` · ${selected.source}` : ''}`}>
          {selected ? (
            <div className="text-sm space-y-1">
              <Row label="Time" value={fmtTs(selected.ts)} />
              <Row label="Title" value={selected.title} />
              <Row label="Source" value={String(selected.source)} />
              <Row label="Tickers" value={(selected.tickers||selected.symbols||[]).join(', ')} />
              <Row label="Sentiment" value={selected.sentiment!=null ? selected.sentiment.toFixed(2) : '—'} />
              <Row label="Importance" value={selected.importance || '—'} />
              <Row label="Language" value={selected.language || '—'} />
              <Row label="Country" value={selected.country || '—'} />
              <div className="pt-2 text-[11px] text-neutral-500">Summary</div>
              <div className="text-sm">{selected.summary || selected.body || '—'}</div>
              <div className="pt-2"><a className="text-xs underline text-indigo-600" href={selected.url} target="_blank" rel="noreferrer">Open original</a></div>
            </div>
          ) : <Empty label="Select a story to see details."/>}
        </Card>
        <Card title="Run Log">
          <div ref={logRef} className="max-h-[26vh] overflow-auto">
            {error ? <div className="text-xs text-rose-600">{error}</div> : <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">{log || 'No streaming output yet.'}</pre>}
          </div>
        </Card>
        <Card title="Tips">
          <ul className="text-xs list-disc pl-4 space-y-1 text-neutral-600 dark:text-neutral-300">
            <li>Use symbols like <code>AAPL, TSLA, RELIANCE.NS</code> to filter ticker-linked headlines.</li>
            <li>Set <em>Min sentiment</em> ≥ 0.2 for bullish screens; ≤ -0.2 for bearish.</li>
            <li><em>Speak</em> uses the browser’s speech engine (if available).</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}

/* ================= Subcomponents ================= */
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
function Badge({ children, tone=0, variant='solid' }:{ children:React.ReactNode; tone?:-1|0|1; variant?:'solid'|'soft' }){
  const base = 'inline-flex items-center rounded px-1.5 py-0.5 text-[10px]';
  const solid = tone>0 ? 'bg-emerald-600 text-white' : tone<0 ? 'bg-rose-600 text-white' : 'bg-neutral-800 text-white';
  const soft = tone>0 ? 'bg-emerald-100 text-emerald-700' : tone<0 ? 'bg-rose-100 text-rose-700' : 'bg-neutral-100 text-neutral-700';
  return <span className={`${base} ${variant==='solid'?solid:soft}`}>{children}</span>;
}