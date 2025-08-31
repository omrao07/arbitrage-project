'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type Side = 'B' | 'S' | 'N'; // buy / sell / neutral
type Print = {
  ts: number;          // epoch ms
  px: number;          // price
  sz: number;          // size (shares/contracts)
  side?: Side;         // B/S/N (if your feed infers aggressor)
  venue?: string;      // exchange/venue code
  id?: string;         // unique id if available
};
type Patch =
  | { rows: Print[] }   // batch
  | { row: Print }      // single
  | { notes: string };

type Props = {
  symbol?: string;           // default symbol shown
  httpEndpoint?: string;     // POST to fetch snapshot/seed (JSON)
  sseEndpoint?: string;      // POST that replies with text/event-stream patches
  title?: string;
  className?: string;
};

/* ================= Utils ================= */
const fmt = (n?: number, d=2) => (n!=null && Number.isFinite(n)) ? n.toFixed(d) : '';
const kfmt = (n?: number) => {
  if (n==null || !Number.isFinite(n)) return '';
  const a = Math.abs(n);
  if (a>=1e9) return (n/1e9).toFixed(2)+'B';
  if (a>=1e6) return (n/1e6).toFixed(2)+'M';
  if (a>=1e3) return (n/1e3).toFixed(2)+'K';
  return String(n);
};
const tsstr = (ms:number) => new Date(ms).toLocaleTimeString();
function download(name:string, text:string, mime='text/plain'){
  const blob = new Blob([text], { type:mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}
function toCSV(rows: Print[]){
  const head = 'ts,price,size,side,venue';
  const body = rows.map(r => [
    new Date(r.ts).toISOString(), r.px, r.sz, r.side||'', r.venue||''
  ].join(','));
  return [head, ...body].join('\n');
}

/* ================= Component ================= */
export default function TimeAndSales({
  symbol: defaultSymbol = 'AAPL',
  httpEndpoint = '/api/timesales/snapshot',
  sseEndpoint  = '/api/timesales/stream',
  title = 'Time & Sales',
  className = '',
}: Props){
  // controls
  const [symbol, setSymbol] = useState(defaultSymbol.toUpperCase());
  const [minSize, setMinSize] = useState<number>(0);
  const [side, setSide] = useState<'ALL'|'B'|'S'>('ALL');
  const [venue, setVenue] = useState<string>('ALL');
  const [limit, setLimit] = useState<number>(1000);
  const [paused, setPaused] = useState(false);
  const [maxRate, setMaxRate] = useState<number>(30); // prints/sec cap (client)

  // data
  const [rows, setRows] = useState<Print[]>([]);
  const [notes, setNotes] = useState('');
  const [error, setError] = useState<string|null>(null);
  const [connected, setConnected] = useState(false);

  // SSE plumbing
  const abortRef = useRef<AbortController|null>(null);
  const lastEmitRef = useRef<number>(0);

  // seed snapshot on mount/symbol change
  useEffect(()=>{ seed(); }, [symbol]);

  async function seed(){
    stop();
    setRows([]); setNotes(''); setError(null);
    try{
      const res = await fetch(httpEndpoint, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ symbol, limit })
      });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Print[] = await res.json();
      setRows(normalize(data, limit));
    }catch(e:any){ setError(e.message); }
  }

  function stop(){ abortRef.current?.abort(); abortRef.current=null; setConnected(false); }

  async function run(){
    stop();
    setError(null); setNotes('');
    const ac = new AbortController(); abortRef.current = ac;
    try{
      const res = await fetch(sseEndpoint, {
        method:'POST',
        headers:{ 'Content-Type':'application/json', 'Accept':'text/event-stream' },
        body: JSON.stringify({ symbol }),
        signal: ac.signal,
      });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      if (!res.body) throw new Error('No body');
      setConnected(true);

      const reader = res.body.getReader(); const dec = new TextDecoder('utf-8');
      let acc = '';
      while(true){
        const { value, done } = await reader.read(); if(done) break;
        const chunk = dec.decode(value, { stream:true }); acc += chunk;
        const lines = acc.split('\n'); acc = lines.pop() || '';
        const now = performance.now();
        for(const line of lines){
          if(!line.startsWith('data:')) continue;
          const data = line.slice(5).trim(); if(!data) continue;
          if (paused) continue;
          if (maxRate>0 && (now - lastEmitRef.current) < (1000/maxRate)) continue;

          try {
            const p = JSON.parse(data) as Patch;
            if ('rows' in p && Array.isArray(p.rows)) {
              setRows(prev => normalize([...p.rows!, ...prev], limit));
            } else if ('row' in p && p.row) {
              setRows(prev => normalize([p.row!, ...prev], limit));
            } else if ('notes' in p && typeof p.notes === 'string') {
              setNotes(n => (n?n+'\n':'') + p.notes);
            }
            lastEmitRef.current = now;
          } catch {
            setNotes(n => (n?n+'\n':'') + data);
          }
        }
      }
    }catch(e:any){
      if (e?.name !== 'AbortError') setError(e.message);
    }finally{
      setConnected(false);
    }
  }

  // derived filters
  const venues = useMemo(()=> Array.from(new Set(rows.map(r=>r.venue).filter(Boolean))) as string[], [rows]);
  const filtered = useMemo(()=>{
    let out = rows;
    if (minSize>0) out = out.filter(r => r.sz >= minSize);
    if (side!=='ALL') out = out.filter(r => (r.side||'N') === side);
    if (venue!=='ALL') out = out.filter(r => (r.venue||'') === venue);
    return out;
  }, [rows, minSize, side, venue]);

  // micro metrics
  const tapeStats = useMemo(()=>{
    const n = filtered.length;
    const v = filtered.reduce((s,r)=>s+r.sz,0);
    const notional = filtered.reduce((s,r)=>s+r.sz*r.px,0);
    const vwap = v ? notional/v : 0;
    const buys = filtered.filter(r=>r.side==='B').reduce((s,r)=>s+r.sz,0);
    const sells= filtered.filter(r=>r.side==='S').reduce((s,r)=>s+r.sz,0);
    const imb = (buys - sells) / Math.max(1, buys + sells);
    return { n, v, vwap, buys, sells, imb };
  }, [filtered]);

  // mini series for cumulative volume & price
  const chart = useMemo(()=>{
    const r = filtered.slice().reverse(); // oldest->newest
    const t = r.map(x=>x.ts);
    const cumV: number[] = [];
    let acc = 0; for (const x of r){ acc += x.sz; cumV.push(acc); }
    const px = r.map(x=>x.px);
    return { t, cumV, px };
  }, [filtered]);

  /* ================= UI ================= */
  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())}
                 className="rounded border px-2 py-1 text-sm" />
          <button onClick={seed} className="px-2 py-1 text-xs rounded border">Seed</button>
          {connected
            ? <button onClick={stop} className="px-3 py-1 text-xs rounded border border-rose-500 text-rose-600">Stop</button>
            : <button onClick={run}  className="px-3 py-1 text-xs rounded border border-indigo-600 bg-indigo-600 text-white">Stream</button>}
          <button onClick={()=>download(`${symbol}_tape.csv`, toCSV(filtered), 'text/csv')}
                  className="px-2 py-1 text-xs rounded border">CSV</button>
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 grid grid-cols-1 md:grid-cols-10 gap-3 border-b border-neutral-200 dark:border-neutral-800 text-sm">
        <Field label="Min Size">
          <input type="number" value={minSize} onChange={e=>setMinSize(Math.max(0, Number(e.target.value)||0))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="Side">
          <select value={side} onChange={e=>setSide(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option><option value="B">Buys</option><option value="S">Sells</option>
          </select>
        </Field>
        <Field label="Venue">
          <select value={venue} onChange={e=>setVenue(e.target.value)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">All</option>
            {venues.map(v=><option key={v} value={v!}>{v}</option>)}
          </select>
        </Field>
        <Field label="Keep last">
          <input type="number" value={limit} onChange={e=>setLimit(Math.max(100, Number(e.target.value)||limit))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="Speed cap (prints/s)">
          <input type="number" value={maxRate} onChange={e=>setMaxRate(Math.max(0, Number(e.target.value)||0))}
                 className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <div className="flex items-end gap-2">
          <button onClick={()=>setPaused(p=>!p)} className="px-3 py-2 text-xs rounded border">{paused?'Resume':'Pause'}</button>
          <div className={`text-[11px] px-2 py-1 rounded ${connected?'bg-emerald-100 text-emerald-700':'bg-neutral-100 text-neutral-700'}`}>
            {connected?'LIVE':'IDLE'}
          </div>
        </div>
      </div>

      {/* Metrics + Mini Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 p-3">
        <Card title="Tape Stats">
          <div className="grid grid-cols-3 gap-2 text-sm">
            <Row label="Prints" value={String(tapeStats.n)} />
            <Row label="Volume" value={kfmt(tapeStats.v)} />
            <Row label="VWAP" value={fmt(tapeStats.vwap, 4)} />
            <Row label="Buy Vol" value={kfmt(tapeStats.buys)} />
            <Row label="Sell Vol" value={kfmt(tapeStats.sells)} />
            <Row label="Imbalance" value={`${(tapeStats.imb*100).toFixed(1)}%`} />
          </div>
        </Card>
        <Card title="Cumulative Volume">
          {chart.cumV.length ? <Line t={chart.t} y={chart.cumV} color="#4f46e5" /> : <Empty label="—" />}
        </Card>
        <Card title="Price (last N)">
          {chart.px.length ? <Line t={chart.t} y={chart.px} color="#10b981" /> : <Empty label="—" />}
        </Card>
      </div>

      {/* Tape Table */}
      <div className="flex-1 overflow-auto px-3 pb-3">
        <table className="min-w-full text-xs">
          <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800">
            <tr>
              <Th>Time</Th><Th align="right">Price</Th><Th align="right">Size</Th><Th align="center">Side</Th><Th>Venue</Th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r,i)=>(
              <tr key={(r.id || r.ts)+'-'+i} className={i%2?'bg-neutral-50 dark:bg-neutral-950':''}>
                <Td>{tsstr(r.ts)}</Td>
                <Td> align="right" className={toneBySide(r.side)}{'>'}{fmt(r.px, 4)}</Td>
                <Td> align="right"{'>'}{kfmt(r.sz)}</Td>
                <Td align="center">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${pillBySide(r.side)}`}>
                    {r.side || 'N'}
                  </span>
                </Td>
                <Td>{r.venue || ''}</Td>
              </tr>
            ))}
          </tbody>
        </table>
        {error && <div className="text-xs text-rose-600 mt-2">{error}</div>}
        {notes && <pre className="text-[11px] whitespace-pre-wrap text-neutral-600 dark:text-neutral-300 mt-2">{notes}</pre>}
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
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="block text-[11px] text-neutral-500 mb-1">{label}</label>{children}</div>;
}
function Empty({ label }: { label: string }) { return <div className="text-xs text-neutral-500">{label}</div>; }
function Th({ children, align='left' }:{ children:React.ReactNode; align?:'left'|'right'|'center' }){
  return <th className={`px-2 py-1 text-${align} font-semibold`}>{children}</th>;
}
function Td({ children, align='left' }:{ children:React.ReactNode; align?:'left'|'right'|'center' }){
  return <td className={`px-2 py-1 text-${align}`}>{children}</td>;
}
function Row({ label, value }:{label:string; value:string}){ return <div className="flex justify-between"><span className="text-[11px] text-neutral-500">{label}</span><span className="text-sm">{value}</span></div>; }
function Line({ t, y, color='#4f46e5' }:{ t:number[]; y:number[]; color?:string }){
  const width=360, height=120;
  const min=Math.min(...y), max=Math.max(...y); const span=(max-min)||1;
  const step = width/Math.max(1,y.length-1);
  const d = y.map((v,i)=>`${i===0?'M':'L'}${(i*step).toFixed(2)},${(height-((v-min)/span)*height).toFixed(2)}`).join(' ');
  return <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}><path d={d} fill="none" stroke={color} strokeWidth={1.5}/></svg>;
}
const toneBySide = (s?:Side) => s==='B' ? 'text-emerald-700' : s==='S' ? 'text-rose-700' : 'text-neutral-800';
const pillBySide = (s?:Side) => s==='B' ? 'bg-emerald-600 text-white' : s==='S' ? 'bg-rose-600 text-white' : 'bg-neutral-200 text-neutral-800';

/* ================= Helpers ================= */
function normalize(xs: Print[], limit:number){
  // dedupe by id+ts+px+sz; keep most recent first
  const key = (r:Print)=> `${r.id||''}_${r.ts}_${r.px}_${r.sz}`;
  const map = new Map<string, Print>();
  for (const r of xs) {
    const k = key(r);
    if (!map.has(k)) map.set(k, r);
  }
  return Array.from(map.values()).sort((a,b)=> b.ts - a.ts).slice(0, Math.max(100, limit));
}