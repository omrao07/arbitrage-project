'use client';
import React, { useEffect, useMemo, useState } from 'react';

/* ============== Types ============== */
type OptionSide = 'call' | 'put';
type OptionRow = {
  strike: number;
  side: OptionSide;
  expiry: string; // ISO date
  iv: number;     // implied vol
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  oi: number;
  vol: number;
  last: number;
  bid?: number;
  ask?: number;
};

type Props = {
  endpoint?: string;       // e.g. /api/options/monitor
  title?: string;
  defaultSymbol?: string;
  className?: string;
};

/* ============== Utils ============== */
const fmt = (x: number|undefined, d=2) => (x!=null && Number.isFinite(x)) ? x.toFixed(d) : '';
const pct = (x: number|undefined) => (x!=null && Number.isFinite(x)) ? (x*100).toFixed(1)+'%' : '';

/* ============== Component ============== */
export default function OptionsMonitor({
  endpoint = '/api/options/monitor',
  title = 'Options Monitor',
  defaultSymbol = 'AAPL',
  className = '',
}: Props){
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [expiry, setExpiry] = useState<string>('');
  const [rows, setRows] = useState<OptionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string|null>(null);

  useEffect(()=>{ fetchData(); }, [symbol, expiry]);

  async function fetchData(){
    setLoading(true); setError(null);
    try{
      const res = await fetch(endpoint, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ symbol, expiry }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: OptionRow[] = await res.json();
      setRows(data);
    }catch(e:any){
      setError(e.message);
    }finally{
      setLoading(false);
    }
  }

  // derived expiries and strikes
  const expiries = useMemo(()=> Array.from(new Set(rows.map(r=>r.expiry))).sort(), [rows]);
  const strikes  = useMemo(()=> Array.from(new Set(rows.map(r=>r.strike))).sort((a,b)=>a-b), [rows]);

  // group calls and puts by strike
  const chain = useMemo(()=>{
    const map = new Map<number, {call?:OptionRow; put?:OptionRow}>();
    for(const r of rows){
      const e = map.get(r.strike) || {};
      if (r.side==='call') e.call=r; else e.put=r;
      map.set(r.strike, e);
    }
    return Array.from(map.entries()).map(([strike, v])=>({ strike, ...v }));
  }, [rows]);

  // metrics
  const putOI = rows.filter(r=>r.side==='put').reduce((s,r)=>s+r.oi,0);
  const callOI= rows.filter(r=>r.side==='call').reduce((s,r)=>s+r.oi,0);
  const pcr = callOI ? putOI/callOI : 0;

  /* ============== UI ============== */
  return (
    <div className={`flex flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex gap-2">
          <input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())}
            className="rounded border px-2 py-1 text-sm"/>
          <select value={expiry} onChange={e=>setExpiry(e.target.value)} className="rounded border px-2 py-1 text-sm">
            <option value="">All expiries</option>
            {expiries.map(e=><option key={e} value={e}>{e}</option>)}
          </select>
          <button onClick={fetchData} className="px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white text-xs">Reload</button>
        </div>
      </div>

      {/* metrics */}
      <div className="grid grid-cols-3 gap-2 p-3 border-b border-neutral-200 dark:border-neutral-800 text-xs">
        <div className="rounded border px-2 py-1">Put OI: {putOI}</div>
        <div className="rounded border px-2 py-1">Call OI: {callOI}</div>
        <div className="rounded border px-2 py-1">Put/Call Ratio: {pcr.toFixed(2)}</div>
      </div>

      {/* chain table */}
      <div className="overflow-auto">
        <table className="min-w-full text-xs">
          <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800">
            <tr>
              <th className="px-2 py-1 text-left">Strike</th>
              <th className="px-2 py-1 text-center">Call IV</th>
              <th className="px-2 py-1 text-center">Call OI</th>
              <th className="px-2 py-1 text-center">Call Δ</th>
              <th className="px-2 py-1 text-center">Put Δ</th>
              <th className="px-2 py-1 text-center">Put OI</th>
              <th className="px-2 py-1 text-center">Put IV</th>
            </tr>
          </thead>
          <tbody>
            {chain.map(r=>(
              <tr key={r.strike} className="odd:bg-white even:bg-neutral-50 dark:odd:bg-neutral-900 dark:even:bg-neutral-950">
                <td className="px-2 py-1 font-semibold">{r.strike}</td>
                <td className="px-2 py-1 text-center text-emerald-700">{pct(r.call?.iv)}</td>
                <td className="px-2 py-1 text-center">{r.call?.oi ?? ''}</td>
                <td className="px-2 py-1 text-center">{fmt(r.call?.delta,2)}</td>
                <td className="px-2 py-1 text-center">{fmt(r.put?.delta,2)}</td>
                <td className="px-2 py-1 text-center">{r.put?.oi ?? ''}</td>
                <td className="px-2 py-1 text-center text-rose-700">{pct(r.put?.iv)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {error && <div className="text-xs text-rose-600 p-2">{error}</div>}
    </div>
  );
}