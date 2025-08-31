'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type Bar = { t: number; c: number }; // unix ms, close
type Series = { symbol: string; bars: Bar[] };
type HedgeMethod = 'OLS' | 'Beta' | 'Fixed 1:1';

type Patch = {
  // one-shot JSON is fine; SSE patches also supported if you extend later
  series?: Series[];            // for selected symbols
  scan?: {                      // optional quick scan triplets
    symbols: string[];
    zscore: number;             // latest abs z
  }[];
  notes?: string;
};

type Props = {
  endpoint?: string;            // POST /api/rv
  title?: string;
  className?: string;
};

/* ================= Utils ================= */
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const fmt = (x:number|undefined, d=2)=> Number.isFinite(x as number) ? (x as number).toFixed(d) : '';
const clamp=(x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
function download(filename:string, text:string, mime='text/plain'){
  const blob = new Blob([text], { type:mime }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}

/* ======= Tiny math helpers (no deps) ======= */
function aligned(series: Series[]): { t:number; cols:number[] }[] {
  if (!series.length) return [];
  const maps = series.map(s => new Map(s.bars.map(b => [b.t, b.c])));
  const times = Array.from(maps[0].keys()).filter(t => maps.every(m => m.has(t))).sort((a,b)=>a-b);
  return times.map(t => ({ t, cols: maps.map(m => m.get(t) as number) }));
}
function mean(xs:number[]){ return xs.length? xs.reduce((a,b)=>a+b,0)/xs.length : 0; }
function std(xs:number[]){ const m=mean(xs); const v=xs.length? xs.reduce((s,x)=>s+(x-m)*(x-m),0)/xs.length : 0; return Math.sqrt(v); }
function cov(xs:number[], ys:number[]){ const mx=mean(xs), my=mean(ys); let s=0; for(let i=0;i<xs.length;i++) s+=(xs[i]-mx)*(ys[i]-my); return xs.length? s/xs.length : 0; }
function corr(xs:number[], ys:number[]){ const sx=std(xs), sy=std(ys); if (sx===0||sy===0) return 0; return cov(xs,ys)/(sx*sy); }
function ols(y:number[], x:number[]){ // y = a + b*x
  const mx=mean(x), my=mean(y);
  let num=0, den=0; for(let i=0;i<x.length;i++){ num+=(x[i]-mx)*(y[i]-my); den+=(x[i]-mx)*(x[i]-mx); }
  const b = den===0 ? 0 : num/den; const a = my - b*mx; return { a, b };
}
function rollingZ(xs:number[], win:number){
  const out:number[]=[]; for(let i=0;i<xs.length;i++){
    const s = Math.max(0, i-win+1); const w = xs.slice(s, i+1);
    const m=mean(w), sd=std(w) || 1e-12; out.push((xs[i]-m)/sd);
  } return out;
}

/* ================= Component ================= */
export default function RelativeValue({
  endpoint = '/api/rv',
  title = 'Relative Value',
  className = '',
}: Props){
  // Controls
  const [symbols, setSymbols] = useState('AAPL, MSFT'); // CSV
  const [lookback, setLookback] = useState<number>(180); // days
  const [zWin, setZWin] = useState<number>(60);          // rolling window
  const [method, setMethod] = useState<HedgeMethod>('OLS');
  const [rebalance, setRebalance] = useState<'static'|'rolling'>('static');
  const [basketMode, setBasketMode] = useState<'pair'|'triplet'>('pair');

  // Data
  const [series, setSeries] = useState<Series[]>([]);
  const [scan, setScan] = useState<{symbols:string[]; zscore:number}[]>([]);
  const [notes, setNotes] = useState('');
  const [error, setError] = useState<string|null>(null);
  const [loading, setLoading] = useState(false);

  // Actions
  async function run(){
    setLoading(true); setError(null); setSeries([]); setScan([]); setNotes('');
    try{
      const res = await fetch(endpoint, {
        method:'POST',
        headers:{'Content-Type':'application/json', Accept:'application/json'},
        body: JSON.stringify({
          id: uid(),
          symbols: parseSymbols(symbols, basketMode),
          lookback, zWin, method, rebalance
        })
      });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const p:Patch = await res.json();
      if (p.series) setSeries(p.series);
      if (p.scan) setScan(p.scan);
      if (p.notes) setNotes(p.notes);
    }catch(e:any){
      setError(e.message);
    }finally{ setLoading(false); }
  }

  // Pair math
  const alignedRows = useMemo(()=> aligned(series), [series]);
  const rv = useMemo(()=>{
    if (!alignedRows.length) return null;
    const cols = alignedRows.map(r => r.cols);
    const k = cols[0].length;
    if (k < 2) return null;

    // support pair (2) or basket (3+ with x as combination)
    const ys = cols.map(c => c[0]);        // first symbol as dependent by default
    const Xs = cols.map(c => c.slice(1));  // others

    // Build synthetic X: for pair X = single series; for triplet X = weighted combo via OLS multi-way (stacked one-factor)
    let hedge:number[] = [];  // weights for each independent series
    let spread:number[] = [];
    let betaVec:number[] = [];

    if (k === 2) {
      // classic pair
      const x = cols.map(c => c[1]);
      const y = cols.map(c => c[0]);
      let b = 1, a = 0;
      if (method==='OLS'){ const m = ols(y, x); a = m.a; b = m.b; }
      else if (method==='Beta'){ b = betaLike(y, x); a = 0; }
      else { b = 1; a = 0; }
      betaVec = [b];
      spread = y.map((yv, i) => yv - (a + b * x[i]));
      hedge = [ -1, b ]; // weights on [Y, X] that make spread ~ 0 (y - a - b x)
    } else {
      // triplet+: regress Y on linear combo of X1..Xn using greedy OLS (Gram-Schmidt-lite)
      const ys1 = cols.map(c => c[0]);
      const X = cols.map(c => c.slice(1)); // array of arrays
      const p = k - 1;
      // simple sequential fit
      const b: number[] = new Array(p).fill(0);
      let resid = ys1.slice();
      for (let j=0;j<p;j++){
        const xj = X.map(row => row[j]);
        const m = ols(resid, xj);
        b[j] += m.b;
        // update residual
        for (let i=0;i<resid.length;i++){ resid[i] -= (m.a + m.b * xj[i]); }
      }
      betaVec = b;
      spread = resid;
      hedge = [ -1, ...b ]; // weights (Y, X1..Xp)
    }

    const z = rollingZ(spread, zWin);
    const last = spread.length ? spread[spread.length-1] : 0;
    const lastZ = z.length ? z[z.length-1] : 0;

    // simple signals
    const enterLong = lastZ <= -2;
    const enterShort= lastZ >=  2;
    const exit = Math.abs(lastZ) <= 0.5;

    // correlation matrix
    const mat = corrMatrix(alignedRows);

    return {
      t: alignedRows.map(r=>r.t),
      spread, z,
      hedge, betaVec,
      last, lastZ, enterLong, enterShort, exit,
      corr: mat
    };
  }, [alignedRows, method, zWin]);

  // ======= UI =======
  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={()=>exportCSV(series)} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export CSV</button>
          {loading
            ? <button className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Running…</button>
            : <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>}
        </div>
      </div>

      {/* controls */}
      <div className="p-3 grid grid-cols-1 lg:grid-cols-10 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <Field label="Symbols">
          <input value={symbols} onChange={e=>setSymbols(e.target.value)} placeholder="AAPL, MSFT  (pair)  or  AAPL, MSFT, GOOG (triplet)"
                 className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
        <Field label="Mode">
          <select value={basketMode} onChange={e=>setBasketMode(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="pair">Pair</option><option value="triplet">Triplet</option>
          </select>
        </Field>
        <Field label="Lookback (days)">
          <input type="number" value={lookback} onChange={e=>setLookback(Math.max(20, Number(e.target.value)||lookback))}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Field label="Z window">
          <input type="number" value={zWin} onChange={e=>setZWin(Math.max(10, Number(e.target.value)||zWin))}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Field label="Hedge method">
          <select value={method} onChange={e=>setMethod(e.target.value as HedgeMethod)} className="w-full rounded border px-2 py-2 text-sm">
            <option>OLS</option><option>Beta</option><option>Fixed 1:1</option>
          </select>
        </Field>
        <Field label="Rebalance">
          <select value={rebalance} onChange={e=>setRebalance(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="static">Static</option><option value="rolling">Rolling</option>
          </select>
        </Field>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* Left: Spread & Z */}
        <div className="space-y-3">
          <Card title="Spread">
            {rv ? <Line t={rv.t} y={rv.spread} color={rv.lastZ>=0?'#ef4444':'#10b981'} /> : <Empty label="Run to compute" />}
            {rv && <div className="mt-2 text-xs text-neutral-600">Hedge weights: <code>[{rv.hedge.map((w,i)=>`${i===0?'Y':'X'+i}=${fmt(w,3)}`).join(', ')}]</code></div>}
          </Card>
          <Card title="Z-Score">
            {rv ? <Line t={rv.t} y={rv.z} color="#4f46e5" bands={[2, -2, 0]} /> : <Empty label="Run to compute" />}
            {rv && (
              <div className="grid grid-cols-3 gap-2 text-xs mt-2">
                <Badge tone={rv.enterLong?1:0}>Long entry: {rv.enterLong ? 'YES' : 'no'}</Badge>
                <Badge tone={rv.enterShort?-1:0}>Short entry: {rv.enterShort ? 'YES' : 'no'}</Badge>
                <Badge tone={rv.exit?0:0}>Exit: {rv.exit ? 'YES (|z|≤0.5)' : 'no'}</Badge>
              </div>
            )}
          </Card>
        </div>

        {/* Middle: Price panel */}
        <Card title="Prices (normalized)">
          {alignedRows.length ? (
            <MultiLine series={normalizeForPlot(series)} />
          ) : <Empty label="No series yet" />}
          {series.length ? <Legend symbols={series.map(s=>s.symbol)} /> : null}
        </Card>

        {/* Right: Correlation + Scan + Notes */}
        <div className="space-y-3">
          <Card title="Correlation (lookback)">
            {alignedRows.length ? <CorrHeat symbols={series.map(s=>s.symbol)} mat={rv?.corr || unit(series.length)} /> : <Empty label="No data" />}
          </Card>
          <Card title="Quick Triplet Scan (optional)">
            {scan.length ? (
              <ul className="text-xs space-y-1">
                {scan.slice(0,12).map((r,i)=>(
                  <li key={i} className="flex justify-between">
                    <span>{r.symbols.join(' / ')}</span>
                    <span className="font-mono">{fmt(r.zscore,2)}</span>
                  </li>
                ))}
              </ul>
            ) : <Empty label="Backend can return a ranked list of RV triplets here."/>}
          </Card>
          <Card title="Notes">
            <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || '—'}</pre>
            {error && <div className="text-xs text-rose-600 mt-2">{error}</div>}
          </Card>
        </div>
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
function Badge({ children, tone=0 }:{children:React.ReactNode; tone?:-1|0|1}){
  const c=tone>0?'bg-emerald-100 text-emerald-700':tone<0?'bg-rose-100 text-rose-700':'bg-neutral-100 text-neutral-700';
  return <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] ${c}`}>{children}</span>;
}

/* ======= Charts (inline SVG) ======= */
function Line({ t, y, color='#4f46e5', bands }:{ t:number[]; y:number[]; color?:string; bands?:number[] }){
  const width=420, height=140;
  const ymin=Math.min(...y,0), ymax=Math.max(...y,0); const span=ymax-ymin||1;
  const step = width / Math.max(1, y.length-1);
  const path = y.map((v,i)=>`${i===0?'M':'L'}${(i*step).toFixed(2)},${(height-((v-ymin)/span)*height).toFixed(2)}`).join(' ');
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {bands?.map((b,i)=>{
        const yb = height - ((b - ymin)/span)*height;
        return <line key={i} x1={0} x2={width} y1={yb} y2={yb} stroke="#9ca3af" strokeDasharray="4 4" />;
      })}
      <path d={path} fill="none" stroke={color} strokeWidth={1.5}/>
    </svg>
  );
}
function MultiLine({ series }:{ series:{ label:string; t:number[]; y:number[]; color:string }[] }){
  const width=520, height=180;
  const allY = series.flatMap(s=>s.y);
  const ymin=Math.min(...allY,0.9), ymax=Math.max(...allY,1.1); const span=ymax-ymin||1;
  const maxN = Math.max(...series.map(s=>s.y.length));
  const step = width / Math.max(1, maxN-1);
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {series.map((s,idx)=>{
        const path = s.y.map((v,i)=>`${i===0?'M':'L'}${(i*step).toFixed(2)},${(height-((v-ymin)/span)*height).toFixed(2)}`).join(' ');
        return <path key={s.label} d={path} fill="none" stroke={color10(idx)} strokeWidth={1.4}/>;
      })}
    </svg>
  );
}
function Legend({ symbols }:{ symbols:string[] }){
  return <div className="flex flex-wrap gap-2 mt-2 text-[11px]">
    {symbols.map((s,i)=>(
      <span key={s} className="inline-flex items-center gap-1"><i className="inline-block w-3 h-2 rounded" style={{backgroundColor:color10(i)}}></i>{s}</span>
    ))}
  </div>;
}
function CorrHeat({ symbols, mat }:{ symbols:string[]; mat:number[][] }){
  const n = symbols.length; const size = 20; const pad = 60; const w = pad + n*size + 10; const h = pad + n*size + 10;
  return (
    <svg width="100%" height={pad + n*size + 30} viewBox={`0 0 ${w} ${h}`}>
      {/* labels */}
      {symbols.map((s,i)=>(
        <text key={'x'+i} x={pad+i*size+size/2} y={50} transform={`rotate(-45 ${pad+i*size+size/2} 50)`} textAnchor="end" fontSize="10" fill="#374151">{s}</text>
      ))}
      {symbols.map((s,i)=>(
        <text key={'y'+i} x={10} y={pad+i*size+size*0.7} fontSize="10" fill="#374151">{s}</text>
      ))}
      {/* cells */}
      {mat.map((row,i)=>
        row.map((v,j)=>{
          const c = heat(v);
          return <rect key={`${i}-${j}`} x={pad+j*size} y={pad+i*size} width={size-1} height={size-1} fill={c} />;
        })
      )}
    </svg>
  );
}
function color10(i:number){ const c=['#4f46e5','#10b981','#f59e0b','#ef4444','#06b6d4','#8b5cf6','#64748b','#a855f7','#22c55e','#fb7185']; return c[i % c.length]; }
function heat(x:number){ // -1..1
  const v = clamp((x+1)/2, 0, 1);
  const r = Math.floor(239*v + 16*(1-v));
  const g = Math.floor(68*(1-v) + 185*v);
  const b = Math.floor(68*(1-v) + 229*v);
  return `rgb(${r},${g},${b})`;
}

/* ======= Helpers ======= */
function parseSymbols(input:string, mode:'pair'|'triplet'){
  const arr = input.split(',').map(s=>s.trim().toUpperCase()).filter(Boolean);
  if (mode==='pair' && arr.length<2) return arr.concat(['SPY']).slice(0,2);
  if (mode==='triplet' && arr.length<3) return [...arr, 'SPY', 'QQQ'].slice(0,3);
  return arr.slice(0, mode==='pair'?2:3);
}
function normalizeForPlot(series: Series[]){
  if (!series.length) return [];
  const rows = aligned(series);
  const cols = series.map((s,idx)=> rows.map(r => r.cols[idx]));
  const base = cols.map(c => c[0] || 1);
  return cols.map((c,i)=>({
    label: series[i].symbol,
    t: rows.map(r=>r.t),
    y: c.map(v => v / (base[i] || 1)),
    color: color10(i)
  }));
}
function corrMatrix(rows:{t:number; cols:number[]}[]){
  const k = rows[0]?.cols.length || 0;
  const mat:number[][] = Array.from({length:k}, ()=> Array(k).fill(1));
  for (let i=0;i<k;i++){
    const xi = rows.map(r=>r.cols[i]);
    for (let j=i+1;j<k;j++){
      const xj = rows.map(r=>r.cols[j]);
      const c = corr(xi,xj);
      mat[i][j]=mat[j][i]=c;
    }
  }
  return mat;
}
function unit(n:number){ return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=> i===j?1:0)); }
function betaLike(y:number[], x:number[]){ // cov/var
  const den = cov(x,x); return den===0 ? 0 : cov(y,x)/den;
}
function exportCSV(series: Series[]){
  if (!series.length) return;
  const rows = aligned(series);
  const head = ['t', ...series.map(s=>s.symbol)].join(',');
  const lines = rows.map(r => [new Date(r.t).toISOString(), ...r.cols.map(v=>String(v))].join(','));
  download('relative_value.csv', [head, ...lines].join('\n'), 'text/csv');
}