'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type AssetClass = 'Equity'|'Rates'|'FX'|'Credit'|'Vol';
type Position = {
  symbol: string;
  assetClass: AssetClass;
  notional: number;       // base currency
  beta?: number;          // equity beta vs market
  dv01?: number;          // PnL per 1bp move (base ccy)
  fxDelta?: number;       // PnL per 1% move in FX
  vega?: number;          // PnL per 1 vol point change
  spr01?: number;         // Credit PnL per 1bp spread move
  liqScore?: number;      // 0..1 higher => less liquid
};

/* ================= Mock portfolio ================= */
function mockPortfolio(): Position[] {
  return [
    { symbol:'AAPL',      assetClass:'Equity', notional:1_200_000, beta:1.10,  liqScore:0.1 },
    { symbol:'RELIANCE',  assetClass:'Equity', notional:900_000,  beta:0.95,   liqScore:0.15 },
    { symbol:'UST 10Y',   assetClass:'Rates',  notional:5_000_000, dv01: 4500, liqScore:0.02 },
    { symbol:'IND10Y',    assetClass:'Rates',  notional:3_000_000, dv01: 3200, liqScore:0.04 },
    { symbol:'USDINR',    assetClass:'FX',     notional:1_500_000, fxDelta: 8_500, liqScore:0.05 },
    { symbol:'IG CDS',    assetClass:'Credit', notional:2_000_000, spr01: 1600, liqScore:0.08 },
    { symbol:'BANKNIFTYv',assetClass:'Vol',    notional:700_000,  vega: 6_800,  liqScore:0.20 },
  ];
}

/* ================= Helpers ================= */
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const fmt0 = (n:number)=> n.toLocaleString(undefined,{maximumFractionDigits:0});
const fmt2 = (n:number)=> n.toLocaleString(undefined,{maximumFractionDigits:2});
const sign = (x:number)=> x>0?'+':'';

/* Shock model */
type Shock = {
  eqPct: number;     // market down % (negative is crash)
  rateBp: number;    // parallel up bp
  fxPct: number;     // base ccy up % (positive -> strengthens)
  volPts: number;    // vol +pts
  sprBp: number;     // credit spread +bp
  liqPct: number;    // haircut %
  corrBreak: boolean;
};

/* Deterministic stress PnL */
function stressPnL(p: Position, s: Shock): { pnl:number, by:{[k:string]:number} } {
  let pnl = 0;
  const by:any = {};
  // Equity beta
  if (p.beta!=null){
    const c = p.notional * (p.beta * (s.eqPct/100));
    pnl += c; by.Equity = (by.Equity||0)+c;
  }
  // Rates DV01: bp up => price down (loss = dv01 * bp)
  if (p.dv01!=null){
    const c = - p.dv01 * s.rateBp;
    pnl += c; by.Rates = (by.Rates||0)+c;
  }
  // FX delta: base ccy +% typically hurts foreign asset if long local; treat delta as PnL/% in base
  if (p.fxDelta!=null){
    const c = p.fxDelta * (s.fxPct);
    pnl += c; by.FX = (by.FX||0)+c;
  }
  // Vol vega: +vol -> vega * +pts
  if (p.vega!=null){
    const c = p.vega * s.volPts;
    pnl += c; by.Vol = (by.Vol||0)+c;
  }
  // Credit spreads: wider spreads -> loss
  if (p.spr01!=null){
    const c = - p.spr01 * s.sprBp;
    pnl += c; by.Credit = (by.Credit||0)+c;
  }
  // Liquidity haircut (apply to loss only if corrBreak on, else mild)
  const haircut = s.liqPct/100 * p.notional * (s.corrBreak ? 1 : 0.25) * (p.liqScore ?? 0);
  pnl -= haircut; by.Liquidity = (by.Liquidity||0) - haircut;

  return { pnl, by };
}

/* Portfolio aggregation */
function runStress(port: Position[], s: Shock){
  const rows = port.map(pos=>{
    const r = stressPnL(pos, s);
    return { ...pos, pnl: r.pnl, by: r.by };
  });
  const total = rows.reduce((a,b)=>a + b.pnl, 0);
  const byFactor = rows.reduce((acc, r)=>{
    for (const k of Object.keys(r.by)) acc[k] = (acc[k]||0)+r.by[k];
    return acc;
  }, {} as Record<string,number>);
  const byAsset = rows.map(r=>({ symbol:r.symbol, pnl:r.pnl, assetClass:r.assetClass }));
  return { rows, total, byFactor, byAsset };
}

/* Quick Monte Carlo around the chosen shock (fat tails) */
function mcSamples(port: Position[], s: Shock, N=2000){
  const out:number[] = [];
  for (let i=0;i<N;i++){
    // Student-ish noise via sum of uniforms (approx)
    const n = ()=> (Math.random()+Math.random()+Math.random()+Math.random()-2); // ~N(0,1)ish
    const shockVar: Shock = {
      eqPct: s.eqPct + 8*n(),         // 8% sigma on equity tail
      rateBp: s.rateBp + 25*n(),      // 25bp sigma on rates
      fxPct: s.fxPct + 1.5*n(),       // 1.5% sigma
      volPts: s.volPts + 4*n(),       // 4 vol pts sigma
      sprBp: s.sprBp + 30*n(),        // 30bp sigma
      liqPct: clamp(s.liqPct + 5*n(), 0, 100),
      corrBreak: s.corrBreak,
    };
    // If corrBreak: bias equity down + spreads up + vol up
    if (s.corrBreak){
      shockVar.eqPct -= Math.abs(6*n());
      shockVar.sprBp += Math.abs(40*n());
      shockVar.volPts += Math.abs(5*n());
    }
    const r = runStress(port, shockVar);
    out.push(r.total);
  }
  out.sort((a,b)=>a-b);
  return out;
}

/* VaR / ES */
function varEs(sortedPnL:number[], alpha=0.99){
  if (sortedPnL.length===0) return { var:0, es:0 };
  const idx = Math.floor((1-alpha)*sortedPnL.length);
  const var99 = sortedPnL[idx] ?? sortedPnL[0];
  const tail = sortedPnL.slice(0, idx+1);
  const es = tail.length ? tail.reduce((a,b)=>a+b,0)/tail.length : var99;
  return { var: var99, es };
}

/* ================= Component ================= */
export default function StressSandbox() {
  const [port] = useState<Position[]>(mockPortfolio());

  const [shock, setShock] = useState<Shock>({
    eqPct: -15,
    rateBp: +100,
    fxPct: +2,
    volPts: +10,
    sprBp: +80,
    liqPct: 5,
    corrBreak: true,
  });

  const stress = useMemo(()=>runStress(port, shock), [port, shock]);
  const sims = useMemo(()=>mcSamples(port, shock, 1500), [port, shock]);
  const { var:var99, es:es99 } = useMemo(()=>varEs(sims, 0.99), [sims]);

  /* ===== Charts ===== */
  const barRef = useRef<HTMLCanvasElement|null>(null);
  const histRef = useRef<HTMLCanvasElement|null>(null);

  // Bar: PnL by asset
  useEffect(()=>{
    const c = barRef.current; if(!c) return; const ctx = c.getContext('2d'); if(!ctx) return;
    ctx.clearRect(0,0,c.width,c.height);
    const W=920,H=300,padL=100,padR=16,padT=24,padB=44;
    const plotW=W-padL-padR, plotH=H-padT-padB;
    const rows = stress.byAsset;
    const bw = rows.length ? plotW/rows.length : plotW;
    const maxAbs = Math.max(1, ...rows.map(r=>Math.abs(r.pnl)));
    // grid
    ctx.strokeStyle='#f3f4f6'; ctx.lineWidth=1;
    ctx.beginPath();
    for(let g=0;g<=4;g++){ const y=padT+(g/4)*plotH; ctx.moveTo(padL,y); ctx.lineTo(padL+plotW,y); }
    ctx.stroke();
    // zero
    const y0 = padT + (0.5)*plotH;
    // bars
    for(let i=0;i<rows.length;i++){
      const v = rows[i].pnl;
      const h = (Math.abs(v)/maxAbs)*(plotH*0.9);
      const x = padL + i*bw + 8;
      const y = v>=0 ? (padT+plotH/2 - h) : (padT+plotH/2);
      ctx.fillStyle = v>=0 ? 'rgba(16,185,129,0.8)' : 'rgba(239,68,68,0.8)';
      ctx.fillRect(x, y, bw-16, h);
      // label
      ctx.save();
      ctx.translate(x + (bw-16)/2, padT + plotH + 14);
      ctx.rotate(-Math.PI/4);
      ctx.fillStyle='#374151'; ctx.font='12px system-ui';
      ctx.fillText(rows[i].symbol, -30, 0);
      ctx.restore();
    }
    // y labels
    ctx.fillStyle='#6b7280'; ctx.font='11px system-ui';
    for(let g=0;g<=4;g++){
      const val = maxAbs - (g/4)*2*maxAbs;
      const y = padT + (g/4)*plotH;
      ctx.fillText(fmt0(val), 8, y+4);
    }
    ctx.fillStyle='#111827'; ctx.font='13px system-ui';
    ctx.fillText('Stressed PnL by asset', padL, padT-6);
  }, [stress]);

  // Histogram: Monte Carlo
  useEffect(()=>{
    const c = histRef.current; if(!c) return; const ctx = c.getContext('2d'); if(!ctx) return;
    ctx.clearRect(0,0,c.width,c.height);
    const W=920,H=280,padL=60,padR=16,padT=20,padB=44;
    const plotW=W-padL-padR, plotH=H-padT-padB;

    if (!sims.length) return;
    const min = sims[0], max = sims[sims.length-1];
    const bins = 40;
    const bw = plotW/bins;
    const counts = new Array(bins).fill(0);
    for(const v of sims){
      const t = clamp(Math.floor(((v-min)/Math.max(1e-9,(max-min)))*bins),0,bins-1);
      counts[t] += 1;
    }
    const maxC = Math.max(...counts, 1);

    // grid
    ctx.strokeStyle='#f3f4f6'; ctx.lineWidth=1;
    ctx.beginPath();
    for(let g=0;g<=3;g++){ const y=padT+(g/3)*plotH; ctx.moveTo(padL,y); ctx.lineTo(padL+plotW,y); }
    ctx.stroke();

    // bars
    for(let i=0;i<bins;i++){
      const x = padL + i*bw + 1;
      const h = (counts[i]/maxC)*plotH;
      ctx.fillStyle='rgba(37,99,235,0.75)';
      ctx.fillRect(x, padT + (plotH - h), bw-2, h);
    }

    // x ticks: min, VaR, median, max
    ctx.fillStyle='#6b7280'; ctx.font='11px system-ui';
    const med = sims[Math.floor(sims.length/2)];
    const ticks = [min, var99, med, max];
    const toX = (v:number)=> padL + ((v-min)/Math.max(1e-9,(max-min)))*plotW;
    for(const v of ticks){
      const x = toX(v);
      ctx.fillText(fmt0(v), x-16, padT+plotH+16);
      // marker
      ctx.strokeStyle = v===var99 ? '#ef4444' : '#9ca3af';
      ctx.setLineDash(v===var99 ? [4,3] : [2,3]);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT+plotH); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle='#111827'; ctx.font='13px system-ui';
    ctx.fillText('PnL distribution (Monte Carlo)', padL, padT-6);
  }, [sims, var99]);

  /* ===== Exports ===== */
  function exportCSV() {
    const head = 'symbol,assetClass,notional,beta,dv01,fxDelta,vega,spr01,liqScore,stressed_pnl';
    const lines = stress.rows.map(r =>
      [r.symbol, r.assetClass, r.notional, r.beta??'', r.dv01??'', r.fxDelta??'', r.vega??'', r.spr01??'', r.liqScore??'', r.pnl.toFixed(2)].join(',')
    );
    const csv = [head, ...lines, `TOTAL,,,,,,,,,${stress.total.toFixed(2)}`].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'stress_results.csv'; a.click();
    URL.revokeObjectURL(a.href);
  }
  function exportPNG() {
    const b1 = barRef.current, b2 = histRef.current;
    if (!b1 || !b2) return;
    const W = 940, H = 600;
    const tmp = document.createElement('canvas'); tmp.width=W; tmp.height=H;
    const ctx = tmp.getContext('2d'); if (!ctx) return;
    ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
    ctx.drawImage(b1, 10, 10);
    ctx.drawImage(b2, 10, 320);
    const a = document.createElement('a');
    a.href = tmp.toDataURL('image/png'); a.download='stress_sandbox.png'; a.click();
  }

  /* ===== UI ===== */
  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>Stress Sandbox</h2>
          <span style={S.badge}>Total: {sign(stress.total)}{fmt0(stress.total)}</span>
          <span style={{...S.badge, background:'#fef2f2', color:'#991b1b'}}>VaR99: {fmt0(var99)}</span>
          <span style={{...S.badge, background:'#fff7ed', color:'#9a3412'}}>ES99: {fmt0(es99)}</span>
        </div>
        <div style={S.controls}>
          <button onClick={exportCSV} style={S.btn}>Export CSV</button>
          <button onClick={exportPNG} style={S.btn}>Export PNG</button>
        </div>
      </div>

      {/* Controls */}
      <div style={{ padding:12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Shocks</div>
          <div style={S.grid}>
            <Slider label="Equity move (%)"  value={shock.eqPct}  min={-40} max={+20} step={1} onChange={v=>setShock(s=>({...s, eqPct:v}))}/>
            <Slider label="Rates (bp)"       value={shock.rateBp} min={-150} max={+300} step={5} onChange={v=>setShock(s=>({...s, rateBp:v}))}/>
            <Slider label="FX move (%)"      value={shock.fxPct}  min={-10} max={+10}  step={0.5} onChange={v=>setShock(s=>({...s, fxPct:v}))}/>
            <Slider label="Vol change (pts)" value={shock.volPts} min={-10} max={+30}  step={1} onChange={v=>setShock(s=>({...s, volPts:v}))}/>
            <Slider label="Spread (bp)"      value={shock.sprBp}  min={-50} max={+250} step={5} onChange={v=>setShock(s=>({...s, sprBp:v}))}/>
            <Slider label="Liquidity haircut (%)" value={shock.liqPct} min={0} max={30} step={1} onChange={v=>setShock(s=>({...s, liqPct:v}))}/>
            <label style={{...S.ctrlItem, gridColumn:'span 2'}}>
              <span style={S.lbl}>Correlation breakdown</span>
              <input type="checkbox" checked={shock.corrBreak} onChange={e=>setShock(s=>({...s, corrBreak:e.target.checked}))}/>
            </label>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div style={{ padding:12, display:'grid', gridTemplateColumns:'1fr', gap:12 }}>
        <div style={S.card}><canvas ref={barRef}  width={920} height={300} style={S.canvas}/></div>
        <div style={S.card}><canvas ref={histRef} width={920} height={280} style={S.canvas}/></div>
      </div>

      {/* Factor table */}
      <div style={{ padding:12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>By factor</div>
          <div style={{ overflow:'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  {Object.keys(stress.byFactor).map(k=><th key={k} style={S.th}>{k}</th>)}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {Object.keys(stress.byFactor).map(k=>(
                    <td key={k} style={S.td}>{sign(stress.byFactor[k])}{fmt0(stress.byFactor[k])}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Positions table */}
      <div style={{ padding:12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Positions</div>
          <div style={{ overflow:'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={{...S.th, textAlign:'left'}}>Symbol</th>
                  <th style={S.th}>Class</th>
                  <th style={S.th}>Notional</th>
                  <th style={S.th}>β</th>
                  <th style={S.th}>DV01</th>
                  <th style={S.th}>FXΔ</th>
                  <th style={S.th}>Vega</th>
                  <th style={S.th}>Spr01</th>
                  <th style={S.th}>Liq</th>
                  <th style={S.th}>Stressed PnL</th>
                </tr>
              </thead>
              <tbody>
                {stress.rows.map((r,i)=>(
                  <tr key={i} style={{ background: r.pnl>=0 ? '#ecfdf5' : '#fef2f2' }}>
                    <td style={{...S.td, textAlign:'left', fontWeight:600}}>{r.symbol}</td>
                    <td style={S.td}>{r.assetClass}</td>
                    <td style={S.td}>{fmt0(r.notional)}</td>
                    <td style={S.td}>{r.beta ?? '-'}</td>
                    <td style={S.td}>{r.dv01 ?? '-'}</td>
                    <td style={S.td}>{r.fxDelta ?? '-'}</td>
                    <td style={S.td}>{r.vega ?? '-'}</td>
                    <td style={S.td}>{r.spr01 ?? '-'}</td>
                    <td style={S.td}>{r.liqScore ?? '-'}</td>
                    <td style={S.td}>{sign(r.pnl)}{fmt0(r.pnl)}</td>
                  </tr>
                ))}
                {stress.rows.length===0 && <tr><td style={S.td} colSpan={10}>No positions</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ===== Subcomponents ===== */
function Slider({label, value, min, max, step, onChange}:{label:string; value:number; min:number; max:number; step:number; onChange:(v:number)=>void}){
  return (
    <label style={S.ctrlItem}>
      <span style={S.lbl}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(parseFloat(e.target.value))}/>
      <div style={{ fontSize:12, color:'#111827', fontWeight:700 }}>{fmt2(value)}</div>
    </label>
  );
}

/* ================= Styles ================= */
const S: Record<string, React.CSSProperties> = {
  wrap: { border:'1px solid #e5e7eb', borderRadius:16, background:'#fff', boxShadow:'0 2px 6px rgba(0,0,0,0.06)', width:'100%', fontFamily:'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding:16, borderBottom:'1px solid #eee' },
  titleRow: { display:'flex', alignItems:'center', gap:8, flexWrap:'wrap', marginBottom:10 },
  title: { margin:0, fontSize:18, fontWeight:700, color:'#111827' },
  badge: { display:'inline-flex', alignItems:'center', borderRadius:999, padding:'4px 10px', fontSize:12, fontWeight:600, background:'#f3f4f6', color:'#111827' },

  controls: { display:'flex', gap:10, alignItems:'center', flexWrap:'wrap' },
  btn: { height:36, padding:'0 12px', borderRadius:10, border:'1px solid #e5e7eb', background:'#fff', cursor:'pointer', fontSize:13 },

  card: { border:'1px solid #e5e7eb', borderRadius:12, background:'#fff', overflow:'hidden', padding:8 },
  cardHeader: { padding:'10px 12px', borderBottom:'1px solid #eee', fontWeight:700, fontSize:14, color:'#111827', display:'flex', alignItems:'center' },
  canvas: { width:'100%', maxWidth:1120, borderRadius:8, background:'#fff' },

  grid: { display:'grid', gridTemplateColumns:'repeat(3, minmax(180px, 1fr))', gap:10, padding:12 },
  ctrlItem: { display:'flex', flexDirection:'column', gap:6, padding:4 },
  lbl: { fontSize:12, color:'#6b7280' },

  table: { width:'100%', borderCollapse:'separate', borderSpacing:0 },
  th: { textAlign:'right', padding:'8px 10px', fontSize:12, color:'#6b7280', borderBottom:'1px solid #eee', position:'sticky' as any, top:0, background:'#fff' },
  td: { textAlign:'right', padding:'8px 10px', fontSize:13, color:'#111827', borderBottom:'1px solid #f3f4f6' },
};