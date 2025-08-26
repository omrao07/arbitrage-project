'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type Comparator = '>' | '<' | '>=' | '<=' | '==' | '!=' | 'crossesAbove' | 'crossesBelow';
type Indicator =
  | 'SMA' | 'EMA' | 'RSI' | 'MACD' | 'Price' | 'ROC' | 'BBUpper' | 'BBLower'
  | 'VWAP' | 'ATR';

type Condition = {
  id: string;
  left: { indicator: Indicator; period?: number; source?: 'close'|'open'|'high'|'low' };
  cmp: Comparator;
  right: { indicator: Indicator; period?: number; value?: number; source?: 'close'|'open'|'high'|'low' };
};

type Rule = {
  id: string;
  name: string;
  logic: 'ALL' | 'ANY';      // AND / OR between its conditions
  weight: number;            // contribution to signal [-1..1]
  side: 'long'|'short'|'both';
  conditions: Condition[];
};

type Risk = {
  capital: number;
  maxPos: number;            // max concurrent positions
  stopLossPct?: number;      // e.g., 0.02
  takeProfitPct?: number;
  cooldownBars?: number;
};

type StrategyConfig = {
  name: string;
  symbol: string;
  timeframe: '1m'|'5m'|'15m'|'1h'|'1d';
  rules: Rule[];
  risk: Risk;
  notes?: string;
};

/* ================= Helpers ================= */
const uid = () => Math.random().toString(36).slice(2, 9);
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const fmt = (n: number, d=2) => n.toLocaleString(undefined, { maximumFractionDigits: d });

/* --- tiny TA helpers on arrays (mock-friendly) --- */
function SMA(arr: number[], p: number) {
  const out = new Array(arr.length).fill(NaN);
  let s = 0;
  for (let i=0;i<arr.length;i++){
    s += arr[i];
    if (i >= p) s -= arr[i-p];
    if (i >= p-1) out[i] = s / p;
  }
  return out;
}
function EMA(arr: number[], p: number) {
  const out = new Array(arr.length).fill(NaN);
  const k = 2/(p+1);
  let prev = arr[0] ?? 0;
  for (let i=0;i<arr.length;i++){
    const v = i===0 ? arr[0] : (arr[i]*k + prev*(1-k));
    out[i] = v; prev = v;
  }
  return out;
}
function ROC(arr: number[], p: number) {
  const out = new Array(arr.length).fill(NaN);
  for (let i=0;i<arr.length;i++){
    out[i] = i>=p ? (arr[i]-arr[i-p])/(Math.abs(arr[i-p])||1e-9) : NaN;
  }
  return out;
}
function RSI(arr: number[], p: number) {
  const out = new Array(arr.length).fill(NaN);
  let gain=0, loss=0;
  for (let i=1;i<arr.length;i++){
    const ch = arr[i]-arr[i-1];
    const g = Math.max(0,ch), l = Math.max(0,-ch);
    if (i<=p){ gain+=g; loss+=l; if(i===p){ const rs=gain/Math.max(1e-9,loss); out[i]=100-100/(1+rs);} }
    else {
      const pg = (!Number.isNaN(out[i - 1])) ? gain/p : gain/p;
      const pl = (!Number.isNaN(out[i - 1])) ? loss/p : loss/p;
      gain = (pg*(p-1))+g; loss=(pl*(p-1))+l;
      const rs=gain/Math.max(1e-9,loss); out[i]=100-100/(1+rs);
    }
  }
  return out;
}
function MACD(arr: number[], fast=12, slow=26, sig=9) {
  const efast = EMA(arr, fast);
  const eslow = EMA(arr, slow);
  const macd = efast.map((v,i)=>v - (eslow[i] ?? NaN));
  const signal = EMA(macd.map(v=>Number.isFinite(v)?v:0), sig);
  return { macd, signal, hist: macd.map((v,i)=>v - (signal[i] ?? NaN)) };
}
function BB(arr: number[], p=20, k=2) {
  const ma = SMA(arr, p);
  const outU = new Array(arr.length).fill(NaN);
  const outL = new Array(arr.length).fill(NaN);
  for (let i=0;i<arr.length;i++){
    if (i>=p-1){
      let s2=0;
      for (let j=i-p+1;j<=i;j++){ const d=arr[j]-ma[i]; s2+=d*d; }
      const sd = Math.sqrt(s2/p);
      outU[i] = ma[i] + k*sd;
      outL[i] = ma[i] - k*sd;
    }
  }
  return { upper: outU, lower: outL, mid: ma };
}

/* --- compute indicator series --- */
function computeIndicator(ind: Indicator, price: number[], p?: number) {
  switch(ind){
    case 'Price': return price;
    case 'SMA': return SMA(price, p ?? 20);
    case 'EMA': return EMA(price, p ?? 20);
    case 'RSI': return RSI(price, p ?? 14);
    case 'MACD': return MACD(price).macd;
    case 'ROC': return ROC(price, p ?? 10);
    case 'BBUpper': return BB(price, p ?? 20, 2).upper;
    case 'BBLower': return BB(price, p ?? 20, 2).lower;
    case 'VWAP': {
      // simple running average as placeholder (no volume in mock)
      const out = new Array(price.length).fill(NaN);
      let s=0;
      for(let i=0;i<price.length;i++){ s+=price[i]; out[i]=s/(i+1); }
      return out;
    }
    case 'ATR': {
      // use price diffs as ATR proxy in mock
      const out = new Array(price.length).fill(NaN);
      const pX = p ?? 14;
      for(let i=1;i<price.length;i++){
        const tr = Math.abs(price[i]-price[i-1]);
        out[i] = i<pX ? NaN : (out[i-1] ?? tr);
      }
      return out;
    }
  }
}

/* --- evaluate a single condition at bar i --- */
function evalCondition(c: Condition, i: number, price: number[], cache: Record<string, number[]>) {
  function seriesFor(side: 'left'|'right'): number[] {
    const node = side==='left' ? c.left : c.right;
    const key = `${side}:${node.indicator}:${node.period ?? 0}`;
    if (!cache[key]) cache[key] = computeIndicator(node.indicator, price, node.period) as number[];
    return cache[key];
  }
  const L = seriesFor('left')[i];
  const R = c.right.value != null ? c.right.value : seriesFor('right')[i];

  if (!Number.isFinite(L) || !Number.isFinite(R)) return false;

  switch(c.cmp){
    case '>': return L >  R!;
    case '<': return L <  R!;
    case '>=': return L >= R!;
    case '<=': return L <= R!;
    case '==': return Math.abs(L - (R as number)) < 1e-9;
    case '!=': return Math.abs(L - (R as number)) > 1e-9;
    case 'crossesAbove': {
      const prevL = seriesFor('left')[i-1], prevR = c.right.value!=null ? c.right.value : seriesFor('right')[i-1];
      if (!Number.isFinite(prevL) || !Number.isFinite(prevR)) return false;
      return prevL <= prevR && L > (R as number);
    }
    case 'crossesBelow': {
      const prevL = seriesFor('left')[i-1], prevR = c.right.value!=null ? c.right.value : seriesFor('right')[i-1];
      if (!Number.isFinite(prevL) || !Number.isFinite(prevR)) return false;
      return prevL >= prevR && L < (R as number);
    }
  }
}

/* --- simple mock price series --- */
function mockPrice(n=500) {
  const out:number[] = [];
  let p=100;
  for (let i=0;i<n;i++){
    const drift = 0.02*Math.sin(i/35);
    const noise = (Math.random()-0.5)*0.8;
    p = Math.max(1, p + drift + noise);
    out.push(+p.toFixed(2));
  }
  return out;
}

/* --- apply strategy → signal [-1..+1] and equity --- */
function simulate(cfg: StrategyConfig, price: number[]) {
  const cache: Record<string, number[]> = {};
  const N = price.length;
  const signal = new Array(N).fill(0);
  const weights = cfg.rules.map(r=>r.weight);
  const W = weights.reduce((s,w)=>s+Math.abs(w), 0) || 1;

  // raw signal
  for (let i=1;i<N;i++){
    let s=0;
    for (const r of cfg.rules){
      if (r.conditions.length===0) continue;
      const hits = r.conditions.map(c=>evalCondition(c, i, price, cache));
      const ok = r.logic==='ALL' ? hits.every(Boolean) : hits.some(Boolean);
      if (ok) s += r.weight * (r.side==='short' ? -1 : 1);
    }
    signal[i] = clamp(s/W, -1, 1);
  }

  // translate to equity with naive execution and risk stops (mock)
  const eq = new Array(N).fill(0);
  let pos = 0; // +1 long, -1 short (simplified)
  let entry = 0;
  for (let i=1;i<N;i++){
    const sig = signal[i];
    // entries/exits
    if (sig > 0.4 && pos<=0){ pos = 1; entry = price[i]; }
    else if (sig < -0.4 && pos>=0){ pos = -1; entry = price[i]; }
    else if (Math.abs(sig) < 0.2 && pos!==0){ pos = 0; entry = 0; }

    // stops (mock as pct from entry)
    if (pos !== 0 && entry>0){
      const pnlPct = pos===1 ? (price[i]-entry)/entry : (entry-price[i])/entry;
      if (cfg.risk.stopLossPct && pnlPct <= -(cfg.risk.stopLossPct)) { pos=0; entry=0; }
      if (cfg.risk.takeProfitPct && pnlPct >= (cfg.risk.takeProfitPct)) { pos=0; entry=0; }
    }
    eq[i] = eq[i-1] + pos * (price[i]-price[i-1]);
  }

  // drawdown
  const peak = new Array(N).fill(0);
  const dd = new Array(N).fill(0);
  let pmax = 0;
  for (let i=0;i<N;i++){ pmax = Math.max(pmax, eq[i]); peak[i]=pmax; dd[i] = eq[i]-pmax; }

  return { signal, eq, dd };
}

/* ================= Component ================= */
export default function StrategyBuilder() {
  // default config
  const [cfg, setCfg] = useState<StrategyConfig>({
    name: 'MeanRevert + Breakout',
    symbol: 'DEMO',
    timeframe: '15m',
    rules: [
      {
        id: uid(),
        name: 'Mean reversion buy',
        logic: 'ALL',
        weight: 1,
        side: 'long',
        conditions: [
          { id: uid(), left:{indicator:'Price'}, cmp:'crossesBelow', right:{indicator:'BBLower', period:20} },
          { id: uid(), left:{indicator:'RSI', period:14}, cmp:'<', right:{indicator:'Price', value:35} }, // RSI<35 (value via right.value)
        ],
      },
      {
        id: uid(),
        name: 'Momentum breakout sell',
        logic: 'ALL',
        weight: 1,
        side: 'short',
        conditions: [
          { id: uid(), left:{indicator:'Price'}, cmp:'crossesBelow', right:{indicator:'EMA', period:50} },
          { id: uid(), left:{indicator:'ROC', period:10}, cmp:'<', right:{indicator:'Price', value:0} },
        ],
      }
    ],
    risk: { capital: 100_000, maxPos: 5, stopLossPct: 0.02, takeProfitPct: 0.04, cooldownBars: 10 },
    notes: 'Demo strategy composed in the UI.',
  });

  const [selectedRule, setSelectedRule] = useState<string | null>(cfg.rules[0]?.id ?? null);
  const sel = useMemo(()=>cfg.rules.find(r=>r.id===selectedRule) ?? null,[cfg.rules, selectedRule]);

  // mock price + sim
  const price = useMemo(()=>mockPrice(480), []);
  const { signal, eq, dd } = useMemo(()=>simulate(cfg, price), [cfg, price]);

  // derived KPIs
  const totalPnL = (eq[eq.length-1] ?? 0) - (eq[0] ?? 0);
  const maxDD = Math.abs(Math.min(0, ...dd));
  const sharpe = useMemo(()=>{
    const rets = eq.map((v,i)=> i===0 ? 0 : (v - eq[i-1]));
    const m = rets.reduce((a,b)=>a+b,0)/Math.max(1,rets.length);
    const s = Math.sqrt(rets.reduce((a,b)=>a+(b-m)*(b-m),0)/Math.max(1,rets.length));
    return s>0 ? (m/s)*Math.sqrt(252) : 0;
  }, [eq]);

  /* ========= Handlers ========= */
  function addRule(side: 'long'|'short'|'both') {
    const r: Rule = { id: uid(), name: `Rule ${cfg.rules.length+1}`, logic: 'ALL', weight: 1, side, conditions: [] };
    setCfg(c => ({ ...c, rules: [...c.rules, r] })); setSelectedRule(r.id);
  }
  function removeRule(id: string) {
    setCfg(c => ({ ...c, rules: c.rules.filter(r => r.id !== id) }));
    setSelectedRule(p => (p===id ? null : p));
  }
  function updateRule(id: string, patch: Partial<Rule>) {
    setCfg(c => ({ ...c, rules: c.rules.map(r => r.id===id ? { ...r, ...patch } : r) }));
  }
  function addCondition(ruleId: string) {
    const nc: Condition = { id: uid(), left:{indicator:'SMA', period:20}, cmp:'>', right:{indicator:'EMA', period:50} };
    setCfg(c => ({ ...c, rules: c.rules.map(r => r.id===ruleId ? { ...r, conditions:[...r.conditions, nc] } : r) }));
  }
  function removeCondition(ruleId: string, condId: string) {
    setCfg(c => ({ ...c, rules: c.rules.map(r => r.id===ruleId ? { ...r, conditions:r.conditions.filter(x=>x.id!==condId) } : r) }));
  }

  function exportJSON() {
    const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${cfg.name.replace(/\s+/g,'_')}.json`; a.click();
    URL.revokeObjectURL(a.href);
  }
  function exportPseudo() {
    const lines: string[] = [];
    lines.push(`# ${cfg.name} (${cfg.symbol}, ${cfg.timeframe})`);
    for (const r of cfg.rules) {
      lines.push(`\n# Rule: ${r.name} [${r.side}] weight=${r.weight} logic=${r.logic}`);
      lines.push(`IF ${r.conditions.map(condToText).join(r.logic==='ALL'?' AND ':' OR ')} THEN signal += ${r.side==='short'?'-':''}${r.weight}`);
    }
    lines.push(`\nRisk: stopLoss=${(cfg.risk.stopLossPct??0)*100}% takeProfit=${(cfg.risk.takeProfitPct??0)*100}% cooldown=${cfg.risk.cooldownBars} bars`);
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${cfg.name.replace(/\s+/g,'_')}.txt`; a.click();
    URL.revokeObjectURL(a.href);
  }
  function condToText(c: Condition) {
    const L = c.left.indicator + (c.left.period?`(${c.left.period})`: '');
    const R = c.right.value!=null ? String(c.right.value) : c.right.indicator + (c.right.period?`(${c.right.period})`:'');
    return `${L} ${c.cmp} ${R}`;
  }

  /* ========= Canvas preview ========= */
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const W = 920, H = 360, padL=60, padR=16, padT=24, padB=40;
  useEffect(()=>{
    const c = canvasRef.current; if(!c) return; const ctx=c.getContext('2d'); if(!ctx) return;
    ctx.clearRect(0,0,W,H);
    const plotW = W-padL-padR, plotH = H-padT-padB;

    // eq axis
    const vmin = Math.min(...eq), vmax = Math.max(...eq, 1);
    const xAt = (i:number)=> padL + (i/(price.length-1))*plotW;
    const yAt = (v:number)=> padT + (1-(v - vmin)/Math.max(1e-9, (vmax-vmin)))*plotH;

    // grid
    ctx.strokeStyle='#f3f4f6'; ctx.lineWidth=1;
    ctx.beginPath();
    for(let g=0; g<=4; g++){ const y=padT+(g/4)*plotH; ctx.moveTo(padL,y); ctx.lineTo(padL+plotW,y); }
    ctx.stroke();

    // equity line
    ctx.beginPath();
    for(let i=0;i<price.length;i++){
      const x=xAt(i), y=yAt(eq[i]);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.strokeStyle='#111827'; ctx.lineWidth=2; ctx.stroke();

    // zero line
    const y0=yAt(0);
    ctx.setLineDash([4,4]); ctx.strokeStyle='#9ca3af';
    ctx.beginPath(); ctx.moveTo(padL,y0); ctx.lineTo(padL+plotW,y0); ctx.stroke();
    ctx.setLineDash([]);

    // overlay signal heat (green/red bars at bottom)
    const h=6;
    for(let i=0;i<price.length;i++){
      const x=xAt(i);
      const s=signal[i];
      if (s===0) continue;
      ctx.strokeStyle = s>0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)';
      ctx.beginPath();
      ctx.moveTo(x, padT+plotH+h+2);
      ctx.lineTo(x, padT+plotH+2);
      ctx.stroke();
    }

    // axes labels
    ctx.fillStyle='#6b7280'; ctx.font='11px system-ui';
    const ticks = 6;
    for(let t=0;t<ticks;t++){
      const i = Math.round((t/(ticks-1))*(price.length-1));
      ctx.fillText(String(i), xAt(i)-6, padT+plotH+18);
    }
    for(let g=0; g<=4; g++){
      const y=padT+(g/4)*plotH;
      const val = vmax - (g/4)*(vmax-vmin);
      ctx.fillText(fmt(val,0), 8, y+4);
    }
  }, [price, eq, signal]);

  /* ========= UI ========= */
  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>Strategy Builder</h2>
          <span style={S.badge}>{cfg.name}</span>
          <span style={{...S.badge, background:'#eef2ff', color:'#3730a3'}}>PnL: {fmt(totalPnL,0)}</span>
          <span style={{...S.badge, background:'#fef2f2', color:'#991b1b'}}>MaxDD: {fmt(maxDD,0)}</span>
          <span style={{...S.badge, background:'#ecfeff', color:'#0369a1'}}>Sharpe≈ {fmt(sharpe,2)}</span>
        </div>
        <div style={S.controls}>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Name</span>
            <input value={cfg.name} onChange={e=>setCfg(c=>({...c, name:e.target.value}))} style={S.input}/>
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Symbol</span>
            <input value={cfg.symbol} onChange={e=>setCfg(c=>({...c, symbol:e.target.value.toUpperCase()}))} style={S.input}/>
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Timeframe</span>
            <select value={cfg.timeframe} onChange={e=>setCfg(c=>({...c, timeframe:e.target.value as any}))} style={S.select}>
              <option value="1m">1m</option><option value="5m">5m</option><option value="15m">15m</option>
              <option value="1h">1h</option><option value="1d">1d</option>
            </select>
          </label>
          <button onClick={exportJSON} style={S.btn}>Export JSON</button>
          <button onClick={exportPseudo} style={S.btn}>Export Pseudo</button>
        </div>
      </div>

      {/* Preview */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Mock Backtest (equity & signals)</div>
          <canvas ref={canvasRef} width={W} height={H} style={S.canvas}/>
        </div>
      </div>

      {/* Risk */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Risk</div>
          <div style={S.grid}>
            <Field label="Capital"  value={cfg.risk.capital}  onChange={v=>setCfg(c=>({...c, risk:{...c.risk, capital:v}}))}/>
            <Field label="Max Positions" value={cfg.risk.maxPos} onChange={v=>setCfg(c=>({...c, risk:{...c.risk, maxPos:v}}))}/>
            <Field label="Stop Loss %" value={(cfg.risk.stopLossPct??0)*100} onChange={v=>setCfg(c=>({...c, risk:{...c.risk, stopLossPct:v/100}}))}/>
            <Field label="Take Profit %" value={(cfg.risk.takeProfitPct??0)*100} onChange={v=>setCfg(c=>({...c, risk:{...c.risk, takeProfitPct:v/100}}))}/>
            <Field label="Cooldown bars" value={(cfg.risk.cooldownBars??0)} onChange={v=>setCfg(c=>({...c, risk:{...c.risk, cooldownBars:v}}))}/>
          </div>
        </div>
      </div>

      {/* Rules list + editor */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>
            Rules
            <div style={{ marginLeft:'auto', display:'inline-flex', gap:8 }}>
              <button onClick={()=>addRule('long')}  style={S.btnSmall}>+ Long</button>
              <button onClick={()=>addRule('short')} style={S.btnSmall}>+ Short</button>
              <button onClick={()=>addRule('both')}  style={S.btnSmall}>+ Both</button>
            </div>
          </div>

          <div style={S.rulesWrap}>
            {/* Sidebar list */}
            <div style={S.sidebar}>
              {cfg.rules.map(r=>(
                <div key={r.id} style={{...S.ruleItem, background: r.id===selectedRule ? '#eef2ff' : '#fff' }}>
                  <div style={{ display:'flex', gap:8, alignItems:'center' }}>
                    <button onClick={()=>setSelectedRule(r.id)} style={S.btnMini}>Edit</button>
                    <div style={{ fontWeight:700 }}>{r.name}</div>
                  </div>
                  <div style={{ display:'flex', gap:6, alignItems:'center' }}>
                    <span style={S.tag}>{r.side}</span>
                    <span style={S.tag}>w={r.weight}</span>
                    <span style={S.tag}>{r.logic}</span>
                    <button onClick={()=>removeRule(r.id)} style={{...S.btnMini, background:'#fee2e2', color:'#991b1b'}}>Delete</button>
                  </div>
                </div>
              ))}
              {cfg.rules.length===0 && <div style={{ padding:8, color:'#6b7280' }}>No rules.</div>}
            </div>

            {/* Editor */}
            <div style={S.editor}>
              {!sel ? <div style={{ padding:8, color:'#6b7280' }}>Select a rule to edit.</div> : (
                <div>
                  <div style={S.row}>
                    <label style={S.lbl2}>Name</label>
                    <input value={sel.name} onChange={e=>updateRule(sel.id,{name:e.target.value})} style={S.input}/>
                    <label style={S.lbl2}>Side</label>
                    <select value={sel.side} onChange={e=>updateRule(sel.id,{side:e.target.value as any})} style={S.select}>
                      <option value="long">long</option><option value="short">short</option><option value="both">both</option>
                    </select>
                    <label style={S.lbl2}>Logic</label>
                    <select value={sel.logic} onChange={e=>updateRule(sel.id,{logic:e.target.value as any})} style={S.select}>
                      <option value="ALL">ALL</option><option value="ANY">ANY</option>
                    </select>
                    <label style={S.lbl2}>Weight</label>
                    <input type="number" step="0.1" value={sel.weight} onChange={e=>updateRule(sel.id,{weight: parseFloat(e.target.value)})} style={S.inputSmall}/>
                  </div>

                  <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', margin:'8px 0' }}>
                    <div style={{ fontWeight:700 }}>Conditions</div>
                    <button onClick={()=>addCondition(sel.id)} style={S.btnSmall}>+ Condition</button>
                  </div>

                  {sel.conditions.map((c, idx)=>(
                    <div key={c.id} style={S.condRow}>
                      {/* left */}
                      <select value={c.left.indicator} onChange={e=>{
                        const indicator = e.target.value as Indicator;
                        updateRule(sel.id, { conditions: sel.conditions.map(k=> k.id===c.id ? { ...k, left:{...k.left, indicator} } : k ) });
                      }} style={S.select}>
                        {IND_OPTS.map(o=><option key={o} value={o}>{o}</option>)}
                      </select>
                      <input type="number" placeholder="period" value={c.left.period ?? ''} onChange={e=>{
                        const period = e.target.value==='' ? undefined : parseInt(e.target.value);
                        updateRule(sel.id, { conditions: sel.conditions.map(k=> k.id===c.id ? { ...k, left:{...k.left, period} } : k ) });
                      }} style={S.inputSmall}/>

                      {/* comparator */}
                      <select value={c.cmp} onChange={e=>{
                        const cmp = e.target.value as Comparator;
                        updateRule(sel.id, { conditions: sel.conditions.map(k=> k.id===c.id ? { ...k, cmp } : k ) });
                      }} style={S.select}>
                        {CMP_OPTS.map(o=><option key={o} value={o}>{o}</option>)}
                      </select>

                      {/* right */}
                      <select value={c.right.indicator} onChange={e=>{
                        const indicator = e.target.value as Indicator;
                        updateRule(sel.id, { conditions: sel.conditions.map(k=> k.id===c.id ? { ...k, right:{...k.right, indicator}, } : k ) });
                      }} style={S.select}>
                        {IND_OPTS.map(o=><option key={o} value={o}>{o}</option>)}
                      </select>
                      <input type="number" placeholder="period/value" value={c.right.value ?? c.right.period ?? ''} onChange={e=>{
                        const v = e.target.value==='' ? undefined : Number(e.target.value);
                        // if comparator is against numeric value, store as value; otherwise treat as period
                        const numericCmp = ['>','<','>=','<=','==','!='].includes(c.cmp);
                        const nextRight = { ...c.right };
                        if (numericCmp) { nextRight.value = v; nextRight.period = nextRight.period; }
                        else { nextRight.period = Number.isFinite(v as number) ? (v as number) : undefined; nextRight.value = undefined; }
                        updateRule(sel.id, { conditions: sel.conditions.map(k=> k.id===c.id ? { ...k, right: nextRight } : k ) });
                      }} style={S.inputSmall}/>

                      <button onClick={()=>removeCondition(sel.id, c.id)} style={{...S.btnMini, background:'#fee2e2', color:'#991b1b'}}>Delete</button>
                    </div>
                  ))}
                  {sel.conditions.length===0 && <div style={{ padding:8, color:'#6b7280' }}>No conditions yet.</div>}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Notes */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Notes</div>
          <textarea value={cfg.notes ?? ''} onChange={e=>setCfg(c=>({...c, notes:e.target.value}))} style={S.textarea} rows={4}/>
        </div>
      </div>
    </div>
  );
}

/* ===== Small subcomponent ===== */
function Field({label, value, onChange}:{label:string; value:number; onChange:(v:number)=>void}){
  return (
    <label style={S.ctrlItem}>
      <span style={S.lbl}>{label}</span>
      <input type="number" value={Number.isFinite(value) ? value : 0} onChange={e=>onChange(parseFloat(e.target.value))} style={S.input}/>
    </label>
  );
}

/* ===== Consts & Styles ===== */
const IND_OPTS: Indicator[] = ['Price','SMA','EMA','RSI','MACD','ROC','BBUpper','BBLower','VWAP','ATR'];
const CMP_OPTS: Comparator[] = ['>','<','>=','<=','==','!=','crossesAbove','crossesBelow'];

const S: Record<string, React.CSSProperties> = {
  wrap: { border:'1px solid #e5e7eb', borderRadius:16, background:'#fff', boxShadow:'0 2px 6px rgba(0,0,0,0.06)', width:'100%', fontFamily:'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding:16, borderBottom:'1px solid #eee' },
  titleRow: { display:'flex', alignItems:'center', gap:8, flexWrap:'wrap', marginBottom:10 },
  title: { margin:0, fontSize:18, fontWeight:700, color:'#111827' },
  badge: { display:'inline-flex', alignItems:'center', borderRadius:999, padding:'4px 10px', fontSize:12, fontWeight:600, background:'#f3f4f6', color:'#111827' },
  controls: { display:'flex', gap:10, alignItems:'center', flexWrap:'wrap' },
  ctrlItem: { display:'flex', flexDirection:'column', gap:6 },
  lbl: { fontSize:12, color:'#6b7280' },
  lbl2: { fontSize:12, color:'#6b7280', minWidth:60 },
  input: { height:36, border:'1px solid #e5e7eb', borderRadius:10, padding:'0 10px', fontSize:14, outline:'none', minWidth:140 },
  inputSmall: { height:36, border:'1px solid #e5e7eb', borderRadius:10, padding:'0 8px', fontSize:13, outline:'none', width:110 },
  select: { height:36, border:'1px solid #e5e7eb', borderRadius:10, padding:'0 8px', fontSize:14, background:'#fff', minWidth:120 },
  btn: { height:36, padding:'0 12px', borderRadius:10, border:'1px solid #e5e7eb', background:'#fff', cursor:'pointer', fontSize:13 },
  btnSmall: { height:30, padding:'0 10px', borderRadius:8, border:'1px solid #e5e7eb', background:'#fff', cursor:'pointer', fontSize:12 },
  btnMini: { height:26, padding:'0 8px', borderRadius:8, border:'1px solid #e5e7eb', background:'#fff', cursor:'pointer', fontSize:12 },

  card: { border:'1px solid #e5e7eb', borderRadius:12, background:'#fff', overflow:'hidden' },
  cardHeader: { padding:'10px 12px', borderBottom:'1px solid #eee', fontWeight:700, fontSize:14, color:'#111827', display:'flex', alignItems:'center' },
  canvas: { width:'100%', maxWidth:1120, borderRadius:8, background:'#fff' },

  grid: { display:'grid', gridTemplateColumns:'repeat(5, minmax(140px, 1fr))', gap:10, padding:12 },
  rulesWrap: { display:'grid', gridTemplateColumns:'320px 1fr', gap:12, padding:12 },
  sidebar: { borderRight:'1px solid #eee', paddingRight:12, maxHeight:420, overflow:'auto' },
  editor: { paddingLeft:12 },
  ruleItem: { border:'1px solid #e5e7eb', borderRadius:10, padding:8, marginBottom:8, display:'flex', alignItems:'center', justifyContent:'space-between' },
  tag: { fontSize:11, padding:'2px 8px', borderRadius:999, background:'#f3f4f6', color:'#111827' },

  row: { display:'flex', gap:8, alignItems:'center', margin:'6px 0' },
  condRow: { display:'flex', gap:8, alignItems:'center', padding:'8px 0', borderBottom:'1px solid #f3f4f6' },

  textarea: { width:'100%', border:'1px solid #e5e7eb', borderRadius:10, padding:'8px 10px', fontSize:14, outline:'none' },
};