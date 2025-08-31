'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ===================== Types ===================== */

type GNode = {
  id: string;
  label?: string;
  group?: string;
  size?: number;         // visual radius multiplier
  score?: number;        // e.g., centrality score
  color?: string;        // optional fixed color
  pinned?: boolean;
  x?: number; y?: number;
  vx?: number; vy?: number;
};
type GEdge = {
  id?: string;
  source: string;
  target: string;
  weight?: number;       // spring strength
  directed?: boolean;
};

type Patch = {
  nodes?: GNode[];
  edges?: GEdge[];
  node?: GNode;
  edge?: GEdge;
  notes?: string;
  kpis?: { label: string; value: string | number }[];
};

type Props = {
  endpoint?: string; // e.g. '/api/graph/pro'
  title?: string;
  className?: string;
};

/* ===================== Utils ===================== */

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const dist2 = (ax:number, ay:number, bx:number, by:number)=> {
  const dx = ax-bx, dy=ay-by; return dx*dx+dy*dy;
};
const rand = (a:number,b:number)=> a + Math.random()*(b-a);
const themeColors = ['#4f46e5','#059669','#dc2626','#f59e0b','#06b6d4','#8b5cf6','#10b981','#64748b','#a855f7','#ef4444'];

/* ===================== Component ===================== */

export default function GraphPro({
  endpoint = '/api/graph/pro',
  title = 'Graph Pro',
  className = '',
}: Props) {
  // controls
  const [query, setQuery] = useState('');
  const [minW, setMinW] = useState<number>(0);
  const [layout, setLayout] = useState<'force'|'static'>('force');
  const [showLabels, setShowLabels] = useState(true);
  const [highlightPath, setHighlightPath] = useState(false);

  // data
  const [nodes, setNodes] = useState<GNode[]>([]);
  const [edges, setEdges] = useState<GEdge[]>([]);
  const [kpis, setKpis] = useState<{label:string; value:string|number}[]>([]);
  const [notes, setNotes] = useState('');

  // run state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string|null>(null);
  const [log, setLog] = useState('');
  const abortRef = useRef<AbortController|null>(null);

  // selection + path
  const [selA, setSelA] = useState<string|undefined>();
  const [selB, setSelB] = useState<string|undefined>();
  const [path, setPath] = useState<string[]>([]);

  // canvas (SVG) pan/zoom + drag
  const svgRef = useRef<SVGSVGElement|null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({x:0, y:0});
  const panning = useRef<{on:boolean; sx:number; sy:number; ox:number; oy:number}>({on:false,sx:0,sy:0,ox:0,oy:0});
  const dragging = useRef<{id:string|null; ox:number; oy:number}>({id:null, ox:0, oy:0});

  // physics loop
  const animRef = useRef<number|undefined>(undefined);
  const simRef = useRef({ alpha: 0.95, running: false });

  // adjacency for quick ops
  const adj = useMemo(()=>{
    const m = new Map<string, Set<string>>();
    for (const e of edges) {
      if (!m.has(e.source)) m.set(e.source, new Set());
      if (!m.has(e.target)) m.set(e.target, new Set());
      m.get(e.source)!.add(e.target);
      m.get(e.target)!.add(e.source);
    }
    return m;
  }, [edges]);

  // derived: filtered edges by minW & existing nodes
  const nodeMap = useMemo(()=> new Map(nodes.map(n=>[n.id,n])), [nodes]);
  const visEdges = useMemo(()=> edges.filter(e => (e.weight ?? 0) >= minW && nodeMap.has(e.source) && nodeMap.has(e.target)), [edges, minW, nodeMap]);

  // search subset
  const q = query.trim().toLowerCase();
  const matches = useMemo(()=> new Set(nodes.filter(n =>
    (n.id.toLowerCase().includes(q) || (n.label||'').toLowerCase().includes(q) || (n.group||'').toLowerCase().includes(q))
  ).map(n=>n.id)), [nodes, q]);

  // KPIs auto
  useEffect(()=>{
    const N = nodes.length, E = visEdges.length;
    const avgDeg = N ? (E*2)/N : 0;
    const density = N>1 ? (E)/(N*(N-1)/2) : 0;
    setKpis((prev)=>{
      const base = [
        {label:'Nodes', value:N},
        {label:'Edges', value:E},
        {label:'Avg Deg', value:avgDeg.toFixed(2)},
        {label:'Density', value: density.toFixed(4)},
      ];
      // keep any custom KPIs appended by backend (those whose label not in base)
      const custom = prev.filter(k => !['Nodes','Edges','Avg Deg','Density'].includes(k.label));
      return [...base, ...custom];
    });
  }, [nodes, visEdges.length]);

  // physics init if nodes change
  useEffect(()=>{
    // seed positions if missing
    setNodes(prev=>{
      const hasPos = (n:GNode)=> Number.isFinite(n.x) && Number.isFinite(n.y);
      if (prev.length) return prev;
      return nodes.map(n => hasPos(n) ? n : ({...n, x: rand(-200,200), y: rand(-150,150), vx:0, vy:0}));
    });
    // ensure any new node has a spot
    setNodes(ns => ns.map(n => ({
      ...n,
      x: Number.isFinite(n.x) ? n.x : rand(-200,200),
      y: Number.isFinite(n.y) ? n.y : rand(-150,150),
      vx: n.vx ?? 0, vy: n.vy ?? 0
    })));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeMap.size]);

  // simulation tick
  useEffect(()=>{
    if (layout !== 'force') { stopSim(); return; }
    startSim();
    return stopSim;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, nodes.length, visEdges.length]);

  function startSim(){
    if (simRef.current.running) return;
    simRef.current.running = true;
    simRef.current.alpha = 0.95;
    const tick = ()=> {
      if (!simRef.current.running) return;
      const alpha = simRef.current.alpha = Math.max(0.02, simRef.current.alpha * 0.98);
      const kRepel = 500;    // repulsive strength
      const kSpring = 0.02;  // attractive along edges
      const restLen = 40;    // ideal edge length
      const kCenter = 0.002; // centering
      const fric = 0.85;

      // work on a mutable copy
      setNodes(ns => {
        const arr = ns.map(n => ({...n}));
        // repulsion O(N^2) — fine for few thousand
        for (let i=0; i<arr.length; i++){
          const a = arr[i]; if (a.pinned) continue;
          for (let j=i+1; j<arr.length; j++){
            const b = arr[j];
            const dx = (a.x!-b.x!), dy = (a.y!-b.y!);
            let d2 = dx*dx + dy*dy + 0.01;
            const f = (kRepel * alpha) / d2;
            const fx = f*dx, fy=f*dy;
            if (!a.pinned){ a.vx = (a.vx ?? 0) + fx; a.vy = (a.vy ?? 0) + fy; }
            if (!b.pinned){ b.vx = (b.vx ?? 0) - fx; b.vy = (b.vy ?? 0) - fy; }
          }
        }
        // springs
        for (const e of visEdges){
          const a = arr.find(n=>n.id===e.source); const b = arr.find(n=>n.id===e.target);
          if (!a || !b) continue;
          const dx = (b.x!-a.x!), dy=(b.y!-a.y!);
          const d = Math.sqrt(dx*dx + dy*dy) || 0.001;
          const w = (e.weight ?? 1);
          const f = kSpring * alpha * w * (d - restLen);
          const fx = f * (dx/d), fy = f * (dy/d);
          if (!a.pinned){ a.vx = (a.vx ?? 0) + fx; a.vy = (a.vy ?? 0) + fy; }
          if (!b.pinned){ b.vx = (b.vx ?? 0) - fx; b.vy = (b.vy ?? 0) - fy; }
        }
        // center + integrate
        for (const n of arr){
          if (!n.pinned){
            n.vx = (n.vx ?? 0) + (-n.x!)*kCenter;
            n.vy = (n.vy ?? 0) + (-n.y!)*kCenter;
            n.x! += (n.vx ?? 0); n.y! += (n.vy ?? 0);
            n.vx! *= fric; n.vy! *= fric;
          }
        }
        return arr;
      });
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
  }

  function stopSim(){
    simRef.current.running = false;
    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = undefined;
  }

  /* ========== Streaming / Fetching ========== */

  async function run(){
    setLoading(true); setError(null); setLog(''); setNotes('');
    setNodes([]); setEdges([]); setPath([]); setSelA(undefined); setSelB(undefined);

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;
    const payload = { id: uid() };

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type':'application/json', Accept:'text/event-stream, application/json, text/plain' },
        body: JSON.stringify(payload),
        signal,
      });
      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && ctype.includes('text/event-stream')) {
        const reader = res.body.getReader();
        const dec = new TextDecoder('utf-8');
        let acc = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = dec.decode(value, { stream: true });
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
    } catch (e:any) {
      if (e?.name !== 'AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }
  function stop(){ abortRef.current?.abort(); abortRef.current=null; setLoading(false); }

  function applyPatch(p: Patch){
    if (Array.isArray(p.kpis)) setKpis(prev => {
      // merge by label
      const map = new Map(prev.map(k=>[k.label,k.value]));
      for (const k of p.kpis!) map.set(k.label, k.value);
      return Array.from(map.entries()).map(([label, value])=>({label, value}));
    });
    if (Array.isArray(p.nodes)) {
      setNodes(prev => mergeNodes(prev, p.nodes!));
    }
    if (p.node) {
      setNodes(prev => mergeNodes(prev, [p.node!]));
    }
    if (Array.isArray(p.edges)) {
      setEdges(prev => mergeEdges(prev, p.edges!));
    }
    if (p.edge) {
      setEdges(prev => mergeEdges(prev, [p.edge!]));
    }
    if (typeof p.notes === 'string') setNotes(n => (n ? n + '\n' : '') + p.notes);
  }
  function mergeNodes(prev:GNode[], incoming:GNode[]){
    const map = new Map<string,GNode>(prev.map(n=>[n.id,n]));
    for (const n of incoming){
      const o = map.get(n.id);
      map.set(n.id, { ...(o||{}), ...n, x: o?.x ?? n.x ?? rand(-200,200), y: o?.y ?? n.y ?? rand(-150,150), vx: o?.vx ?? 0, vy: o?.vy ?? 0 });
    }
    return Array.from(map.values());
  }
  function mergeEdges(prev:GEdge[], incoming:GEdge[]){
    const hash = (e:GEdge)=> `${e.source}::${e.target}`;
    const map = new Map<string,GEdge>(prev.map(e=>[hash(e), e]));
    for (const e of incoming){ map.set(hash(e), { ...(map.get(hash(e))||{}), ...e }); }
    return Array.from(map.values());
  }

  /* ========== Export ========== */

  function exportJSON(){
    const out = JSON.stringify({ nodes, edges }, null, 2);
    download('graph.json', out, 'application/json');
  }
  function exportPNG(){
    const svg = svgRef.current; if (!svg) return;
    const s = new XMLSerializer().serializeToString(svg);
    const blob = new Blob([s], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = ()=> {
      const canvas = document.createElement('canvas');
      const w = svg.clientWidth || 1000, h = svg.clientHeight || 600;
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,w,h);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((png)=> {
        if (!png) return;
        const purl = URL.createObjectURL(png);
        const a = document.createElement('a'); a.href = purl; a.download='graph.png'; a.click();
        URL.revokeObjectURL(purl); URL.revokeObjectURL(url);
      });
    };
    img.src = url;
  }
  function download(filename:string, text:string, mime='text/plain'){
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download=filename; a.click();
    URL.revokeObjectURL(url);
  }

  /* ========== Shortest path (BFS) ========== */
  useEffect(()=>{
    if (!highlightPath || !selA || !selB) { setPath([]); return; }
    const queue = [selA];
    const prev = new Map<string,string|null>(); prev.set(selA, null);
    while (queue.length){
      const v = queue.shift()!;
      if (v === selB) break;
      for (const w of (adj.get(v) || [])){
        if (!prev.has(w)){ prev.set(w, v); queue.push(w); }
      }
    }
    if (!prev.has(selB)) { setPath([]); return; }
    const stack:string[] = []; let cur:string|null = selB;
    while (cur){ stack.push(cur); cur = prev.get(cur) ?? null; }
    setPath(stack.reverse());
  }, [highlightPath, selA, selB, adj]);

  /* ========== UI Handlers ========== */

  // wheel zoom with focus point compensation
  function onWheel(e: React.WheelEvent<SVGSVGElement>) {
    e.preventDefault();
    const factor = Math.pow(1.001, -e.deltaY);
    const rect = (e.target as SVGElement).getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const newZoom = clamp(zoom * factor, 0.2, 4);
    const k = newZoom / zoom;
    setPan(p => ({ x: mx - k*(mx - p.x), y: my - k*(my - p.y) }));
    setZoom(newZoom);
  }

  function toGraphCoords(clientX:number, clientY:number){
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const x = (clientX - rect.left - pan.x) / zoom;
    const y = (clientY - rect.top - pan.y) / zoom;
    return { x, y };
  }

  function onBgMouseDown(e: React.MouseEvent){
    // start panning if not dragging a node
    if ((e.target as HTMLElement).dataset?.role === 'node') return;
    panning.current = { on:true, sx:e.clientX, sy:e.clientY, ox:pan.x, oy:pan.y };
  }
  function onBgMouseMove(e: React.MouseEvent){
    if (!panning.current.on) return;
    const dx = e.clientX - panning.current.sx;
    const dy = e.clientY - panning.current.sy;
    setPan({ x: panning.current.ox + dx, y: panning.current.oy + dy });
  }
  function onBgMouseUp(){ panning.current.on = false; }

  function onNodeMouseDown(e: React.MouseEvent, id:string){
    e.stopPropagation();
    dragging.current.id = id;
    const n = nodeMap.get(id)!;
    const { x, y } = toGraphCoords(e.clientX, e.clientY);
    dragging.current.ox = x - (n.x || 0);
    dragging.current.oy = y - (n.y || 0);
  }
  function onNodeMouseMove(e: React.MouseEvent){
    const id = dragging.current.id; if (!id) return;
    const n = nodeMap.get(id); if (!n) return;
    const { x, y } = toGraphCoords(e.clientX, e.clientY);
    n.x = x - dragging.current.ox; n.y = y - dragging.current.oy;
    n.vx = 0; n.vy = 0; n.pinned = true;
    setNodes(ns => ns.map(m => m.id===id ? {...n} : m));
  }
  function onNodeMouseUp(){ dragging.current.id = null; }

  function colorFor(n:GNode, idx:number){
    if (n.color) return n.color;
    if (n.group) {
      // stable hash color by group
      let h = 0; for (let i=0;i<n.group.length;i++) h = (h*31 + n.group.charCodeAt(i))|0;
      return themeColors[Math.abs(h)%themeColors.length];
    }
    return themeColors[idx % themeColors.length];
  }

  const selectedSet = useMemo(()=> new Set([selA, selB].filter(Boolean) as string[]), [selA, selB]);
  const pathSet = useMemo(()=> new Set(path), [path]);

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={exportJSON} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export JSON</button>
          <button onClick={exportPNG} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export PNG</button>
          {loading
            ? <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
            : <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>
          }
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 grid grid-cols-1 md:grid-cols-8 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <Field label="Search">
          <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="id / label / group"
                 className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
        <Field label="Min weight">
          <input type="number" value={minW} onChange={e=>setMinW(Number(e.target.value)||0)}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Field label="Layout">
          <select value={layout} onChange={e=>setLayout(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="force">Force</option><option value="static">Static</option>
          </select>
        </Field>
        <Field label="Labels">
          <label className="inline-flex items-center gap-2 text-sm mt-2">
            <input type="checkbox" checked={showLabels} onChange={e=>setShowLabels(e.target.checked)} className="accent-indigo-600"/>
            Show labels
          </label>
        </Field>
        <Field label="Path highlight">
          <label className="inline-flex items-center gap-2 text-sm mt-2">
            <input type="checkbox" checked={highlightPath} onChange={e=>setHighlightPath(e.target.checked)} className="accent-indigo-600"/>
            Shortest path A↔B
          </label>
        </Field>
        <Field label="Node A">
          <select value={selA || ''} onChange={e=>setSelA(e.target.value || undefined)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="">(none)</option>
            {nodes.map(n => <option key={n.id} value={n.id}>{n.id}</option>)}
          </select>
        </Field>
        <Field label="Node B">
          <select value={selB || ''} onChange={e=>setSelB(e.target.value || undefined)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="">(none)</option>
            {nodes.map(n => <option key={n.id} value={n.id}>{n.id}</option>)}
          </select>
        </Field>
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* Left: KPIs + Notes */}
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
            ) : <Empty label="No KPIs"/>}
          </Card>
          <Card title="Notes">
            <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || 'No notes.'}</pre>
          </Card>
        </div>

        {/* Graph */}
        <div className="lg:col-span-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Graph (drag to pan • wheel to zoom • drag nodes to pin)</div>
          <div
            onMouseMove={(e)=>{ onBgMouseMove(e); onNodeMouseMove(e); }}
            onMouseUp={()=>{ onBgMouseUp(); onNodeMouseUp(); }}
            onMouseLeave={()=>{ onBgMouseUp(); onNodeMouseUp(); }}
            className="h-[60vh]"
          >
            <svg
              ref={svgRef}
              width="100%" height="100%"
              onWheel={onWheel}
              onMouseDown={onBgMouseDown}
            >
              <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
                {/* Edges */}
                {visEdges.map((e, i)=>{
                  const a = nodeMap.get(e.source), b = nodeMap.get(e.target);
                  if (!a || !b) return null;
                  const onPath = pathSet.has(e.source) && pathSet.has(e.target); // crude but fine
                  const w = 0.5 + (e.weight ?? 1)*0.5;
                  return (
                    <line key={i}
                      x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke={onPath ? '#f59e0b' : '#9ca3af'}
                      strokeWidth={onPath ? (w+1.5) : w}
                      opacity={matches.size && !(matches.has(e.source)||matches.has(e.target)) ? 0.25 : 0.9}
                    />
                  );
                })}
                {/* Nodes */}
                {nodes.map((n, i)=>{
                  const r = 4 + (n.size ?? 1)*2;
                  const sel = selectedSet.has(n.id);
                  const onPath = pathSet.has(n.id);
                  const fill = sel ? '#4f46e5' : onPath ? '#f59e0b' : colorFor(n, i);
                  const stroke = n.pinned ? '#111827' : '#ffffff';
                  const alpha = (matches.size && !matches.has(n.id)) ? 0.3 : 1.0;
                  return (
                    <g key={n.id} transform={`translate(${n.x},${n.y})`} opacity={alpha}>
  <circle
    r={r}
    fill={fill}
    stroke={stroke}
    strokeWidth={1.5}
    data-role="node"
    onMouseDown={(e) => onNodeMouseDown(e, n.id)}
    onDoubleClick={() =>
      setNodes((ns) =>
        ns.map((x) => (x.id === n.id ? { ...x, pinned: !x.pinned } : x))
      )
    }
  >
    <title>
      {`${n.id}${n.label ? ' ' + n.label : ''}${n.group ? ' ' + n.group : ''}`}
    </title>
  </circle>

  {showLabels && (
    <text x={r + 3} y={3} fontSize="10" fill="#374151">
      {n.label || n.id}
    </text>
  )}
</g>
                  );
                })}
              </g>
            </svg>
          </div>
        </div>

        {/* Right: Inspect */}
        <div className="space-y-3">
          <Card title="Inspect">
            <div className="text-xs text-neutral-500 mb-2">Pick Node A & B for path; double-click a node to toggle pin.</div>
            <div className="space-y-2">
              <Button onClick={()=>{ setZoom(1); setPan({x:0,y:0}); }}>Reset View</Button>
              <Button onClick={()=>{ setNodes(ns=>ns.map(n=>({...n, pinned:false}))); }}>Unpin All</Button>
            </div>
            <div className="mt-3 text-[11px] text-neutral-500">
              Matches: {matches.size ? `${matches.size} nodes` : '—'}
              {highlightPath && selA && selB ? <div>Path length: {path.length? path.length-1 : 'unreachable'}</div> : null}
            </div>
          </Card>
          <Card title="Run Log">
            <div className="max-h-[24vh] overflow-auto">
              {error ? <div className="text-xs text-rose-600">{error}</div> : <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">{log || 'No streaming output yet.'}</pre>}
            </div>
          </Card>
        </div>
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
function Empty({ label }: { label: string }) { return <div className="text-xs text-neutral-500">{label}</div>; }
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="block text-[11px] text-neutral-500 mb-1">{label}</label>{children}</div>;
}
function Button({ onClick, children }:{ onClick:()=>void; children:React.ReactNode }){
  return <button onClick={onClick} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">{children}</button>;
}