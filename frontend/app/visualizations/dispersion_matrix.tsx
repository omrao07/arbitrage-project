'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Backend contract (example):
 * POST /api/dispersion
 * body: { universe: string[], window: number, sector?: string|'ALL', stat: 'corr'|'distance' }
 * resp: { series: {symbol:string, sector?:string, bars:{t:number,c:number}[]}[], sectors?: string[], notes?: string }
 */

type Bar = { t: number; c: number };
type Series = { symbol: string; sector?: string; bars: Bar[] };

type Payload = {
  universe?: string[];
  window?: number;
  sector?: string | 'ALL';
  stat?: 'corr' | 'distance';
};

type Patch = {
  series?: Series[];
  sectors?: string[];
  notes?: string;
};

type Props = {
  endpoint?: string;
  title?: string;
  className?: string;
};

/* ------------ tiny utils ------------ */
const fmt = (x?: number, d = 2) => (x != null && Number.isFinite(x) ? x.toFixed(d) : '');
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const std = (xs: number[]) => {
  const m = mean(xs);
  const v = xs.length ? xs.reduce((s, x) => s + (x - m) * (x - m), 0) / xs.length : 0;
  return Math.sqrt(v);
};
const corr = (x: number[], y: number[]) => {
  const mx = mean(x), my = mean(y);
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < x.length; i++) {
    const a = x[i] - mx, b = y[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  const den = Math.sqrt(dx * dy) || 1;
  return num / den;
};
const z = (x: number[], eps = 1e-12) => {
  const m = mean(x), s = std(x) || eps;
  return x.map(v => (v - m) / s);
};
const pctReturns = (xs: number[]) => {
  const out: number[] = [];
  for (let i = 1; i < xs.length; i++) out.push((xs[i] - xs[i - 1]) / (xs[i - 1] || 1));
  return out;
};

function aligned(series: Series[]): { t: number; cols: number[] }[] {
  if (!series.length) return [];
  const maps = series.map(s => new Map(s.bars.map(b => [b.t, b.c])));
  const times = Array.from(maps[0].keys()).filter(t => maps.every(m => m.has(t))).sort((a, b) => a - b);
  return times.map(t => ({ t, cols: maps.map(m => m.get(t) as number) }));
}

function tsDispersion(matrix: number[][]) {
  const n = matrix.length ? matrix[0].length : 0;
  const out: number[] = [];
  for (let t = 0; t < n; t++) {
    const slice = matrix.map(row => row[t]).filter(Number.isFinite);
    out.push(std(slice));
  }
  return out;
}

function toCSV(series: Series[], dispTS: { t: number[]; val: number[] }) {
  const rows = aligned(series);
  const head = ['time', ...series.map(s => s.symbol)].join(',');
  const pxLines = rows.map(r => [new Date(r.t).toISOString(), ...r.cols.map(v => String(v))].join(','));
  const dHead = 'time,dispersion';
  const dLines = dispTS.t.map((t, i) => [new Date(t).toISOString(), String(dispTS.val[i])].join(','));
  return `# prices\n${head}\n${pxLines.join('\n')}\n\n# dispersion\n${dHead}\n${dLines.join('\n')}\n`;
}

function download(name: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}

/* ------------ component ------------ */
export default function DispersionMatrix({
  endpoint = '/api/dispersion',
  title = 'Dispersion Matrix',
  className = '',
}: Props) {
  const [universe, setUniverse] = useState('AAPL, MSFT, AMZN, GOOGL, META, NVDA');
  const [windowDays, setWindowDays] = useState(120);
  const [stat, setStat] = useState<'corr' | 'distance'>('corr');
  const [sector, setSector] = useState<'ALL' | string>('ALL');

  const [series, setSeries] = useState<Series[]>([]);
  const [sectors, setSectors] = useState<string[]>([]);
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // debounce (browser safe)
  const ref = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (ref.current) clearTimeout(ref.current);
    ref.current = setTimeout(() => run(), 250);
    return () => { if (ref.current) clearTimeout(ref.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [universe, windowDays, stat, sector]);

  async function run() {
    setLoading(true); setError(null);
    try {
      const payload: Payload = {
        universe: parseSymbols(universe),
        window: windowDays,
        sector,
        stat,
      };
      const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const p: Patch = await res.json();
      if (p.series) setSeries(p.series);
      if (p.sectors) setSectors(p.sectors);
      if (p.notes) setNotes(p.notes);
    } catch (e: any) {
      setError(e.message);
    } finally { setLoading(false); }
  }

  /* ----- derived ----- */
  const rows = useMemo(() => aligned(series), [series]);

  const retMatrix = useMemo(() => {
    if (!rows.length) return [] as number[][];
    const k = series.length;
    const mat: number[][] = Array.from({ length: k }, () => []);
    for (let i = 0; i < k; i++) {
      const px = rows.map(r => r.cols[i]);
      mat[i] = pctReturns(px);
    }
    const n = Math.min(...mat.map(r => r.length));
    return mat.map(r => r.slice(-n));
  }, [rows, series.length]);

  const dispTS = useMemo(() => {
    if (!retMatrix.length) return { t: [] as number[], val: [] as number[] };
    const t = rows.slice(1).map(r => r.t).slice(-(retMatrix[0]?.length || 0));
    const val = tsDispersion(retMatrix);
    return { t, val };
  }, [retMatrix, rows]);

  const matrix = useMemo(() => {
    const m = retMatrix.length;
    if (!m) return [] as number[][];
    const zed = retMatrix.map(r => z(r));
    const out: number[][] = Array.from({ length: m }, () => Array(m).fill(1));
    for (let i = 0; i < m; i++) {
      for (let j = i; j < m; j++) {
        const v = stat === 'corr' ? corr(zed[i], zed[j]) : 1 - corr(zed[i], zed[j]);
        out[i][j] = out[j][i] = v;
      }
    }
    return out;
  }, [retMatrix, stat]);

  const ranks = useMemo(() => {
    if (!retMatrix.length) return [] as { sym: string; score: number }[];
    const zed = retMatrix.map(r => z(r));
    const scores = zed.map(row => mean(row.map(v => Math.abs(v))));
    return series.map((s, i) => ({ sym: s.symbol, score: scores[i] || 0 }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 12);
  }, [retMatrix, series]);

  /* ----- UI ----- */
  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={() => download('dispersion.csv', toCSV(series, dispTS), 'text/csv')}
            className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700">CSV</button>
          <button onClick={() => run()}
            className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white">
            {loading ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>

      {/* controls */}
      <div className="p-3 grid grid-cols-1 xl:grid-cols-12 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <Field label="Universe (CSV)">
          <input value={universe} onChange={e => setUniverse(e.target.value)} className="w-full rounded border px-3 py-2 text-sm" />
        </Field>
        <Field label="Lookback (days)">
          <input type="number" value={windowDays} onChange={e => setWindowDays(Math.max(20, Number(e.target.value) || windowDays))}
            className="w-full rounded border px-2 py-2 text-sm" />
        </Field>
        <Field label="Stat">
          <select value={stat} onChange={e => setStat(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="corr">Correlation (−1..+1)</option>
            <option value="distance">Distance (1 − corr)</option>
          </select>
        </Field>
        <Field label="Sector">
          <select value={sector} onChange={e => setSector(e.target.value)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="ALL">ALL</option>
            {sectors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 2xl:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        <Card title={`Pairwise ${stat === 'corr' ? 'Correlation' : 'Distance'}`}>
          {matrix.length ? (
            <Heatmap labels={series.map(s => s.symbol)} mat={matrix} stat={stat} />
          ) : <Empty label="Run to compute matrix" />}
        </Card>

        <Card title="Cross-sectional Dispersion (std of returns across names)">
          {dispTS.t.length ? <Line t={dispTS.t} y={dispTS.val} color="#4f46e5" /> : <Empty label="Run to compute timeseries" />}
          {dispTS.t.length ? (
            <div className="grid grid-cols-3 gap-2 mt-2 text-xs">
              <Badge>Last: {fmt(dispTS.val.at(-1), 4)}</Badge>
              <Badge>Avg: {fmt(mean(dispTS.val), 4)}</Badge>
              <Badge>Vol: {fmt(std(dispTS.val), 4)}</Badge>
            </div>
          ) : null}
          {error && <div className="text-xs text-rose-600 mt-2">{error}</div>}
          {notes && <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300 mt-2">{notes}</pre>}
        </Card>

        <Card title="Top Contributors (avg |z-ret|)">
          {ranks.length ? (
            <ul className="text-xs space-y-1">
              {ranks.map((r, i) => (
                <li key={r.sym} className="flex justify-between">
                  <span>{i + 1}. {r.sym}</span>
                  <span className="font-mono">{fmt(r.score, 3)}</span>
                </li>
              ))}
            </ul>
          ) : <Empty label="Run to see ranks" />}
        </Card>
      </div>
    </div>
  );
}

/* ------------ subcomponents ------------ */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="block text-[11px] text-neutral-500 mb-1">{label}</label>{children}</div>;
}
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}
function Empty({ label }: { label: string }) { return <div className="text-xs text-neutral-500">{label}</div>; }
function Badge({ children }: { children: React.ReactNode }) {
  return <span className="inline-block bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-200 rounded px-1.5 py-0.5 text-[10px]">{children}</span>;
}

function Line({ t, y, color = '#4f46e5' }: { t: number[]; y: number[]; color?: string }) {
  const width = 480, height = 160;
  const min = Math.min(...y), max = Math.max(...y), span = (max - min) || 1;
  const step = width / Math.max(1, y.length - 1);
  const d = y.map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(2)},${(height - ((v - min) / span) * height).toFixed(2)}`).join(' ');
  return <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}><path d={d} fill="none" stroke={color} strokeWidth={1.5} /></svg>;
}

function Heatmap({ labels, mat, stat }: { labels: string[]; mat: number[][]; stat: 'corr' | 'distance' }) {
  const n = labels.length;
  const size = Math.max(18, Math.min(36, Math.floor(320 / Math.max(3, n))));
  const pad = 80;
  const w = pad + n * size + 12;
  const h = pad + n * size + 12;

  function color(v: number) {
    if (stat === 'corr') { // -1..+1 → red→white→blue
      const x = clamp((v + 1) / 2, 0, 1);
      const r = Math.round(220 * (1 - x) + 40 * x);
      const g = Math.round(80 + 140 * x);
      const b = Math.round(60 + 180 * x);
      return `rgb(${r},${g},${b})`;
    } else { // 0..2 → white→purple
      const x = clamp(v / 2, 0, 1);
      const r = Math.round(250 * (1 - x) + 110 * x);
      const g = Math.round(250 * (1 - x) + 70 * x);
      const b = Math.round(250 * (1 - x) + 200 * x);
      return `rgb(${r},${g},${b})`;
    }
  }

  return (
    <svg width="100%" height={pad + n * size + 24} viewBox={`0 0 ${w} ${h}`}>
      {labels.map((s, i) => (
        <text key={'x' + i} x={pad + i * size + size / 2} y={50}
          transform={`rotate(-45 ${pad + i * size + size / 2} 50)`}
          textAnchor="end" fontSize="10" fill="#374151">{s}</text>
      ))}
      {labels.map((s, i) => (
        <text key={'y' + i} x={10} y={pad + i * size + size * 0.7}
          fontSize="10" fill="#374151">{s}</text>
      ))}
      {mat.map((row, i) =>
        row.map((v, j) => (
          <rect key={`${i}-${j}`} x={pad + j * size} y={pad + i * size}
            width={size - 1} height={size - 1} fill={color(v)} />
        ))
      )}
    </svg>
  );
}

/* ------------ helpers ------------ */
function parseSymbols(input: string) {
  return input.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
}