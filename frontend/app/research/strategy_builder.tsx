'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ===================== Types ===================== */

type KPI = { label: string; value: string | number };
type SeriesPoint = { t: number; v: number };
type Row = { [key: string]: string | number | null };

type BlockType =
  | 'signal'
  | 'condition'
  | 'risk'
  | 'sizer'
  | 'exit'
  | 'schedule'
  | 'universe'
  | 'costs'
  | 'slippage';

type ParamKind = 'number' | 'string' | 'boolean' | 'select';

type ParamMeta = {
  key: string;
  label: string;
  kind: ParamKind;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  options?: { label: string; value: string }[];
  help?: string;
  default?: number | string | boolean;
};

type BlockTemplate = {
  id: string;
  type: BlockType;
  title: string;
  description?: string;
  params: ParamMeta[];
};

type BlockInstance = {
  uid: string;            // runtime id
  templateId: string;     // refers to BlockTemplate.id
  type: BlockType;
  title: string;
  params: Record<string, number | string | boolean>;
};

type StrategySpec = {
  name: string;
  description?: string;
  timeframe: '1m' | '5m' | '15m' | '1h' | '1d';
  universe: string[]; // tickers/symbols
  blocks: BlockInstance[];
  meta?: Record<string, any>;
};

type BacktestResult = {
  kpis?: KPI[];
  series?: SeriesPoint[];  // equity curve or PnL
  trades?: Row[];          // trade log table
  columns?: string[];      // optional explicit columns for trades
  notes?: string;
};

type Props = {
  compileEndpoint?: string;  // e.g. '/api/strategy/compile'
  backtestEndpoint?: string; // e.g. '/api/strategy/backtest'
  title?: string;
  className?: string;
  storageKey?: string;
};

/* ===================== Palettes ===================== */

const PALETTE: BlockTemplate[] = [
  {
    id: 'ema-cross',
    type: 'signal',
    title: 'EMA Crossover',
    description: 'Go long when fast EMA crosses above slow EMA; short on cross down.',
    params: [
      { key: 'fast', label: 'Fast EMA', kind: 'number', min: 2, max: 200, step: 1, default: 12 },
      { key: 'slow', label: 'Slow EMA', kind: 'number', min: 5, max: 400, step: 1, default: 26 },
      { key: 'confirm', label: 'Bars Confirm', kind: 'number', min: 0, max: 10, step: 1, default: 1 },
    ],
  },
  {
    id: 'rsi-meanrevert',
    type: 'condition',
    title: 'RSI Mean Revert',
    description: 'Buy when RSI < lower; sell when RSI > upper.',
    params: [
      { key: 'window', label: 'RSI Window', kind: 'number', min: 2, max: 100, step: 1, default: 14 },
      { key: 'lower', label: 'Lower', kind: 'number', min: 0, max: 50, step: 1, default: 30 },
      { key: 'upper', label: 'Upper', kind: 'number', min: 50, max: 100, step: 1, default: 70 },
    ],
  },
  {
    id: 'boll-revert',
    type: 'signal',
    title: 'Bollinger Revert',
    description: 'Fade 2σ moves back to the middle band.',
    params: [
      { key: 'window', label: 'Window', kind: 'number', min: 5, max: 200, step: 1, default: 20 },
      { key: 'sigma', label: 'Sigma', kind: 'number', min: 1, max: 4, step: 0.1, default: 2 },
    ],
  },
  {
    id: 'news-sentiment',
    type: 'signal',
    title: 'News Sentiment',
    description: 'Trade with positive/negative news sentiment.',
    params: [
      { key: 'threshold', label: 'Score Threshold', kind: 'number', min: -1, max: 1, step: 0.05, default: 0.2 },
      { key: 'lookback_m', label: 'Lookback (min)', kind: 'number', min: 5, max: 720, step: 5, default: 60 },
    ],
  },
  {
    id: 'risk-basic',
    type: 'risk',
    title: 'Risk Limits',
    description: 'Stop loss / take profit / max position.',
    params: [
      { key: 'stop_pct', label: 'Stop %', kind: 'number', min: 0, max: 0.5, step: 0.005, default: 0.02, help: '0.02 = 2%' },
      { key: 'take_pct', label: 'Take %', kind: 'number', min: 0, max: 1.0, step: 0.01, default: 0.05, help: '0.05 = 5%' },
      { key: 'max_pos', label: 'Max Position (qty)', kind: 'number', min: 0, max: 1e6, step: 1, default: 100 },
    ],
  },
  {
    id: 'sizer-fixed',
    type: 'sizer',
    title: 'Fixed Size',
    description: 'Trade a fixed quantity per signal.',
    params: [{ key: 'qty', label: 'Quantity', kind: 'number', min: 0.0001, max: 1e9, step: 0.0001, default: 1 }],
  },
  {
    id: 'exit-time',
    type: 'exit',
    title: 'Time-based Exit',
    description: 'Exit after N bars.',
    params: [{ key: 'bars', label: 'Bars', kind: 'number', min: 1, max: 2000, step: 1, default: 50 }],
  },
  {
    id: 'schedule-rth',
    type: 'schedule',
    title: 'Schedule (RTH)',
    description: 'Only trade during regular trading hours.',
    params: [
      { key: 'start', label: 'Start HH:MM', kind: 'string', default: '09:30' },
      { key: 'end', label: 'End HH:MM', kind: 'string', default: '16:00' },
      { key: 'timezone', label: 'Timezone', kind: 'select', default: 'America/New_York', options: [
        { label: 'New York', value: 'America/New_York' },
        { label: 'UTC', value: 'UTC' },
        { label: 'Mumbai', value: 'Asia/Kolkata' },
      ] },
    ],
  },
  {
    id: 'universe-list',
    type: 'universe',
    title: 'Universe',
    description: 'Manual list of symbols.',
    params: [{ key: 'tickers', label: 'Tickers CSV', kind: 'string', placeholder: 'AAPL, MSFT, NVDA', default: '' }],
  },
  {
    id: 'tca-simple',
    type: 'costs',
    title: 'Cost Model',
    description: 'Flat bps per trade.',
    params: [{ key: 'bps', label: 'Fees (bps)', kind: 'number', min: 0, max: 100, step: 0.5, default: 1.0 }],
  },
  {
    id: 'slip-linear',
    type: 'slippage',
    title: 'Slippage Model',
    description: 'Linear slippage vs ADV.',
    params: [
      { key: 'coeff', label: 'Coeff', kind: 'number', min: 0, max: 5, step: 0.01, default: 0.05, help: 'slip = coeff * (qty/ADV)' },
    ],
  },
];

/* ===================== Utils ===================== */
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const ts = () => Date.now();

function templateById(id: string): BlockTemplate | undefined {
  return PALETTE.find(p => p.id === id);
}

function instantiate(t: BlockTemplate): BlockInstance {
  const params: Record<string, any> = {};
  for (const p of t.params) {
    params[p.key] = p.default ?? (p.kind === 'boolean' ? false : p.kind === 'number' ? 0 : '');
  }
  return { uid: uid(), templateId: t.id, type: t.type, title: t.title, params };
}

function parseCSVTickers(s: string): string[] {
  return (s || '')
    .split(',')
    .map(x => x.trim().toUpperCase())
    .filter(Boolean);
}

function toSpec(state: {
  name: string;
  description: string;
  timeframe: StrategySpec['timeframe'];
  rootUniverse: string;
  blocks: BlockInstance[];
}): StrategySpec {
  // If a Universe block exists, use it; else fall back to rootUniverse input
  const uniBlock = state.blocks.find(b => b.type === 'universe' && typeof b.params.tickers === 'string');
  const universe = uniBlock ? parseCSVTickers(String(uniBlock.params.tickers || '')) : parseCSVTickers(state.rootUniverse);
  return {
    name: state.name || 'unnamed-strategy',
    description: state.description || '',
    timeframe: state.timeframe,
    universe,
    blocks: state.blocks,
    meta: { generated_ms: ts(), version: 1 },
  };
}

function download(filename: string, text: string, mime = 'application/json') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

/* ===================== Component ===================== */

export default function StrategyBuilder({
  compileEndpoint = '/api/strategy/compile',
  backtestEndpoint = '/api/strategy/backtest',
  title = 'Strategy Builder',
  className = '',
  storageKey = 'strategy-builder-v1',
}: Props) {
  // Header/meta
  const [name, setName] = useState('my-strategy');
  const [description, setDescription] = useState('');
  const [timeframe, setTimeframe] = useState<StrategySpec['timeframe']>('5m');
  const [rootUniverse, setRootUniverse] = useState('AAPL, MSFT, NVDA');

  // Blocks
  const [blocks, setBlocks] = useState<BlockInstance[]>([]);
  // Results
  const [compiled, setCompiled] = useState<string>(''); // JSON preview
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [log, setLog] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  // Persist
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const j = JSON.parse(raw);
        if (j.name) setName(j.name);
        if (j.description) setDescription(j.description);
        if (j.timeframe) setTimeframe(j.timeframe);
        if (j.rootUniverse) setRootUniverse(j.rootUniverse);
        if (Array.isArray(j.blocks)) setBlocks(j.blocks);
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify({ name, description, timeframe, rootUniverse, blocks }));
    } catch {}
  }, [name, description, timeframe, rootUniverse, blocks, storageKey]);

  // Autoscroll
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [log, loading]);

  /* ----- block ops ----- */
  function addBlock(templateId: string) {
    const t = templateById(templateId);
    if (!t) return;
    setBlocks(prev => [...prev, instantiate(t)]);
  }
  function removeBlock(uid: string) {
    setBlocks(prev => prev.filter(b => b.uid !== uid));
  }
  function move(uid: string, dir: -1 | 1) {
    setBlocks(prev => {
      const i = prev.findIndex(b => b.uid === uid);
      if (i < 0) return prev;
      const j = i + dir;
      if (j < 0 || j >= prev.length) return prev;
      const copy = [...prev];
      const [it] = copy.splice(i, 1);
      copy.splice(j, 0, it);
      return copy;
    });
  }
  function patchParam(uid: string, key: string, val: any) {
    setBlocks(prev => prev.map(b => (b.uid === uid ? { ...b, params: { ...b.params, [key]: val } } : b)));
  }

  /* ----- compile preview ----- */
  function compileSpec() {
    const spec = toSpec({ name, description, timeframe, rootUniverse, blocks });
    const json = JSON.stringify(spec, null, 2);
    setCompiled(json);
    setError(null);
  }

  /* ----- backtest ----- */
  async function backtest() {
    setResult(null);
    setError(null);
    setLog('');
    setCompiled(prev => prev || JSON.stringify(toSpec({ name, description, timeframe, rootUniverse, blocks }), null, 2));

    const spec = toSpec({ name, description, timeframe, rootUniverse, blocks });

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    setLoading(true);
    try {
      const res = await fetch(backtestEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream, application/json, text/plain' },
        body: JSON.stringify({ id: uid(), spec }),
        signal,
      });
      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
        let accResult: BacktestResult = {};
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          if (ctype.includes('text/event-stream')) {
            const lines = (acc + chunk).split('\n');
            acc = lines.pop() || '';
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;
              const data = line.slice(5).trim();
              if (!data) continue;
              if (data === '[DONE]') continue;
              setLog(s => s + data + '\n');
              try {
                const patch = JSON.parse(data);
                accResult = mergeResult(accResult, normalizeResult(patch));
                setResult({ ...accResult });
              } catch {
                setResult(prev => {
                  const base = prev || {};
                  const notes = ((base.notes || '') + (base.notes ? '\n' : '') + data);
                  return { ...base, notes };
                });
              }
            }
          } else {
            setLog(s => s + chunk);
          }
        }
      } else if (ctype.includes('application/json')) {
        const j = await res.json();
        setResult(normalizeResult(j));
      } else {
        const txt = await res.text();
        setResult({ notes: txt });
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }
  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  const canBacktest = useMemo(() => blocks.some(b => b.type === 'signal'), [blocks]);

  /* ===================== UI ===================== */

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-200 dark:border-neutral-800 px-3 py-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button
            onClick={compileSpec}
            className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900"
          >
            Compile JSON
          </button>
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
          ) : (
            <button
              onClick={backtest}
              disabled={!canBacktest}
              className={`text-xs px-3 py-1 rounded border ${canBacktest ? 'border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700' : 'border-neutral-300 bg-neutral-200 text-neutral-500 cursor-not-allowed'}`}
            >
              Backtest
            </button>
          )}
        </div>
      </div>

      {/* Meta */}
      <div className="p-3 border-b border-neutral-200 dark:border-neutral-800 grid grid-cols-1 md:grid-cols-6 gap-3">
        <TextField label="Name" value={name} setValue={setName} />
        <SelectField
          label="Timeframe"
          value={timeframe}
          setValue={(v) => setTimeframe(v as any)}
          options={[
            { label: '1m', value: '1m' }, { label: '5m', value: '5m' }, { label: '15m', value: '15m' },
            { label: '1h', value: '1h' }, { label: '1d', value: '1d' },
          ]}
        />
        <TextField className="md:col-span-4" label="Universe (CSV)" value={rootUniverse} setValue={setRootUniverse} placeholder="AAPL, MSFT, NVDA" />
        <TextArea label="Description" value={description} setValue={setDescription} rows={2} className="md:col-span-6" />
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* Palette */}
        <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Blocks</div>
          <div className="p-2 grid grid-cols-1 gap-2">
            {PALETTE.map((p) => (
              <div key={p.id} className="rounded border border-neutral-200 dark:border-neutral-800 p-2">
                <div className="text-sm font-semibold">{p.title}</div>
                {p.description && <div className="text-[11px] text-neutral-500">{p.description}</div>}
                <button
                  onClick={() => addBlock(p.id)}
                  className="mt-2 text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800"
                >
                  + Add
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Canvas / Blocks editor */}
        <div className="lg:col-span-2 min-h-0 grid grid-rows-[minmax(0,1fr)_200px] gap-3">
          <div className="min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-auto p-2 space-y-2">
            {blocks.length === 0 ? (
              <div className="p-3 text-xs text-neutral-500">Pick blocks from the left to build your strategy.</div>
            ) : (
              blocks.map((b, idx) => (
                <BlockCard
                  key={b.uid}
                  block={b}
                  onRemove={() => removeBlock(b.uid)}
                  onMoveUp={() => move(b.uid, -1)}
                  onMoveDown={() => move(b.uid, +1)}
                >
                  <ParamEditor block={b} onChange={(k, v) => patchParam(b.uid, k, v)} />
                </BlockCard>
              ))
            )}
          </div>

          {/* Right-bottom: compile preview + run log */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
              <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">
                Compiled JSON
                <button
                  onClick={() => download(`${name || 'strategy'}.json`, compiled || JSON.stringify(toSpec({ name, description, timeframe, rootUniverse, blocks }), null, 2))}
                  className="ml-2 text-[11px] underline decoration-dotted"
                >
                  download
                </button>
              </div>
              <pre className="p-3 text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-200">
                {compiled || '// Click "Compile JSON" to preview'}
              </pre>
            </div>

            <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
              <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Run Log</div>
              <div ref={logRef} className="flex-1 overflow-auto p-3">
                {error ? (
                  <div className="text-xs text-rose-600">{error}</div>
                ) : (
                  <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">
                    {log || 'No streaming output yet.'}
                  </pre>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="border-t border-neutral-200 dark:border-neutral-800 p-3 grid grid-cols-1 lg:grid-cols-3 gap-3">
        <Card title="KPIs">
          {result?.kpis?.length ? (
            <div className="grid grid-cols-2 gap-2">
              {result.kpis.map((k, i) => (
                <div key={i} className="rounded border border-neutral-200 dark:border-neutral-800 px-3 py-2">
                  <div className="text-[11px] text-neutral-500">{k.label}</div>
                  <div className="text-sm font-semibold">{String(k.value)}</div>
                </div>
              ))}
            </div>
          ) : (
            <Empty label="No KPIs yet" />
          )}
        </Card>

        <Card title="Equity Curve">
          {result?.series?.length ? <LineChart series={result.series} /> : <Empty label="No series yet" />}
        </Card>

        <Card title="Trades">
          {result?.trades?.length ? (
            <TradesTable rows={result.trades} columns={result.columns} />
          ) : (
            <Empty label="No trades yet" />
          )}
        </Card>
      </div>
    </div>
  );
}

/* ===================== Subcomponents ===================== */

function BlockCard({
  block,
  children,
  onRemove,
  onMoveUp,
  onMoveDown,
}: {
  block: BlockInstance;
  children: React.ReactNode;
  onRemove: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800">
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{block.title} <span className="text-[10px] text-neutral-500">({block.type})</span></div>
        <div className="flex items-center gap-2">
          <button onClick={onMoveUp} className="text-[11px] px-2 py-[2px] rounded border border-neutral-300 dark:border-neutral-700">↑</button>
          <button onClick={onMoveDown} className="text-[11px] px-2 py-[2px] rounded border border-neutral-300 dark:border-neutral-700">↓</button>
          <button onClick={onRemove} className="text-[11px] px-2 py-[2px] rounded border border-rose-300 text-rose-700">Remove</button>
        </div>
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function ParamEditor({ block, onChange }: { block: BlockInstance; onChange: (k: string, v: any) => void }) {
  const t = templateById(block.templateId);
  if (!t) return <div className="text-xs text-rose-600">Unknown template: {block.templateId}</div>;
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
      {t.params.map((p) => {
        const val = block.params[p.key];
        switch (p.kind) {
          case 'number':
            return (
              <NumberField
                key={p.key}
                label={p.label}
                value={typeof val === 'number' ? val : Number(p.default ?? 0)}
                setValue={(v) => onChange(p.key, v)}
                min={p.min} max={p.max} step={p.step}
                help={p.help}
              />
            );
          case 'string':
            return (
              <TextField
                key={p.key}
                label={p.label}
                value={typeof val === 'string' ? val : String(p.default ?? '')}
                setValue={(v) => onChange(p.key, v)}
                placeholder={p.placeholder}
              />
            );
          case 'boolean':
            return (
              <SwitchField
                key={p.key}
                label={p.label}
                checked={!!val}
                setChecked={(v) => onChange(p.key, v)}
              />
            );
          case 'select':
            return (
              <SelectField
                key={p.key}
                label={p.label}
                value={typeof val === 'string' ? val : String(p.default ?? '')}
                setValue={(v) => onChange(p.key, v)}
                options={p.options || []}
              />
            );
          default:
            return null;
        }
      })}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}
function Empty({ label }: { label: string }) {
  return <div className="text-xs text-neutral-500">{label}</div>;
}

/* ---- Inputs ---- */
function TextField({
  label, value, setValue, placeholder, className = '',
}: { label: string; value: string; setValue: (v: string) => void; placeholder?: string; className?: string }) {
  return (
    <div className={className}>
      <label className="block text-[11px] text-neutral-500 mb-1">{label}</label>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
      />
    </div>
  );
}
function TextArea({
  label, value, setValue, rows = 3, className = '',
}: { label: string; value: string; setValue: (v: string) => void; rows?: number; className?: string }) {
  return (
    <div className={className}>
      <label className="block text-[11px] text-neutral-500 mb-1">{label}</label>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        rows={rows}
        className="w-full resize-y rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
      />
    </div>
  );
}
function NumberField({
  label, value, setValue, min, max, step = 1, help,
}: { label: string; value: number; setValue: (v: number) => void; min?: number; max?: number; step?: number; help?: string }) {
  return (
    <div>
      <label className="block text-[11px] text-neutral-500 mb-1">{label}</label>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        min={min} max={max} step={step}
        onChange={(e) => setValue(parseFloat(e.target.value || '0'))}
        className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
      />
      {help ? <div className="text-[10px] text-neutral-500 mt-1">{help}</div> : null}
    </div>
  );
}
function SelectField({
  label, value, setValue, options, className = '',
}: { label: string; value: string; setValue: (v: string) => void; options: { label: string; value: string }[]; className?: string }) {
  return (
    <div className={className}>
      <label className="block text-[11px] text-neutral-500 mb-1">{label}</label>
      <select
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}
function SwitchField({
  label, checked, setChecked,
}: { label: string; checked: boolean; setChecked: (v: boolean) => void }) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-[11px] text-neutral-500">{label}</label>
      <button
        onClick={() => setChecked(!checked)}
        className={`h-5 w-9 rounded-full border ${checked ? 'bg-indigo-600 border-indigo-600' : 'bg-neutral-200 border-neutral-300'}`}
        aria-pressed={checked}
      >
        <span className={`block h-4 w-4 bg-white rounded-full mt-[2px] transition-transform ${checked ? 'translate-x-4' : 'translate-x-1'}`} />
      </button>
    </div>
  );
}

/* ---- Results ---- */
function TradesTable({ rows, columns }: { rows: Row[]; columns?: string[] }) {
  const cols = useMemo(() => {
    if (columns?.length) return columns;
    const keys = new Set<string>();
    rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)));
    return Array.from(keys);
  }, [rows, columns]);

  return (
    <div className="overflow-auto max-h-[40vh]">
      <table className="min-w-full text-sm">
        <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
          <tr>
            {cols.map(c => (
              <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="odd:bg-white even:bg-neutral-50 dark:odd:bg-neutral-900 dark:even:bg-neutral-950">
              {cols.map(c => (
                <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800 align-top">
                  {formatCell(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function formatCell(v: any): string {
  if (v == null) return '';
  if (typeof v === 'number') {
    if (Math.abs(v) < 1 && Math.abs(v) > 0) return v.toFixed(4);
    if (!Number.isInteger(v)) return v.toFixed(2);
  }
  return String(v);
}

/* ---- tiny SVG chart ---- */
function LineChart({ series, width = 320, height = 120 }: { series: SeriesPoint[]; width?: number; height?: number }) {
  if (!series.length) return null;
  const xs = series.map((_, i) => i);
  const ys = series.map(p => p.v);
  const xmin = 0, xmax = Math.max(xs[xs.length - 1] ?? 1, 1);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const spanY = ymax - ymin || 1;
  const stepX = width / (xmax - xmin || 1);
  const path = series.map((p, i) => {
    const x = i * stepX;
    const y = height - ((p.v - ymin) / spanY) * height;
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
  const tone = (ys[ys.length - 1] ?? 0) >= (ys[0] ?? 0) ? '#059669' : '#dc2626';
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="block">
      <path d={path} fill="none" stroke={tone} strokeWidth={1.5} />
    </svg>
  );
}

/* ---- Result normalization/merge ---- */
function mergeResult(base: BacktestResult, patch: BacktestResult): BacktestResult {
  const out: BacktestResult = { ...(base || {}) };
  if (patch.kpis) out.kpis = patch.kpis;
  if (patch.series) out.series = patch.series;
  if (patch.trades) out.trades = [...(out.trades || []), ...patch.trades];
  if (patch.columns) out.columns = patch.columns;
  if (patch.notes) out.notes = ((out.notes || '') + (out.notes ? '\n' : '') + patch.notes);
  return out;
}
function normalizeResult(j: any): BacktestResult {
  if (!j || typeof j !== 'object') return { notes: String(j ?? '') };
  const d = j.data && typeof j.data === 'object' ? j.data : j;
  const r: BacktestResult = {};
  if (Array.isArray(d.kpis)) r.kpis = d.kpis;
  if (Array.isArray(d.series)) r.series = d.series;
  if (Array.isArray(d.trades)) r.trades = d.trades;
  if (Array.isArray(d.columns)) r.columns = d.columns;
  if (typeof d.notes === 'string') r.notes = d.notes;
  return r;
}