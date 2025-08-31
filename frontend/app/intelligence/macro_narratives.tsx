'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ---------- types ---------- */

type SourceKey =
  | 'CPI'
  | 'PPI'
  | 'PMI'
  | 'Jobs'
  | 'FOMC'
  | 'YieldCurve'
  | 'Housing'
  | 'Earnings'
  | 'Energy'
  | 'FX';

type Driver = {
  name: string;
  impact: 'positive' | 'negative' | 'neutral';
  evidence?: string;
};

type Point = { t: number; v: number };

type Narrative = {
  id: string;
  title: string;
  score: number;      // -1 .. +1 (composite stance: bullish/bearish)
  momentum: number;   // -1 .. +1 (improving/deteriorating)
  confidence: number; // 0 .. 1
  tags?: string[];
  summary?: string;
  drivers?: Driver[];
  series?: Point[];
  linked?: string[];  // tickers or tick groups
  region?: string;    // 'US','EU','IN', etc
  updated_ms?: number;
};

type ResultPayload = {
  kpis?: { label: string; value: string | number }[];
  narratives?: Narrative[];
  notes?: string;
};

type Props = {
  endpoint?: string; // e.g. '/api/macro/narratives'
  title?: string;
  className?: string;
  defaultRegion?: string;
  defaultHorizon?: '1M' | '3M' | '6M' | '1Y';
  storageKey?: string;
};

/* ---------- util ---------- */

function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
function clamp(x: number, a: number, b: number) {
  return Math.max(a, Math.min(b, x));
}
function fmtPct(x: number, dp = 0) {
  return `${(x * 100).toFixed(dp)}%`;
}
function ts() {
  return Date.now();
}
function sanitizePatch(n: any): Narrative | null {
  if (!n || typeof n !== 'object') return null;
  const id = String(n.id ?? uid());
  const title = String(n.title ?? 'Untitled narrative');
  const score = clamp(Number(n.score ?? 0), -1, 1);
  const momentum = clamp(Number(n.momentum ?? 0), -1, 1);
  const confidence = clamp(Number(n.confidence ?? 0), 0, 1);
  const tags = Array.isArray(n.tags) ? (n.tags as string[]) : [];
  const series: Point[] = Array.isArray(n.series)
    ? n.series.map((p: any) => ({ t: Number(p.t ?? ts()), v: Number(p.v ?? 0) }))
    : [];
  const drivers: Driver[] = Array.isArray(n.drivers)
    ? n.drivers.map((d: any) => ({
        name: String(d.name ?? 'driver'),
        impact: (['positive', 'negative', 'neutral'] as const).includes(d.impact)
          ? d.impact
          : 'neutral',
        evidence: d.evidence ? String(d.evidence) : undefined,
      }))
    : [];
  const linked = Array.isArray(n.linked) ? (n.linked as string[]) : [];
  const region = n.region ? String(n.region) : undefined;

  return {
    id,
    title,
    score,
    momentum,
    confidence,
    tags,
    series,
    drivers,
    linked,
    region,
    summary: n.summary ? String(n.summary) : undefined,
    updated_ms: Number(n.updated_ms ?? ts()),
  };
}

/* ---------- main component ---------- */

export default function MacroNarratives({
  endpoint = '/api/macro/narratives',
  title = 'Macro Narratives',
  className = '',
  defaultRegion = 'US',
  defaultHorizon = '3M',
  storageKey = 'macro-narratives-v1',
}: Props) {
  const [region, setRegion] = useState(defaultRegion);
  const [horizon, setHorizon] = useState<'1M' | '3M' | '6M' | '1Y'>(defaultHorizon);
  const [sources, setSources] = useState<Record<SourceKey, boolean>>({
    CPI: true,
    PPI: false,
    PMI: true,
    Jobs: true,
    FOMC: true,
    YieldCurve: true,
    Housing: false,
    Earnings: true,
    Energy: false,
    FX: false,
  });
  const [query, setQuery] = useState('');
  const [narratives, setNarratives] = useState<Narrative[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notes, setNotes] = useState<string>('');
  const [kpis, setKpis] = useState<{ label: string; value: string | number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const listRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Persist to localStorage
  useEffect(() => {
    try {
      const payload = { region, horizon, sources, narratives, kpis, notes, ts: ts() };
      localStorage.setItem(storageKey, JSON.stringify(payload));
    } catch {}
  }, [region, horizon, sources, narratives, kpis, notes, storageKey]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const j = JSON.parse(raw);
        if (j.region) setRegion(j.region);
        if (j.horizon) setHorizon(j.horizon);
        if (j.sources) setSources(j.sources);
        if (Array.isArray(j.narratives)) setNarratives(j.narratives);
        if (Array.isArray(j.kpis)) setKpis(j.kpis);
        if (typeof j.notes === 'string') setNotes(j.notes);
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Autoscroll log
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [log, loading]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let arr = narratives;
    if (q) {
      arr = arr.filter(
        (n) =>
          n.title.toLowerCase().includes(q) ||
          (n.summary || '').toLowerCase().includes(q) ||
          (n.tags || []).some((t) => t.toLowerCase().includes(q)) ||
          (n.linked || []).some((t) => t.toLowerCase().includes(q)),
      );
    }
    // sort: highest confidence * |score|, then momentum
    return [...arr].sort(
      (a, b) =>
        b.confidence * Math.abs(b.score) - a.confidence * Math.abs(a.score) ||
        b.momentum - a.momentum,
    );
  }, [narratives, query]);

  function toggleSource(s: SourceKey) {
    setSources((prev) => ({ ...prev, [s]: !prev[s] }));
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  async function run() {
    setError(null);
    setLoading(true);
    setLog('');
    setNotes('');
    setKpis([]);
    setNarratives([]);

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    const body = {
      id: uid(),
      region,
      horizon,
      sources: Object.entries(sources)
        .filter(([, v]) => !!v)
        .map(([k]) => k),
    };

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream, application/json, text/plain',
        },
        body: JSON.stringify(body),
        signal,
      });

      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        // streaming
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
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
              setLog((s) => s + data + '\n');
              // try parse json patch
              try {
                const obj = JSON.parse(data);
                applyPatch(obj);
              } catch {
                setNotes((n) => (n ? n + '\n' : '') + data);
              }
            }
          } else {
            setLog((s) => s + chunk);
          }
        }
      } else if (ctype.includes('application/json')) {
        const json = await res.json();
        applyPatch(json);
      } else {
        const txt = await res.text();
        setNotes(txt);
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function applyPatch(patch: any) {
    // kpis
    if (Array.isArray(patch.kpis)) setKpis(patch.kpis);
    if (patch.kpi && patch.kpi.label) {
      setKpis((prev) => [...prev, patch.kpi]);
    }
    // narratives array or single
    if (Array.isArray(patch.narratives)) {
      for (const n of patch.narratives) upsertNarrative(n);
    } else if (patch.narrative) {
      upsertNarrative(patch.narrative);
    }
    // notes
    if (typeof patch.notes === 'string') {
      setNotes((n) => (n ? n + '\n' : '') + patch.notes);
    }
  }

  function upsertNarrative(n: any) {
    const sane = sanitizePatch(n);
    if (!sane) return;
    setNarratives((prev) => {
      const i = prev.findIndex((x) => x.id === sane.id);
      if (i === -1) return [...prev, sane];
      const merged = { ...prev[i], ...sane, updated_ms: ts() };
      const copy = [...prev];
      copy[i] = merged;
      return copy;
    });
  }

  function exportJSON() {
    const blob = new Blob(
      [
        JSON.stringify(
          { region, horizon, sources, narratives, kpis, notes, exported_ms: ts() },
          null,
          2,
        ),
      ],
      { type: 'application/json' },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `macro-narratives-${region}-${horizon}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const selected = narratives.find((n) => n.id === selectedId) || null;

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button
            onClick={exportJSON}
            className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900"
          >
            Export
          </button>
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-red-400 text-red-600 hover:bg-red-50">
              Stop
            </button>
          ) : (
            <button
              onClick={run}
              className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700"
            >
              Run
            </button>
          )}
        </div>
      </div>

      {/* controls */}
      <div className="p-3 border-b border-neutral-200 dark:border-neutral-800">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          <div>
            <label className="block text-[11px] text-neutral-500 mb-1">Region</label>
            <select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"
            >
              <option>US</option>
              <option>EU</option>
              <option>UK</option>
              <option>IN</option>
              <option>JP</option>
              <option>CN</option>
              <option>Global</option>
            </select>
          </div>
          <div>
            <label className="block text-[11px] text-neutral-500 mb-1">Horizon</label>
            <select
              value={horizon}
              onChange={(e) => setHorizon(e.target.value as any)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"
            >
              <option value="1M">1M</option>
              <option value="3M">3M</option>
              <option value="6M">6M</option>
              <option value="1Y">1Y</option>
            </select>
          </div>
          <div className="md:col-span-3">
            <label className="block text-[11px] text-neutral-500 mb-1">Search</label>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by title, tag, or ticker"
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"
            />
          </div>
        </div>

        {/* sources */}
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {(Object.keys(sources) as SourceKey[]).map((k) => (
            <label key={k} className={`inline-flex items-center gap-1 px-2 py-1 rounded-full border cursor-pointer
              ${sources[k] ? 'border-indigo-600 text-indigo-700 bg-indigo-50 dark:bg-indigo-900/30' : 'border-neutral-300 dark:border-neutral-700'}`}>
              <input
                type="checkbox"
                checked={!!sources[k]}
                onChange={() => toggleSource(k)}
                className="accent-indigo-600"
              />
              {k}
            </label>
          ))}
        </div>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* list */}
        <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">
            Narratives {filtered.length ? `(${filtered.length})` : ''}
          </div>
          <div className="flex-1 overflow-auto divide-y divide-neutral-100 dark:divide-neutral-800">
            {filtered.length === 0 ? (
              <div className="p-3 text-xs text-neutral-500">No narratives yet. Click <span className="font-medium">Run</span>.</div>
            ) : (
              filtered.map((n) => (
                <NarrativeRow
                  key={n.id}
                  n={n}
                  selected={selectedId === n.id}
                  onSelect={() => setSelectedId(n.id)}
                />
              ))
            )}
          </div>
        </div>

        {/* right pane: details + log */}
        <div className="min-h-0 space-y-3">
          <div className="min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
            <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Details</div>
            {selected ? (
              <NarrativeDetail n={selected} />
            ) : (
              <div className="p-3 text-xs text-neutral-500">Select a narrative to see details.</div>
            )}
          </div>
          <div className="min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
            <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Run Log</div>
            <div ref={listRef} className="flex-1 overflow-auto p-3">
              {error ? (
                <div className="text-xs text-red-600">{error}</div>
              ) : (
                <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">
                  {log || 'No streaming output yet.'}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* footer KPI pills */}
      {kpis.length > 0 && (
        <div className="border-t border-neutral-200 dark:border-neutral-800 px-3 py-2 flex flex-wrap gap-2">
          {kpis.map((k, i) => (
            <div key={i} className="text-[11px] rounded-full border border-neutral-300 dark:border-neutral-700 px-2 py-1">
              <span className="text-neutral-500">{k.label}: </span>
              <span className="font-medium">{String(k.value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- subcomponents ---------- */

function NarrativeRow({
  n,
  selected,
  onSelect,
}: {
  n: Narrative;
  selected: boolean;
  onSelect: () => void;
}) {
  const vals = useMemo(() => (n.series || []).map((p) => p.v), [n.series]);
  const last = vals.length ? vals[vals.length - 1] : n.score;
  const color =
    (n.score >= 0.2 && 'text-emerald-700') ||
    (n.score <= -0.2 && 'text-rose-700') ||
    'text-neutral-700';

  const momentumBadge =
    n.momentum > 0.15 ? '↑ improving' : n.momentum < -0.15 ? '↓ deteriorating' : '→ flat';

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-2 hover:bg-neutral-50 dark:hover:bg-neutral-800/60 ${selected ? 'bg-neutral-50 dark:bg-neutral-800/60' : ''}`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium truncate">{n.title}</div>
            <div className="text-[11px] text-neutral-500">{n.region || '—'}</div>
          </div>
          <div className="mt-1 grid grid-cols-[140px_1fr] gap-3 items-center">
            <Sparkline values={vals.length ? vals : [n.score]} />
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className={`font-semibold ${color}`}>score {n.score.toFixed(2)}</span>
              <span className="text-neutral-500">·</span>
              <span className="text-neutral-700">{momentumBadge}</span>
              <span className="text-neutral-500">·</span>
              <span className="text-neutral-700">confidence {fmtPct(n.confidence, 0)}</span>
              {n.tags?.slice(0, 4).map((t) => (
                <span key={t} className="ml-1 rounded-full border border-neutral-300 dark:border-neutral-700 px-2 py-[2px]">
                  {t}
                </span>
              ))}
            </div>
          </div>
          {n.summary && (
            <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-300 line-clamp-2">
              {n.summary}
            </div>
          )}
          {n.linked && n.linked.length > 0 && (
            <div className="mt-1 text-[11px] text-neutral-500">
              linked: {n.linked.slice(0, 8).join(', ')}
              {n.linked.length > 8 ? '…' : ''}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

function NarrativeDetail({ n }: { n: Narrative }) {
  return (
    <div className="p-3 text-sm space-y-3">
      <div className="flex items-center justify-between">
        <div className="font-semibold">{n.title}</div>
        <div className="text-[11px] text-neutral-500">
          updated {n.updated_ms ? new Date(n.updated_ms).toLocaleString() : '—'}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs">
        <Metric label="Score" value={n.score.toFixed(2)} tone={toneFromScore(n.score)} />
        <Metric
          label="Momentum"
          value={n.momentum.toFixed(2)}
          tone={toneFromScore(n.momentum)}
        />
        <Metric label="Confidence" value={fmtPct(n.confidence, 0)} />
      </div>

      <div>
        <div className="text-xs font-medium mb-1">Trajectory</div>
        <Sparkline values={(n.series || []).map((p) => p.v)} height={40} />
      </div>

      {n.summary && (
        <div>
          <div className="text-xs font-medium mb-1">Summary</div>
          <div className="text-sm whitespace-pre-wrap">{n.summary}</div>
        </div>
      )}

      {n.drivers && n.drivers.length > 0 && (
        <div>
          <div className="text-xs font-medium mb-1">Drivers</div>
          <ul className="list-disc pl-5 space-y-1">
            {n.drivers.map((d, i) => (
              <li key={i} className="text-sm">
                <span
                  className={`mr-2 text-[11px] px-1.5 py-[2px] rounded border ${
                    d.impact === 'positive'
                      ? 'border-emerald-400 text-emerald-700'
                      : d.impact === 'negative'
                      ? 'border-rose-400 text-rose-700'
                      : 'border-neutral-300 text-neutral-700'
                  }`}
                >
                  {d.impact}
                </span>
                <span className="font-medium">{d.name}</span>
                {d.evidence ? <span className="text-neutral-500"> — {d.evidence}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      )}

      {n.linked && n.linked.length > 0 && (
        <div>
          <div className="text-xs font-medium mb-1">Linked tickers/themes</div>
          <div className="flex flex-wrap gap-2">
            {n.linked.map((t) => (
              <span key={t} className="text-xs rounded-full border border-neutral-300 dark:border-neutral-700 px-2 py-[2px]">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: string | number; tone?: 'pos' | 'neg' | 'neu' }) {
  const cls =
    tone === 'pos'
      ? 'text-emerald-700'
      : tone === 'neg'
      ? 'text-rose-700'
      : 'text-neutral-800 dark:text-neutral-200';
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-3 py-2">
      <div className="text-[11px] text-neutral-500">{label}</div>
      <div className={`text-sm font-semibold ${cls}`}>{String(value)}</div>
    </div>
  );
}

function toneFromScore(x: number): 'pos' | 'neg' | 'neu' {
  if (x > 0.15) return 'pos';
  if (x < -0.15) return 'neg';
  return 'neu';
}

/* small inline sparkline using pure SVG */
function Sparkline({
  values,
  width = 140,
  height = 32,
  strokeWidth = 1.5,
}: {
  values: number[];
  width?: number;
  height?: number;
  strokeWidth?: number;
}) {
  const pts = values && values.length ? values : [0];
  const vmin = Math.min(...pts);
  const vmax = Math.max(...pts);
  const span = vmax - vmin || 1;
  const step = pts.length > 1 ? width / (pts.length - 1) : width;
  const path = pts
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - vmin) / span) * height;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
  const last = pts[pts.length - 1] ?? 0;
  const first = pts[0] ?? 0;
  const tone = last > first + 1e-9 ? '#059669' : last < first - 1e-9 ? '#dc2626' : '#6b7280';

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="block">
      <path d={path} fill="none" stroke={tone} strokeWidth={strokeWidth} />
    </svg>
  );
}