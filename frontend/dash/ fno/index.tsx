'use client';
import React, { useEffect, useRef, useState } from 'react';

/* ---------------- inline MarginWidget ---------------- */
function MarginWidget({
  balance = 200000,
  used = 85000,
  currency = 'USD',
  warningPct = 70,
  dangerPct = 90,
  title = 'Margin Overview',
}: {
  balance?: number; used?: number; currency?: string; warningPct?: number; dangerPct?: number; title?: string;
}) {
  const free = Math.max(0, balance - used);
  const ratio = balance > 0 ? (used / balance) * 100 : 0;
  const color = ratio < warningPct ? '#10b981' : ratio < dangerPct ? '#f59e0b' : '#ef4444';

  return (
    <div style={S.card}>
      <div style={S.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={S.h1}>{title}</span>
          <span style={{ ...S.badge, background: '#eef2ff', color: '#3730a3' }}>{currency}</span>
        </div>
      </div>
      <div style={{ padding: 20 }}>
        <div style={S.grid4}>
          <KPI label="Account Balance" value={balance} currency={currency} />
          <KPI label="Used Margin" value={used} currency={currency} />
          <KPI label="Free Margin" value={free} currency={currency} />
          <KPI label="Margin Ratio" value={`${ratio.toFixed(1)}%`} />
        </div>

        <div style={{ marginTop: 18 }}>
          <div style={S.row}>
            <span style={S.rowLabel}>Utilization</span>
            <span style={{ ...S.rowValue, color }}>{ratio.toFixed(1)}%</span>
          </div>
          <div style={S.progressOuter}>
            <div style={{ ...S.progressInner, width: `${Math.min(100, Math.max(0, ratio))}%`, background: color }} />
          </div>
          <div style={S.legend}>
            <Legend swatch="#10b981" text="< 70%" />
            <Legend swatch="#f59e0b" text="70–90%" />
            <Legend swatch="#ef4444" text="≥ 90%" />
          </div>
        </div>
      </div>
    </div>
  );
}
function KPI({ label, value, currency }: { label: string; value: number | string; currency?: string }) {
  const f = typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value;
  return (
    <div style={S.kpiCard}>
      <div style={S.kpiLabel}>{label}</div>
      <div style={S.kpiValue}>{f}{currency && typeof value === 'number' ? ` ${currency}` : ''}</div>
    </div>
  );
}
function Legend({ swatch, text }: { swatch: string; text: string }) {
  return <div style={S.legendItem}><span style={{ ...S.swatch, background: swatch }} />{text}</div>;
}

/* ---------------- inline Chat ---------------- */
type Role = 'user' | 'assistant';
type ChatMessage = { id: string; role: Role; content: string; createdAt: number };
const uid = (p = 'm') => `${p}_${Math.random().toString(36).slice(2, 10)}`;
const now = () => Date.now();

function Chat({ apiEndpoint = '/api/chat', title = 'AI Assistant' }: { apiEndpoint?: string; title?: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight + 9999; }, [messages, busy]);

  async function send() {
    const content = input.trim(); if (!content) return;
    setError(null);
    const userMsg: ChatMessage = { id: uid('u'), role: 'user', content, createdAt: now() };
    setMessages((m) => [...m, userMsg]); setInput(''); setBusy(true);
    try {
      const payload = { messages: [...messages, userMsg].map(({ role, content }) => ({ role, content })) };
      const res = await fetch(apiEndpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = res.ok ? await res.json().catch(() => ({})) : {};
      const reply = String((data as any)?.reply ?? (data as any)?.text ?? (data as any)?.message ?? (res.ok ? 'Okay.' : `Server ${res.status}.`));
      setMessages((m) => [...m, { id: uid('a'), role: 'assistant', content: reply, createdAt: now() }]);
    } catch (e: any) {
      setError(e?.message || 'Request failed');
      setMessages((m) => [...m, { id: uid('a'), role: 'assistant', content: 'Network error. Try again.', createdAt: now() }]);
    } finally { setBusy(false); }
  }

  return (
    <div style={S.card}>
      <div style={S.header}><span style={S.h1}>{title}</span></div>
      <div style={{ padding: 12 }}>
        <div style={{ ...S.box }}>
          <div ref={scrollRef} style={S.scroll}>
            <div style={{ padding: 12 }}>
              {messages.length === 0 && <div style={S.placeholder}>Say hi 👋</div>}
              {messages.map((m) => <Bubble key={m.id} msg={m} />)}
              {busy && <div style={S.thinking}>Thinking…</div>}
            </div>
          </div>
          <div style={S.composer}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Type a message…"
              style={S.input}
            />
            <button onClick={send} disabled={busy || input.trim() === ''} style={S.btn}>Send</button>
          </div>
        </div>
        {error && <div style={S.err}>⚠️ {error}</div>}
      </div>
    </div>
  );
}
function Bubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user';
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', margin: '8px 0' }}>
      <div style={{ ...S.bubble, ...(isUser ? S.bubbleUser : S.bubbleBot) }}>
        <div style={S.meta}>
          <span>{isUser ? 'You' : 'Assistant'}</span>
          <span style={S.metaTime}>{new Date(msg.createdAt).toLocaleTimeString()}</span>
        </div>
        <div style={S.text}>{msg.content}</div>
      </div>
    </div>
  );
}

/* ---------------- page layout ---------------- */
export default function IndexPage() {
  return (
    <div style={S.page}>
      <header style={S.appHeader}><h1 style={{ margin: 0, fontSize: 22 }}>Hedge Fund Dashboard</h1></header>
      <main style={S.grid2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <MarginWidget balance={200000} used={85000} currency="USD" />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <Chat apiEndpoint="/api/chat" />
        </div>
      </main>
      <footer style={S.footer}>⚡ Powered by your Arbitrage Platform</footer>
    </div>
  );
}

/* ---------------- styles ---------------- */
const S: Record<string, React.CSSProperties> = {
  page: { display: 'flex', flexDirection: 'column', minHeight: '100vh', background: '#f9fafb', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  appHeader: { padding: '16px 24px', borderBottom: '1px solid #e5e7eb', background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' },
  grid2: { flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, padding: 24, alignItems: 'flex-start' },
  footer: { padding: '12px 20px', borderTop: '1px solid #e5e7eb', background: '#fff', fontSize: 13, color: '#6b7280', textAlign: 'center' },

  card: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%' },
  header: { padding: '16px 20px', borderBottom: '1px solid #eee' },
  h1: { fontWeight: 700, fontSize: 18, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600 },

  grid4: { display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 12 },
  kpiCard: { border: '1px solid #eef2f7', borderRadius: 12, padding: '14px 12px', background: '#fafafa' },
  kpiLabel: { fontSize: 12, color: '#6b7280', marginBottom: 6 },
  kpiValue: { fontSize: 18, fontWeight: 700, color: '#111827' },

  row: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
  rowLabel: { fontSize: 14, color: '#374151', fontWeight: 600 },
  rowValue: { fontSize: 14, fontWeight: 700 },

  progressOuter: { height: 16, borderRadius: 10, overflow: 'hidden', background: '#f3f4f6', border: '1px solid #e5e7eb' },
  progressInner: { height: '100%', borderRadius: 10, transition: 'width .3s ease' },
  legend: { display: 'flex', gap: 16, alignItems: 'center', marginTop: 8, color: '#6b7280', fontSize: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 8 },
  swatch: { width: 10, height: 10, borderRadius: 3, border: '1px solid rgba(0,0,0,0.1)' },

  box: { border: '1px solid #e5e7eb', borderRadius: 12, overflow: 'hidden' },
  scroll: { height: 420, overflow: 'auto', background: '#fff' },
  placeholder: { color: '#9ca3af', textAlign: 'center', padding: '36px 0', fontSize: 14 },
  thinking: { padding: '6px 8px', fontSize: 12, color: '#6b7280' },
  composer: { borderTop: '1px solid #e5e7eb', padding: 8, display: 'flex', gap: 8, alignItems: 'center' },
  input: { height: 40, border: '1px solid #e5e7eb', borderRadius: 12, padding: '0 10px', flex: 1, fontSize: 14, outline: 'none' },
  btn: { height: 36, padding: '0 12px', borderRadius: 12, border: '1px solid transparent', background: '#111', color: '#fff', cursor: 'pointer', fontSize: 13 },
  err: { marginTop: 10, padding: '8px 10px', borderRadius: 10, border: '1px solid #fecaca', background: '#fff1f2', color: '#991b1b', fontSize: 13 },

  bubble: { maxWidth: '86%', border: '1px solid #e5e7eb', borderRadius: 16, padding: '8px 10px', boxShadow: '0 1px 2px rgba(0,0,0,0.04)' },
  bubbleUser: { background: '#fff7ed', borderColor: '#fde68a' },
  bubbleBot: { background: '#fff' },
  meta: { display: 'flex', alignItems: 'center', gap: 8, color: '#6b7280', fontSize: 12, marginBottom: 4 },
  metaTime: { color: '#9ca3af', fontSize: 11 },
  text: { whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: 14 },
};