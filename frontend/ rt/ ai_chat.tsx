'use client';
import React, { useEffect, useRef, useState } from 'react';

type Role = 'user' | 'assistant';
type ChatMessage = { id: string; role: Role; content: string; createdAt: number };

type Props = {
  apiEndpoint?: string;      // your POST endpoint; returns { reply: string }
  title?: string;
  placeholder?: string;
  starter?: string[];        // optional quick prompts
};

const uid = (p = 'm') => `${p}_${Math.random().toString(36).slice(2, 10)}`;
const now = () => Date.now();

export default function AIChat({
  apiEndpoint = '/api/chat',
  title = 'AI Chat',
  placeholder = 'Type a message…',
  starter = ['What can you do?', 'Summarize today’s market', 'Explain my PnL'],
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight + 9999;
    }
  }, [messages, busy]);

  async function send(text?: string) {
    const content = (text ?? input).trim();
    if (!content) return;
    setError(null);

    const userMsg: ChatMessage = { id: uid('u'), role: 'user', content, createdAt: now() };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setBusy(true);

    try {
      const payload = { messages: [...messages, userMsg].map(({ role, content }) => ({ role, content })) };
      const res = await fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      let reply = '';
      if (res.ok) {
        const data = await res.json().catch(() => ({}));
        reply = String(data?.reply ?? data?.text ?? data?.message ?? 'Okay.');
      } else {
        reply = `Server responded ${res.status}. (Echo) ${content}`;
      }

      const botMsg: ChatMessage = { id: uid('a'), role: 'assistant', content: reply, createdAt: now() };
      setMessages((m) => [...m, botMsg]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Request failed';
      setError(msg);
      // Fallback echo so UI still feels responsive
      setMessages((m) => [...m, { id: uid('a'), role: 'assistant', content: `Echo: ${content}`, createdAt: now() }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.title}>{title}</div>
        {starter?.length ? (
          <div style={styles.startersWrap}>
            {starter.map((s, i) => (
              <button key={i} style={{ ...styles.btn, ...styles.btnSecondary }} onClick={() => send(s)} disabled={busy}>
                {s}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div style={styles.body}>
        {/* Transcript */}
        <div style={styles.transcriptBox}>
          <div ref={scrollRef} style={styles.scroll}>
            <div style={{ padding: 12 }}>
              {messages.length === 0 && (
                <div style={styles.placeholder}>Start the conversation with a prompt, or type below.</div>
              )}
              {messages.map((m) => (
                <MessageBubble key={m.id} msg={m} />
              ))}
              {busy && <div style={styles.thinking}>Thinking…</div>}
            </div>
          </div>

          {/* Composer */}
          <div style={styles.composer}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder={placeholder}
              style={styles.input}
            />
            <button onClick={() => send()} disabled={busy || input.trim() === ''} style={styles.btn}>
              Send
            </button>
          </div>
        </div>

        {/* Error (non-blocking) */}
        {error ? <div style={styles.errorBox}>⚠️ {error}</div> : null}
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user';
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', margin: '8px 0' }}>
      <div
        style={{
          ...styles.bubble,
          ...(isUser ? styles.bubbleUser : styles.bubbleBot),
        }}
      >
        <div style={styles.meta}>
          <span>{isUser ? 'You' : 'Assistant'}</span>
          <span style={styles.metaTime}>{new Date(msg.createdAt).toLocaleTimeString()}</span>
        </div>
        <div style={styles.text}>{msg.content}</div>
      </div>
    </div>
  );
}

/* ---------------- minimal inline styles ---------------- */
const styles: Record<string, React.CSSProperties> = {
  card: {
    border: '1px solid #e5e7eb',
    borderRadius: 16,
    background: '#fff',
    boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    maxWidth: 920,
    width: '100%',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    justifyContent: 'space-between',
    padding: '12px 14px',
    borderBottom: '1px solid #eee',
  },
  title: { fontWeight: 600, fontSize: 16 },
  startersWrap: { display: 'flex', flexWrap: 'wrap', gap: 8 },
  body: { padding: 12 },
  transcriptBox: {
    border: '1px solid #e5e7eb',
    borderRadius: 12,
    overflow: 'hidden',
  },
  scroll: {
    height: 460,
    overflow: 'auto',
    background: '#fff',
  },
  placeholder: {
    color: '#9ca3af',
    textAlign: 'center',
    padding: '36px 0',
    fontSize: 14,
  },
  thinking: { padding: '6px 8px', fontSize: 12, color: '#6b7280' },
  composer: {
    borderTop: '1px solid #e5e7eb',
    padding: 8,
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  },
  input: {
    height: 40,
    border: '1px solid #e5e7eb',
    borderRadius: 12,
    padding: '0 10px',
    flex: 1,
    fontSize: 14,
    outline: 'none',
  },
  btn: {
    height: 36,
    padding: '0 12px',
    borderRadius: 12,
    border: '1px solid transparent',
    background: '#111',
    color: '#fff',
    cursor: 'pointer',
    fontSize: 13,
  },
  btnSecondary: { background: '#f3f4f6', color: '#111' },
  errorBox: {
    marginTop: 10,
    padding: '8px 10px',
    borderRadius: 10,
    border: '1px solid #fecaca',
    background: '#fff1f2',
    color: '#991b1b',
    fontSize: 13,
  },
  bubble: {
    maxWidth: '86%',
    border: '1px solid #e5e7eb',
    borderRadius: 16,
    padding: '8px 10px',
    boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
  },
  bubbleUser: { background: '#fff7ed', borderColor: '#fde68a' },
  bubbleBot: { background: '#fff' },
  meta: { display: 'flex', alignItems: 'center', gap: 8, color: '#6b7280', fontSize: 12, marginBottom: 4 },
  metaTime: { color: '#9ca3af', fontSize: 11 },
  text: { whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: 14 },
};