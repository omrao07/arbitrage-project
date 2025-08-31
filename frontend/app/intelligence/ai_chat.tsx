'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

type Role = 'user' | 'assistant' | 'system' | 'tool';

type Message = {
  id: string;
  role: Role;
  content: string;
  ts: number;
};

type Props = {
  title?: string;
  /** HTTP endpoint that returns either JSON: {reply:string} OR streams text */
  endpoint?: string; // e.g. '/api/ai/chat'
  placeholder?: string;
  storageKey?: string;
  className?: string;
};

function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function now() {
  return Date.now();
}

export default function AIChat({
  title = 'AI Chat',
  endpoint = '/api/ai/chat',
  placeholder = 'Type a message…',
  storageKey = 'ai-chat-history',
  className = '',
}: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Load saved chat
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) setMessages(JSON.parse(raw));
    } catch {}
  }, [storageKey]);

  // Persist chat
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(messages));
    } catch {}
  }, [messages, storageKey]);

  // Auto-scroll to last message
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  const canSend = input.trim().length > 0 && !loading;

  async function sendMessage(userText?: string) {
    const text = (userText ?? input).trim();
    if (!text) return;

    setError(null);
    setLoading(true);

    // push user msg
    const u: Message = { id: uid(), role: 'user', content: text, ts: now() };
    // create assistant placeholder (for streaming)
    const a: Message = { id: uid(), role: 'assistant', content: '', ts: now() };

    setMessages((prev) => [...prev, u, a]);
    setInput('');
    inputRef.current?.focus();

    // Prepare abort
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;
    setStreamingId(a.id);

    try {
      // Try streaming first (SSE or chunked text)
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream, text/plain, application/json' },
        body: JSON.stringify({ messages: sanitizeForServer([...messages, u]) }),
        signal,
      });

      const ctype = res.headers.get('content-type') || '';

      if (!res.ok) {
        const msg = `HTTP ${res.status}`;
        throw new Error(msg);
      }

      // SSE (text/event-stream) or chunked text
      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          // If it's SSE, parse "data:" lines; else treat as plain text stream
          if (ctype.includes('text/event-stream')) {
            const lines = (acc + chunk).split('\n');
            acc = lines.pop() || '';
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;
              const data = line.slice(5).trim();
              if (data === '[DONE]') continue;
              appendToAssistant(a.id, data);
            }
          } else {
            appendToAssistant(a.id, chunk);
          }
        }
      } else if (ctype.includes('application/json')) {
        // JSON { reply: string } or OpenAI-style { choices:[{message:{content}}] }
        const json = await res.json();
        const reply =
          json?.reply ??
          json?.choices?.[0]?.message?.content ??
          json?.content ??
          (typeof json === 'string' ? json : '');
        replaceAssistant(a.id, reply || '(empty)');
      } else {
        // Fallback: treat as text
        const txt = await res.text();
        replaceAssistant(a.id, txt || '(empty)');
      }
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        replaceAssistant(a.id, '(stopped)');
      } else {
        setError(e?.message || 'Request failed');
        replaceAssistant(a.id, '(error)');
      }
    } finally {
      setLoading(false);
      setStreamingId(null);
      abortRef.current = null;
    }
  }

  function appendToAssistant(id: string, delta: string) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, content: (m.content || '') + delta } : m)),
    );
  }

  function replaceAssistant(id: string, full: string) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content: full } : m)));
  }

  function stopStreaming() {
    abortRef.current?.abort();
  }

  function clearChat() {
    abortRef.current?.abort();
    setMessages([]);
    setInput('');
    setStreamingId(null);
    setError(null);
  }

  function copyText(txt: string) {
    try {
      navigator.clipboard.writeText(txt);
    } catch {}
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canSend) sendMessage();
    }
  }

  const header = useMemo(
    () => (
      <div className="flex items-center justify-between border-b border-neutral-200 dark:border-neutral-800 px-3 py-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button
            onClick={clearChat}
            className="text-xs rounded px-2 py-1 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900"
            title="Clear"
          >
            Clear
          </button>
          {streamingId ? (
            <button
              onClick={stopStreaming}
              className="text-xs rounded px-2 py-1 border border-red-400 text-red-600 hover:bg-red-50"
              title="Stop"
            >
              Stop
            </button>
          ) : null}
        </div>
      </div>
    ),
    [title, streamingId],
  );

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {header}

      <div ref={listRef} className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map((m) => (
            <MessageBubble
              key={m.id}
              role={m.role}
              content={m.content}
              ts={m.ts}
              onCopy={() => copyText(m.content)}
            />
          ))
        )}
        {error && (
          <div className="text-xs text-red-600 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded p-2">
            {error}
          </div>
        )}
      </div>

      <div className="border-t border-neutral-200 dark:border-neutral-800 p-2">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            rows={2}
            className="w-full resize-none rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            disabled={!canSend}
            onClick={() => sendMessage()}
            className={`shrink-0 rounded-lg px-3 py-2 text-sm font-medium border ${canSend ? 'bg-indigo-600 text-white border-indigo-600 hover:bg-indigo-700' : 'bg-neutral-200 text-neutral-500 border-neutral-300 cursor-not-allowed'}`}
          >
            Send
          </button>
        </div>
        <div className="mt-1 text-[11px] text-neutral-500">
          Enter to send · Shift+Enter for newline
        </div>
      </div>
    </div>
  );
}

/* ---------- helpers & subcomponents ---------- */

function sanitizeForServer(msgs: Message[]) {
  // Strip client-only fields before sending to backend
  return msgs.map(({ role, content }) => ({ role, content }));
}

function EmptyState() {
  return (
    <div className="text-xs text-neutral-500 px-1">
      Start chatting. Your history is saved locally.
    </div>
  );
}

function MessageBubble({
  role,
  content,
  ts,
  onCopy,
}: {
  role: Role;
  content: string;
  ts: number;
  onCopy: () => void;
}) {
  const isUser = role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap break-words border
        ${isUser
            ? 'bg-indigo-600 text-white border-indigo-600'
            : 'bg-neutral-50 dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 border-neutral-200 dark:border-neutral-800'}`}
      >
        <div>{content || (isUser ? '' : '…')}</div>
        <div className="mt-1 flex items-center gap-2 opacity-70">
          <span className="text-[10px]">{new Date(ts).toLocaleTimeString()}</span>
          {!isUser && (
            <button onClick={onCopy} className="text-[10px] underline decoration-dotted">
              copy
            </button>
          )}
        </div>
      </div>
    </div>
  );
}