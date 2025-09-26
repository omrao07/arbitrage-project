"use client";

/**
 * terminal/page.tsx
 * Zero-import, self-contained terminal UI (client component).
 *
 * Features
 * - Fake shell with prompt, command history, autocomplete hint
 * - Built-in commands: help, clear, echo, date, time, now, whoami, pwd, ls, cat, env, json, theme, about
 * - Multi-line output with ANSI-ish color tags: <red>, <green>, <yellow>, <blue>, <dim>
 * - Copy line, copy all, download log, clear screen
 * - History ↑/↓, Tab to cycle suggestions, Cmd/Ctrl+L to clear
 * - Auto-scroll, preserves session in localStorage
 *
 * Drop this under /terminal/page.tsx. It renders inside terminal/layout.tsx.
 */

type Line = { id: string; html: string; ts: string };
type ThemeMode = "system" | "light" | "dark";

export default function TerminalPage() {
  const [lines, setLines] = useState<Line[]>(() => loadLog());
  const [input, setInput] = useState("");
  const [cwd, setCwd] = useState<string>(() => loadKV("term:cwd", "~/project"));
  const [user, setUser] = useState<string>(() => loadKV("term:user", "guest"));
  const [theme, setTheme] = useState<ThemeMode>(() => (loadKV("term:theme", "system") as ThemeMode));
  const [history, setHistory] = useState<string[]>(() => loadKV("term:history", []) as string[]);
  const [hIndex, setHIndex] = useState<number>(-1);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const viewRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  /* Persist */
  useEffect(() => saveLog(lines), [lines]);
  useEffect(() => saveKV("term:cwd", cwd), [cwd]);
  useEffect(() => saveKV("term:user", user), [user]);
  useEffect(() => saveKV("term:theme", theme), [theme]);
  useEffect(() => saveKV("term:history", history), [history]);

  /* Autosroll */
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines]);

  /* Keyboard shortcuts */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Clear (Cmd/Ctrl + L)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "l") {
        e.preventDefault();
        clearScreen();
      }
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  /* Suggest commands on input */
  useEffect(() => {
    const all = builtinCommands();
    const q = input.trim();
    if (!q) return setSuggestions([]);
    const s = all.filter((c) => c.startsWith(q)).slice(0, 6);
    setSuggestions(s);
  }, [input]);

  function prompt() {
    return `<green>${user}</green>@host:<blue>${cwd}</blue>$`;
  }

  function print(html: string) {
    setLines((L) => [...L, { id: uid(), html, ts: ts() }]);
  }

  function printBlock(lines: string[]) {
    const batch = lines.map((h) => ({ id: uid(), html: h, ts: ts() }));
    setLines((L) => [...L, ...batch]);
  }

  function clearScreen() {
    setLines([]);
  }

  async function run(raw: string) {
    const cmd = raw.trim();
    if (!cmd) return;
    // show prompt + command
    print(`${prompt()} ${escapeHtml(raw)}`);

    // push history (dedupe consecutive)
    setHistory((H) => (H[H.length - 1] === raw ? H : [...H, raw]));
    setHIndex(-1);

    // dispatch
    const [name, ...args] = tokenize(cmd);
    const out = await dispatch(name, args);
    if (out && out.length) printBlock(out.map(renderColors));
    setInput("");
  }

  /* ------------------------------ Built-ins ------------------------------ */

  async function dispatch(name: string, args: string[]): Promise<string[]> {
    switch (name) {
      case "help":
      case "?":
        return [
          "<yellow>Available commands</yellow>",
          " help, clear, echo, date, time, now, whoami, pwd, ls, cat, env, json, theme, about",
          " usage: echo hello | json {\"a\":1} | theme dark",
        ];
      case "clear":
        clearScreen();
        return [];
      case "echo":
        return [escapeHtml(args.join(" ")) || ""];
      case "date":
      case "time":
      case "now":
        return [`<dim>${new Date().toString()}</dim>`];
      case "whoami":
        return [user];
      case "pwd":
        return [cwd];
      case "ls":
        return ["README.md", "src/", "package.json", "node_modules/"];
      case "cat":
        if (!args[0]) return ["<red>error:</red> file required"];
        if (args[0] === "README.md") {
          return [
            "# Demo",
            "",
            "This is a <dim>mock</dim> terminal. Try <yellow>help</yellow>.",
          ];
        }
        return [`<red>not found:</red> ${escapeHtml(args[0])}`];
      case "env":
        return [
          `USER=${user}`,
          `CWD=${cwd}`,
          `THEME=${theme}`,
          `AGENT=web-term/1.0`,
        ];
      case "json":
        try {
          const text = args.join(" ");
          const obj = JSON.parse(text);
          return ["<dim>parsed:</dim>", "```json", escapeHtml(JSON.stringify(obj, null, 2)), "```"];
        } catch (e: any) {
          return [`<red>parse error:</red> ${escapeHtml(e?.message || "invalid JSON")}`];
        }
      case "theme": {
        const t = (args[0] || "").toLowerCase();
        const next: ThemeMode = t === "light" ? "light" : t === "dark" ? "dark" : "system";
        setTheme(next);
        return [`theme → <blue>${next}</blue>`];
      }
      case "cd": {
        const dir = args[0] || "~";
        setCwd(resolveCwd(cwd, dir));
        return [];
      }
      case "setuser": {
        if (!args[0]) return ["usage: setuser <name>"];
        setUser(args[0]);
        return [`user → <green>${escapeHtml(args[0])}</green>`];
      }
      case "about":
        return [
          "<green>Web Terminal</green> – zero-import demo UI.",
          " Shortcuts: <dim>Ctrl/Cmd+L</dim> clears screen. Use ↑/↓ for history.",
        ];
      default:
        if (!name) return [];
        return [`<red>command not found:</red> ${escapeHtml(name)}`];
    }
  }

  /* ----------------------------- UI Handlers ----------------------------- */

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      run(input);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      const H = history;
      if (!H.length) return;
      const next = hIndex === -1 ? H[H.length - 1] : H[Math.max(0, hIndex - 1)];
      setHIndex((i) => (i === -1 ? H.length - 1 : Math.max(0, i - 1)));
      setInput(next);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      const H = history;
      if (!H.length) return;
      if (hIndex === -1) return;
      const nextIdx = Math.min(H.length - 1, hIndex + 1);
      setHIndex(nextIdx === H.length - 1 ? -1 : nextIdx);
      setInput(nextIdx === H.length - 1 ? "" : H[nextIdx]);
      return;
    }
    if (e.key === "Tab") {
      e.preventDefault();
      if (!suggestions.length) return;
      // cycle suggestions
      const idx = suggestions.findIndex((s) => s === input.trim());
      const next = idx === -1 ? suggestions[0] : suggestions[(idx + 1) % suggestions.length];
      setInput(next + " ");
    }
  }

  function copyAll() {
    try {
      const text = lines.map((l) => stripTags(l.html)).join("\n");
      navigator.clipboard?.writeText(text);
      toast("Copied log.");
    } catch {}
  }

  function downloadLog() {
    const text = lines.map((l) => `[${l.ts}] ${stripTags(l.html)}`).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "terminal.log";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

    function toast(arg0: string) {
        throw new Error("Function not implemented.");
    }

  /* -------------------------------- Render ------------------------------- */

  return (
    <div className={`flex h-full min-h-[60vh] flex-col ${themeClass(theme)}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-neutral-800 bg-neutral-950 px-3 py-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="rounded border border-neutral-800 bg-neutral-900 px-2 py-0.5">
            <span className="text-emerald-400">●</span> connected
          </span>
          <span className="hidden md:inline text-neutral-400">
            {user}@host:{cwd}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setTheme(cycleTheme(theme))} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">
            theme: {theme}
          </button>
          <button onClick={copyAll} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">Copy</button>
          <button onClick={downloadLog} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">Download</button>
          <button onClick={clearScreen} className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 hover:bg-neutral-800">Clear</button>
        </div>
      </div>

      {/* Output */}
      <div ref={viewRef} className="flex-1 overflow-auto bg-neutral-950 p-3 font-mono text-sm leading-6 text-neutral-200">
        {lines.map((ln) => (
          <div key={ln.id} className="group flex items-start gap-2">
            <div className="mt-1 select-none text-[10px] text-neutral-600 w-16">{timeOnly(ln.ts)}</div>
            <div className="whitespace-pre-wrap break-words" dangerouslySetInnerHTML={{ __html: renderColors(ln.html) }} />
            <button
              className="ml-auto hidden rounded border border-neutral-800 bg-neutral-900 px-1.5 text-[10px] text-neutral-400 group-hover:block hover:bg-neutral-800"
              onClick={() => { try { navigator.clipboard?.writeText(stripTags(ln.html)); toast("Copied line."); } catch {} }}
            >
              copy
            </button>
          </div>
        ))}
      </div>

      {/* Prompt */}
      <div className="flex items-center gap-2 border-t border-neutral-800 bg-neutral-950 px-3 py-2">
        <div className="font-mono text-sm" dangerouslySetInnerHTML={{ __html: renderColors(prompt()) }} />
        <input
          ref={inputRef}
          className="ml-2 flex-1 bg-transparent font-mono text-sm text-neutral-100 outline-none placeholder:text-neutral-600"
          placeholder="Type a command… (try 'help')"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          autoFocus
        />
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div className="border-t border-neutral-800 bg-neutral-925 px-3 py-1 text-xs text-neutral-400">
          <span className="mr-2 text-neutral-500">suggest:</span>
          {suggestions.map((s, i) => (
            <button
              key={s + i}
              className="mr-1 rounded border border-neutral-800 bg-neutral-900 px-2 py-0.5 hover:bg-neutral-800"
              onClick={() => setInput(s + " ")}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------ Helpers --------------------------------- */

function builtinCommands() {
  return ["help","clear","echo","date","time","now","whoami","pwd","ls","cat","env","json","theme","about","cd","setuser"];
}

function tokenize(s: string): string[] {
  const out: string[] = [];
  let cur = "", q: string | null = null;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (q) {
      if (ch === q) { q = null; continue; }
      cur += ch; continue;
    }
    if (ch === '"' || ch === "'") { q = ch; continue; }
    if (/\s/.test(ch)) { if (cur) { out.push(cur); cur=""; } continue; }
    cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

function renderColors(s: string): string {
  // very small formatter for <red>text</red> etc
  return s
    .replaceAll("&lt;", "<").replaceAll("&gt;", ">") // allow pre-escaped blocks (e.g. from json)
    .replace(/<dim>(.*?)<\/dim>/g, `<span class="text-neutral-400">$1</span>`)
    .replace(/<red>(.*?)<\/red>/g, `<span class="text-rose-400">$1</span>`)
    .replace(/<green>(.*?)<\/green>/g, `<span class="text-emerald-400">$1</span>`)
    .replace(/<yellow>(.*?)<\/yellow>/g, `<span class="text-amber-300">$1</span>`)
    .replace(/<blue>(.*?)<\/blue>/g, `<span class="text-sky-400">$1</span>`);
}

function stripTags(s: string) { return s.replace(/<[^>]+>/g, ""); }
function escapeHtml(s: string) {
  return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
function uid() { return Math.random().toString(36).slice(2, 10); }
function ts() { try { return new Date().toISOString(); } catch { return "" + Date.now(); } }
function timeOnly(iso: string) { try { return new Date(iso).toLocaleTimeString(); } catch { return ""; } }

function resolveCwd(cur: string, arg: string): string {
  if (!arg || arg === "~") return "~/project";
  if (arg.startsWith("/")) return arg;
  const base = cur.replace(/\/+$/,"");
  const parts = (base + "/" + arg).split("/").filter(Boolean);
  const out: string[] = [];
  for (const p of parts) {
    if (p === ".") continue;
    if (p === "..") out.pop();
    else out.push(p);
  }
  return "/" + out.join("/");
}

function themeClass(t: ThemeMode) {
  // This demo only adjusts subtle hues; real theming would toggle a root class
  switch (t) {
    case "light": return "bg-white text-neutral-900";
    case "dark": return "bg-neutral-950 text-neutral-100";
    default: return ""; // system (inherit)
  }
}

function cycleTheme(t: ThemeMode): ThemeMode {
  return t === "system" ? "dark" : t === "dark" ? "light" : "system";
}

/* ------------------------------ Persistence ------------------------------ */

function loadLog(): Line[] {
  try {
    const raw = localStorage.getItem("term:log");
    if (!raw) return seedLog();
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return seedLog();
    return arr.map((x: any) => ({ id: String(x.id || uid()), html: String(x.html || ""), ts: String(x.ts || ts()) })).slice(-2000);
  } catch { return seedLog(); }
}
function saveLog(list: Line[]) {
  try { localStorage.setItem("term:log", JSON.stringify(list.slice(-2000))); } catch {}
}
function seedLog(): Line[] {
  return [
    { id: uid(), ts: ts(), html: renderColors("<dim>Welcome to</dim> <green>Web Terminal</green> — type <yellow>help</yellow> to get started.") },
  ];
}
function loadKV<T>(k: string, fallback: T): T {
  try { const raw = localStorage.getItem(k); return raw ? (JSON.parse(raw) as T) : fallback; } catch { return fallback; }
}
function saveKV(k: string, v: any) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} }

/* ----------------------- Ambient React (no imports) ---------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useRef<T>(i: T | null): { current: T | null };