"use client";

/**
 * terminal/layout.tsx
 * Zero-import, self-contained layout shell for the Terminal section (Next.js route layout).
 *
 * Features
 * - Header with title + quick actions
 * - Collapsible sidebar (commands/history/files)
 * - Resizable sidebar (drag handle)
 * - Main content area renders route children (your terminal UI)
 * - Bottom status bar (connection / cwd / clock)
 * - LocalStorage persistence of UI state (sidebar width + collapsed)
 * - Simple hotkeys:
 *     • Cmd/Ctrl+B → toggle sidebar
 *     • Cmd/Ctrl+K → focus quick input
 */

export default function TerminalLayout({ children }: { children: any }) {
  /* ------------------------------ UI State ------------------------------ */
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => loadBool("term:sidebarOpen", true));
  const [sidebarW, setSidebarW] = useState<number>(() => loadNum("term:sidebarW", 300)); // px
  const [dragging, setDragging] = useState(false);
  const [now, setNow] = useState<string>(() => new Date().toLocaleTimeString());

  useEffect(() => saveBool("term:sidebarOpen", sidebarOpen), [sidebarOpen]);
  useEffect(() => saveNum("term:sidebarW", sidebarW), [sidebarW]);

  /* ------------------------------ Resizing ------------------------------ */
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!dragging) return;
      const min = 220, max = 560;
      const next = clamp(e.clientX, min, max);
      setSidebarW(next);
      e.preventDefault();
    }
    function onUp() { setDragging(false); }
    if (dragging) {
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    }
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging]);

  /* ------------------------------- Hotkeys ------------------------------ */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const isMeta = e.ctrlKey || e.metaKey;
      if (!isMeta) return;
      // Cmd/Ctrl+B → toggle sidebar
      if (e.key.toLowerCase() === "b") {
        e.preventDefault();
        setSidebarOpen((v) => !v);
      }
      // Cmd/Ctrl+K → focus quick input
      if (e.key.toLowerCase() === "k") {
        e.preventDefault();
        focusQuickInput();
      }
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  /* ------------------------------- Clock -------------------------------- */
  useEffect(() => {
    const t = setInterval(() => setNow(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(t);
  }, []);

  /* -------------------------------- UI --------------------------------- */
  return (
    <div className="flex min-h-screen w-full flex-col bg-neutral-950 text-neutral-100">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-neutral-200 hover:bg-neutral-800"
            title="Toggle sidebar (Cmd/Ctrl+B)"
          >
            ☰
          </button>
          <h1 className="text-sm font-semibold">Terminal</h1>
          <span className="ml-2 rounded border border-emerald-800 bg-emerald-900/30 px-2 py-0.5 text-[11px] text-emerald-300">
            connected
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <input
              id="term-quick"
              placeholder="Quick action… (Cmd/Ctrl+K)"
              className="w-64 rounded-md border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-sm placeholder:text-neutral-600 focus:w-80 transition-all"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const v = (e.target as HTMLInputElement).value.trim();
                  if (v) {
                    notify(`Run: ${v}`);
                    (e.target as HTMLInputElement).value = "";
                  }
                }
              }}
            />
          </div>
          <button
            className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm hover:bg-neutral-800"
            onClick={() => notify("New tab")}
            title="New Terminal Tab"
          >
            + Tab
          </button>
          <button
            className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm hover:bg-neutral-800"
            onClick={() => notify("Settings")}
            title="Settings"
          >
            ⚙
          </button>
        </div>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`relative h-full border-r border-neutral-800 bg-neutral-925 ${sidebarOpen ? "block" : "hidden"} md:block`}
          style={{ width: sidebarOpen ? sidebarW : 0 } as any}
        >
          {/* Tabs */}
          <div className="flex items-center gap-1 border-b border-neutral-800 px-3 py-2 text-xs">
            <TabButton id="commands" label="Commands" active />
            <TabButton id="history" label="History" />
            <TabButton id="files" label="Files" />
          </div>

          {/* Content */}
          <div className="h-[calc(100%-40px)] overflow-auto p-3">
            <SidebarSection title="Recent commands">
              <ul className="space-y-1 text-xs">
                {["ls -la", "git status", "yarn dev", "kubectl get pods", "htop"].map((cmd, i) => (
                  <li key={i}>
                    <button
                      className="w-full truncate rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1 text-left hover:bg-neutral-800"
                      onClick={() => notify(`Execute: ${cmd}`)}
                    >
                      <span className="mr-2 text-neutral-500">$</span>{cmd}
                    </button>
                  </li>
                ))}
              </ul>
            </SidebarSection>

            <SidebarSection title="Workspaces">
              <div className="space-y-2 text-xs">
                {["app", "packages/ui", "services/api"].map((w) => (
                  <button
                    key={w}
                    className="block w-full truncate rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1 text-left hover:bg-neutral-800"
                    onClick={() => notify(`Open workspace: ${w}`)}
                  >
                    {w}
                  </button>
                ))}
              </div>
            </SidebarSection>
          </div>

          {/* Drag handle */}
          <div
            onMouseDown={() => setDragging(true)}
            className="absolute right-0 top-0 h-full w-1 cursor-col-resize bg-neutral-800/30 hover:bg-neutral-700/60"
            title="Drag to resize"
          />
        </aside>

        {/* Main */}
        <section className="flex-1 overflow-auto">
          {/* Children route content goes here (your terminal UI) */}
          {children}

          {/* If you want a default placeholder when no child renders */}
          {!children && <EmptyTerminalHint />}
        </section>
      </div>

      {/* Status Bar */}
      <footer className="flex items-center justify-between border-t border-neutral-800 bg-neutral-925 px-3 py-2 text-[11px] text-neutral-400">
        <div className="flex items-center gap-3">
          <span className="truncate">cwd: <code className="rounded bg-neutral-900 px-1">~/project</code></span>
          <span>shell: <code className="rounded bg-neutral-900 px-1">bash</code></span>
          <span className="hidden sm:inline">node: <code className="rounded bg-neutral-900 px-1">v18</code></span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400" title="Connected" />
          <span>{now}</span>
        </div>
      </footer>
    </div>
  );
}

/* -------------------------------- Bits ---------------------------------- */

function TabButton({ id, label, active }: { id: string; label: string; active?: boolean }) {
  return (
    <button
      data-id={id}
      className={`rounded-md border px-2 py-1 ${active ? "border-sky-600 bg-sky-600/20 text-sky-300" : "border-neutral-700 bg-neutral-900 text-neutral-300 hover:bg-neutral-800"}`}
    >
      {label}
    </button>
  );
}

function SidebarSection({ title, children }: { title: string; children: any }) {
  return (
    <section className="mb-4">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-neutral-500">{title}</h3>
      {children}
    </section>
  );
}

function EmptyTerminalHint() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="mx-auto max-w-md rounded-xl border border-dashed border-neutral-800 bg-neutral-925 p-6 text-center">
        <div className="mb-2 text-lg font-semibold text-neutral-200">No terminal mounted</div>
        <p className="text-sm text-neutral-400">
          This is the <code>terminal/layout.tsx</code> shell. Render your terminal UI as a child page
          to replace this placeholder.
        </p>
        <div className="mt-3 text-xs text-neutral-500">
          Tip: Toggle the sidebar with <kbd className="rounded bg-neutral-800 px-1">Cmd/Ctrl</kbd> + <kbd className="rounded bg-neutral-800 px-1">B</kbd>.
        </div>
      </div>
    </div>
  );
}

/* ------------------------------- Helpers -------------------------------- */

function clamp(n: number, min: number, max: number) { return Math.max(min, Math.min(max, n)); }
function saveBool(k: string, v: boolean) { try { localStorage.setItem(k, JSON.stringify(!!v)); } catch {} }
function loadBool(k: string, d = false) { try { const r = localStorage.getItem(k); return r ? !!JSON.parse(r) : d; } catch { return d; } }
function saveNum(k: string, v: number) { try { localStorage.setItem(k, JSON.stringify(v|0)); } catch {} }
function loadNum(k: string, d = 300) { try { const r = localStorage.getItem(k); return r ? (JSON.parse(r)|0) : d; } catch { return d; } }
function focusQuickInput() { try { (document.getElementById("term-quick") as HTMLInputElement)?.focus(); } catch {} }
function notify(s: string) { try { console.log("[terminal]", s); } catch {} }

/* ----------------------- Ambient React (no imports) ---------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;