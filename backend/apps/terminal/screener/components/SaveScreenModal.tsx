"use client";

/**
 * savescreenmodal.tsx
 * Zero-import, self-contained “Save Screen” modal for dashboards.
 *
 * What it does
 * - Lets users name the current view, add description, tags, folder, and visibility
 * - Optional thumbnail preview (pass a dataURL)
 * - Validates required fields, shows inline errors
 * - Keyboard shortcuts: Esc = close, ⌘/Ctrl+S = save
 * - Copies share URL to clipboard on save (optional)
 *
 * Tailwind-only. No external links/imports.
 */

export type SaveScreenPayload = {
  name: string;
  description?: string;
  tags: string[];
  folder?: string;
  visibility: "private" | "team" | "public";
  overwrite: boolean;
  shareUrl?: string;
  thumbnailDataUrl?: string; // optional preview data URL
  metadata?: Record<string, any>;
};

export default function SaveScreenModal({
  open,
  onClose,
  onSave,
  initial,
  shareUrl,
  thumbnailDataUrl,
  metadata,
  canOverwrite = false,
  folders = [],
  title = "Save screen",
}: {
  open: boolean;
  onClose: () => void;
  onSave: (data: SaveScreenPayload) => void;
  initial?: Partial<SaveScreenPayload>;
  shareUrl?: string;
  thumbnailDataUrl?: string;
  metadata?: Record<string, any>;
  canOverwrite?: boolean;
  folders?: string[];
  title?: string;
}) {
  /* ------------------------------- State -------------------------------- */
  const [name, setName] = useState(initial?.name || "");
  const [desc, setDesc] = useState(initial?.description || "");
  const [tags, setTags] = useState<string[]>(initial?.tags || []);
  const [folder, setFolder] = useState(initial?.folder || (folders[0] || ""));
  const [vis, setVis] = useState<"private" | "team" | "public">(initial?.visibility || "private");
  const [overwrite, setOverwrite] = useState<boolean>(initial?.overwrite ?? false);
  const [err, setErr] = useState<{ name?: string }>({});
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        handleSave();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, name, desc, tags, folder, vis, overwrite]);

  useEffect(() => {
    if (!open) return;
    // Reset copied state on open
    setCopied(false);
  }, [open]);

  /* ------------------------------ Actions ------------------------------- */

  function validate(): boolean {
    const e: typeof err = {};
    if (!name.trim()) e.name = "Please enter a name.";
    setErr(e);
    return Object.keys(e).length === 0;
  }

  function handleAddTagFromInput(e: React.KeyboardEvent<HTMLInputElement>) {
    const input = e.currentTarget;
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      const raw = input.value.trim().replace(/,+$/, "");
      if (!raw) return;
      if (!tags.includes(raw)) setTags((t) => [...t, raw].slice(0, 20));
      input.value = "";
    }
    if (e.key === "Backspace" && !input.value && tags.length) {
      setTags((t) => t.slice(0, -1));
    }
  }

  async function handleSave() {
    if (!validate()) return;
    const payload: SaveScreenPayload = {
      name: name.trim(),
      description: desc.trim() || undefined,
      tags,
      folder: folder || undefined,
      visibility: vis,
      overwrite: overwrite && canOverwrite,
      shareUrl,
      thumbnailDataUrl,
      metadata,
    };
    // Optional: copy share URL
    if (shareUrl) {
      try {
        await (navigator as any).clipboard?.writeText(shareUrl);
        setCopied(true);
        setTimeout(() => setCopied(false), 1800);
      } catch {}
    }
    onSave(payload);
  }

  function removeTag(tag: string) {
    setTags((t) => t.filter((x) => x !== tag));
  }

  /* -------------------------------- UI ---------------------------------- */

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      {/* Modal */}
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div
          className="w-full max-w-2xl overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-900 shadow-2xl"
          role="dialog"
          aria-modal="true"
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-neutral-800 px-5 py-3">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-md bg-neutral-800 text-neutral-300">
                <svg width="14" height="14" viewBox="0 0 24 24">
                  <path d="M5 4h14v16H5zM5 8h14M9 4v4" stroke="currentColor" strokeWidth="2" fill="none" />
                </svg>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
            </div>
            <button
              onClick={onClose}
              className="rounded-md p-1 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
              aria-label="Close"
              title="Esc"
            >
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          {/* Body */}
          <div className="grid grid-cols-1 gap-5 p-5 md:grid-cols-3">
            {/* Left form */}
            <div className="md:col-span-2 space-y-4">
              <div>
                <label className="block text-xs text-neutral-400">Name<span className="text-rose-400"> *</span></label>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., FX Dashboard – USD Focus"
                  className={`mt-1 w-full rounded-md border bg-neutral-950 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 outline-none ${
                    err.name ? "border-rose-600" : "border-neutral-700 focus:border-neutral-500"
                  }`}
                  maxLength={120}
                  autoFocus
                />
                {err.name && <p className="mt-1 text-[11px] text-rose-400">{err.name}</p>}
              </div>

              <div>
                <label className="block text-xs text-neutral-400">Description</label>
                <textarea
                  value={desc}
                  onChange={(e) => setDesc(e.target.value)}
                  placeholder="Short summary to help others understand this view…"
                  className="mt-1 h-24 w-full resize-y rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 outline-none focus:border-neutral-500"
                  maxLength={1000}
                />
                <div className="mt-1 text-[11px] text-neutral-500">{desc.length}/1000</div>
              </div>

              <div>
                <label className="block text-xs text-neutral-400">Tags</label>
                <div className="mt-1 flex flex-wrap items-center gap-2 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5">
                  {tags.map((t) => (
                    <span key={t} className="inline-flex items-center gap-1 rounded bg-neutral-800 px-2 py-0.5 text-xs text-neutral-200">
                      {t}
                      <button
                        className="text-neutral-400 hover:text-neutral-200"
                        onClick={() => removeTag(t)}
                        aria-label={`Remove ${t}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  <input
                    onKeyDown={handleAddTagFromInput}
                    placeholder="Type and press Enter"
                    className="min-w-[8ch] flex-1 bg-transparent text-xs text-neutral-200 placeholder:text-neutral-600 outline-none"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label className="block text-xs text-neutral-400">Folder</label>
                  <select
                    value={folder}
                    onChange={(e) => setFolder(e.target.value)}
                    className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 outline-none focus:border-neutral-500"
                  >
                    {folders.length === 0 && <option value="">(No folders)</option>}
                    {folders.map((f) => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-neutral-400">Visibility</label>
                  <div className="mt-1 grid grid-cols-3 gap-2">
                    <RadioChip label="Private" value="private" groupValue={vis} onChange={setVis} />
                    <RadioChip label="Team" value="team" groupValue={vis} onChange={setVis} />
                    <RadioChip label="Public" value="public" groupValue={vis} onChange={setVis} />
                  </div>
                </div>
              </div>

              {canOverwrite && (
                <label className="mt-1 inline-flex items-center gap-2 text-sm text-neutral-300">
                  <input
                    type="checkbox"
                    checked={overwrite}
                    onChange={(e) => setOverwrite(e.target.checked)}
                    className="h-4 w-4 rounded border-neutral-700 bg-neutral-950"
                  />
                  Overwrite if a screen with this name exists
                </label>
              )}
            </div>

            {/* Right preview */}
            <div className="space-y-3">
              <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-neutral-300">Preview</span>
                  {thumbnailDataUrl && (
                    <span className="text-[11px] text-neutral-500">PNG</span>
                  )}
                </div>
                <div className="aspect-video w-full overflow-hidden rounded-md border border-neutral-800 bg-neutral-900">
                  {thumbnailDataUrl ? (
                    <img
                      src={thumbnailDataUrl}
                      alt="Screen preview"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-neutral-500">
                      <svg width="28" height="28" viewBox="0 0 24 24">
                        <rect x="3" y="4" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
                        <path d="M3 14l4-4 5 5 3-3 6 6" stroke="currentColor" strokeWidth="2" fill="none" />
                      </svg>
                      <span className="ml-2 text-xs">No thumbnail</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-neutral-300">Share</span>
                  {copied && <span className="text-[11px] text-emerald-400">Copied!</span>}
                </div>
                <div className="flex items-center gap-2">
                  <input
                    readOnly
                    value={shareUrl || ""}
                    placeholder="No share link yet"
                    className="w-full rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1.5 text-xs text-neutral-300"
                  />
                  <button
                    onClick={async () => {
                      if (!shareUrl) return;
                      try { await (navigator as any).clipboard?.writeText(shareUrl); setCopied(true); setTimeout(()=>setCopied(false), 1500); } catch {}
                    }}
                    className="shrink-0 rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200 hover:bg-neutral-800"
                  >
                    Copy
                  </button>
                </div>
              </div>

              {metadata && (
                <details className="rounded-lg border border-neutral-800 bg-neutral-950 p-3 text-xs text-neutral-300">
                  <summary className="cursor-pointer text-neutral-400">Metadata</summary>
                  <pre className="mt-2 max-h-48 overflow-auto rounded bg-neutral-900 p-2 text-[11px] text-neutral-300">
                    {tryStringify(metadata)}
                  </pre>
                </details>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between border-t border-neutral-800 px-5 py-3">
            <p className="text-[11px] text-neutral-500">Esc to close · ⌘/Ctrl+S to save</p>
            <div className="flex items-center gap-2">
              <button
                onClick={onClose}
                className="rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-800"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="rounded-md border border-emerald-700 bg-emerald-600/20 px-3 py-2 text-sm font-medium text-emerald-300 hover:bg-emerald-600/30"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Subcomponents ------------------------------ */

function RadioChip({
  label, value, groupValue, onChange,
}: {
  label: string;
  value: "private" | "team" | "public";
  groupValue: "private" | "team" | "public";
  onChange: (v: "private" | "team" | "public") => void;
}) {
  const active = groupValue === value;
  return (
    <button
      type="button"
      onClick={() => onChange(value)}
      className={`rounded-md border px-2 py-1 text-xs ${
        active
          ? "border-emerald-600 bg-emerald-600/20 text-emerald-300"
          : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:bg-neutral-800"
      }`}
    >
      {label}
    </button>
  );
}

/* --------------------------------- Utils ---------------------------------- */

function tryStringify(o?: Record<string, any>) {
  try { return JSON.stringify(o ?? {}, null, 2); } catch { return "{}"; }
}

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;