// app/portfolio/error.tsx
// No imports. Minimal client error UI for the /portfolio route.
// - Shows a friendly message with (optional) digest and stack
// - "Try again" uses Next's reset() if provided, otherwise reloads
// - "Copy details" copies a plain-text dump for debugging
// - Inline styles only

"use client";

type Props = {
  error?: { name?: string; message?: string; stack?: string; digest?: string } | any;
  reset?: () => void;
};

export default function PortfolioError({ error, reset }: Props) {
  // best-effort console reporting (safe without hooks)
  try { if (error) console.error("[portfolio:error]", error); } catch {}

  const title = "Something went wrong";
  const msg =
    (error && (error.message || error.toString?.())) ||
    "We couldn't load your portfolio right now.";
  const digest = error?.digest;
  const stack = typeof error?.stack === "string" ? error.stack : "";

  async function onCopy() {
    const text = [
      "Portfolio Error",
      `Time: ${new Date().toISOString()}`,
      error?.name ? `Name: ${error.name}` : "",
      `Message: ${msg}`,
      digest ? `Digest: ${digest}` : "",
      stack ? "\nStack:\n" + stack : "",
    ]
      .filter(Boolean)
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      toast("Details copied");
    } catch {
      toast("Copy failed");
    }
  }

  function onRetry() {
    if (typeof reset === "function") {
      try { reset(); } catch { location.reload(); }
    } else {
      location.reload();
    }
  }

  return (
    <section style={wrap} aria-live="polite" aria-atomic="true">
      <style>{css}</style>
      <div id="perr-toast" style={toastStyle} />

      <div style={card}>
        <header style={head}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span aria-hidden="true" style={icon}>⚠️</span>
            <div>
              <h2 style={h2}>{title}</h2>
              <p style={sub}>Please try again in a moment.</p>
            </div>
          </div>
          <div style={actions}>
            <button onClick={onRetry} style={primaryBtn}>Try again</button>
            <button onClick={onCopy} style={ghostBtn}>Copy details</button>
          </div>
        </header>

        <div style={box}>
          <div style={label}>Message</div>
          <p style={mono}>{msg}</p>
          {digest ? (
            <p style={digestRow}>
              <span style={label}>Digest</span>
              <code style={code}>{digest}</code>
            </p>
          ) : null}

          <details style={detailsBox}>
            <summary style={summary}>Technical details</summary>
            {stack ? (
              <pre style={pre}>{stack}</pre>
            ) : (
              <p style={muted}>No stack trace available.</p>
            )}
          </details>
        </div>
      </div>
    </section>
  );
}

/* -------- tiny helpers & styles -------- */

function toast(msg: string) {
  const el = document.getElementById("perr-toast");
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

const wrap: any = { padding: 16, display: "grid", placeItems: "center" };
const card: any = {
  width: "min(860px, 96vw)",
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 16,
  boxShadow: "0 12px 48px rgba(0,0,0,0.08)",
  padding: 16,
  display: "grid",
  gap: 12,
};

const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" };
const icon: any = { fontSize: 22, lineHeight: 1 };
const h2: any = { margin: 0, fontSize: 18, lineHeight: "24px" };
const sub: any = { margin: "2px 0 0", color: "#555", fontSize: 13 };

const actions: any = { display: "flex", gap: 8, alignItems: "center" };
const primaryBtn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 10, padding: "8px 12px", cursor: "pointer", fontSize: 13 };
const ghostBtn: any = { border: "1px solid #d4d4d8", background: "#fff", color: "#111", borderRadius: 10, padding: "8px 12px", cursor: "pointer", fontSize: 13 };

const box: any = { display: "grid", gap: 8 };
const label: any = { fontWeight: 600, fontSize: 12, color: "#6b7280" };
const mono: any = { fontFamily: "ui-monospace, Menlo, Monaco, monospace", fontSize: 13, margin: 0 };
const digestRow: any = { display: "flex", alignItems: "center", gap: 8, margin: 0 };
const code: any = { background: "#f4f4f5", border: "1px solid #e5e7eb", padding: "2px 6px", borderRadius: 8, fontFamily: "ui-monospace, Menlo, Monaco, monospace", fontSize: 12.5 };

const detailsBox: any = { borderTop: "1px dashed #eee", paddingTop: 8, marginTop: 4 };
const summary: any = { cursor: "pointer", fontWeight: 600, fontSize: 13, color: "#111" };
const pre: any = { margin: 0, overflow: "auto", maxHeight: "40vh", background: "#0b0b0c", color: "#e5e7eb", padding: 10, borderRadius: 12, fontSize: 12.5 };

const muted: any = { color: "#6b7280", fontSize: 13, margin: 0 };

const toastStyle: any = {
  position: "fixed",
  right: 16,
  bottom: 16,
  background: "#111",
  color: "#fff",
  padding: "8px 12px",
  borderRadius: 10,
  opacity: 0,
  transition: "opacity .25s ease",
  pointerEvents: "none",
  zIndex: 50,
};

const css = `
  #perr-toast[data-show="1"] { opacity: 1 !important; }
  @media (prefers-color-scheme: dark) {
    section > div[style] { background: #0b0b0c !important; border-color: rgba(255,255,255,.08) !important; color: #e5e7eb; }
    code { background: #111214 !important; border-color: rgba(255,255,255,.12) !important; color: #e5e7eb !important; }
    summary { color: #e5e7eb !important; }
    button { color: inherit; }
  }
`;
