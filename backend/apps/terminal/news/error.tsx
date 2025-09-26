// app/news/error.tsx (or app/market/news/error.tsx)
// No imports. Client error boundary for News routes.

"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const details =
    (error?.name ? error.name + ": " : "") +
    (error?.message || "Unknown error") +
    (error?.digest ? `\nDigest: ${error.digest}` : "") +
    (error?.stack ? `\n\n${error.stack}` : "");

  function copyDetails() {
    try {
      navigator.clipboard?.writeText(details);
      toast("Copied error details");
    } catch {
      toast("Copy failed");
    }
  }

  function toast(msg: string) {
    const el = document.getElementById("news-error-toast");
    if (!el) return;
    el.textContent = msg;
    el.style.opacity = "1";
    setTimeout(() => (el.style.opacity = "0"), 1200);
  }

  return (
    <section role="alert" aria-live="assertive" style={wrap}>
      <style>{css}</style>
      <div id="news-error-toast" style={toastStyle} />

      <div style={card}>
        <div style={headerRow}>
          <span aria-hidden="true" style={icon}>
            📰
          </span>
          <div>
            <h2 style={h2}>News failed to load</h2>
            <p style={sub}>
              Something went wrong while fetching the latest stories. You can try
              again or view technical details below.
            </p>
          </div>
        </div>

        <div style={btnRow}>
          <button style={primaryBtn} onClick={() => reset()}>
            Try again
          </button>
          <a href="/market" style={btn}>
            Go to Market
          </a>
          <button style={btnAlt} onClick={copyDetails}>
            Copy details
          </button>
        </div>

        <details style={detailsBox}>
          <summary style={summaryStyle}>Technical details</summary>
          <pre style={pre}>{details}</pre>
        </details>
      </div>
    </section>
  );
}

/* ---------------- styles ---------------- */
const wrap: any = {
  display: "grid",
  placeItems: "center",
  minHeight: "60vh",
  padding: 16,
};

const card: any = {
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 16,
  boxShadow: "0 8px 28px rgba(0,0,0,0.06)",
  padding: 18,
  maxWidth: 720,
  width: "100%",
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

const headerRow: any = { display: "flex", gap: 12, alignItems: "center" };
const icon: any = { fontSize: 28 };
const h2: any = { margin: 0, fontSize: 20, lineHeight: "26px" };
const sub: any = { margin: "4px 0 0", color: "#555", fontSize: 13 };

const btnRow: any = { display: "flex", gap: 8, flexWrap: "wrap", marginTop: 4 };
const primaryBtn: any = {
  appearance: "none",
  border: "1px solid #111",
  background: "#111",
  color: "#fff",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
};
const btn: any = {
  appearance: "none",
  border: "1px solid #d4d4d8",
  background: "#f4f4f5",
  color: "#111",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
};
const btnAlt: any = { ...btn };

const detailsBox: any = {
  marginTop: 6,
  borderTop: "1px solid #eee",
  paddingTop: 8,
};
const summaryStyle: any = {
  cursor: "pointer",
  fontWeight: 600,
  marginBottom: 6,
};
const pre: any = {
  margin: 0,
  padding: 10,
  background: "#0b0b0c",
  color: "#fff",
  borderRadius: 10,
  overflowX: "auto",
  fontSize: 12,
  lineHeight: "18px",
};

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
  @media (prefers-color-scheme: dark) {
    pre { background: #111827 !important; }
    a { color: #9ecaff; }
    section > div[style] {
      background: #0b0b0c !important;
      border-color: rgba(255,255,255,0.08) !important;
      box-shadow: 0 8px 28px rgba(0,0,0,0.6) !important;
    }
  }
`;
