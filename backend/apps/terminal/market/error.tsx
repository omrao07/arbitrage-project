"use client";

import React, { useEffect, useState } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    // Optional: send to your logging endpoint
    // fetch("/api/log", { method: "POST", body: JSON.stringify({ message: error.message, stack: error.stack, digest: error.digest, route: "market" }) });
    console.error("[market/error]", error);
  }, [error]);

  return (
    <div style={wrap}>
      <div style={card} role="alert" aria-live="assertive">
        <div style={iconRow}>
          <div style={icon}>!</div>
          <h1 style={title}>Market module crashed</h1>
        </div>

        <p style={msg}>
          {error?.message || "An unexpected error occurred while loading market data."}
        </p>

        {error?.digest ? (
          <p style={meta}>
            <strong>Digest:</strong> <code>{error.digest}</code>
          </p>
        ) : null}

        <div style={actions}>
          <button style={primaryBtn} onClick={() => reset()}>
            Retry
          </button>
          <button
            style={secondaryBtn}
            onClick={() => setShowDetails((s) => !s)}
            aria-expanded={showDetails}
            aria-controls="market-error-details"
          >
            {showDetails ? "Hide details" : "Show details"}
          </button>
          <a href="/" style={linkBtn}>Home</a>
        </div>

        {showDetails && (
          <pre id="market-error-details" style={pre}>
            {error?.stack || "No stack trace available."}
          </pre>
        )}
      </div>
    </div>
  );
}

/* ---------- inline styles (no external deps) ---------- */
const wrap: React.CSSProperties = {
  minHeight: "60vh",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 24,
  background: "linear-gradient(180deg, rgba(0,0,0,0.02) 0%, rgba(0,0,0,0.05) 100%)",
};

const card: React.CSSProperties = {
  maxWidth: 760,
  width: "100%",
  background: "#fff",
  border: "1px solid rgba(0,0,0,0.08)",
  borderRadius: 14,
  boxShadow: "0 6px 24px rgba(0,0,0,0.06)",
  padding: 20,
};

const iconRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  marginBottom: 8,
};

const icon: React.CSSProperties = {
  width: 28,
  height: 28,
  borderRadius: "50%",
  background: "#ffe8e6",
  color: "#b42318",
  display: "grid",
  placeItems: "center",
  fontWeight: 700,
};

const title: React.CSSProperties = {
  margin: 0,
  fontSize: 18,
  lineHeight: "24px",
};

const msg: React.CSSProperties = {
  margin: "8px 0 4px",
  color: "#444",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
};

const meta: React.CSSProperties = {
  margin: "6px 0 16px",
  color: "#666",
  fontSize: 13,
};

const actions: React.CSSProperties = {
  display: "flex",
  gap: 10,
  marginBottom: 12,
  flexWrap: "wrap",
};

const baseBtn: React.CSSProperties = {
  appearance: "none",
  border: "1px solid rgba(0,0,0,0.12)",
  borderRadius: 10,
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: 14,
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
};

const primaryBtn: React.CSSProperties = {
  ...baseBtn,
  background: "#111",
  color: "#fff",
  borderColor: "#111",
};

const secondaryBtn: React.CSSProperties = {
  ...baseBtn,
  background: "#fafafa",
  color: "#111",
};

const linkBtn: React.CSSProperties = {
  ...baseBtn,
  background: "#f4f4f5",
  color: "#111",
};

const pre: React.CSSProperties = {
  marginTop: 8,
  padding: 12,
  background: "#0a0a0a",
  color: "#eaeaea",
  borderRadius: 10,
  overflowX: "auto",
  fontSize: 12.5,
  lineHeight: "18px",
};
