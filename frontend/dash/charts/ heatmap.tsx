'use client';
import React, { useEffect, useRef, useState } from 'react';

export default function HeatmapSafe() {
  // mock 10x14 matrix in 0..1
  const R = 10, C = 14;
  const rows = Array.from({ length: R }, (_, i) => `R${i + 1}`);
  const cols = Array.from({ length: C }, (_, j) => `C${j + 1}`);
  const [mat] = useState<number[][]>(
    Array.from({ length: R }, (_, i) =>
      Array.from({ length: C }, (_, j) => {
        const base = Math.sin(i / 2) * Math.cos(j / 3) * 0.5 + 0.5;
        const noise = (Math.random() - 0.5) * 0.1;
        return clamp(base + noise, 0, 1);
      }),
    ),
  );

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // constants kept inside component to avoid TS config issues
  const W = 920, H = 520;
  const padL = 120, padT = 36, padR = 16, padB = 60;

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const cw = plotW / C;
    const ch = plotH / R;

    // find min/max for colors
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const v = mat[i][j]; if (!isFinite(v)) continue;
      mn = Math.min(mn, v); mx = Math.max(mx, v);
    }
    if (!isFinite(mn)) { mn = 0; mx = 1; }

    // cells
    for (let i = 0; i < R; i++) {
      for (let j = 0; j < C; j++) {
        const v = mat[i][j];
        const x = padL + j * cw;
        const y = padT + i * ch;
        ctx.fillStyle = colorFor(v, mn, mx);
        ctx.fillRect(x, y, cw + 1, ch + 1);
      }
    }

    // row labels
    ctx.font = '12px system-ui';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#374151';
    for (let i = 0; i < R; i++) {
      const y = padT + i * ch + ch / 2;
      ctx.fillText(rows[i], 10, y);
    }

    // col labels (angled)
    ctx.save();
    ctx.textBaseline = 'top';
    for (let j = 0; j < C; j++) {
      const x = padL + j * cw + cw / 2;
      ctx.save();
      ctx.translate(x, padT + plotH + 8);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(cols[j], -20, 0);
      ctx.restore();
    }
    ctx.restore();

    // legend
    const Lx = W - 160, Ly = padT + 8, Lw = 120, Lh = 10;
    for (let i = 0; i < Lw; i++) {
      const v = mn + (i / (Lw - 1)) * (mx - mn);
      ctx.fillStyle = colorFor(v, mn, mx);
      ctx.fillRect(Lx + i, Ly, 1, Lh);
    }
    ctx.strokeStyle = '#9ca3af';
    ctx.strokeRect(Lx, Ly, Lw, Lh);
    ctx.fillStyle = '#6b7280';
    ctx.fillText(mn.toFixed(2), Lx, Ly + 18);
    ctx.fillText(mx.toFixed(2), Lx + Lw - 40, Ly + 18);
  }, [mat]);

  return (
    <div style={S.wrap}>
      <div style={S.header}>
        <h2 style={S.title}>Heatmap (Safe)</h2>
        <span style={S.badge}>{R}×{C}</span>
      </div>
      <div style={{ padding: 12 }}>
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          style={{ width: '100%', maxWidth: 1120, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff' }}
        />
      </div>
    </div>
  );
}

/* --- tiny helpers & styles inline to avoid import issues --- */
function clamp(x: number, a: number, b: number) { return Math.max(a, Math.min(b, x)); }
function colorFor(v: number, vmin: number, vmax: number) {
  const t = clamp((v - vmin) / Math.max(1e-9, vmax - vmin), 0, 1);
  const seg = t < 0.33 ? 0 : t < 0.66 ? 1 : 2;
  const local = seg === 0 ? t / 0.33 : seg === 1 ? (t - 0.33) / 0.33 : (t - 0.66) / 0.34;
  let r = 0, g = 0, b = 0;
  if (seg === 0) { r = 20; g = Math.round(120 * local + 40); b = Math.round(240 * (1 - local) + 15); }
  else if (seg === 1) { r = Math.round(255 * local * 0.85); g = 160 + Math.round(60 * (1 - local)); b = 40; }
  else { r = 200 + Math.round(55 * local); g = Math.round(150 * (1 - local)); b = 30; }
  return `rgb(${r},${g},${b})`;
}
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee', display: 'flex', alignItems: 'center', gap: 8 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },
};