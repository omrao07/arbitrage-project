"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

// -----------------------------
// Types
// -----------------------------
type CurrencyPair = `${string}/${string}`;
type DeltaBucket = "10P" | "25P" | "ATM" | "25C" | "10C";
type Expiry =
  | "1W"
  | "2W"
  | "1M"
  | "2M"
  | "3M"
  | "6M"
  | "9M"
  | "1Y"
  | "18M"
  | "2Y";

type VolSmile = {
  pair: CurrencyPair;
  expiry: Expiry;
  asOf: string;
  vols: Record<DeltaBucket, number>;
};

type VolSurface = {
  pair: CurrencyPair;
  asOf: string;
  smiles: VolSmile[];
};

type FetchParams = {
  pair: CurrencyPair;
  asOf?: string;
  atmMethod?: "delta-neutral" | "vega-weighted";
};

// -----------------------------
// Mock fetcher (replace with real server call)
// -----------------------------
async function fetchVolSurface(params: FetchParams): Promise<VolSurface> {
  const expiries: Expiry[] = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"];
  const deltas: DeltaBucket[] = ["10P", "25P", "ATM", "25C", "10C"];
  const baseLevel = 9.5;
  const now = new Date().toISOString();

  const smiles: VolSmile[] = expiries.map((e, i) => {
    const termBump = Math.log1p(i / 8) * 2.2;
    const skew = 0.6 + i * 0.03;
    const smile: Record<DeltaBucket, number> = {
      "10P": +(baseLevel + termBump + 0.9 * skew + rand(0.15)).toFixed(2),
      "25P": +(baseLevel + termBump + 0.45 * skew + rand(0.12)).toFixed(2),
      ATM: +(baseLevel + termBump + rand(0.1)).toFixed(2),
      "25C": +(baseLevel + termBump + 0.35 * skew + rand(0.12)).toFixed(2),
      "10C": +(baseLevel + termBump + 0.8 * skew + rand(0.15)).toFixed(2),
    };
    return { pair: params.pair, expiry: e, asOf: now, vols: smile };
  });

  return { pair: params.pair, asOf: now, smiles };
}

function rand(mag = 0.1) {
  return (Math.random() - 0.5) * 2 * mag;
}

// -----------------------------
// Constants
// -----------------------------
const DELTAS: DeltaBucket[] = ["10P", "25P", "ATM", "25C", "10C"];
const EXPIRIES: Expiry[] = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"];
const DEFAULT_PAIRS: CurrencyPair[] = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/INR"];

// -----------------------------
// Utilities
// -----------------------------
function toCsv(surface: VolSurface): string {
  const header = ["pair", "asOf", "expiry", ...DELTAS].join(",");
  const rows = surface.smiles.map((s) => {
    return [
      s.pair,
      s.asOf,
      s.expiry,
      ...DELTAS.map((d) => s.vols[d]?.toFixed(4) ?? ""),
    ].join(",");
  });
  return [header, ...rows].join("\n");
}

function downloadFile(filename: string, contents: string, mime = "text/csv") {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function paintHeatmap(
  canvas: HTMLCanvasElement,
  surface: VolSurface,
  atmMethod: "delta-neutral" | "vega-weighted",
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  const rows = DELTAS.length;
  const cols = EXPIRIES.length;
  const cellW = width / cols;
  const cellH = height / rows;

  let vmin = Number.POSITIVE_INFINITY;
  let vmax = Number.NEGATIVE_INFINITY;
  for (const s of surface.smiles) {
    for (const d of DELTAS) {
      const v = s.vols[d];
      if (v !== undefined) {
        vmin = Math.min(vmin, v);
        vmax = Math.max(vmax, v);
      }
    }
  }
  if (!isFinite(vmin) || !isFinite(vmax)) return;
  if (vmax === vmin) vmax = vmin + 0.01;

  const colorAt = (t: number) => {
    if (t <= 0.5) {
      const k = t / 0.5;
      const r = 20 * (1 - k);
      const g = 180 * k + 40;
      const b = 220 * (1 - k) + 40 * k;
      return `rgb(${r|0},${g|0},${b|0})`;
    } else {
      const k = (t - 0.5) / 0.5;
      const r = 60 + 180 * k;
      const g = 220;
      const b = 40;
      return `rgb(${r|0},${g|0},${b|0})`;
    }
  };

  ctx.clearRect(0, 0, width, height);

  for (let y = 0; y < rows; y++) {
    const delta = DELTAS[y];
    for (let x = 0; x < cols; x++) {
      const exp = EXPIRIES[x];
      const s = surface.smiles.find((sm) => sm.expiry === exp);
      if (!s) continue;
      let v = s.vols[delta];
      if (delta === "ATM" && atmMethod === "vega-weighted") {
        v = v + 0.05;
      }
      if (v == null) continue;
      const t = (v - vmin) / (vmax - vmin);
      ctx.fillStyle = colorAt(t);
      ctx.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

// -----------------------------
// Component
// -----------------------------
export default function FxVolSurface({
  initialPair = "EUR/USD",
  supportedPairs = DEFAULT_PAIRS,
}: {
  initialPair?: CurrencyPair;
  supportedPairs?: CurrencyPair[];
}) {
  const [pair, setPair] = useState<CurrencyPair>(initialPair);
  const [expiry, setExpiry] = useState<Expiry>("1M");
  const [delta, setDelta] = useState<DeltaBucket>("ATM");
  const [atmMethod, setAtmMethod] = useState<"delta-neutral" | "vega-weighted">("delta-neutral");
  const [surface, setSurface] = useState<VolSurface | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const load = async () => {
    const data = await fetchVolSurface({ pair, atmMethod });
    setSurface(data);
  };

  useEffect(() => {
    load();
  }, [pair, atmMethod]);

  useEffect(() => {
    if (!surface || !canvasRef.current) return;
    paintHeatmap(canvasRef.current, surface, atmMethod);
  }, [surface, atmMethod]);

  return (
    <div>
      <h2>FX Vol Surface</h2>

      <div>
        <label>
          Pair:
          <select value={pair} onChange={(e) => setPair(e.target.value as CurrencyPair)}>
            {supportedPairs.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>

        <label>
          ATM Method:
          <select value={atmMethod} onChange={(e) => setAtmMethod(e.target.value as any)}>
            <option value="delta-neutral">Delta-neutral</option>
            <option value="vega-weighted">Vega-weighted</option>
          </select>
        </label>

        <button onClick={() => surface && downloadFile("fxvol.csv", toCsv(surface))}>
          Export CSV
        </button>
      </div>

      <canvas ref={canvasRef} style={{ width: "100%", height: "300px", border: "1px solid #ccc" }} />

      <p>Expiry: {expiry} | Delta: {delta}</p>
    </div>
  );
}
