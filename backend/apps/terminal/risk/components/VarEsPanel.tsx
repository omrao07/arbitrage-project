// app/components/varespanel.tsx
// No imports. Self-contained VaR / ES panel with historical & parametric methods.
// - Paste a series of daily returns (%) or daily P&L (absolute)
// - Choose method (Historical / Parametric-Normal), confidence, horizon (days)
// - Shows VaR and ES (CVaR), mean, σ, N; tiny SVG histogram with VaR/ES markers
// - Export visible metrics + distribution to CSV; load sample data
// - Dark-mode friendly; inline styles only

"use client";

type Method = "historical" | "parametric";
type Kind = "returns" | "pnl";

type Props = {
  title?: string;
  unit?: string;            // currency symbol for P&L (e.g., "₹", "$")
  defaultKind?: Kind;       // "returns" | "pnl"
  defaultMethod?: Method;   // "historical" | "parametric"
  defaultAlphaPct?: number; // e.g., 99
  defaultHorizon?: number;  // days (sqrt-time scaling)
  series?: number[];        // optional initial series (numbers). If defaultKind === "returns", interpret as % (e.g., 0.5 = 0.5%)
  notional?: number;        // used to translate returns into absolute
  dense?: boolean;
};

export default function VaREsPanel({
  title = "VaR · ES",
  unit = "₹",
  defaultKind = "returns",
  defaultMethod = "historical",
  defaultAlphaPct = 99,
  defaultHorizon = 1,
  series,
  notional = 0,
  dense = false,
}: Props) {
  if (typeof window !== "undefined") {
    queueMicrotask(() => {
      const root = document.getElementById("vr-root");
      if (!root) return;

      // Seed inputs
      setVal(root, 'select[name="vr-kind"]', defaultKind);
      setVal(root, 'select[name="vr-method"]', defaultMethod);
      setVal(root, 'input[name="vr-alpha"]', String(defaultAlphaPct));
      setVal(root, 'input[name="vr-h"]', String(defaultHorizon));
      setVal(root, 'input[name="vr-unit"]', unit || "");
      setVal(root, 'input[name="vr-notional"]', notional ? String(notional) : "");

      const ta = root.querySelector<HTMLTextAreaElement>("#vr-series");
      if (ta && !ta.value.trim()) {
        if (series && series.length) {
          ta.value = pretty(series);
        } else {
          // Put a gentle hint only (no sample yet)
          ta.value = "";
        }
      }

      // Wire controls
      const recalc = () => run(root);
      root.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select').forEach((el) => {
        el.addEventListener("input", () => {
          if (el.getAttribute("name") === "vr-kind") syncKindVisibility(root);
          recalc();
        });
        el.addEventListener("change", recalc);
      });
      ta?.addEventListener("input", recalc);

      root.querySelector<HTMLButtonElement>("#vr-load")?.addEventListener("click", () => {
        const kind = getSel<Kind>(root, 'select[name="vr-kind"]') || "returns";
        const t = root.querySelector<HTMLTextAreaElement>("#vr-series");
        if (!t) return;
        if (kind === "returns") {
          t.value = sampleReturnsCSV();
          setVal(root, 'input[name="vr-notional"]', "10000000"); // 1 crore demo
          setVal(root, 'input[name="vr-unit"]', unit || "₹");
        } else {
          t.value = samplePnLCSV(unit || "₹");
        }
        toast(root, "Loaded sample");
        run(root);
      });

      root.querySelector<HTMLButtonElement>("#vr-clear")?.addEventListener("click", () => {
        const t = root.querySelector<HTMLTextAreaElement>("#vr-series"); if (t) t.value = "";
        run(root);
      });

      root.querySelector<HTMLButtonElement>("#vr-export")?.addEventListener("click", () => exportCsv(root));

      // First render
      syncKindVisibility(root);
      run(root);
    });
  }

  return (
    <section id="vr-root" style={{ ...wrap, ...(dense ? { fontSize: 13 } : null) }} data-busy="0">
      <style>{css}</style>
      <div id="vr-toast" style={toastStyle} />

      <header style={head}>
        <div>
          <h3 style={h3}>{title}</h3>
          <p style={sub}>Paste a daily series (returns % or absolute P&amp;L). Choose method, confidence, horizon.</p>
        </div>
        <div style={ctrls}>
          <button id="vr-export" style={btnGhost}>Export CSV</button>
          <button id="vr-load" style={btnGhost}>Load Sample</button>
          <button id="vr-clear" style={btnGhost}>Clear</button>
        </div>
      </header>

      <section style={grid2}>
        {/* Inputs */}
        <div style={card}>
          <h4 style={h4}>Inputs</h4>

          <div style={grid3}>
            <div>
              <label style={lbl}>Series Kind</label>
              <select name="vr-kind" style={input}>
                <option value="returns">Returns (%)</option>
                <option value="pnl">P&amp;L (absolute)</option>
              </select>
            </div>
            <div>
              <label style={lbl}>Method</label>
              <select name="vr-method" style={input}>
                <option value="historical">Historical</option>
                <option value="parametric">Parametric (Normal)</option>
              </select>
            </div>
            <div>
              <label style={lbl}>Confidence %</label>
              <input name="vr-alpha" type="number" step="0.1" min="50" max="99.99" defaultValue={String(defaultAlphaPct)} style={input} />
            </div>
          </div>

          <div style={grid3}>
            <div>
              <label style={lbl}>Horizon (days)</label>
              <input name="vr-h" type="number" step="1" min="1" defaultValue={String(defaultHorizon)} style={input} />
            </div>
            <div data-kind="returns">
              <label style={lbl}>Portfolio Notional</label>
              <input name="vr-notional" type="number" step="1" placeholder="Optional" defaultValue={notional ? String(notional) : ""} style={input} />
            </div>
            <div data-kind="pnl">
              <label style={lbl}>Currency</label>
              <input name="vr-unit" placeholder="₹, $, €…" defaultValue={unit || ""} style={input} />
            </div>
          </div>

          <div>
            <label style={lbl}>Series (comma/space/newline separated)</label>
            <textarea id="vr-series" rows={9} style={ta} placeholder={'Returns (%): e.g.\n0.3\n-0.8\n1.2\n...\n\nor P&L: e.g.\n-45000\n12000\n...'} />
          </div>

          <p style={hint}>Historical VaR/ES use the empirical left tail. Parametric assumes normality. Returns are interpreted in PERCENT units (e.g., <b>0.5</b> = 0.5%).</p>
        </div>

        {/* Results */}
        <div style={card}>
          <h4 style={h4}>Results</h4>

          <div id="vr-summary" style={sumRow}>
            <div style={pill}><span style={pillLbl}>VaR</span><b id="vr-var">—</b></div>
            <div style={pill}><span style={pillLbl}>ES (CVaR)</span><b id="vr-es">—</b></div>
            <div style={pill}><span style={pillLbl}>Mean</span><b id="vr-mean">—</b></div>
            <div style={pill}><span style={pillLbl}>σ</span><b id="vr-sd">—</b></div>
            <div style={pill}><span style={pillLbl}>N</span><b id="vr-n">—</b></div>
            <div style={pill}><span style={pillLbl}>Updated</span><b id="vr-upd">—</b></div>
          </div>

          <div id="vr-chart-wrap" style={chartWrap}>
            <svg id="vr-chart" viewBox="0 0 600 220" preserveAspectRatio="none" style={svgBox} />
            <div style={legend}>
              <span style={legItem}><i style={{ ...legSwatch, background: "#111" }} /> VaR</span>
              <span style={legItem}><i style={{ ...legSwatch, background: "#b42318" }} /> ES</span>
            </div>
          </div>

          <div style={{ marginTop: 6, color: "#6b7280", fontSize: 12 }}>
            <span id="vr-foot">—</span>
          </div>
        </div>
      </section>
    </section>
  );
}

/* --------------------------------- Run/Render --------------------------------- */

function run(root: HTMLElement) {
  try {
    root.setAttribute("data-busy", "1");

    const kind = getSel<Kind>(root, 'select[name="vr-kind"]') || "returns";
    const method = getSel<Method>(root, 'select[name="vr-method"]') || "historical";
    const alphaPct = clampNum(numVal(root, 'input[name="vr-alpha"]'), 50, 99.99) ?? 99;
    const horizon = Math.max(1, Math.floor(numVal(root, 'input[name="vr-h"]') ?? 1));
    const unit = textVal(root, 'input[name="vr-unit"]') || "₹";
    const notional = numVal(root, 'input[name="vr-notional"]') ?? 0;

    const parsed = parseSeries(getText(root, "#vr-series"));
    const data = parsed.values;
    const N = data.length;
    setTxt(root, "#vr-n", String(N));
    setTxt(root, "#vr-upd", new Date().toLocaleString());

    if (!N) {
      setTxt(root, "#vr-var", "—");
      setTxt(root, "#vr-es", "—");
      setTxt(root, "#vr-mean", "—");
      setTxt(root, "#vr-sd", "—");
      setTxt(root, "#vr-foot", parsed.msg || "Paste a series to compute metrics.");
      drawEmpty(root);
      return;
    }

    const alpha = alphaPct / 100;

    // Normalize to a distribution of daily losses (positive = loss)
    // For returns: numbers are in percent units (e.g., 0.5 => 0.5%), convert to fraction
    // Loss series L:
    //   kind=returns:   L = -(r% / 100) * (notional || 1)  (absolute if notional>0 else fraction)
    //   kind=pnl:       L = max(0, -pnl)? No — P&L includes gains/losses; define loss = -pnl
    const scale = kind === "returns" ? ((notional || 0) > 0 ? notional : 1) : 1;
    const isAbs = kind === "pnl" || (kind === "returns" && notional > 0);

    const losses = data.map((x) => {
      if (kind === "returns") {
        const frac = x / 100; // percent -> fraction
        return -(frac) * scale;
      }
      return -x; // P&L -> loss
    });

    // Summary stats (on the native daily series used for method)
    let mu: number, sd: number;
    if (method === "parametric") {
      // For parametric, base the model on the same transform as losses/scale.
      // If isAbs (currency), we'll model absolute losses via transformed returns/P&L.
      const x = losses.slice();
      mu = mean(x);
      sd = stdev(x, mu);
    } else {
      // Historical stats for display only (on losses)
      const x = losses.slice();
      mu = mean(x);
      sd = stdev(x, mu);
    }

    // sqrt-time scaling
    const t = Math.sqrt(Math.max(1, horizon));

    // Compute VaR/ES
    let varAbs = 0, esAbs = 0;

    if (method === "historical") {
      const L = losses.slice().sort((a, b) => a - b); // ascending (more negative pnl -> bigger loss if negative? Our losses already + for losses)
      // Our 'losses' are positive for actual losses when P&L negative; gains are negative numbers.
      // Historical left tail: the alpha-quantile of LOSS distribution.
      varAbs = quantile(L, alpha) * t;
      // ES = average of tail beyond VaR
      const idx = tailStartIndex(L, alpha);
      const tail = L.slice(idx).map((v) => v * t);
      esAbs = tail.length ? mean(tail) : varAbs;
    } else {
      // Parametric Normal on losses
      const z = invNorm(alpha);
      // For losses L ~ N(mu, sd). VaR_alpha = mu + z*sd; ES = mu + sd * φ(z) / (1-α)
      varAbs = (mu + z * sd) * t;
      esAbs = (mu + (phi(z) / (1 - alpha)) * sd) * t;
    }

    // If returns without notional, display % numbers (fractional of 1)
    const showUnit = isAbs ? (textVal(root, 'input[name="vr-unit"]') || unit || "") : "";
    const toDisplay = (x: number) => isAbs ? money(x, showUnit) : pct(x); // note: x is loss in absolute or fraction

    setTxt(root, "#vr-var", paintLoss(toDisplay(varAbs), varAbs));
    setTxt(root, "#vr-es", paintLoss(toDisplay(esAbs), esAbs));
    setTxt(root, "#vr-mean", toDisplay(mu));
    setTxt(root, "#vr-sd", toDisplay(sd));
    setTxt(root, "#vr-foot", footnote(kind, method, alphaPct, horizon, isAbs ? showUnit : "%"));

    // Draw histogram with markers
    drawHist(root, losses, { varAbs, esAbs, isAbs, unit: showUnit });

  } catch (e) {
    console.error(e);
    toast(root, "Computation error");
  } finally {
    root.removeAttribute("data-busy");
  }
}

/* --------------------------------- Drawing --------------------------------- */

function drawEmpty(root: HTMLElement) {
  const svg = root.querySelector<SVGSVGElement>("#vr-chart");
  if (!svg) return;
  svg.innerHTML = "";
  // Frame only
  const w = 600, h = 220, pad = 28;
  const frame = `<rect x="${pad}" y="${10}" width="${w - pad * 2}" height="${h - pad - 20}" fill="none" stroke="var(--b)" />`;
  svg.innerHTML = frame;
}

function drawHist(
  root: HTMLElement,
  losses: number[],
  opts: { varAbs: number; esAbs: number; isAbs: boolean; unit: string },
) {
  const svg = root.querySelector<SVGSVGElement>("#vr-chart");
  if (!svg) return;
  const w = 600, h = 220;
  const padL = 40, padR = 16, padT = 10, padB = 28;

  const minV = Math.min(...losses);
  const maxV = Math.max(...losses);
  // Expand a bit
  const lo = minV - (maxV - minV) * 0.05;
  const hi = maxV + (maxV - minV) * 0.05;

  const bins = 28;
  const bw = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of losses) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / bw)));
    counts[i]++;
  }
  const maxC = Math.max(1, ...counts);

  const x = (v: number) => padL + ((v - lo) / (hi - lo)) * (w - padL - padR);
  const y = (c: number) => padT + (1 - c / maxC) * (h - padT - padB);

  let bars = "";
  for (let i = 0; i < bins; i++) {
    const x0 = x(lo + i * bw);
    const x1 = x(lo + (i + 1) * bw);
    const y1 = y(counts[i]);
    bars += `<rect x="${x0 + 0.5}" y="${y1}" width="${Math.max(0, x1 - x0 - 1)}" height="${h - padB - y1}" fill="var(--bar)" />`;
  }

  const axes = `
    <rect x="${padL}" y="${padT}" width="${w - padL - padR}" height="${h - padT - padB}" fill="none" stroke="var(--b)" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="var(--b)" />
  `;

  // Markers
  const vx = x(opts.varAbs);
  const ex = x(opts.esAbs);
  const varLine = `<line x1="${vx}" y1="${padT}" x2="${vx}" y2="${h - padB}" stroke="#111" stroke-width="1.5" />
                   <text x="${vx + 4}" y="${padT + 12}" font-size="11" fill="currentColor">VaR</text>`;
  const esLine = `<line x1="${ex}" y1="${padT}" x2="${ex}" y2="${h - padB}" stroke="#b42318" stroke-dasharray="4,3" stroke-width="1.5" />
                  <text x="${ex + 4}" y="${padT + 24}" font-size="11" fill="currentColor">ES</text>`;

  // X ticks: 5 ticks
  const ticks = 5;
  let xt = "";
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (i / ticks) * (hi - lo);
    const tx = x(v);
    const label = opts.isAbs ? money(v, opts.unit) : pct(v);
    xt += `<line x1="${tx}" y1="${h - padB}" x2="${tx}" y2="${h - padB + 4}" stroke="var(--b)"/>
           <text x="${tx}" y="${h - padB + 16}" font-size="10.5" text-anchor="middle" fill="currentColor">${label}</text>`;
  }

  svg.innerHTML = `<g>${axes}${bars}${varLine}${esLine}${xt}</g>`;
}

/* --------------------------------- Export --------------------------------- */

function exportCsv(root: HTMLElement) {
  const kind = getSel<Kind>(root, 'select[name="vr-kind"]') || "returns";
  const method = getSel<Method>(root, 'select[name="vr-method"]') || "historical";
  const alphaPct = clampNum(numVal(root, 'input[name="vr-alpha"]'), 50, 99.99) ?? 99;
  const horizon = Math.max(1, Math.floor(numVal(root, 'input[name="vr-h"]') ?? 1));
  const unit = textVal(root, 'input[name="vr-unit"]') || "₹";
  const notional = numVal(root, 'input[name="vr-notional"]') ?? 0;

  const parsed = parseSeries(getText(root, "#vr-series"));
  const data = parsed.values;

  if (!data.length) return toast(root, "Nothing to export");

  // Recompute losses for export
  const scale = kind === "returns" ? ((notional || 0) > 0 ? notional : 1) : 1;
  const losses = data.map((x) => kind === "returns" ? -(x / 100) * scale : -x);

  const mu = mean(losses);
  const sd = stdev(losses, mu);
  const t = Math.sqrt(Math.max(1, horizon));
  const z = invNorm(alphaPct / 100);
  const varH = quantile(losses.slice().sort((a, b) => a - b), alphaPct / 100) * t;
  const esH = (function () {
    const L = losses.slice().sort((a, b) => a - b);
    const idx = tailStartIndex(L, alphaPct / 100);
    const tail = L.slice(idx).map((v) => v * t);
    return tail.length ? mean(tail) : varH;
  })();
  const varP = (mu + z * sd) * t;
  const esP = (mu + (phi(z) / (1 - alphaPct / 100)) * sd) * t;

  const head = ["kind", "method", "alphaPct", "horizonDays", "unit", "notional", "VaR_hist", "ES_hist", "VaR_param", "ES_param", "mean", "sd", "N"];
  const row = [kind, method, alphaPct, horizon, unit, notional, varH, esH, varP, esP, mu, sd, data.length];

  const rows = [["series"], ...data.map((v) => [v])];

  const csv =
    [head, row].map((r) => r.map(csvEsc).join(",")).join("\n") +
    "\n\n" +
    rows.map((r) => r.map(csvEsc).join(",")).join("\n");

  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `vares_${stamp(new Date())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* --------------------------------- Helpers --------------------------------- */

function syncKindVisibility(root: HTMLElement) {
  const kind = getSel<Kind>(root, 'select[name="vr-kind"]') || "returns";
  root.querySelectorAll<HTMLElement>('[data-kind="returns"]').forEach((el) => (el.style.display = kind === "returns" ? "" : "none"));
  root.querySelectorAll<HTMLElement>('[data-kind="pnl"]').forEach((el) => (el.style.display = kind === "pnl" ? "" : "none"));
}

function footnote(kind: Kind, method: Method, alphaPct: number, h: number, unit: string) {
  const typ = method === "historical" ? "Empirical" : "Normal";
  const label = kind === "returns" && unit === "%" ? "%" : unit;
  return `${typ} VaR/ES at ${alphaPct}% over ${h} day(s). Units: ${label}. Positive = loss.`;
}

function parseSeries(s: string): { values: number[]; msg?: string } {
  const raw = (s || "").trim();
  if (!raw) return { values: [] };
  // Allow CSV with currency symbols or %; strip any non-number tokens except minus, dot, exponent, and percent
  const lines = raw.split(/[\s,;]+/).map((t) => t.trim()).filter(Boolean);
  const vals: number[] = [];
  for (const tok of lines) {
    const clean = tok.replace(/[^0-9eE+.\-%]/g, "");
    if (!clean) continue;
    const isPct = /%$/.test(clean);
    const num = Number(clean.replace(/%$/, ""));
    if (!Number.isFinite(num)) continue;
    vals.push(isPct ? num : num);
  }
  return { values: vals, msg: undefined };
}

function quantile(sortedAsc: number[], p: number) {
  const n = sortedAsc.length;
  if (!n) return NaN;
  const idx = (n - 1) * p;
  const i = Math.floor(idx);
  const frac = idx - i;
  if (i + 1 < n) return sortedAsc[i] * (1 - frac) + sortedAsc[i + 1] * frac;
  return sortedAsc[n - 1];
}
function tailStartIndex(sortedAsc: number[], p: number) {
  const n = sortedAsc.length;
  const idx = Math.floor((n - 1) * p);
  return Math.max(0, Math.min(n - 1, idx));
}

function mean(a: number[]) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }
function stdev(a: number[], m = mean(a)) {
  if (a.length < 2) return 0;
  let acc = 0;
  for (const v of a) { const d = v - m; acc += d * d; }
  return Math.sqrt(acc / (a.length - 1));
}

// Standard normal helpers
function phi(z: number) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }
// Acklam's inverse-normal approximation
function invNorm(p: number) {
  // clamp
  const pp = Math.max(1e-12, Math.min(1 - 1e-12, p));
  const a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,  1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00];
  const b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,  6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00];
  const d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,  3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r, x;
  if (pp < plow) {
    q = Math.sqrt(-2 * Math.log(pp));
    x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (pp > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - pp));
    x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
          ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else {
    q = pp - 0.5;
    r = q * q;
    x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }
  // one step Halley refinement
  const e = 0.5 * (1 + erf(x / Math.SQRT2)) - pp;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
  return x - u / (1 + x * u / 2);
}
// erf approximation
function erf(x: number) {
  const sgn = x < 0 ? -1 : 1;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sgn * y;
}

function pct(x: number) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}
function money(n: number, unit: string) {
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  if (v >= 1_000_000_000) return `${sign}${unit}${(v / 1_000_000_000).toFixed(2)}B`;
  if (v >= 1_000_000) return `${sign}${unit}${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${sign}${unit}${(v / 1_000).toFixed(2)}k`;
  return `${sign}${unit}${v.toFixed(2)}`;
}
function paintLoss(s: string, n: number) {
  const color = n >= 0 ? "#b42318" : "#067647";
  return `<span style="color:${color};font-weight:700">${s}</span>`;
}

function getText(root: HTMLElement, sel: string) {
  const el = root.querySelector<HTMLTextAreaElement>(sel);
  return el ? el.value : "";
}
function setVal(root: HTMLElement, sel: string, v: any) {
  const el = root.querySelector<HTMLInputElement | HTMLSelectElement>(sel);
  if (el) (el as any).value = v ?? "";
}
function setTxt(root: HTMLElement, sel: string, v: string) {
  const el = root.querySelector<HTMLElement>(sel);
  if (el) el.innerHTML = v;
}
function textVal(root: HTMLElement, sel: string) {
  const el = root.querySelector<HTMLInputElement>(sel);
  return el ? (el.value || "").trim() : "";
}
function numVal(root: HTMLElement, sel: string) {
  const v = textVal(root, sel);
  if (!v) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
function getSel<T>(root: HTMLElement, sel: string): T | undefined {
  const el = root.querySelector<HTMLSelectElement>(sel);
  return el ? (el.value as any as T) : undefined;
}
function clampNum(n: number | undefined, lo: number, hi: number) {
  if (!Number.isFinite(n as number)) return undefined;
  return Math.max(lo, Math.min(hi, n!));
}
function csvEsc(s: any) {
  const str = String(s ?? "");
  const needs = /[",\n\r]/.test(str) || /^\s|\s$/.test(str);
  return needs ? `"${str.replace(/"/g, '""')}"` : str;
}
function pretty(x: any) { return JSON.stringify(x, null, 2); }
function stamp(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function toast(root: HTMLElement, msg: string) {
  const el = root.querySelector("#vr-toast") as HTMLElement | null;
  if (!el) return;
  el.textContent = msg;
  el.setAttribute("data-show", "1");
  setTimeout(() => el.removeAttribute("data-show"), 1200);
}

/* --------------------------------- Samples --------------------------------- */

function sampleReturnsCSV() {
  // ~250 pseudo "daily" returns in %, centered near 0 with fat-ish tails
  const out: string[] = [];
  let s = 12345;
  const rand = () => (s = (1103515245 * s + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 260; i++) {
    // Box-Muller
    const u1 = Math.max(1e-6, rand());
    const u2 = Math.max(1e-6, rand());
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const r = z * 1.1 + (rand() - 0.5) * 0.2; // mean≈0, sd≈1.1%
    out.push((r).toFixed(3)); // percent units
  }
  return out.join("\n");
}
function samplePnLCSV(u: string) {
  const out: string[] = [];
  let s = 54321;
  const rand = () => (s = (1664525 * s + 1013904223) % 4294967296) / 4294967296;
  for (let i = 0; i < 260; i++) {
    // heavy-ish tails around 0, scale ₹50k
    const x = (rand() - 0.5) * 2;
    const y = x * (1 + 1.8 * (rand() - 0.5));
    out.push((y * 50000).toFixed(0));
  }
  return out.join("\n");
}

/* --------------------------------- Styles --------------------------------- */

const wrap: any = { display: "grid", gap: 12, padding: 12 };
const head: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" };
const h3: any = { margin: 0, fontSize: 18 };
const h4: any = { margin: 0, fontSize: 16 };
const sub: any = { margin: "4px 0 0", color: "#6b7280", fontSize: 12.5 };

const ctrls: any = { display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" };
const btnGhost: any = { border: "1px solid #e5e7eb", background: "#fff", color: "#111", borderRadius: 10, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const grid2: any = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 };
const grid3: any = { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 };

const card: any = { border: "1px solid var(--b)", borderRadius: 14, background: "var(--bg)", padding: 12, display: "grid", gap: 10 };

const lbl: any = { fontSize: 12, color: "#6b7280" };
const input: any = { width: "100%", height: 32, padding: "4px 8px", borderRadius: 10, border: "1px solid var(--b)", outline: "none", background: "#fff" };
const ta: any = { width: "100%", border: "1px solid var(--b)", borderRadius: 10, padding: 8, minHeight: 120, fontFamily: "ui-monospace,Menlo,monospace", fontSize: 12.5 };
const hint: any = { margin: 0, color: "#6b7280", fontSize: 12 };

const sumRow: any = { display: "flex", gap: 8, flexWrap: "wrap" };
const pill: any = { display: "grid", gap: 2, border: "1px solid #e5e7eb", background: "#fff", borderRadius: 10, padding: "6px 10px", minWidth: 120, textAlign: "right" };
const pillLbl: any = { color: "#6b7280", fontSize: 11 };

const chartWrap: any = { position: "relative", border: "1px solid var(--b)", borderRadius: 12, padding: 8 };
const svgBox: any = { width: "100%", height: 240, display: "block" };
const legend: any = { position: "absolute", right: 10, top: 8, display: "flex", gap: 10, fontSize: 12 };
const legItem: any = { display: "inline-flex", alignItems: "center", gap: 6 };
const legSwatch: any = { width: 10, height: 10, borderRadius: 2, display: "inline-block", border: "1px solid rgba(0,0,0,.12)" };

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
  zIndex: 60,
};

const css = `
  :root { --b:#e5e7eb; --bg:#fff; --bar:#dbeafe; }
  #vr-root[data-busy="1"] { opacity:.6; pointer-events:none; }
  #vr-toast[data-show="1"] { opacity: 1; }

  @media (max-width: 1000px) { section > .grid2 { grid-template-columns: 1fr !important; } }

  @media (prefers-color-scheme: dark) {
    :root { --b:rgba(255,255,255,.12); --bg:#0b0b0c; --bar:#111827; }
    section, table, th, td, input, textarea { color:#e5e7eb !important; }
    input, select, textarea { background:#0b0b0c !important; border-color:var(--b) !important; }
  }
`;
