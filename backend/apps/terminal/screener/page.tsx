"use client";

/**
 * screener/page.tsx
 * Clean, zero-import rewrite — fully self-contained screener page.
 *
 * Highlights
 * - Equities / Bonds tabs
 * - Simple filters (text + numeric ranges)
 * - Client-side synthetic universes (replace with your API later)
 * - Sortable columns, pagination, CSV copy/download
 * - No imports. Tailwind only.
 */

export default function ScreenerPage() {
  /* ------------------------------ Page State ------------------------------ */
  const [tab, setTab] = useState<"equities" | "bonds">("equities");
  const [asOf, setAsOf] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Table state
  const [rows, setRows] = useState<any[]>([]);
  const [sortKey, setSortKey] = useState<string>("marketCap");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);
  const pageSize = 50;

  // Equities filters
  const [eqSym, setEqSym] = useState("");
  const [eqSector, setEqSector] = useState("");
  const [eqCountry, setEqCountry] = useState("");
  const [eqMktLo, setEqMktLo] = useState("");
  const [eqMktHi, setEqMktHi] = useState("");
  const [eqPriceLo, setEqPriceLo] = useState("");
  const [eqPriceHi, setEqPriceHi] = useState("");

  // Bonds filters
  const [boIssuer, setBoIssuer] = useState("");
  const [boSector, setBoSector] = useState("");
  const [boCountry, setBoCountry] = useState("");
  const [boRatMin, setBoRatMin] = useState<Rating | "">("");
  const [boRatMax, setBoRatMax] = useState<Rating | "">("");
  const [boOASLo, setBoOASLo] = useState("");
  const [boOASHi, setBoOASHi] = useState("");

  /* ------------------------------- Columns -------------------------------- */

  const EQ_COLUMNS: ColumnDef[] = [
    { key: "symbol", label: "Symbol" },
    { key: "name", label: "Name" },
    { key: "sector", label: "Sector" },
    { key: "country", label: "Country" },
    { key: "price", label: "Price", align: "right" },
    { key: "marketCap", label: "MktCap", align: "right" },
    { key: "volume", label: "Volume", align: "right" },
    { key: "pe", label: "P/E", align: "right" },
    { key: "ivPct", label: "IV%", align: "right" },
    { key: "score", label: "Score", align: "right" },
  ];

  const BO_COLUMNS: ColumnDef[] = [
    { key: "cusip", label: "CUSIP" },
    { key: "issuer", label: "Issuer" },
    { key: "sector", label: "Sector" },
    { key: "country", label: "Country" },
    { key: "coupon", label: "Cp%", align: "right" },
    { key: "maturity", label: "Maturity" },
    { key: "price", label: "Price", align: "right" },
    { key: "yieldToMaturity", label: "YTM%", align: "right" },
    { key: "spreadOAS", label: "OAS bp", align: "right" },
    { key: "rating", label: "Rating" },
  ];

  const columns = tab === "equities" ? EQ_COLUMNS : BO_COLUMNS;

  /* --------------------------------- Run ---------------------------------- */

  async function run() {
    setLoading(true);
    setError(null);
    setRows([]);
    setPage(1);
    const now = new Date().toISOString();
    setAsOf(now);

    try {
      if (tab === "equities") {
        const rng = mulberry32(42);
        const uni = buildEquityMock(2000, rng);
        let out = uni.filter((r) => {
          if (eqSym && !r.symbol.toUpperCase().includes(eqSym.trim().toUpperCase())) return false;
          if (eqSector && !r.sector.toLowerCase().includes(eqSector.trim().toLowerCase())) return false;
          if (eqCountry && r.country.toUpperCase() !== eqCountry.trim().toUpperCase()) return false;
          if (!rangeOk(r.marketCap, [numOrEmpty(eqMktLo), numOrEmpty(eqMktHi)])) return false;
          if (!rangeOk(r.price, [numOrEmpty(eqPriceLo), numOrEmpty(eqPriceHi)])) return false;
          return true;
        });
        out.sort((a: any, b: any) => cmp(a[sortKey], b[sortKey], sortDir));
        setRows(out.slice(0, 1000));
      } else {
        const rng = mulberry32(7);
        const uni = buildBondMock(1500, rng);
        const order: Rating[] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"];
        const idx = (r: Rating) => order.indexOf(r);

        const minIdx = boRatMin ? idx(boRatMin) : -Infinity;
        const maxIdx = boRatMax ? idx(boRatMax) : +Infinity;

        let out = uni.filter((r) => {
          if (boIssuer && !r.issuer.toLowerCase().includes(boIssuer.trim().toLowerCase())) return false;
          if (boSector && r.sector.toLowerCase() !== boSector.trim().toLowerCase()) return false;
          if (boCountry && r.country.toUpperCase() !== boCountry.trim().toUpperCase()) return false;
          const ri = idx(r.rating);
          if (ri < (Number.isFinite(minIdx) ? minIdx : -1e9)) return false;
          if (ri > (Number.isFinite(maxIdx) ? maxIdx : 1e9)) return false;
          if (!rangeOk(r.spreadOAS, [numOrEmpty(boOASLo), numOrEmpty(boOASHi)])) return false;
          return true;
        });
        // Bonds default sort to OAS desc if current sortKey not in schema
        const key = BO_COLUMNS.find((c) => c.key === sortKey) ? sortKey : "spreadOAS";
        out.sort((a: any, b: any) => cmp(a[key], b[key], sortDir));
        setRows(out.slice(0, 1000));
      }
    } catch (e: any) {
      setError(e?.message || String(e) || "Screen failed.");
    } finally {
      setLoading(false);
    }
  }

  /* ---------------------------- Sorting & Paging --------------------------- */

  function toggleSort(k: string) {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir("desc");
    }
  }

  const sorted = useMemo(() => {
    const arr = [...rows];
    const k = sortKey as any;
    arr.sort((a, b) => cmp(a?.[k], b?.[k], sortDir));
    return arr;
  }, [rows, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const view = sorted.slice((page - 1) * pageSize, page * pageSize);

  /* -------------------------------- Actions -------------------------------- */

  function copyCSV() {
    const head = columns.map((c) => c.label).join(",");
    const lines = [head];
    for (const r of view) lines.push(columns.map((c) => csv(formatCell(r[c.key]))).join(","));
    try {
      (navigator as any).clipboard?.writeText(lines.join("\n"));
    } catch {}
  }

  function downloadCSV() {
    const head = columns.map((c) => c.label).join(",");
    const lines = [head];
    for (const r of sorted) lines.push(columns.map((c) => csv(formatCell(r[c.key]))).join(","));
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${tab}-screener-${asOf.slice(0, 19).replace(/[:T]/g, "-")}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function resetFilters() {
    if (tab === "equities") {
      setEqSym(""); setEqSector(""); setEqCountry("");
      setEqMktLo(""); setEqMktHi(""); setEqPriceLo(""); setEqPriceHi("");
    } else {
      setBoIssuer(""); setBoSector(""); setBoCountry("");
      setBoRatMin(""); setBoRatMax(""); setBoOASLo(""); setBoOASHi("");
    }
    setRows([]); setError(null); setPage(1);
  }

  /* --------------------------------- Render -------------------------------- */

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      {/* Header */}
      <header className="border-b border-neutral-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Screener</h1>
            <p className="mt-1 text-sm text-neutral-400">
              Quick filters on a synthetic universe. {asOf && <span>As of {asOf.slice(0, 19).replace("T", " ")}</span>}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={run}
              className="rounded-md border border-emerald-700 bg-emerald-600/20 px-3 py-2 font-medium text-emerald-300 hover:bg-emerald-600/30"
            >
              Run Screen
            </button>
            <button
              onClick={copyCSV}
              className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-200 hover:bg-neutral-800"
            >
              Copy CSV (page)
            </button>
            <button
              onClick={downloadCSV}
              className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-200 hover:bg-neutral-800"
            >
              Download CSV (all)
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 p-6">
        {/* Tabs */}
        <div className="flex items-center gap-2">
          <Tab label="Equities" active={tab === "equities"} onClick={() => { setTab("equities"); setSortKey("marketCap"); setSortDir("desc"); resetFilters(); }} />
          <Tab label="Bonds" active={tab === "bonds"} onClick={() => { setTab("bonds"); setSortKey("spreadOAS"); setSortDir("desc"); resetFilters(); }} />
          {loading && <span className="ml-auto text-xs text-neutral-400">Running…</span>}
        </div>

        {/* Filters */}
        {tab === "equities" ? (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Field label="Symbol">
              <input value={eqSym} onChange={(e) => setEqSym(e.target.value)} placeholder="e.g., AAPL"
                     className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </Field>
            <Field label="Sector">
              <input value={eqSector} onChange={(e) => setEqSector(e.target.value)} placeholder="e.g., Tech"
                     className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </Field>
            <Field label="Country">
              <input value={eqCountry} onChange={(e) => setEqCountry(e.target.value)} placeholder="e.g., US"
                     className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </Field>

            <Range label="Mkt Cap (USD)" lo={eqMktLo} hi={eqMktHi} setLo={setEqMktLo} setHi={setEqMktHi} />
            <Range label="Price" lo={eqPriceLo} hi={eqPriceHi} setLo={setEqPriceLo} setHi={setEqPriceHi} />
            <Tips text={`Sort: ${sortKey} (${sortDir}). Click headers to change.`} />
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Field label="Issuer">
              <input value={boIssuer} onChange={(e) => setBoIssuer(e.target.value)} placeholder="contains…"
                     className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </Field>
            <Field label="Sector">
              <input value={boSector} onChange={(e) => setBoSector(e.target.value)} placeholder="e.g., IG Corp"
                     className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </Field>
            <Field label="Country">
              <input value={boCountry} onChange={(e) => setBoCountry(e.target.value)} placeholder="e.g., US"
                     className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </Field>

            <Field label="Rating range">
              <div className="flex items-center gap-2">
                <select
                  value={boRatMin}
                  onChange={(e) => setBoRatMin(e.target.value as Rating | "")}
                  className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
                >
                  <option value="">Min</option>
                  {["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"].map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
                <span className="text-neutral-400">to</span>
              </div>
            </Field>
            <Range label="OAS (bp)" lo={boOASLo} hi={boOASHi} setLo={setBoOASLo} setHi={setBoOASHi} />
            <Tips text="Bonds default sort: OAS (desc). Narrow by issuer/sector/country." />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-rose-900 bg-rose-950/40 p-4 text-sm text-rose-300">
            {error}
          </div>
        )}

        {/* Results */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900">
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
            <div className="space-y-0.5">
              <h3 className="text-sm font-semibold">{tab === "equities" ? "Equities" : "Bonds"} Results</h3>
              <p className="text-xs text-neutral-400">{rows.length ? `${rows.length} matches` : "No results yet"}</p>
            </div>
            <div className="text-xs text-neutral-400">Page {page}/{totalPages}</div>
          </div>

          <div className="max-h-[60vh] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-neutral-950 text-[11px] uppercase text-neutral-500">
                <tr>
                  {columns.map((c) => (
                    <th
                      key={c.key}
                      onClick={() => toggleSort(c.key)}
                      className={`cursor-pointer select-none px-3 py-2 ${c.align === "right" ? "text-right" : "text-left"} hover:text-neutral-300`}
                    >
                      {c.label}
                      {sortKey === c.key && (
                        <span className="ml-1 text-neutral-400">{sortDir === "asc" ? "▲" : "▼"}</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800">
                {!view.length && (
                  <tr>
                    <td colSpan={columns.length} className="px-4 py-10 text-center text-xs text-neutral-500">
                      {loading ? "Scanning…" : "No rows to show. Click “Run Screen”."}
                    </td>
                  </tr>
                )}
                {view.map((r, i) => (
                  <tr key={(r.symbol || r.cusip || i) + "-" + i} className="hover:bg-neutral-800/40">
                    {columns.map((c) => (
                      <td key={c.key} className={`px-3 py-2 ${c.align === "right" ? "text-right font-mono" : ""}`}>
                        {formatCell(r[c.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between border-t border-neutral-800 px-4 py-2 text-xs">
            <span className="text-neutral-400">Page {page} of {totalPages}</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 disabled:opacity-50 hover:bg-neutral-800"
              >
                Prev
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 disabled:opacity-50 hover:bg-neutral-800"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

/* ------------------------------ Subcomponents ------------------------------ */

function Tab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md border px-3 py-2 text-sm ${
        active
          ? "border-sky-600 bg-sky-600/20 text-sky-300"
          : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:bg-neutral-800"
      }`}
    >
      {label}
    </button>
  );
}

function Field({ label, children }: { label: string; children: any }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="mb-1 text-xs text-neutral-400">{label}</div>
      {children}
    </div>
  );
}

function Range({
  label, lo, hi, setLo, setHi,
}: { label: string; lo: string; hi: string; setLo: (v: string) => void; setHi: (v: string) => void }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4">
      <div className="mb-1 text-xs text-neutral-400">{label}</div>
      <div className="flex items-center gap-2">
        <input value={lo} onChange={(e) => setLo(e.target.value)} placeholder="min"
               className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
        <span className="text-neutral-500">–</span>
        <input value={hi} onChange={(e) => setHi(e.target.value)} placeholder="max"
               className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
      </div>
    </div>
  );
}

function Tips({ text }: { text: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-4 text-xs text-neutral-400">
      {text}
    </div>
  );
}

function Select({
  value, onChange, options,
}: {
  value: string | "";
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
    >
      {options.map((o) => (
        <option key={o.value || "_"} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

/* ---------------------------------- Types --------------------------------- */
type Rating = "AAA" | "AA" | "A" | "BBB" | "BB" | "B" | "CCC" | "NR";
type ColumnDef = { key: string; label: string; align?: "left" | "right" };

/* --------------------------------- Utils ---------------------------------- */

function numOrEmpty(s: string): number | "" {
  const n = Number(s);
  return s.trim() === "" || !Number.isFinite(n) ? "" : n;
}
function rangeOk(v: number, p?: [number | "", number | ""]) {
  if (!p) return true;
  const [lo, hi] = p;
  if (lo !== "" && v < Number(lo)) return false;
  if (hi !== "" && v > Number(hi)) return false;
  return true;
}
function cmp(a: any, b: any, dir: "asc" | "desc") {
  const ax = toSortVal(a), bx = toSortVal(b);
  if (ax === bx) return 0;
  const r = ax > bx ? 1 : -1;
  return dir === "asc" ? r : -r;
}
function toSortVal(v: any) {
  if (v == null) return -Infinity;
  if (typeof v === "number") return v;
  const d = Date.parse(v);
  if (!isNaN(d)) return d;
  return String(v);
}
function formatCell(v: any) {
  if (v == null || v === "") return "";
  if (typeof v === "number") {
    // Pretty print large numbers + percents
    if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(1) + "T";
    if (Math.abs(v) >= 1e9)  return (v / 1e9).toFixed(1) + "B";
    if (Math.abs(v) >= 1e6)  return (v / 1e6).toFixed(1) + "M";
    return String(v);
  }
  return String(v);
}
function csv(x: any) {
  const s = String(x ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/* ---------------------------- Mock Generators ----------------------------- */

function buildEquityMock(n: number, rng: () => number) {
  const sectors = ["Tech", "Financials", "Health Care", "Energy", "Materials", "Industrials", "Utilities", "Consumer", "Real Estate"];
  const countries = ["US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "CA", "AU"];
  const out: any[] = [];
  for (let i = 0; i < n; i++) {
    out.push({
      symbol: genSym(i),
      name: `Company ${i}`,
      sector: pick(sectors, rng),
      country: pick(countries, rng),
      price: round(5 + rng() * 995, 2),
      marketCap: Math.floor(100e6 + rng() * 900e9),
      volume: Math.floor(100e3 + rng() * 5e7),
      pe: round(5 + rng() * 35, 1),
      ivPct: round(10 + rng() * 70, 1),
      score: round(rng() * 100, 1),
    });
  }
  return out;
}

function buildBondMock(n: number, rng: () => number) {
  const sectors = ["IG Corp", "HY Corp", "Sovereign", "Muni"];
  const countries = ["US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "CA", "AU"];
  const ratings: Rating[] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"];
  const issuers = ["Globex", "Initech", "Vandelay", "Acme", "Stark", "Wayne", "Umbrella", "Hooli", "Dynamic", "Soylent"];
  const out: any[] = [];
  for (let i = 0; i < n; i++) {
    const issueYear = 2008 + Math.floor(rng() * 15);
    const issueMonth = 1 + Math.floor(rng() * 12);
    const issueDay = 1 + Math.floor(rng() * 28);
    const mtyYears = 2 + Math.floor(rng() * 25);
    out.push({
      cusip: genCUSIP(i, rng),
      issuer: pick(issuers, rng) + " Holdings",
      sector: pick(sectors, rng),
      country: pick(countries, rng),
      coupon: round(2 + rng() * 7, 2),
      maturity: isoDate(issueYear + mtyYears, issueMonth, issueDay),
      price: round(70 + rng() * 50, 2),
      yieldToMaturity: round(2 + rng() * 8, 2),
      spreadOAS: Math.floor(40 + rng() * 450),
      rating: pick(ratings, rng),
    });
  }
  return out;
}

/* -------------------------------- Routines -------------------------------- */

function round(x: number, d = 2) { const p = 10 ** d; return Math.round(x * p) / p; }
function pick<T>(arr: T[], rng: () => number) { return arr[Math.floor(rng() * arr.length)]; }
function genSym(i: number) { const L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; return L[i % 26] + L[(i * 7) % 26] + L[(i * 13) % 26]; }
function genCUSIP(i: number, rng: () => number) {
  const L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", D = "0123456789";
  const part = (len: number, pool: string) => Array.from({ length: len }, () => pool[Math.floor(rng() * pool.length)]).join("");
  return part(3, L) + part(5, D) + (i % 10);
}
function isoDate(y: number, m: number, d: number) { return `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`; }
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;