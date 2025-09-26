"use client";

/**
 * screenerform.tsx
 * Zero-import, self-contained Screener Form for building filter queries.
 *
 * Features
 * - Asset class selector (Equities, Crypto, FX, Fixed Income, Commodities)
 * - Symbol search + watchlist picker
 * - Multi-select sectors / countries (chips)
 * - Numeric filters with min/max (Price, MktCap, Volume, Yield, PE, Beta, OI, IV%)
 * - Date range (IPO/listing or last update)
 * - “Add Rule” builder for advanced fields (field · op · value)
 * - Presets (load/apply), Save/Reset, Copy link (query string) & Export JSON
 * - Emits structured payload via onChange/onSubmit
 *
 * Tailwind-only. No external imports.
 */

export type ScreenerRule = {
  field: string;                // e.g., "pe", "marketCap", "iv", "beta"
  op: ">" | ">=" | "<" | "<=" | "=" | "between" | "contains" | "in";
  value: string | number | [number | "", number | ""];
};

export type ScreenerPayload = {
  assetClass: "equities" | "crypto" | "fx" | "fixed_income" | "commodities";
  symbols: string[];            // explicit tickers/pairs if any
  watchlist?: string;           // optional watchlist key
  sectors: string[];
  countries: string[];
  ranges: {
    price?: [number | "", number | ""];
    marketCap?: [number | "", number | ""];
    volume?: [number | "", number | ""];
    yield?: [number | "", number | ""];
    pe?: [number | "", number | ""];
    beta?: [number | "", number | ""];
    openInterest?: [number | "", number | ""];
    ivPct?: [number | "", number | ""];
  };
  dateFrom?: string;            // ISO date
  dateTo?: string;              // ISO date
  rules: ScreenerRule[];        // advanced rules
  sort?: { key: string; dir: "asc" | "desc" };
  limit?: number;               // max results hint
  meta?: Record<string, any>;
};

export type ScreenerPreset = {
  id: string;
  name: string;
  payload: Partial<ScreenerPayload>;
};

export default function ScreenerForm({
  title = "Build a Screen",
  presets = [],
  initial,
  onChange,
  onSubmit,
  onCancel,
  className = "",
}: {
  title?: string;
  presets?: ScreenerPreset[];
  initial?: Partial<ScreenerPayload>;
  onChange?: (p: ScreenerPayload) => void;
  onSubmit?: (p: ScreenerPayload) => void;
  onCancel?: () => void;
  className?: string;
}) {
  /* --------------------------------- State --------------------------------- */

  const [assetClass, setAssetClass] = useState<ScreenerPayload["assetClass"]>(initial?.assetClass || "equities");
  const [symbols, setSymbols] = useState<string[]>(initial?.symbols || []);
  const [watchlist, setWatchlist] = useState<string>(initial?.watchlist || "");
  const [qSymbol, setQSymbol] = useState("");

  const [sectors, setSectors] = useState<string[]>(initial?.sectors || []);
  const [countries, setCountries] = useState<string[]>(initial?.countries || []);

  const [ranges, setRanges] = useState<ScreenerPayload["ranges"]>({
    price: initial?.ranges?.price || ["", ""],
    marketCap: initial?.ranges?.marketCap || ["", ""],
    volume: initial?.ranges?.volume || ["", ""],
    yield: initial?.ranges?.yield || ["", ""],
    pe: initial?.ranges?.pe || ["", ""],
    beta: initial?.ranges?.beta || ["", ""],
    openInterest: initial?.ranges?.openInterest || ["", ""],
    ivPct: initial?.ranges?.ivPct || ["", ""],
  });

  const [dateFrom, setDateFrom] = useState<string | undefined>(initial?.dateFrom);
  const [dateTo, setDateTo] = useState<string | undefined>(initial?.dateTo);

  const [rules, setRules] = useState<ScreenerRule[]>(initial?.rules || []);
  const [sort, setSort] = useState<ScreenerPayload["sort"]>(initial?.sort || { key: "marketCap", dir: "desc" });
  const [limit, setLimit] = useState<number>(initial?.limit ?? 200);

  const payload = useMemo<ScreenerPayload>(() => ({
    assetClass, symbols, watchlist: watchlist || undefined, sectors, countries, ranges,
    dateFrom, dateTo, rules, sort, limit, meta: { version: 1 }
  }), [assetClass, symbols, watchlist, sectors, countries, ranges, dateFrom, dateTo, rules, sort, limit]);

  useEffect(() => { onChange?.(payload); }, [payload]);

  /* --------------------------------- Actions -------------------------------- */

  function addSymbol() {
    const s = normSym(qSymbol);
    if (!s) return;
    if (!symbols.includes(s)) setSymbols((arr) => [...arr, s].slice(0, 100));
    setQSymbol("");
  }
  function removeSymbol(s: string) { setSymbols((arr) => arr.filter((x) => x !== s)); }

  function toggleChip(list: "sectors" | "countries", v: string) {
    if (list === "sectors") setSectors((a) => toggle(a, v));
    else setCountries((a) => toggle(a, v));
  }

  function updateRange(key: keyof ScreenerPayload["ranges"], idx: 0 | 1, val: string) {
    setRanges((r) => {
      const next = { ...(r as any) };
      const pair = [...(next[key] || ["", ""])];
      pair[idx] = toNumOrEmpty(val);
      next[key] = pair as any;
      return next;
    });
  }

  function addRule() {
    setRules((rs) => [...rs, { field: "pe", op: "<=", value: 20 }]);
  }
  function updateRule(i: number, patch: Partial<ScreenerRule>) {
    setRules((rs) => rs.map((r, k) => (k === i ? { ...r, ...patch } : r)));
  }
  function removeRule(i: number) {
    setRules((rs) => rs.filter((_, k) => k !== i));
  }

  function applyPreset(id: string) {
    const p = presets.find((x) => x.id === id);
    if (!p) return;
    const pl = p.payload;
    if (pl.assetClass) setAssetClass(pl.assetClass);
    if (pl.symbols) setSymbols(pl.symbols);
    if (pl.watchlist !== undefined) setWatchlist(pl.watchlist || "");
    if (pl.sectors) setSectors(pl.sectors);
    if (pl.countries) setCountries(pl.countries);
    if (pl.ranges) setRanges((r) => ({ ...r, ...pl.ranges }));
    if (pl.dateFrom !== undefined) setDateFrom(pl.dateFrom);
    if (pl.dateTo !== undefined) setDateTo(pl.dateTo);
    if (pl.rules) setRules(pl.rules);
    if (pl.sort) setSort(pl.sort);
    if (pl.limit !== undefined) setLimit(pl.limit);
  }

  function resetAll() {
    setSymbols([]); setWatchlist("");
    setSectors([]); setCountries([]);
    setRanges({ price: ["", ""], marketCap: ["", ""], volume: ["", ""], yield: ["", ""], pe: ["", ""], beta: ["", ""], openInterest: ["", ""], ivPct: ["", ""], });
    setDateFrom(undefined); setDateTo(undefined);
    setRules([]); setSort({ key: "marketCap", dir: "desc" }); setLimit(200);
  }

  function handleSubmit() { onSubmit?.(payload); }

  function copyLink() {
    const q = encodePayload(payload);
    try { (navigator as any).clipboard?.writeText(q); } catch {}
  }
  function exportJSON() {
    const s = JSON.stringify(payload, null, 2);
    try { (navigator as any).clipboard?.writeText(s); } catch {}
  }

  /* --------------------------------- Render --------------------------------- */

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-100 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold">{title}</h3>
          <p className="text-xs text-neutral-400">Define filters and build a query. {symbols.length} symbols, {rules.length} rules.</p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {presets.length > 0 && (
            <Select
              label="Preset"
              value=""
              onChange={(id) => id && applyPreset(id)}
              options={[{ value: "", label: "Choose…" }, ...presets.map((p) => ({ value: p.id, label: p.name }))]}
            />
          )}
          <button onClick={copyLink} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Copy Link</button>
          <button onClick={exportJSON} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Export JSON</button>
        </div>
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-3">
        {/* Column A: Basics */}
        <div className="space-y-4">
          <Field label="Asset class">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {([
                ["equities","Equities"],
                ["crypto","Crypto"],
                ["fx","FX"],
                ["fixed_income","Fixed Income"],
                ["commodities","Commodities"],
              ] as const).map(([val, label]) => (
                <RadioChip
                  key={val}
                  value={val}
                  label={label}
                  groupValue={assetClass}
                  onChange={(v) => setAssetClass(v as any)}
                />
              ))}
            </div>
          </Field>

          <Field label="Symbols">
            <div className="flex items-center gap-2">
              <input
                value={qSymbol}
                onChange={(e) => setQSymbol(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") addSymbol(); }}
                placeholder={assetClass === "fx" ? "e.g., EURUSD" : "e.g., AAPL"}
                className="flex-1 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm text-neutral-100 placeholder:text-neutral-500"
              />
              <button onClick={addSymbol} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm hover:bg-neutral-800">Add</button>
            </div>
            {symbols.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {symbols.map((s) => (
                  <span key={s} className="inline-flex items-center gap-1 rounded bg-neutral-800 px-2 py-0.5 text-xs">
                    {s}
                    <button className="text-neutral-400 hover:text-neutral-200" onClick={() => removeSymbol(s)} title="Remove">×</button>
                  </span>
                ))}
              </div>
            )}
          </Field>

          <Field label="Watchlist (optional)">
            <input
              value={watchlist}
              onChange={(e) => setWatchlist(e.target.value)}
              placeholder="e.g., MyTop100"
              className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm text-neutral-100 placeholder:text-neutral-500"
            />
          </Field>
        </div>

        {/* Column B: Filters */}
        <div className="space-y-4">
          <Field label="Sectors">
            <ChipGroup
              options={SECTORS}
              value={sectors}
              onToggle={(v) => toggleChip("sectors", v)}
            />
          </Field>
          <Field label="Countries">
            <ChipGroup
              options={COUNTRIES}
              value={countries}
              onToggle={(v) => toggleChip("countries", v)}
            />
          </Field>

          <Field label="Numeric ranges">
            <div className="grid grid-cols-2 gap-3">
              <Range label="Price" unit="" pair={ranges.price!} onChange={(i, v) => updateRange("price", i, v)} />
              <Range label="Mkt Cap" unit="B" pair={ranges.marketCap!} onChange={(i, v) => updateRange("marketCap", i, v)} />
              <Range label="Volume" unit="M" pair={ranges.volume!} onChange={(i, v) => updateRange("volume", i, v)} />
              <Range label="Yield" unit="%" pair={ranges.yield!} onChange={(i, v) => updateRange("yield", i, v)} />
              <Range label="P/E" unit="" pair={ranges.pe!} onChange={(i, v) => updateRange("pe", i, v)} />
              <Range label="Beta" unit="" pair={ranges.beta!} onChange={(i, v) => updateRange("beta", i, v)} />
              {assetClass !== "equities" && (
                <>
                  <Range label="Open Interest" unit="k" pair={ranges.openInterest!} onChange={(i, v) => updateRange("openInterest", i, v)} />
                  <Range label="IV %" unit="%" pair={ranges.ivPct!} onChange={(i, v) => updateRange("ivPct", i, v)} />
                </>
              )}
            </div>
          </Field>
        </div>

        {/* Column C: Advanced + Dates */}
        <div className="space-y-4">
          <Field label="Date range">
            <div className="grid grid-cols-2 gap-2">
              <input type="date" value={dateFrom || ""} onChange={(e) => setDateFrom(e.target.value || undefined)} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
              <input type="date" value={dateTo || ""} onChange={(e) => setDateTo(e.target.value || undefined)} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm" />
            </div>
          </Field>

          <Field label="Advanced rules">
            {rules.length === 0 && <Empty text="No advanced rules. Add some conditions." />}
            {rules.map((r, i) => (
              <div key={i} className="mb-2 grid grid-cols-12 items-center gap-2">
                <div className="col-span-4">
                  <select
                    value={r.field}
                    onChange={(e) => updateRule(i, { field: e.target.value })}
                    className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
                  >
                    {FIELDS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
                  </select>
                </div>
                <div className="col-span-3">
                  <select
                    value={r.op}
                    onChange={(e) => updateRule(i, { op: e.target.value as any })}
                    className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
                  >
                    {OPS.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <div className="col-span-4">
                  {r.op === "between" ? (
                    <div className="flex items-center gap-2">
                      <input
                        placeholder="min"
                        defaultValue={(Array.isArray(r.value) ? r.value[0] : "") as any}
                        onBlur={(e) => {
                          const v = toNumOrEmpty(e.target.value);
                          const hi = (Array.isArray(r.value) ? r.value[1] : "") as any;
                          updateRule(i, { value: [v, hi] });
                        }}
                        className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
                      />
                      <input
                        placeholder="max"
                        defaultValue={(Array.isArray(r.value) ? r.value[1] : "") as any}
                        onBlur={(e) => {
                          const v = toNumOrEmpty(e.target.value);
                          const lo = (Array.isArray(r.value) ? r.value[0] : "") as any;
                          updateRule(i, { value: [lo, v] });
                        }}
                        className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
                      />
                    </div>
                  ) : (
                    <input
                      placeholder="value"
                      defaultValue={String(r.value ?? "")}
                      onBlur={(e) => {
                        const raw = e.target.value;
                        const val = isFinite(+raw) && raw.trim() !== "" ? +raw : raw.trim();
                        updateRule(i, { value: val as any });
                      }}
                      className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
                    />
                  )}
                </div>
                <div className="col-span-1 text-right">
                  <button onClick={() => removeRule(i)} className="rounded-md border border-neutral-800 bg-neutral-950 px-2 py-1 text-neutral-300 hover:bg-neutral-800" title="Remove">
                    ×
                  </button>
                </div>
              </div>
            ))}
            <button onClick={addRule} className="mt-1 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-sm text-neutral-200 hover:bg-neutral-800">
              + Add Rule
            </button>
          </Field>
        </div>
      </div>

      {/* Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-neutral-800 px-4 py-3">
        <div className="text-[11px] text-neutral-500">
          {limit} result cap · Sort by <strong className="text-neutral-300">{sort?.key}</strong> ({sort?.dir})
        </div>
        <div className="flex items-center gap-2">
          <Select
            label="Sort"
            value={sort?.key || "marketCap"}
            onChange={(v) => setSort((s) => ({ key: v, dir: s?.dir || "desc" }))}
            options={SORT_KEYS.map((k) => ({ value: k.value, label: k.label }))}
          />
          <Select
            label="Dir"
            value={sort?.dir || "desc"}
            onChange={(v) => setSort((s) => ({ key: s?.key || "marketCap", dir: v as any }))}
            options={[{ value: "asc", label: "Asc" }, { value: "desc", label: "Desc" }]}
          />
          <label className="flex items-center gap-2 text-xs">
            <span className="text-neutral-400">Limit</span>
            <input
              type="number"
              min={1}
              max={5000}
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Math.min(5000, Math.round(+e.target.value || 1))))}
              className="w-20 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-right"
            />
          </label>
          <button onClick={resetAll} className="rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-800">
            Reset
          </button>
          {onCancel && (
            <button onClick={onCancel} className="rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-800">
              Cancel
            </button>
          )}
          <button onClick={handleSubmit} className="rounded-md border border-emerald-700 bg-emerald-600/20 px-3 py-2 text-sm font-medium text-emerald-300 hover:bg-emerald-600/30">
            Run Screen
          </button>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------- Subcomponents ------------------------------ */

function Field({ label, children }: { label: string; children: any }) {
  return (
    <div>
      <div className="mb-1 text-xs text-neutral-400">{label}</div>
      {children}
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-xs text-neutral-400">
      {text}
    </div>
  );
}

function RadioChip({
  label, value, groupValue, onChange,
}: {
  label: string;
  value: string;
  groupValue: string;
  onChange: (v: string) => void;
}) {
  const active = value === groupValue;
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

function ChipGroup({
  options, value, onToggle,
}: {
  options: string[];
  value: string[];
  onToggle: (v: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((o) => {
        const on = value.includes(o);
        return (
          <button
            key={o}
            type="button"
            onClick={() => onToggle(o)}
            className={`rounded-md border px-2 py-1 text-xs ${
              on
                ? "border-sky-600 bg-sky-600/20 text-sky-300"
                : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:bg-neutral-800"
            }`}
          >
            {o}
          </button>
        );
      })}
    </div>
  );
}

function Range({
  label, unit, pair, onChange,
}: {
  label: string;
  unit?: string;
  pair: [number | "", number | ""];
  onChange: (idx: 0 | 1, val: string) => void;
}) {
  return (
    <div>
      <div className="mb-1 text-xs text-neutral-400">{label}{unit ? ` (${unit})` : ""}</div>
      <div className="flex items-center gap-2">
        <input
          defaultValue={pair?.[0] ?? ""}
          onBlur={(e) => onChange(0, e.target.value)}
          placeholder="min"
          className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
        />
        <span className="text-neutral-500">–</span>
        <input
          defaultValue={pair?.[1] ?? ""}
          onBlur={(e) => onChange(1, e.target.value)}
          placeholder="max"
          className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1.5 text-sm"
        />
      </div>
    </div>
  );
}

function Select({
  label, value, onChange, options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex items-center gap-2 text-xs">
      <span className="text-neutral-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
      >
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

/* ---------------------------------- Data ---------------------------------- */

const SECTORS = [
  "Tech","Financials","Health Care","Energy","Materials","Industrials",
  "Utilities","Consumer Disc.","Consumer Staples","Real Estate","Telecom"
];

const COUNTRIES = [
  "US","CA","GB","DE","FR","JP","CN","IN","BR","AU","KR","HK","SG","NL","SE","CH"
];

const FIELDS = [
  { value: "price", label: "Price" },
  { value: "marketCap", label: "Market Cap" },
  { value: "volume", label: "Volume" },
  { value: "pe", label: "P/E" },
  { value: "pb", label: "P/B" },
  { value: "ps", label: "P/S" },
  { value: "yield", label: "Dividend Yield %" },
  { value: "beta", label: "Beta" },
  { value: "oi", label: "Open Interest" },
  { value: "iv", label: "Implied Volatility %" },
  { value: "chg1d", label: "Change 1D %" },
  { value: "chg1w", label: "Change 1W %" },
  { value: "rsi", label: "RSI" },
  { value: "atr", label: "ATR" },
  { value: "score", label: "Composite Score" },
];

const OPS: ScreenerRule["op"][] = [">",">=","<","<=","=","between","contains","in"];

const SORT_KEYS = [
  { value: "marketCap", label: "Mkt Cap" },
  { value: "price", label: "Price" },
  { value: "volume", label: "Volume" },
  { value: "pe", label: "P/E" },
  { value: "yield", label: "Yield" },
  { value: "beta", label: "Beta" },
  { value: "iv", label: "IV %" },
  { value: "chg1d", label: "1D %" },
  { value: "score", label: "Score" },
];

/* --------------------------------- Utils --------------------------------- */

function toggle(a: string[], v: string) {
  return a.includes(v) ? a.filter((x) => x !== v) : [...a, v];
}
function normSym(s: string) {
  const t = (s || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
  return t.length > 0 ? t : "";
}
function toNumOrEmpty(v: string): number | "" {
  if (v.trim() === "") return "";
  const n = Number(v);
  return Number.isFinite(n) ? n : (v as unknown as number | "");
}
function encodePayload(p: ScreenerPayload) {
  const o: any = { ...p };
  // Make URL-ish compact string (no real URL encoding to keep it simple)
  return JSON.stringify(o);
}

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;

/* ---------------------------------- Example ---------------------------------
<ScreenerForm
  presets={[
    { id: "megacaps", name: "US Mega Caps", payload: { assetClass: "equities", countries: ["US"], ranges: { marketCap: [200, "" as any] }, sort: { key: "marketCap", dir: "desc" } } },
  ]}
  onSubmit={(p) => console.log("submit", p)}
/>
-------------------------------------------------------------------------------- */