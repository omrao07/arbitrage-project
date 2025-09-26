"use client";

/**
 * PositionsSnapshot.tsx
 * - No external libs
 * - KPIs, Top Gainers/Losers, Exposures by asset class & region
 * - Strict normalization without object spread (...p)
 */

type Position = {
  id: string;
  symbol: string;
  name?: string;
  assetClass?: string;
  region?: string;
  quantity: number;
  avgPrice?: number;
  marketPrice?: number;
  marketValue?: number;
  pnl?: number;            // unrealized, if provided
  currency?: string;
  tags?: string[];
  updatedAt?: string;
  // optional intraday
  dayPnl?: number;
  dayPct?: number;
};

type NormalizedPosition = Required<Position> & {
  cost: number;   // qty * avg
  mv: number;     // qty * last (or provided marketValue)
  chg: number;    // intraday PnL
  chgPct: number; // intraday %
};

type Props = {
  positions: Position[];
  baseCurrency?: string;
  title?: string;
  showTop?: number;
  onSelect?: (pos: Position) => void;
};

/* =================== component =================== */

export default function PositionsSnapshot({
  positions,
  baseCurrency = "USD",
  title = "Positions Snapshot",
  showTop = 5,
  onSelect,
}: Props) {
  const clean = (positions || []).map(normalize);
  const totals = aggregateTotals(clean);
  const topGainers = rankBy(changePct, clean).slice(0, showTop);
  const topLosers = rankBy((p) => -changePct(p), clean).slice(0, showTop);
  const byClass = bucketWeights(clean, (p) => p.assetClass || "Other");
  const byRegion = bucketWeights(clean, (p) => p.region || "Other");

  const maxClass = Math.max(1, ...byClass.map((x) => x.value));
  const maxRegion = Math.max(1, ...byRegion.map((x) => x.value));

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-4 py-3 border-b border-[#222] flex items-center justify-between">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">
          {totals.count} positions · {formatCc(baseCurrency)}
        </div>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3">
        <KPI label="Market Value" value={fmtMoney(totals.mv, baseCurrency)} tone="neutral" />
        <KPI
          label="Unrealized P&L"
          value={fmtMoney(totals.unrealized, baseCurrency)}
          tone={totals.unrealized >= 0 ? "pos" : "neg"}
          sub={pctText(totals.unrealized, totals.cost)}
        />
        <KPI
          label="Day P&L"
          value={fmtMoney(totals.dayPnl, baseCurrency)}
          tone={totals.dayPnl >= 0 ? "pos" : "neg"}
          sub={pctText(totals.dayPnl, totals.mv)}
        />
        <KPI
          label="Gross Exposure"
          value={fmtMoney(totals.gross, baseCurrency)}
          tone="neutral"
          sub={`Net ${fmtMoney(totals.net, baseCurrency)}`}
        />
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3">
        {/* Top Gainers */}
        <div className="bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg overflow-hidden">
          <Header label="Top Gainers" />
          {topGainers.length ? (
            <ul className="divide-y divide-[#1f1f1f]">
              {topGainers.map((p) => (
                <Row
                  key={p.id}
                  pos={p}
                  onClick={() => onSelect?.(p)}
                  tone="pos"
                  right={fmtChange(p, baseCurrency)}
                />
              ))}
            </ul>
          ) : (
            <Empty />
          )}
        </div>

        {/* Top Losers */}
        <div className="bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg overflow-hidden">
          <Header label="Top Losers" />
          {topLosers.length ? (
            <ul className="divide-y divide-[#1f1f1f]">
              {topLosers.map((p) => (
                <Row
                  key={p.id}
                  pos={p}
                  onClick={() => onSelect?.(p)}
                  tone="neg"
                  right={fmtChange(p, baseCurrency)}
                />
              ))}
            </ul>
          ) : (
            <Empty />
          )}
        </div>

        {/* Exposures */}
        <div className="bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg overflow-hidden">
          <Header label="Exposures" />
          <div className="p-3 grid grid-cols-1 gap-3">
            <BarGroup title="By Asset Class" items={byClass} max={maxClass} currency={baseCurrency} />
            <BarGroup title="By Region" items={byRegion} max={maxRegion} currency={baseCurrency} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================== small components ================== */

function KPI({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "neutral" | "pos" | "neg";
}) {
  const color =
    tone === "pos" ? "text-emerald-300" : tone === "neg" ? "text-red-300" : "text-gray-100";
  const chip =
    tone === "pos" ? "bg-emerald-700/30 text-emerald-200 border-emerald-700/60" :
    tone === "neg" ? "bg-red-700/30 text-red-200 border-red-700/60" :
    "bg-gray-700/30 text-gray-200 border-gray-700/60";
  return (
    <div className="bg-[#0e0e0e] border border-[#1f1f1f] rounded-lg p-3">
      <div className="text-[11px] text-gray-400">{label}</div>
      <div className={`text-base font-semibold ${color}`}>{value}</div>
      {sub ? (
        <span className={`mt-1 inline-block text-[10px] px-1.5 py-[1px] rounded border ${chip}`}>
          {sub}
        </span>
      ) : null}
    </div>
  );
}

function Header({ label }: { label: string }) {
  return <div className="px-3 py-2 border-b border-[#1f1f1f] text-sm text-gray-200 font-semibold">{label}</div>;
}

function Empty() {
  return <div className="p-3 text-xs text-gray-500">No data.</div>;
}

function Row({
  pos,
  right,
  onClick,
  tone,
}: {
  pos: NormalizedPosition;
  right?: string;
  onClick?: () => void;
  tone?: "pos" | "neg";
}) {
  const col =
    tone === "pos" ? "text-emerald-300" : tone === "neg" ? "text-red-300" : "text-gray-300";
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-3 py-2 hover:bg-[#101010] transition-colors"
      title={pos.name || pos.symbol}
    >
      <div className="flex items-center gap-3">
        <div className="min-w-[72px]">
          <div className="text-sm text-gray-100">{pos.symbol}</div>
          <div className="text-[11px] text-gray-500 truncate">{pos.name || "—"}</div>
        </div>
        <div className="flex-1 text-[11px] text-gray-400">
          Qty {fmtQty(pos.quantity)} @ {fmtNum(pos.avgPrice)} → MV {fmtNum(pos.marketValue)}
        </div>
        <div className={`text-sm font-semibold ${col}`}>{right}</div>
      </div>
    </button>
  );
}

function BarGroup({
  title,
  items,
  max,
  currency,
}: {
  title: string;
  items: Array<{ key: string; value: number; count: number }>;
  max: number;
  currency: string;
}) {
  return (
    <div>
      <div className="text-[11px] text-gray-400 mb-2">{title}</div>
      <div className="flex flex-col gap-1.5">
        {items.length === 0 ? (
          <div className="text-xs text-gray-500">—</div>
        ) : (
          items.map((x) => {
            const w = Math.max(2, Math.round((x.value / max) * 100));
            return (
              <div key={x.key} className="flex items-center gap-2">
                <div className="w-28 text-[11px] text-gray-300 truncate">{x.key}</div>
                <div className="flex-1 h-3 rounded bg-[#161616] overflow-hidden border border-[#1f1f1f]">
                  <div
                    className="h-full"
                    style={{ width: `${w}%`, background: "linear-gradient(90deg, rgba(34,197,94,0.25), rgba(34,197,94,0.5))" }}
                  />
                </div>
                <div className="w-28 text-[11px] text-gray-300 text-right">
                  {fmtMoney(x.value, currency)} <span className="text-gray-500">({x.count})</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

/* ===================== logic ===================== */

function normalize(p: Position): NormalizedPosition {
  const qty = toNum(p.quantity);
  const avg = toNum(p.avgPrice);
  const last = toNum(p.marketPrice);

  const mv = p.marketValue != null ? toNum(p.marketValue) : (qty && last ? qty * last : 0);
  const cost = qty && avg ? qty * avg : 0;

  const unreal = p.pnl != null ? toNum(p.pnl) : (mv - cost);
  const day = p.dayPnl != null ? toNum(p.dayPnl) : 0;
  const dPct = p.dayPct != null ? toNum(p.dayPct) : (mv ? (day / mv) * 100 : 0);

  return {
    // normalized Position fields (no spread)
    id: String(p.id ?? p.symbol ?? ""),
    symbol: String(p.symbol ?? ""),
    name: p.name ?? "",
    assetClass: p.assetClass ?? "Other",
    region: p.region ?? "Other",
    quantity: qty,
    avgPrice: avg,
    marketPrice: last,
    marketValue: mv,
    pnl: unreal,
    currency: p.currency ?? "USD",
    tags: p.tags ?? [],
    updatedAt: p.updatedAt ?? new Date().toISOString(),
    // intraday normalized
    dayPnl: day,
    dayPct: dPct,
    // derived
    cost,
    mv,
    chg: day,
    chgPct: dPct,
  };
}

function aggregateTotals(arr: NormalizedPosition[]) {
  let mv = 0, cost = 0, unreal = 0, dayPnl = 0, long = 0, short = 0;
  for (const p of arr) {
    mv += p.mv;
    cost += p.cost;
    unreal += p.pnl ?? 0;
    dayPnl += p.chg ?? 0;
    const side = Math.sign(p.quantity);
    if (side >= 0) long += Math.abs(p.mv); else short += Math.abs(p.mv);
  }
  return {
    mv,
    cost,
    unrealized: unreal,
    dayPnl,
    gross: long + short,
    net: long - short,
    count: arr.length,
  };
}

function bucketWeights(arr: NormalizedPosition[], keyFn: (p: NormalizedPosition) => string) {
  const map = new Map<string, { value: number; count: number }>();
  for (const p of arr) {
    const k = keyFn(p);
    const prev = map.get(k) || { value: 0, count: 0 };
    prev.value += Math.abs(p.mv);
    prev.count += 1;
    map.set(k, prev);
  }
  return Array.from(map, ([key, v]) => ({ key, ...v })).sort((a, b) => b.value - a.value);
}

function changePct(p: NormalizedPosition) {
  if (p.chgPct && Number.isFinite(p.chgPct)) return p.chgPct;
  if (p.cost) return ((p.mv - p.cost) / p.cost) * 100;
  return 0;
}

function rankBy(fn: (p: NormalizedPosition) => number, arr: NormalizedPosition[]) {
  return [...arr].sort((a, b) => fn(b) - fn(a));
}

/* ================ utils / format ================ */

function toNum(v: any): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function fmtMoney(n: number, ccy: string) {
  const s = fmtNum(n);
  return `${s} ${formatCc(ccy)}`;
}

function fmtChange(p: NormalizedPosition, ccy: string) {
  const val = p.chg ?? 0;
  const pct = p.chgPct ?? 0;
  const sign = val >= 0 ? "+" : "";
  return `${sign}${fmtNum(val)} ${formatCc(ccy)}  (${pct.toFixed(2)}%)`;
}

function fmtQty(q: number) {
  if (Math.abs(q) >= 1_000_000) return (q / 1_000_000).toFixed(2) + "M";
  if (Math.abs(q) >= 1_000) return (q / 1_000).toFixed(2) + "K";
  return q.toString();
}

function fmtNum(n?: number) {
  if (n == null || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (a >= 1_000_000_000) return `${sign}${(a / 1_000_000_000).toFixed(2)}B`;
  if (a >= 1_000_000) return `${sign}${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 1_000) return `${sign}${(a / 1_000).toFixed(2)}K`;
  return `${sign}${a.toFixed(2)}`;
}

function pctText(num: number, den: number) {
  if (!den) return "—";
  const pct = (num / den) * 100;
  const s = (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%";
  return s;
}

function formatCc(s?: string) {
  return s && s.length <= 4 ? s.toUpperCase() : (s || "USD");
}