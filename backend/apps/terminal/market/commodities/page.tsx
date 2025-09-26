// app/commodities/page.tsx
// Single-file, ZERO imports. Pure JSX + inline SVG charts + Tailwind classes.
// Drop in as-is. No external UI libs required.

type TermPoint = {
  symbol: string;
  expiry: string; // YYYY-MM-DD
  daysToExp: number;
  futPrice: number;
  spot: number;
  basisPct: number;
  annualizedCarryPct: number;
  state: "CONTANGO" | "BACKWARDATION";
};

type FuturesContract = {
  symbol: string;
  expiry: string;
  strike: number;
  type: "CALL" | "PUT" | "FUT";
  last: number;
  change: number;
  volume: number;
  oi: number;
};

type ShippingRatePoint = {
  date: string;
  lane: string;
  rateUsd: number;
  change?: number;
};

type InventoryPoint = {
  date: string;
  level: number;
  change: number;
  daysCover?: number;
};

/* ------------------------------ utils ------------------------------ */

const fmtN = (n: number, d = 2) =>
  Number(n).toLocaleString("en-IN", { maximumFractionDigits: d });

const shortDate = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
  });
};

/* --------------------------- tiny chart lib -------------------------- */
/** Line chart SVG (no libs). dataY is array of numbers. Labels show under x. */
function LineSVG({
  width = 640,
  height = 200,
  dataY,
  labels,
  stroke = "#10b981",
  guide = true,
}: {
  width?: number;
  height?: number;
  dataY: number[];
  labels?: string[];
  stroke?: string;
  guide?: boolean;
}) {
  const w = width;
  const h = height;
  const pad = 24;
  const n = Math.max(1, dataY.length);
  const min = Math.min(...dataY);
  const max = Math.max(...dataY);
  const y = (v: number) =>
    h - pad - ((v - min) / (max - min || 1)) * (h - pad * 2);
  const x = (i: number) => pad + (i * (w - pad * 2)) / (n - 1 || 1);
  const d =
    dataY
      .map((v, i) => `${i ? "L" : "M"} ${x(i).toFixed(2)} ${y(v).toFixed(2)}`)
      .join(" ") || "";
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) =>
    min + ((max - min) * i) / ticks
  );

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
      {guide &&
        yTicks.map((tv, idx) => {
          const yy = y(tv);
          return (
            <g key={idx}>
              <line
                x1={pad}
                y1={yy}
                x2={w - pad}
                y2={yy}
                stroke="#27272a"
                strokeDasharray="3 3"
              />
              <text
                x={4}
                y={yy + 4}
                fill="#9ca3af"
                fontSize="10"
              >{`${fmtN(tv)}`}</text>
            </g>
          );
        })}
      <path d={d} fill="none" stroke={stroke} strokeWidth="2" />
      {labels &&
        labels.map((lab, i) => (
          <text
            key={i}
            x={x(i)}
            y={h - 4}
            textAnchor="middle"
            fill="#9ca3af"
            fontSize="10"
          >
            {lab}
          </text>
        ))}
    </svg>
  );
}

/** Bar chart SVG (vertical bars). */
function BarsSVG({
  width = 640,
  height = 200,
  data,
  labels,
  positive = "#10b981",
  negative = "#ef4444",
}: {
  width?: number;
  height?: number;
  data: number[];
  labels?: string[];
  positive?: string;
  negative?: string;
}) {
  const w = width;
  const h = height;
  const pad = 24;
  const maxAbs = Math.max(...data.map((v) => Math.abs(v)), 1);
  const zeroY = h / 2;
  const bw = (w - pad * 2) / (data.length || 1);
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="block">
      <line
        x1={pad}
        y1={zeroY}
        x2={w - pad}
        y2={zeroY}
        stroke="#6b7280"
        strokeDasharray="4 4"
      />
      {data.map((v, i) => {
        const barH = ((h / 2 - pad) * Math.abs(v)) / maxAbs;
        const x = pad + i * bw + bw * 0.1;
        const y = v >= 0 ? zeroY - barH : zeroY;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={bw * 0.8}
            height={barH}
            fill={v >= 0 ? positive : negative}
            rx="4"
          />
        );
      })}
      {labels &&
        labels.map((lab, i) => (
          <text
            key={i}
            x={pad + i * bw + bw / 2}
            y={h - 4}
            textAnchor="middle"
            fill="#9ca3af"
            fontSize="10"
          >
            {lab}
          </text>
        ))}
    </svg>
  );
}

/* -------------------------- UI: term structure ------------------------- */

function TermStructurePane({ title, data }: { title: string; data: TermPoint[] }) {
  const rows = [...data].sort((a, b) => a.daysToExp - b.daysToExp);
  const prices = rows.map((r) => r.futPrice);
  const labels = rows.map((r) => shortDate(r.expiry));
  const spot = rows[0]?.spot ?? 0;
  const front = rows[0];
  const back = rows[rows.length - 1];
  const slopePct =
    front && back ? ((back.futPrice / front.futPrice - 1) * 100) : 0;

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {rows[0]?.symbol ?? "—"} · {rows.length} expiries
          </p>
        </div>
        <div className="text-xs text-neutral-400">
          Slope:{" "}
          <span className={slopePct >= 0 ? "text-emerald-400" : "text-rose-400"}>
            {slopePct >= 0 ? "+" : ""}
            {fmtN(slopePct, 2)}%
          </span>
        </div>
      </div>

      <div className="px-3 pt-2">
        <LineSVG dataY={prices} labels={labels} />
        {spot ? (
          <div className="px-1 pb-2 text-[11px] text-neutral-400">
            Spot ref: ₹ {fmtN(spot)}
          </div>
        ) : null}
      </div>

      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Expiry</th>
              <th className="px-3 py-2 text-right font-medium">Futures</th>
              <th className="px-3 py-2 text-right font-medium">Basis %</th>
              <th className="px-3 py-2 text-right font-medium">Carry % (ann)</th>
              <th className="px-3 py-2 text-left font-medium">State</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.expiry} className="border-t border-neutral-800">
                <td className="px-3 py-2">{shortDate(r.expiry)}</td>
                <td className="px-3 py-2 text-right">₹ {fmtN(r.futPrice)}</td>
                <td className={`px-3 py-2 text-right ${r.basisPct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.basisPct.toFixed(2)}%
                </td>
                <td className={`px-3 py-2 text-right ${r.annualizedCarryPct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.annualizedCarryPct.toFixed(2)}%
                </td>
                <td className={`px-3 py-2 ${r.state === "CONTANGO" ? "text-emerald-300" : "text-rose-300"}`}>
                  {r.state}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-6 text-center text-neutral-400">
                  No data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------------------ UI: chain ------------------------------ */

function FutChainPane({
  title,
  data,
}: {
  title: string;
  data: FuturesContract[];
}) {
  const rows = [...data].sort((a, b) =>
    a.expiry === b.expiry ? a.strike - b.strike : a.expiry.localeCompare(b.expiry)
  );
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="border-b border-neutral-800 px-4 py-3">
        <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
        <p className="text-xs text-neutral-400">{rows.length} contracts</p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Symbol</th>
              <th className="px-3 py-2 text-left font-medium">Expiry</th>
              <th className="px-3 py-2 text-left font-medium">Type</th>
              <th className="px-3 py-2 text-right font-medium">Strike</th>
              <th className="px-3 py-2 text-right font-medium">Last</th>
              <th className="px-3 py-2 text-right font-medium">Δ</th>
              <th className="px-3 py-2 text-right font-medium">Vol</th>
              <th className="px-3 py-2 text-right font-medium">OI</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.symbol}-${r.expiry}-${r.type}-${r.strike}`} className="border-t border-neutral-800">
                <td className="px-3 py-2 font-mono text-xs text-neutral-300">{r.symbol}</td>
                <td className="px-3 py-2">{shortDate(r.expiry)}</td>
                <td className="px-3 py-2">{r.type}</td>
                <td className="px-3 py-2 text-right">{fmtN(r.strike, 0)}</td>
                <td className="px-3 py-2 text-right">{fmtN(r.last)}</td>
                <td className={`px-3 py-2 text-right ${r.change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {fmtN(r.change)}
                </td>
                <td className="px-3 py-2 text-right">{fmtN(r.volume, 0)}</td>
                <td className="px-3 py-2 text-right">{fmtN(r.oi, 0)}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={8} className="px-3 py-6 text-center text-neutral-400">
                  No contracts
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* --------------------------- UI: shipping rates --------------------------- */

function ShippingRatesPane({
  title,
  unit,
  data,
  defaultLane,
}: {
  title: string;
  unit: string;
  data: ShippingRatePoint[];
  defaultLane: string;
}) {
  const lanes = Array.from(new Set(data.map((d) => d.lane))).sort();
  const lane = lanes.includes(defaultLane) ? defaultLane : lanes[0] || "";
  const rows = data.filter((d) => (lane ? d.lane === lane : true)).sort((a, b) => a.date.localeCompare(b.date));
  const labels = rows.map((r) => shortDate(r.date));
  const series = rows.map((r) => r.rateUsd);
  const last = rows[rows.length - 1];

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">{lane || "—"} · {rows.length} pts</p>
        </div>
        <div className="text-xs text-neutral-400">
          Latest: {last ? `$${fmtN(last.rateUsd, 0)} ${unit}` : "—"}
        </div>
      </div>
      <div className="px-3 py-2">
        <LineSVG dataY={series} labels={labels} />
      </div>
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Date</th>
              <th className="px-3 py-2 text-left font-medium">Lane</th>
              <th className="px-3 py-2 text-right font-medium">Rate ({unit})</th>
              <th className="px-3 py-2 text-right font-medium">Δ (USD)</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-20).reverse().map((r) => (
              <tr key={`${r.date}-${r.lane}`} className="border-t border-neutral-800">
                <td className="px-3 py-2">{shortDate(r.date)}</td>
                <td className="px-3 py-2">{r.lane}</td>
                <td className="px-3 py-2 text-right">${fmtN(r.rateUsd, 0)}</td>
                <td className={`px-3 py-2 text-right ${(r.change ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.change != null ? `${r.change >= 0 ? "+" : ""}$${fmtN(r.change, 0)}` : "—"}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={4} className="px-3 py-6 text-center text-neutral-400">
                  No data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* --------------------------- UI: inventories --------------------------- */

function InventoryFlowsPane({
  title,
  units,
  data,
}: {
  title: string;
  units: string;
  data: InventoryPoint[];
}) {
  const rows = [...data].sort((a, b) => a.date.localeCompare(b.date));
  const labels = rows.map((r) => shortDate(r.date));
  const builds = rows.map((r) => r.change);
  const last = rows[rows.length - 1];

  const mtd = (() => {
    if (!last) return 0;
    const ym = last.date.slice(0, 7);
    return rows
      .filter((r) => r.date.slice(0, 7) === ym)
      .reduce((s, r) => s + r.change, 0);
  })();

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {last ? `Latest: ${shortDate(last.date)} • ${fmtN(last.level)} ${units}` : "No data"}
          </p>
        </div>
        <div className="text-xs">
          <span className="text-neutral-400">MTD: </span>
          <span className={mtd >= 0 ? "text-emerald-400" : "text-rose-400"}>
            {mtd >= 0 ? "+" : ""}
            {fmtN(mtd)} {units}
          </span>
        </div>
      </div>
      <div className="px-3 py-2">
        <BarsSVG data={builds} labels={labels} />
      </div>
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Week</th>
              <th className="px-3 py-2 text-right font-medium">Level ({units})</th>
              <th className="px-3 py-2 text-right font-medium">WoW Δ</th>
              <th className="px-3 py-2 text-right font-medium">Days Cover</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-12).reverse().map((r) => (
              <tr key={r.date} className="border-top border-neutral-800">
                <td className="px-3 py-2">{shortDate(r.date)}</td>
                <td className="px-3 py-2 text-right">{fmtN(r.level)}</td>
                <td className={`px-3 py-2 text-right ${r.change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.change >= 0 ? "+" : ""}
                  {fmtN(r.change)}
                </td>
                <td className="px-3 py-2 text-right">{r.daysCover != null ? fmtN(r.daysCover) : "—"}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={4} className="px-3 py-6 text-center text-neutral-400">
                  No data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ----------------------------- PAGE (server) ---------------------------- */

export default async function CommoditiesPage() {
  const symbol = "CRUDE";

  // mock term structure
  const term: TermPoint[] = [
    { symbol, expiry: "2025-09-25", daysToExp: 7, futPrice: 22100, spot: 22000, basisPct: 0.45, annualizedCarryPct: 6.2, state: "CONTANGO" },
    { symbol, expiry: "2025-10-02", daysToExp: 14, futPrice: 22250, spot: 22000, basisPct: 1.10, annualizedCarryPct: 6.5, state: "CONTANGO" },
    { symbol, expiry: "2025-10-30", daysToExp: 42, futPrice: 22400, spot: 22000, basisPct: 1.82, annualizedCarryPct: 7.0, state: "CONTANGO" },
  ];

  // mock futures chain
  const chain: FuturesContract[] = [
    { symbol: `${symbol}-2025-09-25-FUT`, expiry: "2025-09-25", strike: 22000, type: "FUT", last: 22100, change: 25, volume: 12000, oi: 55000 },
    { symbol: `${symbol}-2025-09-25-C22000`, expiry: "2025-09-25", strike: 22000, type: "CALL", last: 120, change: -5, volume: 8000, oi: 42000 },
    { symbol: `${symbol}-2025-09-25-P22000`, expiry: "2025-09-25", strike: 22000, type: "PUT", last: 95, change: 8, volume: 9500, oi: 46000 },
  ];

  // mock shipping
  const shipping: ShippingRatePoint[] = Array.from({ length: 30 }).map((_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (29 - i));
    const prev = 3000 + Math.sin((i - 1) / 6) * 200 + Math.random() * 80;
    const rate = 3000 + Math.sin(i / 6) * 200 + Math.random() * 80;
    return {
      date: d.toISOString().slice(0, 10),
      lane: "CN->USWC",
      rateUsd: Math.round(rate),
      change: Math.round(rate - prev),
    };
  });

  // mock inventories
  const inventories: InventoryPoint[] = (() => {
    const start = new Date();
    start.setDate(start.getDate() - 7 * 20);
    const rows: InventoryPoint[] = [];
    let level = 420;
    for (let i = 0; i < 20; i++) {
      const d = new Date(start);
      d.setDate(d.getDate() + 7 * i);
      const change = Math.round((Math.random() - 0.5) * 6 * 10) / 10;
      level = Math.max(380, level + change);
      rows.push({
        date: d.toISOString().slice(0, 10),
        level: Math.round(level * 10) / 10,
        change,
        daysCover: Math.round((28 + Math.random() * 4) * 10) / 10,
      });
    }
    return rows;
  })();

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      <div className="sticky top-0 z-20 border-b border-neutral-800 bg-neutral-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
          <h1 className="text-lg font-semibold tracking-tight">Commodities Dashboard</h1>
          <p className="text-xs text-neutral-400">Symbol: {symbol}</p>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <TermStructurePane title={`${symbol} Term Structure`} data={term} />
          </div>
          <div>
            <FutChainPane title={`${symbol} Futures/Options Chain`} data={chain} />
          </div>
        </section>

        <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ShippingRatesPane title="Freight (Spot Rates)" unit="USD/FEU" data={shipping} defaultLane="CN->USWC" />
          <InventoryFlowsPane title="Inventories (Weekly)" units="mn bbl" data={inventories} />
        </section>
      </main>
    </div>
  );
}