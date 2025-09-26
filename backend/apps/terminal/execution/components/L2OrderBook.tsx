"use client";

/**
 * L2OrderBook.tsx
 * - Stateless: no imports, no hooks
 * - Side-by-side Bids/Asks with size & cumulative bars
 * - Computes spread/mid + totals
 * - Click handlers for price/level selection
 */

type Level = {
  price: number;
  size: number;       // aggregated size at price
  orders?: number;    // count at that level (optional)
};

type Props = {
  bids: Level[];          // already sorted DESC by price (best first)
  asks: Level[];          // already sorted ASC by price (best first)
  levels?: number;        // rows per side, default 10
  precision?: number;     // price decimals, default 2
  sizePrecision?: number; // size decimals, default 0
  showCumulative?: boolean; // show cumulative column, default true
  compactionNote?: string;  // e.g. "grouped by $0.05"
  onSelectPrice?: (price: number, side: "BUY" | "SELL") => void;
  onSelectLevel?: (row: { side: "BUY" | "SELL"; price: number; size: number; cum: number; orders?: number; idx: number }) => void;
  title?: string;
};

export default function L2OrderBook({
  bids,
  asks,
  levels = 10,
  precision = 2,
  sizePrecision = 0,
  showCumulative = true,
  compactionNote,
  onSelectPrice,
  onSelectLevel,
  title = "Order Book (L2)",
}: Props) {
  const safeBids = (bids || []).filter(valid).slice(0, levels);
  const safeAsks = (asks || []).filter(valid).slice(0, levels);

  // compute cum sizes
  const bidRows = withCumulative(safeBids, "BUY");
  const askRows = withCumulative(safeAsks, "SELL");

  // scale bars by the max cum on either side so visuals are comparable
  const maxCum = Math.max(
    1,
    ...bidRows.map((r) => r.cum),
    ...askRows.map((r) => r.cum)
  );

  // best prices / spread / mid
  const bestBid = safeBids.length ? safeBids[0].price : undefined;
  const bestAsk = safeAsks.length ? safeAsks[0].price : undefined;
  const spread = bestBid != null && bestAsk != null ? bestAsk - bestBid : undefined;
  const mid = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : undefined;

  // totals
  const totBid = sum(safeBids.map((x) => x.size));
  const totAsk = sum(safeAsks.map((x) => x.size));

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-3 py-2 border-b border-[#222] flex items-center justify-between">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="text-[11px] text-gray-500">
          {bestBid != null && bestAsk != null ? (
            <>
              Bid <span className="text-emerald-300">{fmt(bestBid, precision)}</span> ·
              Ask <span className="text-red-300">{fmt(bestAsk, precision)}</span> ·
              Spread <span className="text-gray-300">{fmt(spread!, precision)}</span>
              {mid != null ? <> · Mid <span className="text-gray-300">{fmt(mid, precision)}</span></> : null}
            </>
          ) : (
            "—"
          )}
        </div>
      </div>

      {/* compaction note */}
      {compactionNote ? (
        <div className="px-3 py-1 text-[11px] text-gray-500 border-b border-[#1f1f1f]">
          {compactionNote}
        </div>
      ) : null}

      {/* tables */}
      <div className="grid grid-cols-2 gap-0">
        {/* BIDS */}
        <SideTable
          side="BUY"
          rows={bidRows}
          precision={precision}
          sizePrecision={sizePrecision}
          maxCum={maxCum}
          showCumulative={showCumulative}
          onSelectPrice={onSelectPrice}
          onSelectLevel={onSelectLevel}
          footerTotal={totBid}
        />

        {/* ASKS */}
        <SideTable
          side="SELL"
          rows={askRows}
          precision={precision}
          sizePrecision={sizePrecision}
          maxCum={maxCum}
          showCumulative={showCumulative}
          onSelectPrice={onSelectPrice}
          onSelectLevel={onSelectLevel}
          footerTotal={totAsk}
        />
      </div>
    </div>
  );
}

/* ---------------- subcomponents ---------------- */

function SideTable({
  side,
  rows,
  precision,
  sizePrecision,
  maxCum,
  showCumulative,
  onSelectPrice,
  onSelectLevel,
  footerTotal,
}: {
  side: "BUY" | "SELL";
  rows: Array<{ price: number; size: number; cum: number; orders?: number; idx: number }>;
  precision: number;
  sizePrecision: number;
  maxCum: number;
  showCumulative: boolean;
  onSelectPrice?: (price: number, side: "BUY" | "SELL") => void;
  onSelectLevel?: (row: { side: "BUY" | "SELL"; price: number; size: number; cum: number; orders?: number; idx: number }) => void;
  footerTotal: number;
}) {
  const isBid = side === "BUY";
  const headTone = isBid ? "text-emerald-300" : "text-red-300";
  const barTone = isBid
    ? "bg-[rgba(34,197,94,0.18)]"
    : "bg-[rgba(239,68,68,0.18)]";

  return (
    <div className="border-r border-[#1f1f1f] last:border-r-0">
      <table className="w-full text-[12px]">
        <thead className="bg-[#0f0f0f] border-b border-[#1f1f1f] text-gray-400">
          <tr>
            <Th className={headTone}>{isBid ? "Bids" : "Asks"}</Th>
            <Th className="text-right">Size</Th>
            {showCumulative ? <Th className="text-right">Cum</Th> : null}
          </tr>
        </thead>
        <tbody className="divide-y divide-[#101010]">
          {rows.map((r) => {
            const pct = Math.max(2, Math.round((r.cum / maxCum) * 100)); // min 2% for visibility
            return (
              <tr
                key={`${side}-${r.idx}`}
                className="relative hover:bg-[#101010] cursor-pointer"
                onClick={() => {
                  onSelectPrice?.(r.price, side);
                  onSelectLevel?.({ side, ...r });
                }}
                title={`${side} ${fmt(r.size, sizePrecision)} @ ${fmt(r.price, precision)} (cum ${fmt(r.cum, sizePrecision)})`}
              >
                {/* depth bar */}
                <td colSpan={showCumulative ? 3 : 2} className="p-0">
                  <div className="relative">
                    <div
                      className={`absolute inset-y-0 ${isBid ? "left-0" : "right-0"} ${barTone}`}
                      style={{ width: `${pct}%` }}
                    />
                    <div className="relative grid grid-cols-[auto_minmax(0,1fr)_auto] items-center px-2 py-1">
                      <span className={isBid ? "text-emerald-300" : "text-red-300"}>{fmt(r.price, precision)}</span>
                      <span className="text-right text-gray-300">{fmt(r.size, sizePrecision)}</span>
                      {showCumulative ? (
                        <span className="ml-2 text-right text-gray-500">{fmt(r.cum, sizePrecision)}</span>
                      ) : null}
                    </div>
                  </div>
                </td>
              </tr>
            );
          })}
          {rows.length === 0 ? (
            <tr><td className="px-2 py-2 text-gray-500" colSpan={showCumulative ? 3 : 2}>No levels.</td></tr>
          ) : null}
        </tbody>
        <tfoot className="bg-[#0f0f0f] border-t border-[#1f1f1f]">
          <tr>
            <Td className="text-[11px] text-gray-400">Total</Td>
            <Td className="text-right text-[11px] text-gray-300">{fmt(footerTotal, sizePrecision)}</Td>
            {showCumulative }
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function Th({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <th className={`px-2 py-1 text-left ${className}`}>{children}</th>;
}
function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-2 py-1 ${className}`}>{children}</td>;
}

/* ---------------- helpers ---------------- */

function valid(l: Level) {
  return l && Number.isFinite(l.price) && Number.isFinite(l.size) && l.size > 0;
}

function withCumulative(levels: Level[], side: "BUY" | "SELL") {
  const out: Array<{ price: number; size: number; cum: number; orders?: number; idx: number }> = [];
  let acc = 0;
  for (let i = 0; i < levels.length; i++) {
    const lv = levels[i];
    acc += lv.size;
    out.push({ price: lv.price, size: lv.size, cum: acc, orders: lv.orders, idx: i });
  }
  // display order: bids usually top best->worse; asks likewise. Keep as provided.
  return out;
}

function sum(arr: number[]) {
  return arr.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
}

function fmt(n: number, d = 2) {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}