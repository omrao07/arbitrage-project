// matrix-orderbook.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

export type MatrixOrderbookLevel = {
  /** Price level */
  price: number;
  /** Per-column bid sizes (aligned with `columns`) */
  bids: number[];
  /** Per-column ask sizes (aligned with `columns`) */
  asks: number[];
};

export interface MatrixOrderbookProps {
  /** Column labels (venues, instruments, etc.) */
  columns: string[];
  /** Aligned ladder (sorted ascending or descending — we’ll detect) */
  ladder: MatrixOrderbookLevel[];
  /** Pixel size */
  width?: number;
  height?: number;
  /** Inner margins (room for axes/labels) */
  margin?: { top: number; right: number; bottom: number; left: number };
  /** Optional fixed [min,max] size domain for color; otherwise inferred */
  sizeDomain?: [number, number] | null;
  /** If true, show a mid-price line (median of best bid/ask across columns) */
  showMid?: boolean;
  /** Called on cell click: (rowIndex, colIndex, side, price, size) */
  onCellClick?: (row: number, col: number, side: "bid" | "ask", price: number, size: number) => void;
  /** Optional formatter for price labels */
  formatPrice?: (p: number) => string;
  /** Optional formatter for size labels (tooltip) */
  formatSize?: (s: number) => string;
}

/**
 * MatrixOrderbook
 *  - Rows are price levels, columns are markets/venues.
 *  - Each cell encodes bid (left triangle) and ask (right triangle) depth by color intensity.
 *  - Best bid/ask per column are outlined; global mid line optional.
 *  - Canvas for performance; SVG for axes/overlays.
 *  - Wheel/drag to zoom & pan vertically.
 */
const MatrixOrderbook: React.FC<MatrixOrderbookProps> = ({
  columns,
  ladder,
  width = 980,
  height = 640,
  margin = { top: 28, right: 16, bottom: 32, left: 56 },
  sizeDomain = null,
  showMid = true,
  onCellClick,
  formatPrice = (p) => d3.format(",.2f")(p),
  formatSize = (s) => (Math.abs(s) >= 1000 ? d3.format(".2~s")(s) : d3.format(",~f")(s)),
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const [zoomT, setZoomT] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  const innerW = Math.max(120, width - margin.left - margin.right);
  const innerH = Math.max(80, height - margin.top - margin.bottom);
  const C = columns.length;
  const R = ladder.length;

  // Guard: ensure arrays align
  const safeLadder: MatrixOrderbookLevel[] = useMemo(() => {
    return ladder.map((lv) => ({
      price: lv.price,
      bids: Array.from({ length: C }, (_, j) => +((lv.bids ?? [])[j] ?? 0)),
      asks: Array.from({ length: C }, (_, j) => +((lv.asks ?? [])[j] ?? 0)),
    }));
  }, [ladder, C]);

  // Determine if prices ascend or descend, and build a y scale (band by row index)
  const ascending = useMemo(() => {
    if (R < 2) return true;
    return safeLadder[0].price <= safeLadder[R - 1].price;
  }, [safeLadder, R]);

  const rowIndexToPrice = (ri: number) => (ascending ? safeLadder[ri].price : safeLadder[R - 1 - ri].price);

  // x scale: one column per market (band)
  const xBand = useMemo(
    () => d3.scaleBand<number>().domain(d3.range(C)).range([0, innerW]).paddingInner(0.08).paddingOuter(0.04),
    [C, innerW]
  );

  // y scale: price rows (band)
  const yBand = useMemo(
    () => d3.scaleBand<number>().domain(d3.range(R)).range([0, innerH]).padding(0),
    [R, innerH]
  );

  // Color scales (separate for bid/ask)
  const [minSize, maxSize] = useMemo(() => {
    if (sizeDomain) return sizeDomain;
    let lo = Infinity,
      hi = 0;
    for (let i = 0; i < R; i++) {
      const bMax = d3.max(safeLadder[i].bids) ?? 0;
      const aMax = d3.max(safeLadder[i].asks) ?? 0;
      const m = Math.max(bMax, aMax);
      if (m > hi) hi = m;
      if (m < lo) lo = Math.min(lo, m);
    }
    if (!Number.isFinite(lo)) lo = 0;
    return [0, hi || 1];
  }, [safeLadder, R, sizeDomain]);

  const bidColor = useMemo(
    () =>
      d3
        .scaleSqrt<string>()
        .domain([0, maxSize || 1e-9])
        .range(["#0b1320", "#60a5fa"]), // dark → blue
    [maxSize]
  );
  const askColor = useMemo(
    () =>
      d3
        .scaleSqrt<string>()
        .domain([0, maxSize || 1e-9])
        .range(["#1a0f10", "#f87171"]), // dark → red
    [maxSize]
  );

  // Best bid/ask per column (row indices)
  const bestByCol = useMemo(() => {
    const out: { bid: number | null; ask: number | null }[] = [];
    for (let j = 0; j < C; j++) {
      let bestBid: { price: number; ri: number } | null = null;
      let bestAsk: { price: number; ri: number } | null = null;
      for (let i = 0; i < R; i++) {
        const p = rowIndexToPrice(i);
        if ((safeLadder[i].bids[j] ?? 0) > 0) {
          if (!bestBid || p > bestBid.price) bestBid = { price: p, ri: i };
        }
        if ((safeLadder[i].asks[j] ?? 0) > 0) {
          if (!bestAsk || p < bestAsk.price) bestAsk = { price: p, ri: i };
        }
      }
      out.push({ bid: bestBid?.ri ?? null, ask: bestAsk?.ri ?? null });
    }
    return out;
  }, [C, R, safeLadder, ascending]);

  // Mid price (median of midpoints across columns where available)
  const midPrice = useMemo(() => {
    const mids: number[] = [];
    for (let j = 0; j < C; j++) {
      const bb = bestByCol[j].bid;
      const aa = bestByCol[j].ask;
      if (bb != null && aa != null) {
        const pb = rowIndexToPrice(bb);
        const pa = rowIndexToPrice(aa);
        if (Number.isFinite(pb) && Number.isFinite(pa)) mids.push((pb + pa) / 2);
      }
    }
    return mids.length ? d3.median(mids) ?? null : null;
  }, [bestByCol]);

  // Drawing (Canvas)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = innerW * ratio;
    canvas.height = innerH * ratio;
    canvas.style.width = `${innerW}px`;
    canvas.style.height = `${innerH}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, innerW, innerH);

    // Apply zoom transform
    ctx.save();
    ctx.translate(zoomT.x, zoomT.y);
    ctx.scale(zoomT.k, zoomT.k);

    // Draw cells: two triangles per cell (left=bid, right=ask)
    for (let i = 0; i < R; i++) {
      const y = yBand(i)!;
      const h = yBand.bandwidth();
      for (let j = 0; j < C; j++) {
        const x = xBand(j)!;
        const w = xBand.bandwidth();
        const b = safeLadder[i].bids[j] || 0;
        const a = safeLadder[i].asks[j] || 0;

        // bid (left triangle)
        if (b > 0) {
          ctx.fillStyle = bidColor(b);
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + w * 0.5, y + h * 0.5);
          ctx.lineTo(x, y + h);
          ctx.closePath();
          ctx.fill();
        }

        // ask (right triangle)
        if (a > 0) {
          ctx.fillStyle = askColor(a);
          ctx.beginPath();
          ctx.moveTo(x + w, y);
          ctx.lineTo(x + w * 0.5, y + h * 0.5);
          ctx.lineTo(x + w, y + h);
          ctx.closePath();
          ctx.fill();
        }

        // faint cell border
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.strokeRect(x, y, w, h);
      }
    }

    // Outline best bid/ask per column
    ctx.lineWidth = 1;
    for (let j = 0; j < C; j++) {
      const w = xBand.bandwidth();
      const x = xBand(j)!;
      const bb = bestByCol[j].bid;
      const aa = bestByCol[j].ask;
      if (bb != null) {
        const y = yBand(bb)!;
        ctx.strokeStyle = "#60a5fa";
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, yBand.bandwidth() - 1);
      }
      if (aa != null) {
        const y = yBand(aa)!;
        ctx.strokeStyle = "#f87171";
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, yBand.bandwidth() - 1);
      }
    }

    ctx.restore();
  }, [innerW, innerH, xBand, yBand, zoomT, safeLadder, bidColor, askColor, bestByCol, R, C]);

  // Axes / interactions (SVG)
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svgRef.current) return;
    svg.selectAll("*").remove();

    const root = svg.attr("viewBox", `0 0 ${width} ${height}`).append("g");

    // Titles / column labels
    const gTop = root.append("g").attr("transform", `translate(${margin.left},${margin.top - 8})`);
    gTop
      .selectAll("text.col")
      .data(columns)
      .join("text")
      .attr("class", "col")
      .attr("x", (_d, j) => (xBand(j) ?? 0) + xBand.bandwidth() / 2)
      .attr("y", 0)
      .attr("text-anchor", "middle")
      .attr("fill", "#e5e7eb")
      .attr("font-size", 11)
      .text((d) => d);

    // Price axis (left)
    const gLeft = root.append("g").attr("transform", `translate(${margin.left - 6},${margin.top})`);
    // Choose ~12 ticks by row index; render price text
    const ticks = 12;
    const tickRows = d3.range(0, R, Math.max(1, Math.floor(R / ticks)));
    gLeft
      .selectAll("text.tick")
      .data(tickRows)
      .join("text")
      .attr("class", "tick")
      .attr("x", 0)
      .attr("y", (i) => (yBand(i) ?? 0) + yBand.bandwidth() / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .attr("fill", "#cbd5e1")
      .attr("font-size", 10)
      .text((i) => formatPrice(rowIndexToPrice(i)));

    // Mid price line
    if (showMid && midPrice != null) {
      // find nearest row (for placement)
      let nearest = 0;
      let best = Infinity;
      for (let i = 0; i < R; i++) {
        const p = rowIndexToPrice(i);
        const d = Math.abs(p - (midPrice as number));
        if (d < best) {
          best = d;
          nearest = i;
        }
      }
      const y = (yBand(nearest) ?? 0) + margin.top + yBand.bandwidth() / 2 + zoomT.y;
      root
        .append("line")
        .attr("x1", margin.left)
        .attr("x2", margin.left + innerW)
        .attr("y1", y)
        .attr("y2", y)
        .attr("stroke", "#f5f5f5")
        .attr("stroke-dasharray", "2,3")
        .attr("stroke-opacity", 0.6);
      root
        .append("text")
        .attr("x", margin.left + innerW + 4)
        .attr("y", y)
        .attr("dy", "0.32em")
        .attr("fill", "#f5f5f5")
        .attr("font-size", 10)
        .text(`mid~${formatPrice(midPrice)}`);
    }

    // Legend
    const legend = root
      .append("g")
      .attr("transform", `translate(${margin.left}, ${height - margin.bottom + 16})`);
    legend
      .append("rect")
      .attr("x", 0)
      .attr("y", -10)
      .attr("width", 12)
      .attr("height", 12)
      .attr("fill", bidColor(maxSize));
    legend.append("text").attr("x", 16).attr("y", 0).attr("dy", "-0.1em").attr("fill", "#cbd5e1").attr("font-size", 11).text("bid depth");
    legend
      .append("rect")
      .attr("x", 90)
      .attr("y", -10)
      .attr("width", 12)
      .attr("height", 12)
      .attr("fill", askColor(maxSize));
    legend
      .append("text")
      .attr("x", 106)
      .attr("y", 0)
      .attr("dy", "-0.1em")
      .attr("fill", "#cbd5e1")
      .attr("font-size", 11)
      .text("ask depth");

    // Zoom / pan (vertical)
    const zoom = d3
      .zoom<SVGRectElement, unknown>()
      .scaleExtent([0.5, 8])
      .on("zoom", (e) => {
        // Restrict to vertical panning/zooming by overriding x
        const t = e.transform;
        setZoomT(new d3.ZoomTransform(1, t.y, t.k));
      });

    // Interaction capture rect (over canvas area)
    root
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
      .append("rect")
      .attr("width", innerW)
      .attr("height", innerH)
      .attr("fill", "transparent")
      .style("cursor", "ns-resize")
      .call(zoom as any)
      .on("mousemove", (ev) => handleHover(ev))
      .on("mouseleave", () => {
        if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
      })
      .on("click", (ev) => handleClick(ev));

    function handleHover(ev: any) {
      const rect = (ev.currentTarget as SVGRectElement).getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;

      // invert zoom (vertical only)
      const zx = cx; // x unaffected
      const zy = (cy - zoomT.y) / zoomT.k;

      // map to indices
      const j = d3.range(C).find((jj) => {
        const x = xBand(jj)!;
        return zx >= x && zx < x + xBand.bandwidth();
      });
      const i = d3.range(R).find((ii) => {
        const y = yBand(ii)!;
        return zy >= y && zy < y + yBand.bandwidth();
      });

      if (j == null || i == null) {
        if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
        return;
      }

      const price = rowIndexToPrice(i);
      const b = safeLadder[i].bids[j] || 0;
      const a = safeLadder[i].asks[j] || 0;

      const side = b >= a ? "bid" : "ask";
      const size = side === "bid" ? b : a;

      if (tooltipRef.current) {
        const el = tooltipRef.current;
        el.style.opacity = "1";
        el.style.left = `${ev.clientX + 12}px`;
        el.style.top = `${ev.clientY + 12}px`;
        el.innerHTML = `<div class="text-xs">
          <div><strong>${columns[j]}</strong> @ <strong>${formatPrice(price)}</strong></div>
          <div>bid: <span class="tabular-nums">${formatSize(b)}</span> | ask: <span class="tabular-nums">${formatSize(a)}</span></div>
        </div>`;
      }
    }

    function handleClick(ev: any) {
      if (!onCellClick) return;
      const rect = (ev.currentTarget as SVGRectElement).getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const zx = cx;
      const zy = (cy - zoomT.y) / zoomT.k;
      const j = d3.range(C).find((jj) => {
        const x = xBand(jj)!;
        return zx >= x && zx < x + xBand.bandwidth();
      });
      const i = d3.range(R).find((ii) => {
        const y = yBand(ii)!;
        return zy >= y && zy < y + yBand.bandwidth();
      });
      if (j == null || i == null) return;
      const price = rowIndexToPrice(i);
      const b = safeLadder[i].bids[j] || 0;
      const a = safeLadder[i].asks[j] || 0;
      const side = b >= a ? "bid" : "ask";
      const size = side === "bid" ? b : a;
      onCellClick(i, j, side, price, size);
    }
  }, [
    width,
    height,
    margin,
    innerW,
    innerH,
    columns,
    xBand,
    yBand,
    rowIndexToPrice,
    showMid,
    midPrice,
    bidColor,
    askColor,
    maxSize,
    onCellClick,
    zoomT,
    C,
    R,
    safeLadder,
    formatPrice,
    formatSize,
  ]);

  return (
    <div className="relative">
      {/* SVG overlay (axes, labels, legend, interaction) */}
      <svg ref={svgRef} width={width} height={height} className="absolute top-0 left-0 pointer-events-auto" />

      {/* Canvas drawing area */}
      <div
        className="bg-zinc-900 rounded-2xl overflow-hidden"
        style={{ width, height, padding: 0 }}
      >
        <div
          style={{
            transform: `translate(${margin.left}px, ${margin.top}px)`,
            width: innerW,
            height: innerH,
          }}
        >
          <canvas ref={canvasRef} />
        </div>
      </div>

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className="pointer-events-none absolute z-10 rounded-md bg-zinc-800/95 px-3 py-2 text-zinc-100 shadow-lg opacity-0 transition-opacity"
      />
    </div>
  );
};

export default MatrixOrderbook;