// heatmap.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

export type HeatmapOrder = "none" | "rowsum" | "colsum" | "both";

export interface HeatmapProps {
  rows: string[];
  cols: string[];
  values: number[][]; // shape rows x cols
  width?: number; // total component width
  height?: number; // total component height
  margin?: { top: number; right: number; bottom: number; left: number };
  /** Color scale domain. If omitted, inferred from data min/max. */
  domain?: [number, number];
  /** Diverging mid-point (e.g., 0 for correlations). If set, uses diverging scale. */
  divergingMid?: number | null;
  /** D3 interpolator — e.g., d3.interpolateTurbo, interpolateRdBu, etc. */
  interpolator?: (t: number) => string;
  /** Reorder strategy */
  order?: HeatmapOrder;
  /** Show row/col labels */
  showLabels?: boolean;
  /** Called on click of a cell */
  onCellClick?: (rIndex: number, cIndex: number, value: number) => void;
}

const defaultMargin = { top: 50, right: 20, bottom: 80, left: 120 };

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const isFiniteNum = (x: any) => typeof x === "number" && Number.isFinite(x);

function computeOrder(
  rows: string[],
  cols: string[],
  values: number[][],
  order: HeatmapOrder
) {
  const rIdx = rows.map((_, i) => i);
  const cIdx = cols.map((_, j) => j);
  if (order === "none") return { rOrder: rIdx, cOrder: cIdx };

  const rowSum = (i: number) =>
    d3.sum(values[i].map((v) => (isFiniteNum(v) ? v : 0)));
  const colSum = (j: number) =>
    d3.sum(values.map((row) => (isFiniteNum(row[j]) ? row[j] : 0)));

  if (order === "rowsum" || order === "both")
    rIdx.sort((a, b) => d3.descending(rowSum(a), rowSum(b)));
  if (order === "colsum" || order === "both")
    cIdx.sort((a, b) => d3.descending(colSum(a), colSum(b)));

  return { rOrder: rIdx, cOrder: cIdx };
}

const Heatmap: React.FC<HeatmapProps> = ({
  rows,
  cols,
  values,
  width = 980,
  height = 640,
  margin = defaultMargin,
  domain,
  divergingMid = null,
  interpolator = d3.interpolateTurbo,
  order = "none",
  showLabels = true,
  onCellClick,
}) => {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  // refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  // dims
  const innerW = Math.max(50, width - margin.left - margin.right);
  const innerH = Math.max(50, height - margin.top - margin.bottom);

  // guard
  const R = rows.length;
  const C = cols.length;
  const safeValues = values && values.length === R ? values : Array.from({ length: R }, () => Array(C).fill(NaN));

  // ordering
  const { rOrder, cOrder } = useMemo(
    () => computeOrder(rows, cols, safeValues, order),
    [rows, cols, safeValues, order]
  );

  // scales
  const xBand = useMemo(
    () => d3.scaleBand<number>().domain(cOrder).range([0, innerW]).padding(0),
    [cOrder, innerW]
  );
  const yBand = useMemo(
    () => d3.scaleBand<number>().domain(rOrder).range([0, innerH]).padding(0),
    [rOrder, innerH]
  );

  const [vmin, vmax] = useMemo(() => {
    if (domain) return domain;
    let lo = Infinity,
      hi = -Infinity;
    for (let i = 0; i < R; i++) {
      for (let j = 0; j < C; j++) {
        const v = safeValues[i]?.[j];
        if (isFiniteNum(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
    // pad a touch
    const pad = (hi - lo) * 0.02;
    return [lo - pad, hi + pad];
  }, [safeValues, R, C, domain]);

  const color = useMemo(() => {
    if (divergingMid !== null && Number.isFinite(divergingMid)) {
      const lo = Math.min(vmin, divergingMid);
      const hi = Math.max(vmax, divergingMid);
      const s = d3.scaleDiverging(interpolator as any).domain([lo, divergingMid, hi]);
      return (v: number) => (isFiniteNum(v) ? s(v) : "transparent");
    }
    const s = d3.scaleSequential(interpolator).domain([vmin, vmax]);
    return (v: number) => (isFiniteNum(v) ? s(v) : "transparent");
  }, [vmin, vmax, divergingMid, interpolator]);

  // draw canvas
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
    ctx.clearRect(0, 0, innerW, innerH);

    // apply zoom/pan transform
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    // draw cells
    const cellW = xBand.bandwidth();
    const cellH = yBand.bandwidth();
    for (let ri = 0; ri < R; ri++) {
      const rIdx = rOrder[ri];
      const y = yBand(rIdx)!;
      for (let ci = 0; ci < C; ci++) {
        const cIdx = cOrder[ci];
        const x = xBand(cIdx)!;
        const v = safeValues[rIdx]?.[cIdx];
        ctx.fillRect(x, y, cellW, cellH);
      }
    }

    ctx.restore();
  }, [innerW, innerH, R, C, rOrder, cOrder, safeValues, xBand, yBand, color, transform]);

  // axes + zoom
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svgRef.current) return;

    svg.selectAll("*").remove();

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // X labels
    if (rows.length && cols.length && showLabels) {
      const xAxis = g
        .append("g")
        .attr("transform", `translate(0, ${innerH})`)
        .attr("class", "text-[10px] fill-zinc-300");

      const ticks = xAxis
        .selectAll("g.tick")
        .data(cOrder)
        .join("g")
        .attr("transform", (d) => `translate(${(xBand(d) ?? 0) + xBand.bandwidth() / 2},0)`);

      ticks
        .append("text")
        .attr("text-anchor", "end")
        .attr("transform", "rotate(-60)")
        .attr("y", 12)
        .text((d) => cols[d]);
    }

    // Y labels
    if (showLabels) {
      const yAxis = g.append("g").attr("class", "text-[11px] fill-zinc-300");
      const ticks = yAxis
        .selectAll("g.tick")
        .data(rOrder)
        .join("g")
        .attr("transform", (d) => `translate(0, ${(yBand(d) ?? 0) + yBand.bandwidth() / 2})`);

      ticks
        .append("text")
        .attr("text-anchor", "end")
        .attr("x", -6)
        .attr("dy", "0.32em")
        .text((d) => rows[d]);
    }

    // zoom/pan (affects canvas transform)
    const zoom = d3
      .zoom<SVGRectElement, unknown>()
      .scaleExtent([0.5, 12])
      .translateExtent([
        [0, 0],
        [innerW, innerH],
      ])
      .on("zoom", (e) => setTransform(e.transform));

    // invisible rect to capture zoom
    g.append("rect")
      .attr("width", innerW)
      .attr("height", innerH)
      .attr("fill", "transparent")
      .style("cursor", "grab")
      .call(zoom as any);

    // overlay for picking (we’ll compute mapping client->cell)
    g.append("rect")
      .attr("width", innerW)
      .attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (e) => handleHover(e))
      .on("mouseleave", () => {
        setHover(null);
        if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
      })
      .on("click", (e) => handleClick(e));

    // legend
    const legendW = Math.min(240, innerW);
    const legendH = 10;
    const legend = svg
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top - 18})`);

    const gradId = `grad-${Math.random().toString(36).slice(2)}`;
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", gradId);
    grad.attr("x1", "0%").attr("x2", "100%");
    const stops = 12;
    for (let i = 0; i <= stops; i++) {
      const t = i / stops;
      grad.append("stop").attr("offset", `${t * 100}%`).attr("stop-color", (interpolator as any)(t));
    }
    legend
      .append("rect")
      .attr("width", legendW)
      .attr("height", legendH)
      .attr("fill", `url(#${gradId})`)
      .attr("rx", 2);

    const axisScale = d3.scaleLinear().domain([vmin, vmax]).range([0, legendW]);
    const axis = d3.axisBottom(axisScale).ticks(5).tickSize(3);
    legend
      .append("g")
      .attr("transform", `translate(0, ${legendH})`)
      .attr("class", "text-[10px] fill-zinc-300")
      .call(axis as any);

    function handleHover(ev: any) {
      const rect = (ev.currentTarget as SVGRectElement).getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;

      // invert zoom
      const zx = (cx - transform.x) / transform.k;
      const zy = (cy - transform.y) / transform.k;

      // map into indices
      const colIdxBand = cOrder.find((d) => {
        const x0 = xBand(d)!;
        return zx >= x0 && zx < x0 + xBand.bandwidth();
      });
      const rowIdxBand = rOrder.find((d) => {
        const y0 = yBand(d)!;
        return zy >= y0 && zy < y0 + yBand.bandwidth();
      });

      if (colIdxBand === undefined || rowIdxBand === undefined) {
        setHover(null);
        if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
        return;
      }

      const r = rowIdxBand;
      const c = colIdxBand;
      setHover({ r, c });

      const v = safeValues[r]?.[c];
      if (tooltipRef.current) {
        const el = tooltipRef.current;
        el.style.opacity = "1";
        el.style.left = `${ev.clientX + 12}px`;
        el.style.top = `${ev.clientY + 12}px`;
        el.innerHTML = `<div class="text-xs">
          <div><strong>${rows[r]}</strong> × <strong>${cols[c]}</strong></div>
          <div>value: <span class="tabular-nums">${isFiniteNum(v) ? d3.format(".4~f")(v) : "NA"}</span></div>
        </div>`;
      }
    }

    function handleClick(ev: any) {
      if (!onCellClick) return;
      // reuse hover mapping
      const rect = (ev.currentTarget as SVGRectElement).getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const zx = (cx - transform.x) / transform.k;
      const zy = (cy - transform.y) / transform.k;

      const colIdxBand = cOrder.find((d) => {
        const x0 = xBand(d)!;
        return zx >= x0 && zx < x0 + xBand.bandwidth();
      });
      const rowIdxBand = rOrder.find((d) => {
        const y0 = yBand(d)!;
        return zy >= y0 && zy < y0 + yBand.bandwidth();
      });
      if (colIdxBand === undefined || rowIdxBand === undefined) return;
      const r = rowIdxBand;
      const c = colIdxBand;
      const v = safeValues[r]?.[c];
      onCellClick(r, c, v);
    }
  }, [
    width,
    height,
    margin,
    innerW,
    innerH,
    rows,
    cols,
    rOrder,
    cOrder,
    xBand,
    yBand,
    vmin,
    vmax,
    interpolator,
    safeValues,
    transform,
    showLabels,
    onCellClick,
  ]);

  // crosshair highlight (SVG overlay)
  const cross = useMemo(() => {
    if (!hover) return null;
    const x = xBand(hover.c);
    const y = yBand(hover.r);
    if (x == null || y == null) return null;
    const bw = xBand.bandwidth();
    const bh = yBand.bandwidth();
    // forward-apply zoom transform for overlay alignment
    const tx = transform.x + x * transform.k;
    const ty = transform.y + y * transform.k;
    return { x: tx, y: ty, w: bw * transform.k, h: bh * transform.k };
  }, [hover, xBand, yBand, transform]);

  return (
    <div className="relative">
      {/* Titles / Controls placeholder (if needed, wrap externally) */}

      {/* Axes + handlers */}
      <svg ref={svgRef} width={width} height={height} className="absolute top-0 left-0 pointer-events-auto" />

      {/* Canvas for cells */}
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

      {/* Crosshair highlight */}
      {cross && (
        <svg
          width={innerW}
          height={innerH}
          className="absolute"
          style={{ left: margin.left, top: margin.top, pointerEvents: "none" }}
        >
          <rect
            x={cross.x - transform.x}
            y={cross.y - transform.y}
            width={cross.w}
            height={cross.h}
            fill="none"
            stroke="white"
            strokeOpacity={0.9}
            strokeWidth={1}
            rx={2}
          />
        </svg>
      )}

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className="pointer-events-none absolute z-10 rounded-md bg-zinc-800/95 px-3 py-2 text-zinc-100 shadow-lg opacity-0 transition-opacity"
      />
    </div>
  );
};

export default Heatmap;