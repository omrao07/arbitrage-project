// corr-galaxy.tsx
import React, { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

/**
 * Correlation Galaxy (Force Graph)
 * - Robust TypeScript typings for D3 v7 (drag/zoom/force)
 * - Pan/zoom, drag-to-reposition, optional "pin" on drag end (hold Shift)
 * - Node size by degree, link width by |weight|
 * - Clean teardown of forces and event handlers
 */

export type GalaxyNode = {
  id: string;
  label?: string;
  group?: string;       // e.g., sector/bucket
  degree?: number;      // optional precomputed degree
  pinned?: boolean;     // if true: fx/fy are respected
  // d3 simulation props (mutated by d3)
  x?: number; y?: number; vx?: number; vy?: number; fx?: number | null; fy?: number | null;
};

export type GalaxyLink = {
  source: string;       // node id
  target: string;       // node id
  weight: number;       // correlation in [-1, 1]
};

export interface CorrGalaxyProps {
  width?: number;
  height?: number;
  nodes?: GalaxyNode[];
  links: GalaxyLink[];
  /** if nodes not provided, they are inferred from links */
  colorBy?: (n: GalaxyNode) => string; // custom color map
  onNodeClick?: (n: GalaxyNode) => void;
  pinOnShiftEnd?: boolean;             // hold Shift while dragging to pin
  strength?: number;                   // link strength base (default 0.4)
}

type DragEvt = d3.D3DragEvent<SVGCircleElement, GalaxyNode, unknown>;

const defaultPalette = d3.schemeTableau10;

const CorrGalaxy: React.FC<CorrGalaxyProps> = ({
  width = 900,
  height = 600,
  nodes,
  links,
  colorBy,
  onNodeClick,
  pinOnShiftEnd = true,
  strength = 0.4,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  const data = useMemo(() => {
    // Build node set if not provided
    const idSet = new Set<string>();
    links.forEach(l => { idSet.add(l.source); idSet.add(l.target); });
    const baseNodes: GalaxyNode[] =
      nodes && nodes.length
        ? nodes.map(n => ({ ...n }))
        : Array.from(idSet).map(id => ({ id }));

    // degree (if not given)
    const deg = new Map<string, number>();
    links.forEach(l => {
      deg.set(l.source, (deg.get(l.source) || 0) + 1);
      deg.set(l.target, (deg.get(l.target) || 0) + 1);
    });
    baseNodes.forEach(n => (n.degree = n.degree ?? deg.get(n.id) ?? 0));

    // Copy links (D3 mutates)
    const baseLinks = links.map(l => ({ ...l }));

    return { nodes: baseNodes, links: baseLinks };
  }, [nodes, links]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Canvas
    svg
      .attr("viewBox", `${-width / 2} ${-height / 2} ${width} ${height}`)
      .attr("width", width)
      .attr("height", height)
      .style("background", "#0a0a0a")
      .style("border", "1px solid #27272a")
      .style("borderRadius", "14px");

    // Layers
    const gRoot = svg.append("g");
    const gLinks = gRoot.append("g").attr("stroke", "#444").attr("stroke-opacity", 0.7);
    const gNodes = gRoot.append("g");
    const gLabels = gRoot.append("g");

    // Zoom/pan
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.25, 5])
      .on("zoom", (ev) => {
        gRoot.attr("transform", String(ev.transform));
      });
    svg.call(zoom as any);

    // Scales
    const abs = (x: number) => Math.abs(x);
    const maxAbsW = d3.max(data.links, d => abs(d.weight)) ?? 1;
    const linkW = d3.scaleLinear().domain([0, maxAbsW]).range([0.5, 3.5]);
    const nodeR = d3.scaleSqrt<number, number>()
      .domain(d3.extent(data.nodes, d => d.degree ?? 0) as [number, number])
      .range([5, 18]);

    const groups = Array.from(new Set(data.nodes.map(d => d.group).filter(Boolean))) as string[];
    const color = colorBy
      ? colorBy
      : (n: GalaxyNode) => {
          if (!groups.length) return defaultPalette[0];
          const idx = Math.abs(groups.indexOf(n.group || "")) % defaultPalette.length;
          return defaultPalette[idx] || defaultPalette[0];
        };

    // Force simulation
    const sim = d3.forceSimulation<GalaxyNode>(data.nodes)
      .force("charge", d3.forceManyBody<GalaxyNode>().strength(-40))
      .force("center", d3.forceCenter(0, 0))
      .force(
        "link",
        d3
          .forceLink<GalaxyNode, any>(data.links as any)
          .id((d: GalaxyNode) => d.id)
          .strength((l: any) => strength * Math.max(0.05, abs(l.weight)))
          .distance((l: any) => 120 * (1 - abs(l.weight)) + 30)
      )
      .force("collide", d3.forceCollide<GalaxyNode>().radius(d => nodeR(d.degree || 0) + 2))
      .alpha(1);

    // Links
    const link = gLinks
      .selectAll("line")
      .data(data.links)
      .enter()
      .append("line")
      .attr("stroke-width", d => linkW(abs(d.weight)))
      .attr("stroke", d => (d.weight >= 0 ? "#22c55e" : "#ef4444"))
      .attr("opacity", 0.85);

    // Nodes
    const node = gNodes
      .selectAll("circle")
      .data(data.nodes)
      .enter()
      .append("circle")
      .attr("r", d => nodeR(d.degree || 0))
      .attr("fill", d => color(d))
      .attr("stroke", "#111")
      .attr("stroke-width", 1.5)
      .style("cursor", "grab")
      .on("click", (_, d) => onNodeClick?.(d));

    // Labels (optional: only for larger nodes to reduce clutter)
    const label = gLabels
      .selectAll("text")
      .data(data.nodes.filter(n => (n.label ?? n.id).length > 0))
      .enter()
      .append("text")
      .text(d => d.label ?? d.id)
      .attr("font-size", "10px")
      .attr("fill", "#c7c7c7")
      .attr("stroke", "none")
      .attr("text-anchor", "middle")
      .attr("dy", d => (nodeR(d.degree || 0) + 10))
      .style("pointer-events", "none");

    // Drag — use function() to get correct `this` (the circle element)
    const drag = d3
      .drag<SVGCircleElement, GalaxyNode>()
      .on("start", function (event: DragEvt, d: GalaxyNode) {
        if (!event.active) sim.alphaTarget(0.3).restart();
        // set initial fixed position at current location so drag feels sticky
        d.fx = d.x;
        d.fy = d.y;
        d3.select(this).style("cursor", "grabbing");
      })
      .on("drag", function (event: DragEvt, d: GalaxyNode) {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", function (event: DragEvt, d: GalaxyNode) {
        if (!event.active) sim.alphaTarget(0);
        // pin only if Shift is held (or if pinOnShiftEnd=false then always leave pinned)
        const shouldPin = pinOnShiftEnd ? (event.sourceEvent instanceof MouseEvent ? event.sourceEvent.shiftKey : false) : true;
        if (shouldPin) {
          d.pinned = true;
          d.fx = event.x;
          d.fy = event.y;
        } else {
          d.pinned = false;
          d.fx = null;
          d.fy = null;
        }
        d3.select(this).style("cursor", "grab");
      });

    node.call(drag as any);

    // Tick
    sim.on("tick", () => {
      link
        .attr("x1", (d: any) => (d.source as GalaxyNode).x!)
        .attr("y1", (d: any) => (d.source as GalaxyNode).y!)
        .attr("x2", (d: any) => (d.target as GalaxyNode).x!)
        .attr("y2", (d: any) => (d.target as GalaxyNode).y!);

      node.attr("cx", d => d.x!).attr("cy", d => d.y!);
      label.attr("x", d => d.x!).attr("y", d => d.y!);
    });

    // Cleanup
    return () => {
      sim.stop();
      svg.on(".zoom", null);
    };
  }, [data, width, height, colorBy, onNodeClick, pinOnShiftEnd, strength]);

  return <svg ref={svgRef} />;
};

export default CorrGalaxy;