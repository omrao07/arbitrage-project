// frontend/components/KnowledgeGraphView.tsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export interface KGNode {
  y: any;
  x: any;
  id: string;
  label: string;
  group?: string;
}

export interface KGLink {
  source: string;
  target: string;
  label?: string;
}

interface Props {
  nodes: KGNode[];
  links: KGLink[];
  width?: number;
  height?: number;
}

export default function KnowledgeGraphView({ nodes, links, width = 800, height = 500 }: Props) {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    d3.select(ref.current).selectAll("*").remove();

    const svg = d3.select(ref.current)
      .attr("width", width)
      .attr("height", height)
      .call(d3.zoom<SVGSVGElement, unknown>().on("zoom", (event) => {
        g.attr("transform", event.transform);
      }));

    const g = svg.append("g");

    // simulation
    const sim = d3.forceSimulation(nodes as any)
      .force("link", d3.forceLink(links).id((d: any) => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-250))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // links
    const link = g.append("g")
      .attr("stroke", "#aaa")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke-width", 1.5);

    // nodes
    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 18)
      .attr("fill", (d) => d.group === "risk" ? "#FF3B30" : d.group === "alpha" ? "#34C759" : "#0A84FF")
      .call(d3.drag<SVGCircleElement, KGNode>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // labels
    const label = g.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.label)
      .attr("text-anchor", "middle")
      .attr("dy", ".35em")
      .style("font-size", "10px")
      .style("pointer-events", "none");

    sim.on("tick", () => {
      link
        .attr("x1", (d: any) => (d.source as KGNode).x!)
        .attr("y1", (d: any) => (d.source as KGNode).y!)
        .attr("x2", (d: any) => (d.target as KGNode).x!)
        .attr("y2", (d: any) => (d.target as KGNode).y!);

      node
        .attr("cx", (d: any) => d.x!)
        .attr("cy", (d: any) => d.y!);

      label
        .attr("x", (d: any) => d.x!)
        .attr("y", (d: any) => d.y!);
    });

    function dragstarted(event: any, d: any) {
      if (!event.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event: any, d: any) {
      if (!event.active) sim.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    return () => { sim.stop(); };
  }, [nodes, links, width, height]);

  return (
    <div className="rounded-2xl shadow-md bg-white dark:bg-gray-900 p-4">
      <h2 className="text-lg font-semibold mb-2">Knowledge Graph</h2>
      <svg ref={ref}></svg>
    </div>
  );
}