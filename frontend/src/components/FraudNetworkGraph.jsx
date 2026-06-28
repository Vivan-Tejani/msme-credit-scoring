import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const FraudNetworkGraph = ({ nodes, edges }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!nodes || nodes.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 600;
    const height = 400;

    svg.attr('width', width).attr('height', height);

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3.forceLink(edges).id((d) => d.id).distance(100)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(25));

    const g = svg.append('g');

    const link = g
      .append('g')
      .selectAll('line')
      .data(edges)
      .enter()
      .append('line')
      .attr('stroke', '#94a3b8')
      .attr('stroke-width', (d) => Math.max(1, d.weight / 50000))
      .attr('stroke-opacity', 0.6);

    const node = g
      .append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .call(
        d3
          .drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    node
      .append('circle')
      .attr('r', (d) => Math.max(8, Math.min(25, d.volume / 50000)))
      .attr('fill', (d) => (d.is_fraudulent ? '#ef4444' : d.type === 'target' ? '#3b82f6' : '#22c55e'))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    node
      .append('text')
      .text((d) => d.id.slice(0, 8) + '...')
      .attr('x', 0)
      .attr('y', (d) => Math.max(8, Math.min(25, d.volume / 50000)) + 14)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#374151');

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges]);

  if (!nodes || nodes.length === 0) {
    return <div className="p-4 text-gray-500">No network data available</div>;
  }

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Transaction Network</h3>
      <svg ref={svgRef} className="w-full h-96 border rounded-lg bg-gray-50" />
      <div className="flex gap-4 mt-3 justify-center text-sm">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-gray-600">Target</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="text-gray-600">Related</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-600">Fraudulent</span>
        </div>
      </div>
    </div>
  );
};

export default FraudNetworkGraph;