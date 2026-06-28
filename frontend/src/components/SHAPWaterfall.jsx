import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const SHAPWaterfall = ({ explanations }) => {
  if (!explanations || explanations.length === 0) {
    return <div className="p-4 text-gray-500">No explanations available</div>;
  }

  const data = explanations.map((exp) => ({
    name: exp.feature.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    impact: exp.impact,
    direction: exp.direction,
    plainText: exp.plain_text,
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg max-w-xs">
          <p className="font-semibold text-sm mb-1">{item.name}</p>
          <p className="text-xs text-gray-600">{item.plainText}</p>
          <p className={`text-xs font-bold mt-1 ${item.direction === 'positive' ? 'text-green-600' : 'text-red-600'}`}>
            Impact: {item.impact > 0 ? '+' : ''}{item.impact}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Top Factors Affecting Score</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={['auto', 'auto']} />
            <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 12 }} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'transparent' }} />
            <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.direction === 'positive' ? '#22c55e' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 mt-3 justify-center text-sm">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 rounded-sm" />
          <span className="text-gray-600">Positive</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded-sm" />
          <span className="text-gray-600">Negative</span>
        </div>
      </div>
    </div>
  );
};

export default SHAPWaterfall;