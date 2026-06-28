import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const ScoreTrend = ({ history }) => {
  if (!history || history.length === 0) {
    return <div className="p-4 text-gray-500">No score history available</div>;
  }

  const data = history.map((h) => ({
    month: h.month,
    score: h.score,
    band: h.risk_band,
  }));

  const getBandColor = (score) => {
    if (score >= 750) return '#22c55e';
    if (score >= 650) return '#eab308';
    if (score >= 500) return '#f97316';
    return '#ef4444';
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-semibold text-sm">{item.month}</p>
          <p className="text-lg font-bold text-gray-800">{item.score}</p>
          <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
            item.score >= 750 ? 'bg-green-100 text-green-700' :
            item.score >= 650 ? 'bg-yellow-100 text-yellow-700' :
            item.score >= 500 ? 'bg-orange-100 text-orange-700' :
            'bg-red-100 text-red-700'
          }`}>
            {item.band}
          </span>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Score Trend (12 Months)</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="month" tick={{ fontSize: 12 }} />
            <YAxis domain={[300, 900]} tick={{ fontSize: 12 }} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={650} stroke="#f97316" strokeDasharray="4 4" label={{ value: 'Approval Threshold', position: 'insideTopRight', fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#3b82f6"
              strokeWidth={3}
              dot={{ r: 4, fill: '#3b82f6', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 6, fill: '#1d4ed8' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ScoreTrend;