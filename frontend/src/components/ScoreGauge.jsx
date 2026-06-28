import React from 'react';

const ScoreGauge = ({ score, riskBand }) => {
  const getColor = () => {
    if (score >= 750) return 'text-green-500';
    if (score >= 650) return 'text-yellow-500';
    if (score >= 500) return 'text-orange-500';
    return 'text-red-500';
  };

  const getBgColor = () => {
    if (score >= 750) return 'bg-green-100 border-green-300';
    if (score >= 650) return 'bg-yellow-100 border-yellow-300';
    if (score >= 500) return 'bg-orange-100 border-orange-300';
    return 'bg-red-100 border-red-300';
  };

  const percentage = ((score - 300) / 600) * 100;

  return (
    <div className={`p-6 rounded-xl border-2 ${getBgColor()} text-center`}>
      <h3 className="text-sm font-medium text-gray-600 mb-2">Credit Score</h3>
      <div className={`text-5xl font-bold ${getColor()}`}>{score}</div>
      <div className="mt-2">
        <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${getBgColor()}`}>
          {riskBand}
        </span>
      </div>
      <div className="mt-4 w-full bg-gray-200 rounded-full h-3">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${getColor().replace('text-', 'bg-')}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>300</span>
        <span>900</span>
      </div>
    </div>
  );
};

export default ScoreGauge;