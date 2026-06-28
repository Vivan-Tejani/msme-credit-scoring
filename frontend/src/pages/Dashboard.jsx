import React, { useState } from 'react';
import { scoreGSTIN, getScoreHistory, getFraudNetwork } from '../services/api';
import ScoreGauge from '../components/ScoreGauge';
import SHAPWaterfall from '../components/SHAPWaterfall';
import FraudNetworkGraph from '../components/FraudNetworkGraph';
import ScoreTrend from '../components/ScoreTrend';

const Dashboard = () => {
  const [gstin, setGstin] = useState('');
  const [loanAmount, setLoanAmount] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [scoreData, setScoreData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [networkData, setNetworkData] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!gstin || gstin.length !== 15) {
      setError('Please enter a valid 15-character GSTIN');
      return;
    }
    setError('');
    setLoading(true);
    setScoreData(null);
    setHistoryData(null);
    setNetworkData(null);

    try {
      const amount = loanAmount ? parseInt(loanAmount) : null;
      const [scoreRes, historyRes, networkRes] = await Promise.all([
        scoreGSTIN(gstin, amount, true),
        getScoreHistory(gstin),
        getFraudNetwork(gstin),
      ]);
      setScoreData(scoreRes);
      setHistoryData(historyRes);
      setNetworkData(networkRes);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch data. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (band) => {
    if (band === 'LOW') return 'text-green-600 bg-green-50';
    if (band === 'MEDIUM') return 'text-yellow-600 bg-yellow-50';
    if (band === 'HIGH') return 'text-orange-600 bg-orange-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-800">MSME CreditIQ</h1>
        <p className="text-sm text-gray-500">Real-time Alternative Credit Scoring & Fraud Detection</p>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <form onSubmit={handleSubmit} className="bg-white p-6 rounded-xl border border-gray-200 mb-8">
          <div className="flex flex-col md:flex-row gap-4 items-end">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">GSTIN</label>
              <input
                type="text"
                value={gstin}
                onChange={(e) => setGstin(e.target.value.toUpperCase())}
                placeholder="27AAPFU0939F1ZV"
                maxLength={15}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none uppercase"
              />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">Loan Amount (₹)</label>
              <input
                type="number"
                value={loanAmount}
                onChange={(e) => setLoanAmount(e.target.value)}
                placeholder="2500000"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analyzing...' : 'Score'}
            </button>
          </div>
          {error && <p className="text-red-500 text-sm mt-3">{error}</p>}
        </form>

        {scoreData && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <ScoreGauge score={scoreData.score} riskBand={scoreData.risk_band} />
              
              <div className="bg-white p-6 rounded-xl border border-gray-200">
                <h3 className="text-sm font-medium text-gray-600 mb-2">Probability of Default</h3>
                <p className="text-3xl font-bold text-gray-800">{(scoreData.probability_of_default * 100).toFixed(1)}%</p>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full bg-red-500"
                    style={{ width: `${scoreData.probability_of_default * 100}%` }}
                  />
                </div>
              </div>

              <div className="bg-white p-6 rounded-xl border border-gray-200">
                <h3 className="text-sm font-medium text-gray-600 mb-2">Recommended Loan</h3>
                <p className="text-2xl font-bold text-gray-800">
                  ₹{(scoreData.recommended_loan_amount / 100000).toFixed(2)} L
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Max: ₹{(scoreData.recommended_loan_amount_max / 100000).toFixed(2)} L
                </p>
                <span className={`inline-block mt-2 px-2 py-1 rounded text-xs font-semibold ${getRiskColor(scoreData.risk_band)}`}>
                  {scoreData.risk_band}
                </span>
              </div>

              <div className="bg-white p-6 rounded-xl border border-gray-200">
                <h3 className="text-sm font-medium text-gray-600 mb-2">Fraud Check</h3>
                <p className={`text-2xl font-bold ${scoreData.fraud.fraud_flag ? 'text-red-600' : 'text-green-600'}`}>
                  {scoreData.fraud.fraud_flag ? 'FLAGGED' : 'CLEAR'}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Score: {scoreData.fraud.fraud_score}
                </p>
                {scoreData.fraud.fraud_reasons.length > 0 && (
                  <ul className="mt-2 text-xs text-red-600 list-disc list-inside">
                    {scoreData.fraud.fraud_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <SHAPWaterfall explanations={scoreData.explanations} />
              <ScoreTrend history={historyData?.history || []} />
            </div>

            {networkData && (
              <div className="mb-8">
                <FraudNetworkGraph nodes={networkData.nodes} edges={networkData.edges} />
              </div>
            )}

            <div className="bg-white p-4 rounded-xl border border-gray-200 text-xs text-gray-500 flex justify-between">
              <span>Request ID: {scoreData.request_id}</span>
              <span>Latency: {scoreData.latency_ms}ms</span>
              <span>Freshness: {new Date(scoreData.data_freshness_timestamp).toLocaleString()}</span>
            </div>
          </>
        )}
      </main>
    </div>
  );
};

export default Dashboard;