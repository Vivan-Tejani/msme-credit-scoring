import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const scoreGSTIN = async (gstin, loanAmount = null, includeFraudCheck = true) => {
  const payload = {
    gstin,
    include_fraud_check: includeFraudCheck,
  };
  if (loanAmount) {
    payload.loan_amount_requested = loanAmount;
  }
  const response = await api.post('/v1/score', payload);
  return response.data;
};

export const getScoreHistory = async (gstin) => {
  const response = await api.get(`/v1/score/${gstin}/history`);
  return response.data;
};

export const getFraudNetwork = async (gstin, radius = 2) => {
  const response = await api.get(`/v1/fraud/network/${gstin}`, {
    params: { radius },
  });
  return response.data;
};

export default api;