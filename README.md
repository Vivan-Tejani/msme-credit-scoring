# MSME CreditIQ

An alternative credit scoring and fraud detection system for Indian MSMEs. Enter a GSTIN and loan amount to get a credit score, probability of default, recommended loan amount, and a breakdown of the top factors driving the decision, all derived from the business's digital transaction footprint instead of a CIBIL score.

> Trained on synthetic data (5,000 samples). AUC ~0.66 on held-out test set.

[Dashboard](image.png)

## Features

- **Credit Score (300–900)**: Risk-banded score with HIGH / MEDIUM / LOW classification
- **Probability of Default**: Model output as a percentage with visual indicator
- **Recommended Loan Amount**: Adjusted from requested amount based on repayment capacity
- **SHAP Explainability**: Top factors affecting the score, with positive/negative contribution breakdown
- **Score Trend (12 Months)**: Historical score chart with approval threshold line
- **Transaction Network Graph**: Maps related entities, flags circular transactions and fraudulent nodes
- **Fraud Check**: Graph-based anomaly detection using NetworkX (cycles, PageRank, community detection)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, Vite, Recharts, D3 |
| Backend | FastAPI, Python 3.11 |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Explainability | SHAP |
| Fraud Detection | NetworkX |
| Database | PostgreSQL |
| Cache | Redis |

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL
- Redis

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Train the models first (required before starting the backend):

```bash
cd notebooks
python model_training.py
```

This generates `backend/ml_models/` with 5 `.pkl` files. Takes ~10–15 minutes using multiprocessing.

Start the backend:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

## API Endpoints

```
POST /v1/score
  Body: { "gstin": "27AAPFU0939F1ZV", "loan_amount": 2500000 }

GET  /v1/fraud/network/{gstin}

GET  /v1/score/{gstin}/history
```

## ML Pipeline

**Model:** Stacked ensemble — XGBoost + LightGBM as base learners, Logistic Regression meta-learner trained on out-of-fold predictions (no data leakage).

**53 features across 4 categories:**
- Cash flow — `inflow_ratio`, `working_capital_ratio`, `upi_counterparty_diversity`
- Revenue — `revenue_growth_3m`, `avg_invoice_value`, `eway_bill_value`
- Compliance — `mca_compliance_score`, `gst_filing_timeliness_score`, `not_filed_ratio`
- Risk signals — `customer_concentration_hhi`, `circular_flow_score`, `invoice_regularity_score`
