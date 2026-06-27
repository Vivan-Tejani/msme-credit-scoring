from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ScoreRequest(BaseModel):
    gstin: str = Field(..., pattern=r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")
    loan_amount_requested: Optional[int] = Field(None, ge=50000, le=50000000)
    purpose: Optional[str] = "working_capital"
    include_fraud_check: bool = True


class Explanation(BaseModel):
    rank: int
    feature: str
    direction: str
    impact: float
    plain_text: str


class FraudResult(BaseModel):
    fraud_flag: bool
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    fraud_reasons: List[str] = []
    circular_entities: List[str] = []
    recommended_action: str = "APPROVE"


class ScoreResponse(BaseModel):
    score: int = Field(..., ge=300, le=900)
    risk_band: str
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    recommended_loan_amount: Optional[int] = None
    recommended_loan_amount_max: Optional[int] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    data_freshness_timestamp: datetime
    explanations: List[Explanation]
    fraud: FraudResult
    request_id: str
    latency_ms: int


class ScoreHistory(BaseModel):
    month: str
    score: int
    risk_band: str


class ScoreHistoryResponse(BaseModel):
    gstin: str
    history: List[ScoreHistory]


class FraudNetworkNode(BaseModel):
    id: str
    type: str
    volume: float
    is_fraudulent: bool


class FraudNetworkEdge(BaseModel):
    source: str
    target: str
    weight: float


class FraudNetworkResponse(BaseModel):
    gstin: str
    nodes: List[FraudNetworkNode]
    edges: List[FraudNetworkEdge]
    community_id: Optional[str] = None
    cycle_detected: bool