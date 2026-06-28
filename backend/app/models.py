from sqlalchemy import Column, String, Float, DateTime, Integer, JSON, create_engine #type:ignore
from sqlalchemy.orm import declarative_base, sessionmaker #type: ignore
from datetime import datetime

Base = declarative_base()


class ScoreRecord(Base):
    __tablename__ = "score_records"
    
    id = Column(String, primary_key=True)
    gstin = Column(String, index=True, nullable=False)
    score = Column(Integer, nullable=False)
    risk_band = Column(String, nullable=False)
    probability_of_default = Column(Float, nullable=False)
    recommended_loan_amount = Column(Integer)
    recommended_loan_amount_max = Column(Integer)
    confidence = Column(Float, nullable=False)
    fraud_flag = Column(String, default="false")
    fraud_score = Column(Float, default=0.0)
    explanations = Column(JSON)
    request_id = Column(String, unique=True)
    latency_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class FraudNetworkCache(Base):
    __tablename__ = "fraud_network_cache"
    
    gstin = Column(String, primary_key=True)
    network_data = Column(JSON)
    cycle_detected = Column(String, default="false")
    community_id = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)


class FeatureCache(Base):
    __tablename__ = "feature_cache"
    
    gstin = Column(String, primary_key=True)
    features = Column(JSON)
    data_freshness = Column(DateTime)
    ttl_hours = Column(Integer, default=24)
    updated_at = Column(DateTime, default=datetime.utcnow)