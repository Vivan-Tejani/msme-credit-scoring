from fastapi import APIRouter, HTTPException
from app.schemas import ScoreRequest, ScoreResponse, FraudResult
from app.services.data_generator import MSMEDataGenerator
from app.services.feature_engineering import FeatureEngineer
from app.services.scoring_engine import ScoringEngine
from app.services.fraud_detection import FraudDetector

router = APIRouter(prefix="/v1/score", tags=["scoring"])

scoring_engine = ScoringEngine()
feature_engineer = FeatureEngineer()
fraud_detector = FraudDetector()


@router.post("", response_model=ScoreResponse)
async def score_gstin(request: ScoreRequest):
    try:
        generator = MSMEDataGenerator(request.gstin)
        data = generator.generate_all()  # type: ignore
        
        features = feature_engineer.compute_features(data)
        score_result = scoring_engine.score(
            features, 
            loan_amount=request.loan_amount_requested
        )
        
        fraud_result = FraudResult(
            fraud_flag=False,
            fraud_score=0.0,
            fraud_reasons=[],
            circular_entities=[],
            recommended_action="APPROVE"
        )
        
        if request.include_fraud_check:
            transactions = []
            invoices = data.get("gst_invoices")
            if invoices is not None and not invoices.empty:
                for _, row in invoices.iterrows():
                    transactions.append({
                        "sender_gstin": request.gstin,
                        "receiver_gstin": row["counterparty_gstin"],
                        "amount": float(row["total_amount"]),
                        "date": str(row["date"])
                    })
            
            upi_df = data.get("upi_transactions")
            mca_data = data.get("mca_data")
            
            fraud_analysis = fraud_detector.analyze(
                request.gstin,
                transactions_data=transactions,
                upi_df=upi_df,
                mca_data=mca_data
            )
            
            fraud_result = FraudResult(
                fraud_flag=fraud_analysis["fraud_flag"],
                fraud_score=fraud_analysis["fraud_score"],
                fraud_reasons=fraud_analysis["fraud_reasons"],
                circular_entities=fraud_analysis["circular_entities"],
                recommended_action=fraud_analysis["recommended_action"]
            )
        
        if fraud_result.fraud_flag:
            score_result["recommended_loan_amount"] = int(score_result["recommended_loan_amount"] * 0.5)
            score_result["recommended_loan_amount_max"] = int(score_result["recommended_loan_amount_max"] * 0.5)
        
        return ScoreResponse(
            score=score_result["score"],
            risk_band=score_result["risk_band"],
            probability_of_default=score_result["probability_of_default"],
            recommended_loan_amount=score_result["recommended_loan_amount"],
            recommended_loan_amount_max=score_result["recommended_loan_amount_max"],
            confidence=score_result["confidence"],
            data_freshness_timestamp=score_result["data_freshness_timestamp"],
            explanations=score_result["explanations"],
            fraud=fraud_result,
            request_id=score_result["request_id"],
            latency_ms=score_result["latency_ms"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {str(e)}")