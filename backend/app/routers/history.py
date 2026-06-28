from fastapi import APIRouter
from app.schemas import ScoreHistoryResponse, ScoreHistory
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/v1/score", tags=["history"])


@router.get("/{gstin}/history", response_model=ScoreHistoryResponse)
async def get_score_history(gstin: str):
    history = []
    base_score = random.randint(450, 780)
    
    for i in range(12):
        month_date = datetime.now() - timedelta(days=30 * (11 - i))
        score = min(900, max(300, base_score + random.randint(-60, 60)))
        
        if score >= 750:
            band = "LOW"
        elif score >= 650:
            band = "MEDIUM"
        elif score >= 500:
            band = "HIGH"
        else:
            band = "VERY HIGH"
        
        history.append(ScoreHistory(
            month=month_date.strftime("%Y-%m"),
            score=score,
            risk_band=band
        ))
    
    return ScoreHistoryResponse(gstin=gstin, history=history)