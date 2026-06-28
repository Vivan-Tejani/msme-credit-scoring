from fastapi import APIRouter, HTTPException
from app.schemas import FraudNetworkResponse, FraudNetworkNode, FraudNetworkEdge
from app.services.data_generator import MSMEDataGenerator
from app.services.fraud_detection import FraudDetector

router = APIRouter(prefix="/v1/fraud", tags=["fraud"])

fraud_detector = FraudDetector()


@router.get("/network/{gstin}", response_model=FraudNetworkResponse)
async def get_fraud_network(gstin: str, radius: int = 2):
    try:
        generator = MSMEDataGenerator(gstin)
        data = generator.generate_all() #type:ignore
        
        transactions = []
        invoices = data.get("gst_invoices")
        if invoices is not None and not invoices.empty:
            for _, row in invoices.iterrows():
                transactions.append({
                    "sender_gstin": gstin,
                    "receiver_gstin": row["counterparty_gstin"],
                    "amount": float(row["total_amount"]),
                    "date": str(row["date"])
                })
        
        fraud_detector.build_graph(transactions)
        nodes, edges = fraud_detector.get_network(gstin, radius=radius)
        
        return FraudNetworkResponse(
            gstin=gstin,
            nodes=[FraudNetworkNode(**n) for n in nodes],
            edges=[FraudNetworkEdge(**e) for e in edges],
            cycle_detected=len(fraud_detector.detect_cycles(gstin)) > 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud network error: {str(e)}")