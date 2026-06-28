from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import score, fraud, history

app = FastAPI(
    title="MSME Credit Scoring API",
    description="Real-time alternative credit scoring & fraud detection for Indian MSMEs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score.router)
app.include_router(fraud.router)
app.include_router(history.router)

@app.get("/")
async def root():
    return {
        "service": "MSME Credit Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "msme-credit-scoring"}