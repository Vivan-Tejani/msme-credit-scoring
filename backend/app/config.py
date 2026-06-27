from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "MSME Credit Scoring API"
    APP_VERSION: str = "1.0.0"
    
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/msme_credit"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    MODEL_PATH: str = "ml_models"
    MOCK_DATA_PATH: str = "data/mock_data"
    
    SCORE_MIN: int = 300
    SCORE_MAX: int = 900
    
    class Config:
        env_file = ".env"


settings = Settings()