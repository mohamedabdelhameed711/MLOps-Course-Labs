import logging
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from api.model import load_model
from prometheus_fastapi_instrumentator import Instrumentator

# import hyperdx
# hyperdx.init()  # set HYPERDX_API_KEY env var


# Logging config — verbose for demo; trim in prod
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Bank Customer Churn – FastAPI",
    description="Predict whether a customer will leave the bank.",
    version="0.1.0",
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)


# pydantic input schema 
class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: str  # e.g. 'France'
    Gender: str     # 'Male' / 'Female'
    Age: int = Field(..., ge=18, le=120)
    Tenure: int = Field(..., ge=0)
    Balance: float
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float

    @validator("Gender")
    def gender_cap(cls, v):          # model expects capitalized first letter
        return v.capitalize()

class Customers(BaseModel):
    data: List[Customer]


# endpoints 
@app.get("/", tags=["Home"])
def read_root():
    return {"message": "Bank Churn Prediction API. Go to /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/predict", tags=["Predict"])
def predict(payload: Customers):
    try:
        model = load_model()
        df = pd.DataFrame([c.dict() for c in payload.data])
        proba = model.predict_proba(df)[:, 1]  # churn probability
        preds = (proba >= 0.5).astype(int)

        logger.info("Predicted %d records", len(preds))
        return {
            "predictions": preds.tolist(),
            "probabilities": proba.round(4).tolist(),
        }
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
