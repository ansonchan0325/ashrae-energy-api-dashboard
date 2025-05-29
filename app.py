# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ensemble import run_ensemble
import pandas as pd
import logging

class PredictRequest(BaseModel):
    building_id: int
    meter:       int
    start_date:  str  # YYYY-MM-DD
    end_date:    str  # YYYY-MM-DD

class Prediction(BaseModel):
    building_id: int
    meter:       int
    start_date:  str  # YYYY-MM-DD
    end_date:    str  # YYYY-MM-DD
    row_id: int
    timestamp: str
    xgb2: float
    xgb5: float
    lgbm3: float
    prediction: float

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

@app.post(
  "/predict",
  response_model=List[Prediction],    # <-- expect a JSON array
  summary="Return hourly prediction records"
)
async def predict(req: PredictRequest):
    try:
        records = run_ensemble(req.dict())
        return records
    except Exception as e:
        logger.exception("Error in run_ensemble")
        raise HTTPException(status_code=400, detail=str(e))
