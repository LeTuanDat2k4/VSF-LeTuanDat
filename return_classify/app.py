"""
FastAPI Demo – Return Prediction Service
=========================================
Loads the trained pipeline from MLflow and exposes REST endpoints
for predicting whether an order will be returned.

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

# ── Global model holder ───────────────────────────────────────────
MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model from MLflow on startup."""
    global MODEL
    print("🔄 Loading model from MLflow...")
    experiment = mlflow.get_experiment_by_name("Return_Classification")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'Return_Classification' not found!")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError("No runs found in MLflow experiment!")
    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    MODEL = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded from run {run_id}")
    yield
    # Cleanup (nothing needed)
    print("👋 Shutting down...")

app = FastAPI(
    title="Return Prediction API",
    description="Dự đoán xác suất đơn hàng bị trả lại dựa trên thông tin đơn hàng.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────
class OrderInput(BaseModel):
    order_id: int = Field(..., example=815209)
    order_date: str = Field(..., example="2022-07-01")
    customer_id: int = Field(..., example=140622)
    zip: int = Field(..., example=1109)
    order_status: str = Field("delivered", example="delivered")
    payment_method: str = Field(..., example="credit_card")
    device_type: str = Field("desktop", example="desktop")
    order_source: str = Field("organic_search", example="organic_search")

class PredictionResult(BaseModel):
    order_id: int
    predicted_class: int
    return_probability: float
    label: str

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResult]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Kiểm tra trạng thái API và model."""
    return HealthResponse(status="ok", model_loaded=MODEL is not None)

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(order: OrderInput):
    """Dự đoán xác suất trả hàng cho MỘT đơn hàng."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")
    
    df = pd.DataFrame([order.model_dump()])
    df["order_date"] = pd.to_datetime(df["order_date"])
    
    try:
        pred = int(MODEL.predict(df)[0])
        prob = float(MODEL.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    return PredictionResult(
        order_id=order.order_id,
        predicted_class=pred,
        return_probability=round(prob, 4),
        label="Returned" if pred == 1 else "Not Returned",
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(orders: list[OrderInput]):
    """Dự đoán xác suất trả hàng cho NHIỀU đơn hàng cùng lúc."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")
    if len(orders) == 0:
        raise HTTPException(status_code=400, detail="Danh sách đơn hàng trống.")
    if len(orders) > 1000:
        raise HTTPException(status_code=400, detail="Tối đa 1000 đơn hàng mỗi lần gọi.")
    
    df = pd.DataFrame([o.model_dump() for o in orders])
    df["order_date"] = pd.to_datetime(df["order_date"])
    
    try:
        preds = MODEL.predict(df)
        probs = MODEL.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    results = []
    for i, order in enumerate(orders):
        results.append(PredictionResult(
            order_id=order.order_id,
            predicted_class=int(preds[i]),
            return_probability=round(float(probs[i]), 4),
            label="Returned" if int(preds[i]) == 1 else "Not Returned",
        ))
    
    return BatchPredictionResponse(predictions=results)

# ── Run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
