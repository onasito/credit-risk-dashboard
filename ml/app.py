import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_applicant, list_applicant_ids, get_applicant_profile

app = FastAPI()

DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://localhost:5173"
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    SK_ID_CURR: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/applicants")
def applicants(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return {"applicant_ids": list_applicant_ids(limit=limit, offset=offset)}

@app.post("/predict")
def predict(request: PredictRequest):
    sk_id_curr = request.SK_ID_CURR
    # look up applicant, build features, run model, return prediction
    try:
        prediction = predict_applicant(sk_id_curr)
        return prediction
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/applicants/{sk_id_curr}/profile")
def applicant_profile(sk_id_curr: int):
    try:
        return get_applicant_profile(sk_id_curr)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))