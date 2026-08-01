from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict_applicant

app = FastAPI()

class PredictRequest(BaseModel):
    SK_ID_CURR: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    sk_id_curr = request.SK_ID_CURR
    # look up applicant, build features, run model, return prediction
    try:
        prediction = predict_applicant(sk_id_curr)
        return prediction
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))