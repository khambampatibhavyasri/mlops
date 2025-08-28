import json, os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Retail Baseline Model")
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_linear.json")

with open(MODEL_PATH, "r") as f:
    ART = json.load(f)

W = np.array(ART["weights"]).reshape(-1,1)
MU = np.array(ART["mu"])
SIG = np.array(ART["sigma"])
FEATS = ART["feature_schema"]["order"][1:]  # exclude bias

class Item(BaseModel):
    features: dict  # keys must match FEATS

def vectorize(feat_dict):
    x = np.zeros((len(FEATS),), dtype=float)
    for i, name in enumerate(FEATS):
        x[i] = float(feat_dict.get(name, 0.0))
    x = (x - MU) / np.where(SIG==0, 1.0, SIG)
    x = np.concatenate(([1.0], x))
    return x.reshape(1,-1)

@app.post("/predict")
def predict(item: Item):
    X = vectorize(item.features)
    yhat = float((X @ W)[0,0])
    return {"prediction_units": max(0.0, yhat)}
