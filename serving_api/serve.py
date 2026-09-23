"""
Serving API — servira tekucu verziju modela iz deljenog /models volume-a.
"""
import pickle
import sys

import pandas as pd
from fastapi import FastAPI
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, generate_latest)
from starlette.responses import Response

sys.path.append("/app/shared")
import config as C  # noqa: E402

app = FastAPI(title="poison-guard serving")
M_PRED = Counter("pg_predictions_total", "Broj predikcija")


def _load():
    with open(C.MODEL_FILE, "rb") as f:
        return pickle.load(f)


@app.post("/predict")
def predict(payload: dict):
    bundle = _load()  # uvek najnovija verzija modela
    df = pd.DataFrame(payload["rows"], columns=payload["columns"])
    X = bundle["pre"].transform(df[bundle["features"]])
    preds = bundle["model"].predict(X).tolist()
    M_PRED.inc(len(preds))
    return {"predictions": preds}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"ok": True}
