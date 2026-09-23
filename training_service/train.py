"""
Servis za treniranje.

Trenira pocetni (baseline) model nad cistim referentnim setom, a zatim na
zahtev (/retrain) inkrementalno doucava model na PRIHVACENIM batch-evima
(partial_fit -> online ucenje, kao chaff demo iz predavanja 09).

Posle svakog doucavanja pokrece KALIBRACIONU PROVERU: pusti rucni cisti test
set kroz model i uporedi tacnost sa baseline-om. Pad > praga = alarm.
Ovo je druga tacka odbrane sa slajda Model Poisoning: Defense.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge,
                               generate_latest)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from starlette.responses import Response

sys.path.append("/app/shared")
import config as C  # noqa: E402

app = FastAPI(title="poison-guard training")

M_CALIB = Gauge("pg_calibration_accuracy", "Tacnost na kalibracionom setu")
M_CALIB_DROP = Gauge("pg_calibration_dropped", "1 ako je tacnost pala ispod praga")
M_RETRAIN = Counter("pg_retrain_total", "Broj doucavanja modela")

STATE = {}


def _feature_cols():
    return [c for c in C.COLUMNS if c not in (C.LABEL_COL, "difficulty")]


def _build_preprocessor(ref: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in _feature_cols() if c not in C.CATEGORICAL]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL),
    ])
    pre.fit(ref[_feature_cols()])
    return pre


def _accuracy(model, pre, df: pd.DataFrame) -> float:
    X = pre.transform(df[_feature_cols()])
    y = df[C.TARGET_COL].to_numpy()
    return float((model.predict(X) == y).mean())


def _save(model, pre):
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    with open(C.MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "pre": pre, "features": _feature_cols()}, f)


@app.on_event("startup")
def _train_baseline():
    ref = pd.read_parquet(C.REFERENCE_FILE)
    calib = pd.read_parquet(C.CALIBRATION_FILE)
    # referenca je sav 'normal' -> dodaj malo napada da baseline ima obe klase
    attack_seed = calib[calib[C.TARGET_COL] == 1]
    train_df = pd.concat([ref, attack_seed]).sample(frac=1, random_state=3)

    pre = _build_preprocessor(train_df)
    X = pre.transform(train_df[_feature_cols()])
    y = train_df[C.TARGET_COL].to_numpy()
    model = SGDClassifier(loss="log_loss", random_state=0)
    model.partial_fit(X, y, classes=np.array([0, 1]))

    STATE.update(model=model, pre=pre, calib=calib)
    baseline = _accuracy(model, pre, calib)
    STATE["baseline_acc"] = baseline
    M_CALIB.set(baseline)
    _save(model, pre)
    with open(C.TRAIN_METRICS_FILE, "w") as f:
        json.dump({"baseline_accuracy": baseline}, f)
    print(f"[train] baseline tacnost (kalibracija) = {baseline:.4f}", flush=True)


@app.post("/retrain")
def retrain(payload: dict):
    df = pd.DataFrame(payload["rows"], columns=payload["columns"])
    pre, model = STATE["pre"], STATE["model"]
    X = pre.transform(df[_feature_cols()])
    y = df[C.TARGET_COL].to_numpy()
    model.partial_fit(X, y)            # online doucavanje
    M_RETRAIN.inc()

    acc = _accuracy(model, pre, STATE["calib"])
    M_CALIB.set(acc)
    dropped = (STATE["baseline_acc"] - acc) > C.CALIBRATION_DROP
    M_CALIB_DROP.set(1 if dropped else 0)
    _save(model, pre)

    if dropped:
        print(f"[train] ALARM: kalibraciona tacnost pala {STATE['baseline_acc']:.3f}"
              f" -> {acc:.3f}", flush=True)
    return {"calibration_accuracy": acc, "dropped": bool(dropped)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"ok": True}
