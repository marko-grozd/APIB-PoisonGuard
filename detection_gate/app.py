"""
Detekcioni gate — jezgro projekta.

Prima batch (POST /inspect), pusta ga kroz statisticke detektore i donosi presudu: ACCEPT (prosledi treniranju) ili BLOCK
(karantin + alarm). Izlaze Prometheus metrike na /metrics.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge,
                               generate_latest)
from starlette.responses import Response

sys.path.append("/app/shared")
import config as C  # noqa: E402
from detectors import (GrubbsDetector, MahalanobisDetector)

app = FastAPI(title="poison-guard detection gate")

TRAINING_URL = os.getenv("TRAINING_URL", "http://training:8000/retrain")

# --- Prometheus metrike ---
M_BATCHES = Counter("pg_batches_total", "Ukupno pregledanih batch-eva")
M_BLOCKED = Counter("pg_blocked_total", "Ukupno blokiranih (trovanih) batch-eva")
M_MAHAL = Gauge("pg_mahalanobis_outlier_rate", "Udeo Mahalanobis outliera u batch-u")
M_GRUBBS = Gauge("pg_grubbs_outliers", "Broj Grubbs outliera u batch-u")
M_VERDICT = Gauge("pg_last_verdict_blocked", "1 ako je poslednji batch blokiran")

REF = {}        # ucitane reference statistike
DETECTORS = {}  # instancirani detektori


@app.on_event("startup")
def _load():
    with open(C.REF_STATS_FILE) as f:
        stats = json.load(f)
    REF["mean"] = np.array(stats["mean"])
    REF["cov"] = np.array(stats["cov"])
    reference = pd.read_parquet(C.REFERENCE_FILE)[C.STAT_FEATURES].to_numpy(float)
    DETECTORS["grubbs"] = GrubbsDetector(
    C.GRUBBS_ALPHA,
    stat_features=C.STAT_FEATURES,
    log_features=C.GRUBBS_LOG_FEATURES,
)
    DETECTORS["mahal"] = MahalanobisDetector(REF["mean"], REF["cov"],
                                             C.MAHALANOBIS_CHI2_Q)
    print(f"[gate] ucitano: {len(reference)} referentnih uzoraka, "
          f"{len(C.STAT_FEATURES)} feature-a", flush=True)


def _decide(scores: dict) -> tuple[bool, list[str]]:
    """Vrati (blocked, razlozi)."""
    reasons = []
    if scores["mahalanobis_outlier_rate"] > C.OUTLIER_RATE_BLOCK:
        reasons.append(f"Mahalanobis outlier rate "
                       f"{scores['mahalanobis_outlier_rate']:.2f} "
                       f"> {C.OUTLIER_RATE_BLOCK}")
    if scores["grubbs_outliers_total"] > C.GRUBBS_OUTLIERS_BLOCK:
        reasons.append(f"Grubbs: {scores['grubbs_outliers_total']} outliera "
                   f"> {C.GRUBBS_OUTLIERS_BLOCK}")
    return (len(reasons) > 0), reasons


@app.post("/inspect")
def inspect(payload: dict):
    """payload = {"rows": [...], "columns": [...], "mode": "clean|label_flip|..."}"""
    df = pd.DataFrame(payload["rows"], columns=payload["columns"])
    batch = df[C.STAT_FEATURES].to_numpy(float)

    scores = {}
    scores.update(DETECTORS["grubbs"].score(batch))
    scores.update(DETECTORS["mahal"].score(batch))

    blocked, reasons = _decide(scores)

    M_BATCHES.inc()
    M_MAHAL.set(scores["mahalanobis_outlier_rate"])
    M_GRUBBS.set(scores["grubbs_outliers_total"])
    M_VERDICT.set(1 if blocked else 0)

    if blocked:
        M_BLOCKED.inc()
        print(f"[gate] BLOCK ({payload.get('mode')}): {reasons}", flush=True)
    else:
        try:
            requests.post(TRAINING_URL, json=payload, timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"[gate] training nedostupan: {e}", flush=True)
        print(f"[gate] ACCEPT ({payload.get('mode')})", flush=True)

    return {"blocked": blocked, "reasons": reasons, "scores": scores}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"ok": True}
