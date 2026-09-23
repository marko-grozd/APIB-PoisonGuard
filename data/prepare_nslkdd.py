"""
Priprema podataka: NSL-KDD -> referentni set, kalibracioni set, stream bazen,
i reference statistike (mean/cov/PSI binovi) za detekcioni gate.

Pokrece se jednom pre prvog `docker compose up` (README).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
import config as C  # noqa: E402

RAW_URL = ("https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/"
           "master/KDDTrain+.txt")
OUT_DIR = os.getenv("DATA_OUT", "./data_out")


def load() -> pd.DataFrame:
    print(f"[prep] preuzimam {RAW_URL}", flush=True)
    df = pd.read_csv(RAW_URL, names=C.COLUMNS)
    df[C.TARGET_COL] = (df[C.LABEL_COL] != "normal").astype(int)
    return df


def reference_stats(ref: pd.DataFrame) -> dict:
    X = ref[C.STAT_FEATURES].to_numpy(float)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    bin_edges, expected = [], []
    for j in range(X.shape[1]):
        edges = np.histogram_bin_edges(X[:, j], bins=10)
        edges[0], edges[-1] = -np.inf, np.inf  # robustno na repove
        counts, _ = np.histogram(X[:, j], bins=edges)
        bin_edges.append(edges.tolist())
        expected.append((counts / counts.sum()).tolist())
    return {"mean": mean.tolist(), "cov": cov.tolist(),
            "bin_edges": bin_edges, "expected": expected}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()

    normal = df[df[C.TARGET_COL] == 0].sample(frac=1, random_state=42)
    attack = df[df[C.TARGET_COL] == 1].sample(frac=1, random_state=42)

    def split(part, a, b):
        """Isecak [a, b) kao udeo duzine particije."""
        return part.iloc[int(a * len(part)): int(b * len(part))]

    # Iste proporcije u sve tri particije -> referentna raspodela je
    # reprezentativna za ono sto stize u stream.
    reference = pd.concat([split(normal, 0.0, 0.5),
                           split(attack, 0.0, 0.5)]).sample(frac=1, random_state=17)

    calibration = pd.concat([split(normal, 0.5, 0.6),
                             split(attack, 0.5, 0.6)]).sample(frac=1, random_state=11)

    stream_pool = pd.concat([split(normal, 0.6, 1.0),
                             split(attack, 0.6, 1.0)]).sample(frac=1, random_state=13)

    reference.to_parquet(f"{OUT_DIR}/reference.parquet")
    calibration.to_parquet(f"{OUT_DIR}/calibration.parquet")
    stream_pool.to_parquet(f"{OUT_DIR}/stream_pool.parquet")

    with open(f"{OUT_DIR}/ref_stats.json", "w") as f:
        json.dump(reference_stats(reference), f)

    print(f"[prep] reference={len(reference)} calibration={len(calibration)} "
          f"stream_pool={len(stream_pool)}", flush=True)
    print(f"[prep] fajlovi u {OUT_DIR}/", flush=True)

    for name, part in [("reference", reference), ("calibration", calibration),
                       ("stream_pool", stream_pool)]:
        share = part[C.TARGET_COL].mean()
        print(f"{name}: n={len(part)}, udeo attack={share:.3f}", flush=True)


if __name__ == "__main__":
    main()
