"""
Simulator napada (protivnik iz predavanja 09).

Generise batch-eve i salje ih detekcionom gate-u. Rezim se bira ENV varijablom
ATTACK_MODE:
  clean       -> cist normal saobracaj
  label_flip  -> okrene labele dela uzoraka (gruba trovanja)
  chaff       -> postepeno ubacuje 'chaff' outlier tacke ('boiling frog')
  ood         -> ubacuje tacke iz potpuno druge distribucije

"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.append("/app/shared")
import config as C  # noqa: E402

GATE_URL = os.getenv("GATE_URL", "http://gate:8000/inspect")
MODE = os.getenv("ATTACK_MODE", "clean")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "300"))
INTERVAL = float(os.getenv("INTERVAL", "5"))
POISON_FRAC = float(os.getenv("POISON_FRAC", "0.2"))

rng = np.random.default_rng(0)


def label_flip(batch: pd.DataFrame) -> pd.DataFrame:
    k = int(len(batch) * POISON_FRAC)
    idx = rng.choice(batch.index, size=k, replace=False)
    batch.loc[idx, C.TARGET_COL] = 1 - batch.loc[idx, C.TARGET_COL]
    return batch


def chaff(batch: pd.DataFrame, step: int) -> pd.DataFrame:
    """Postepeno povecava broj outlier tacaka kroz vreme (boiling frog)."""
    frac = min(POISON_FRAC, 0.02 * step)        # raste sa svakim batch-om
    k = int(len(batch) * frac)
    if k == 0:
        return batch
    idx = rng.choice(batch.index, size=k, replace=False)
    for col in C.STAT_FEATURES:
        shift = batch[col].std() * 3 + 1.0
        batch.loc[idx, col] = batch.loc[idx, col] + shift
    return batch


def ood(batch: pd.DataFrame) -> pd.DataFrame:
    k = int(len(batch) * POISON_FRAC)
    idx = rng.choice(batch.index, size=k, replace=False)
    for col in C.STAT_FEATURES:
        batch.loc[idx, col] = rng.normal(batch[col].mean() * 5 + 100,
                                         batch[col].std() * 5 + 50, size=k)
    return batch


def main():
    pool = pd.read_parquet(C.STREAM_FILE)
    print(f"[sim] mode={MODE} batch={BATCH_SIZE} interval={INTERVAL}s", flush=True)
    step = 0
    while True:
        step += 1
        batch = pool.sample(BATCH_SIZE, random_state=step).reset_index(drop=True)
        if MODE == "label_flip":
            batch = label_flip(batch)
        elif MODE == "chaff":
            batch = chaff(batch, step)
        elif MODE == "ood":
            batch = ood(batch)

        payload = {"rows": batch.values.tolist(),
                   "columns": batch.columns.tolist(),
                   "mode": MODE}
        try:
            r = requests.post(GATE_URL, json=payload, timeout=30).json()
            tag = "BLOCK" if r["blocked"] else "accept"
            print(f"[sim] step={step} -> {tag} "
                  f"psi={r['scores']['psi_max']:.3f} "
                  f"mahal={r['scores']['mahalanobis_outlier_rate']:.3f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[sim] gate nedostupan: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
