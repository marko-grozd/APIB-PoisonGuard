# Poison Guard — detekcija data poisoning napada u ML pipeline-u

Mini projektni zadatak: ML pipeline koji se retrenira na dolaznim podacima, sa
**statističkim detekcionim slojem** koji hvata data poisoning napade pre nego
što pokvare model — plus **monitoring** (Prometheus + Grafana).

Dataset: **NSL-KDD** (network intrusion, binarno `normal` / `attack`).
Sve radi lokalno sa jednim `docker compose up`. Azure je opcioni bonus.

Dve tačke odbrane sa slajda *Model Poisoning: Defense*:
1. **Ulazna inspekcija** (detekcioni gate) — pre treniranja.
2. **Kalibraciona provera** (training servis) — posle treniranja.

## Arhitektura

```
simulator --batch--> gate --(cisto)--> training --model--> serving
                       |
                  (trovano) --> karantin + alarm
                       |
            sve metrike --> Prometheus --> Grafana
```

## Pokretanje (lokalno)

```bash
# 1) priprema podataka (jednom) — skida NSL-KDD i pravi referencu/kalibraciju/stream
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow
DATA_OUT=./data_out python data/prepare_nslkdd.py

docker compose up --build

# Grafana:    http://localhost:3000  (dashboard "Poison Guard")
# Prometheus: http://localhost:9090
# Gate API:   http://localhost:8001/metrics
```

```bash

ATTACK_MODE=clean docker compose up -d simulator


ATTACK_MODE=label_flip POISON_FRAC=0.3 docker compose up -d simulator

ATTACK_MODE=chaff docker compose up -d simulator

ATTACK_MODE=ood docker compose up -d simulator
```


## Pragovi (u `shared/config.py`)

- `PSI_BLOCK = 0.25` — značajan pomeraj distribucije
- `OUTLIER_RATE_BLOCK = 0.15` — udeo Mahalanobis outliera
- `CALIBRATION_DROP = 0.05` — dozvoljen pad tačnosti
