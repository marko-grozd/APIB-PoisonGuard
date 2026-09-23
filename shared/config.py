import os

"""Zajednicka konfiguracija za sve servise."""

# Sve 41 kolone NSL-KDD + label + difficulty
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

# Curirani podskup kontinualnih feature-a za STATISTICKI sloj (Grubbs/Mahalanobis).
STAT_FEATURES = [
    "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
    "serror_rate", "same_srv_rate",
]

GRUBBS_LOG_FEATURES = ["src_bytes", "dst_bytes", "count", "srv_count"]

GRUBBS_OUTLIERS_BLOCK = int(os.getenv("GRUBBS_OUTLIERS_BLOCK", "8"))

# Binarna labela: normal -> 0, sve ostalo (attack) -> 1
LABEL_COL = "label"
TARGET_COL = "y"

# --- Pragovi detekcije (mapirano na predavanje 06) ---
GRUBBS_ALPHA = 0.05          # nivo znacajnosti za Grubbs' test
MAHALANOBIS_CHI2_Q = 0.975   # kvantil hi-kvadrat raspodele za prag
OUTLIER_RATE_BLOCK = 0.15    # ako > 15% batch-a outlier po Mahalanobisu -> blokiraj
CALIBRATION_DROP = 0.05      # pad tacnosti na kalibracionom setu > 5pp -> alarm

GRUBBS_OUTLIERS_BLOCK = 250

# --- Putanje (deljeni volume-i u docker-compose) ---
DATA_DIR = "/data"
MODEL_DIR = "/models"
REFERENCE_FILE = f"{DATA_DIR}/reference.parquet"      # cist referentni set
STREAM_FILE = f"{DATA_DIR}/stream_pool.parquet"       # bazen za batch-eve
CALIBRATION_FILE = f"{DATA_DIR}/calibration.parquet"  # rucni cist test set
REF_STATS_FILE = f"{DATA_DIR}/ref_stats.json"         # mean/cov/bins reference
MODEL_FILE = f"{MODEL_DIR}/model.pkl"
TRAIN_METRICS_FILE = f"{MODEL_DIR}/metrics.json"
