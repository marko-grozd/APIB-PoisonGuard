#/bin/bash

for i in $(seq 1 6); do
  echo "--- $(date +%H:%M:%S)"
  curl -s localhost:8001/metrics | grep -E "^pg_grubbs_outliers |^pg_mahalanobis_outlier_rate|^pg_blocked_total|^pg_batches_total"
  curl -s localhost:8002/metrics | grep "^pg_calibration_accuracy"
  sleep 300
done