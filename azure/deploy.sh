#!/usr/bin/env bash
#TODO
set -euo pipefail

RG=${RG:-poison-guard-rg}
LOC=${LOC:-westeurope}
ACR=${ACR:-poisonguard$RANDOM}
ENVNAME=${ENVNAME:-poison-guard-env}

az group create -n "$RG" -l "$LOC"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true
az acr login -n "$ACR"
LOGIN=$(az acr show -n "$ACR" --query loginServer -o tsv)

for svc in detection_gate training_service serving_api attack_simulator; do
  img="$LOGIN/poison-$svc:latest"
  docker build -f "$svc/Dockerfile" -t "$img" .
  docker push "$img"
done

az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC"


az containerapp create -n gate -g "$RG" --environment "$ENVNAME" \
  --image "$LOGIN/poison-detection_gate:latest" \
  --registry-server "$LOGIN" \
  --target-port 8000 --ingress external --min-replicas 0

echo "Gotovo. Ostali servisi se kreiraju istom 'az containerapp create' komandom."
echo "Za stedljivost: --min-replicas 0 (scale-to-zero) gde god moze."
