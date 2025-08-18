#!/bin/bash

CONDA_ENV_NAME="FlightRank"

mkdir -p notebook/logs
TIME_TAG=$(date +%Y%m%d_%H%M%S)

black .

PREFIX="flight"

nohup bash -c "
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
papermill ${PREFIX}.ipynb notebook/${PREFIX}_${TIME_TAG}.ipynb -p TIME_TAG ${TIME_TAG}
" > notebook/logs/${PREFIX}_${TIME_TAG}.log 2>&1 &

echo "Running ${PREFIX}.ipynb in background with papermill..."
echo "Output notebook: notebook/${PREFIX}_${TIME_TAG}.ipynb"
echo "Log file: notebook/logs/${PREFIX}_${TIME_TAG}.log"

git add .
git commit -m "Run flight with tag ${TIME_TAG}"

echo "${TIME_TAG}"