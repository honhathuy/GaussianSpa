#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="./train_imp_score.py"  
BASE_DATASET_DIR="../Dataset"
chkpnt_iter=14999
declare -a run_scenes=(
  "bicycle"
  # "bonsai"
  # "counter"
  # "kitchen"
  # "room"
  # "stump"
  # "garden"
  # "train"
  # "truck"
  # "treehill"
  # "playroom"
  # "drjohnson"
)


PORT="6010"
SPA_INTERVAL="50"

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -v threshold="$mem_threshold" -F', ' '
      $2 < threshold { print $1; exit }
    '
}



run_script(){
  while ss -tuln | grep -q ":$PORT";do
    echo "Port $PORT is in use."
    PORT=$((PORT + 1))
    echo "New port number is $PORT"
  done

  local DATASET_DIR=$1
  local DATASET_NAME=$(basename "$DATASET_DIR")
  
  OUTPUT_DIR="./output/"$DATASET_NAME"/imp_score_57_50"
  echo "Output script for $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  ckpt="$OUTPUT_DIR"/chkpnt"$chkpnt_iter".pth
  SAMPLING_FACTOR=$(echo "0.50" | bc)
  if [ -f "$OUTPUT_DIR/$OUTPUT_FILE" ]; then
      echo "Output file $OUTPUT_FILE already exists. Skipping this iteration."
      continue
  fi

  gpu_id=$(get_available_gpu)
  if [[ -n $gpu_id ]]; then
    echo "GPU $gpu_id is available."
    CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT" \
    --port "$PORT" \
    -s="$DATASET_DIR" \
    -m="$OUTPUT_DIR" \
    --eval \
    --prune_ratio1 "0.57"\
    --prune_ratio2 "$SAMPLING_FACTOR"\
    --imp_metric "indoor"\
    --iterations "40000"
    #--checkpoint_iterations 14999
    #--start_checkpoint "$ckpt"
    else
      echo "No GPU available at the moment. Retrying in 1 minute."
      sleep 60
  fi
}

for view in "${run_scenes[@]}"; do
    echo "Running script for $view"
    run_script "$BASE_DATASET_DIR/$view"
done