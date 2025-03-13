#!/bin/bash
set -e

# This script serves as the container entrypoint and handles different execution modes

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if we're running as root
if [ "$(id -u)" = "0" ]; then
  echo "Warning: Running as root is not recommended. Consider using a non-root user."
fi

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set up environment based on execution mode
MODE=${1:-"streamlit"}

case $MODE in
  "streamlit")
    echo "Starting Streamlit server..."
    exec streamlit run /app/app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    ;;
    
  "train")
    MODEL_TYPE=${2:-"transformer"}
    echo "Starting model training for type: $MODEL_TYPE"
    
    if [ "$MODEL_TYPE" = "transformer" ]; then
      DATASET=${3:-"sample"}
      TEXT_COL=${4:-"Description"}
      TARGET_COL=${5:-"Activity"}
      
      exec python -m src.training.run_training --model_type=transformer \
        --dataset=$DATASET --text_col=$TEXT_COL --target_col=$TARGET_COL
    elif [ "$MODEL_TYPE" = "cnn" ]; then
      DATASET=${3:-"sample"}
      SMILES_COL=${4:-"SMILES"}
      TARGET_COL=${5:-"Activity"}
      
      exec python -m src.training.run_training --model_type=cnn \
        --dataset=$DATASET --smiles_col=$SMILES_COL --target_col=$TARGET_COL
    else
      echo "Unknown model type: $MODEL_TYPE"
      exit 1
    fi
    ;;
    
  "evaluate")
    MODEL_PATH=${2:-"/app/models/latest"}
    echo "Evaluating model at: $MODEL_PATH"
    
    exec python -m src.evaluation.run_evaluation --model_path=$MODEL_PATH
    ;;
    
  "predict")
    MODEL_PATH=${2:-"/app/models/latest"}
    INPUT_PATH=${3:-"/app/data/input.csv"}
    OUTPUT_PATH=${4:-"/app/data/predictions.csv"}
    
    echo "Running prediction with model: $MODEL_PATH"
    
    exec python -m src.prediction.run_prediction \
      --model_path=$MODEL_PATH --input_path=$INPUT_PATH --output_path=$OUTPUT_PATH
    ;;
    
  "shell")
    echo "Starting interactive shell..."
    exec /bin/bash
    ;;
    
  *)
    echo "Unknown mode: $MODE"
    echo "Available modes: streamlit, train, evaluate, predict, shell"
    exit 1
    ;;
esac
