#!/bin/bash

# Exit immediately if any command fails, to avoid meaningless execution after an error
set -e

# 1. Define your 3 model config files (corresponding to 3 LLMs)
CONFIGS=(
    "./hparams/qwen2.5-7b.yaml"
    "./hparams/llama3-8b.yaml"  
    "./hparams/mistral-7b.yaml"
)

# 2. Define your 3 dataset paths
DATASETS=(
    "./data/zsre/zsre_3k.json"
    "./data/counterfact/counterfact_3k.json"
    "./data/wikibigedit/wikibigedit_3k.json"       
)

# Base save directory
BASE_SAVE_DIR="./saves"

# Number of evaluation samples
NUM_SAMPLES=3000

echo "🚀 Starting 3x3 Evaluation Pipeline..."

# Outer loop: iterate over models
for CONFIG_PATH in "${CONFIGS[@]}"; do
    # Extract model name for naming (e.g., extract "qwen2.5-7b" from "./hparams/qwen2.5-7b.yaml")
    MODEL_NAME=$(basename "$CONFIG_PATH" .yaml)

    # Inner loop: iterate over datasets
    for DATA_PATH in "${DATASETS[@]}"; do
        # Extract dataset name for naming (e.g., extract "zsre_eval_3k" from "./data/zsre/zsre_eval_3k.json")
        DATA_NAME=$(basename "$DATA_PATH" .json)
        
        # Dynamically generate the save directory for this combination
        CURRENT_SAVE_DIR="${BASE_SAVE_DIR}/${MODEL_NAME}_${DATA_NAME}"

        echo "================================================================="
        echo "🔥 Current Setup: Model [${MODEL_NAME}] | Dataset [${DATA_NAME}]"
        echo "================================================================="

        # ---------------------------------------------------------
        # Stage A: Fine-Tuning
        # ---------------------------------------------------------
        echo "▶️ [Step 1/2] Running Fine-tuning..."
        python fine-tune.py \
            --config_path "$CONFIG_PATH" \
            --data_path "$DATA_PATH" \
            --save_model_dir "$CURRENT_SAVE_DIR"

        # ---------------------------------------------------------
        # Stage B: Evaluation
        # ---------------------------------------------------------
        echo "▶️ [Step 2/2] Running Evaluation..."
        # Assume GPU 0 is used here; modify CUDA_VISIBLE_DEVICES if multi-GPU is needed
        CUDA_VISIBLE_DEVICES=0 python eval_edit_metric.py \
            --data_path "$DATA_PATH" \
            --model_path "$CURRENT_SAVE_DIR" \
            --tp_size 1 \
            --num_samples $NUM_SAMPLES

        echo "✅ Finished Setup: [${MODEL_NAME}] x [${DATA_NAME}]"
        echo ""
    done
done

echo "🎉 All experiments completed successfully!"