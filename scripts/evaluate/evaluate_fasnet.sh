#!/bin/bash

# Example
# ./evaluate_fasnet.sh 1 XP1

# Get the execution directory
EXECUTION_DIR=$PWD

# Arguments
STAGE=$1
EXPERIMENT_TAG=$2
INPUT_JSON=$3

# Activate the virtual environment
conda activate asteroid

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Choose Experiment Tag

EXP_PATH="external_repos/asteroid/egs/TAC/exp"


BEST_MODELS_FILE="$EXP_PATH/train_TAC_$EXPERIMENT_TAG/best_k_models.json"

# Function to get the best model path from best_k_models.json
get_best_model_path() {
    if [ ! -f "$BEST_MODELS_FILE" ]; then
        echo -e "${RED}Error:${NC} Best models file not found: $BEST_MODELS_FILE"
        exit 1
    fi

    # Use jq to parse JSON and get the checkpoint with the highest value
    MODEL_PATH=$(jq -r 'to_entries | sort_by(.value) | reverse | .[0].key' "$BEST_MODELS_FILE")

    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}Error:${NC} No valid model path found in $BEST_MODELS_FILE"
        exit 1
    fi

    echo "$MODEL_PATH"
}

# Get the best model checkpoint path
MODEL_PATH=$(get_best_model_path)
echo "The model path: ${MODEL_PATH}"

# Check if model path is valid
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error:${NC} Best model checkpoint not found: $MODEL_PATH"
    $STAGE=0
fi

# Stage 1: Inference
if [ $STAGE -eq 1 ]; then
    echo -e "${CYAN}[Stage 1] Enhancing testset mixtures using Fasnet...${NC}"

    cd external_repos/asteroid/egs/TAC

    python3 custom_inference.py \
            --experiment_tag=$EXPERIMENT_TAG \
            --model_path=$MODEL_PATH \
            --input_json=LibriHAid/data/parse_data/$INPUT_JSON \
            --sample_rate=16000 \
            --device='cuda:0'

    echo -e "${GREEN}[Stage 1] Parsed data successfully."
else
    echo -e "${RED}[Stage 1] Skipped!${NC}"
fi

# Stage 2: Utterance-level Evaluation
if [ $STAGE -eq 2 ]; then
    echo -e "${CYAN}[Stage 2] Evaluating Fasnet at the utterance-level...${NC}"

    python3 src/evaluation/utt_eval.py --experiment_tag=$EXPERIMENT_TAG

    echo -e "${CYAN}[Stage 2] Evaluated Fasnet at the utterance-level${NC}"
else
    echo -e "${RED}[Stage 2] Skipped!${NC}"
fi

# Stage 3: ASR 
if [ $STAGE -eq 3 ]; then
    echo -e "${CYAN}[Stage 3] Generating Transcription of Clean Speech, Mixture and Estimated Speech...${NC}"

    python3 src/evaluation/asr.py --experiment_tag="$EXPERIMENT_TAG"

    echo -e "${CYAN}[Stage 3] Generation done.${NC}"
else
    echo -e "${RED}[Stage 3] Skipped!${NC}"
fi

cd $EXECUTION_DIR