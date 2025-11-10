#!/bin/bash

# Get the execution directory
EXECUTION_DIR=$PWD

# Exit immediately if a command exits with a non-zero status
# set -e

# Check if experiment tag is passed as an argument
if [ -z "$1" ]; then
  echo "Error: No experiment tag provided. Usage: ./train_fasnet.sh <EXPERIMENT_TAG>"
  exit 1
fi

# Example
# ./scripts/train/train_fasnet.sh XP1 neg_sisdr gres 2 48 512 256 mel 18 speech_norm

# Arguments
EXPERIMENT_TAG=$1
LOSS=$2
CLUSTER=$3
GPUs=$4
WALLTIME=$5
N_FFT=${6:-1024}       # Default to 1024 if $6 is not set
HOP_LENGTH=${7:-256}   # Default to 512 if $7 is not set
SCALE=${8:-mel}   # Default to mel if $8 is not set
N_BINS=${9:-18}   # Default to 512 if $9 is not set
WEIGHTS_TYPE=${10:-None}   # Default to 512 if $9 is not set

echo ">>>>>>>>>>>>>>>>>>>> EXPERIMENT_TAG: $EXPERIMENT_TAG"
echo ">>>>>>>>>>>>>>>>>>>> LOSS: $LOSS"
echo ">>>>>>>>>>>>>>>>>>>> WALLTIME: $WALLTIME"

# Change directory to the TAC project folder
cd external_repos/asteroid/egs/TAC #|| { echo "Error: Failed to change directory"; exit 1; }

# Log the current directory for verification
echo "Current working directory: $(pwd)"

# Create a log folder
rm -rf "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}"
mkdir -p "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}"

# Save all job parameters to a config file
PARAMS_FILE="$EXECUTION_DIR/logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_params.txt"

echo "EXPERIMENT_TAG=$EXPERIMENT_TAG"   > "$PARAMS_FILE"
echo "LOSS=$LOSS"                      >> "$PARAMS_FILE"
echo "CLUSTER=$CLUSTER"                >> "$PARAMS_FILE"
echo "GPUs=$GPUs"                      >> "$PARAMS_FILE"
echo "WALLTIME=$WALLTIME"              >> "$PARAMS_FILE"
echo "N_FFT=$N_FFT"                    >> "$PARAMS_FILE"
echo "HOP_LENGTH=$HOP_LENGTH"          >> "$PARAMS_FILE"
echo "SCALE=$SCALE"                    >> "$PARAMS_FILE"
echo "N_BINS=$N_BINS"                  >> "$PARAMS_FILE"
echo "WEIGHTS_TYPE=$WEIGHTS_TYPE"      >> "$PARAMS_FILE"

# Create Output and Error log files
touch "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_output.log"
touch "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_error.log"

# Submit a job to train Fasnet-TAC using OAR with the specified parameters
#  -p "cluster in ('grat', 'gruss', 'graffiti', 'grue')" \
oarsub -vv -l "host=1/gpu=$GPUs,walltime=$WALLTIME" -p $CLUSTER -q production \
       -O "../../../../logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_output.log" \
       -E "../../../../logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_error.log" \
       "./run_.sh --stage 3 --tag $EXPERIMENT_TAG --id 0,1,2,3 --loss $LOSS --scale $SCALE --n_fft $N_FFT --hop_length $HOP_LENGTH --n_bins $N_BINS --weights_type $WEIGHTS_TYPE"

# Notify user that the job has been submitted
echo "Job submitted with tag: $EXPERIMENT_TAG on cluster: $CLUSTER"
cd $EXECUTION_DIR