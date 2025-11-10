#!/bin/bash

# Example 
# ./scripts/evaluate/evaluate_fasnet_job.sh XP1 1 1 grappe

# Get the execution directory
EXECUTION_DIR=$PWD

# Arguments
EXPERIMENT_TAG=$1
STAGE=$2
WALLTIME=$3
CLUSTER=$4

# Check if experiment tag is passed as an argument
if [ -z "$1" ]; then
  echo "Error: No experiment tag provided. Usage: ./script.sh <EXPERIMENT_TAG>"
  # exit 1
fi

# Create a log folder and log files
echo "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}"
echo "$EXPERIMENT_TAG"
mkdir -p "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}"
touch "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_evaluation_output.log"
touch "$EXECUTION_DIR/logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_evaluation_error.log"
echo "created" 

# Submit a job to evaluate Fasnet-TAC using OAR with the specified parameters
oarsub -vv -l "host=1/gpu=1,walltime=$WALLTIME" -p $CLUSTER -q production \
          -O "../../../../logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_evaluation_output.log" \
          -E "../../../../logs/${EXPERIMENT_TAG}/${EXPERIMENT_TAG}_evaluation_error.log" \
          "./scripts/evaluate/evaluate_fasnet.sh $STAGE $EXPERIMENT_TAG"
