#!/bin/bash

# Activate the asteroid virtual environment
conda activate asteroid

# Exit on error
# set -e
# set -o pipefail

# Main storage directory where dataset will be stored
storage_dir=$(readlink -m ./datasets)
librispeech_dir=$storage_dir/LibriSpeech
noise_dir=$storage_dir/Nonspeech
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run_.sh --stage 3 --tag TEST --id 0 --loss neg_sisdr --scale mel --n_fft 512 --hop_length 256 --n_bins 18 --weights_type sigmoid

# General
stage=3  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0
eval_use_gpu=1
loss="" # Loss type must be provided
scale="" 
n_fft=1024 # n_fft for STFT
hop_length=256 # hop_length for STFT
n_bins="" 
weights_type="" 

# Help message function
show_help() {
  echo "Usage: ./run_.sh [OPTIONS]"
  echo ""
  echo "Mandatory options:"
  echo "  --loss <loss_type>      Specify the loss function (e.g., xp1, xp2, ...)"
  echo "  --scale <scale>      "
  echo "  --n_fft <n_fft>      Specify the n_fft of the STFT"
  echo "  --hop_length <hop_length>      Specify the hop_length of the STFT"
  echo "  --n_bins <n_bins>      "
  echo "  --weights_type <weights_type>      "
  echo ""
  echo "Optional arguments:"
  echo "  --stage <num>           Set the starting stage (default: 0)"
  echo "  --tag <name>            Experiment tag (used for folder naming)"
  echo "  --id <gpu_id>           Specify GPU ID (default: 0)"
  echo "  --help                  Show this help message and exit"
  echo ""
  echo "Examples:"
  echo "  ./run_.sh --stage 3 --tag TEST --id 0 --loss neg_sisdr --scale mel --n_fft 512 --hop_length 256 --n_bins 18 --weights_type sigmoid"
  exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2;;
    --tag) tag="$2"; shift 2;;
    --id) id="$2"; shift 2;;
    --loss) loss="$2"; shift 2;;
    --scale) scale="$2"; shift 2;;
    --n_fft) n_fft="$2"; shift 2;;
    --hop_length) hop_length="$2"; shift 2;;
    --n_bins) n_bins="$2"; shift 2;;
    --weights_type) weights_type="$2"; shift 2;;
    --help) show_help;;
    --*) echo "Unknown option: $1"; exit 1;;
  esac
done

# Dataset option
dataset_type=adhoc
samplerate=16000

. utils/parse_options.sh

dumpdir=data/$suffix  # directory to put generated json file

# check if gpuRIR installed
if ! ( pip list | grep -F gpuRIR ); then
  echo 'This recipe requires gpuRIR. Please install gpuRIR.'
  exit
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi
expdir=exp/train_TAC_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs

  # Start monitoring GPU usage in the background
  # watch -n 1 nvidia-smi &

  CUDA_VISIBLE_DEVICES=$id $python_path train.py --sample_rate $samplerate --exp_dir ${expdir} --loss ${loss} --scale ${scale} --n_fft ${n_fft} --hop_length ${hop_length} --n_bins ${n_bins} --weights_type ${weights_type}| tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "TAC/TAC" > $expdir/publish_dir/recipe_name.txt
fi
