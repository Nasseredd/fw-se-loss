import os
import sys
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from pathlib import Path

from local.tac_dataset import TACDataset
from asteroid.models.fasnet import FasNetTAC
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from auditus.evaluation import Auditus

def load_model(model_path, sample_rate, device='cpu'):
    """Load the trained model."""
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model arguments from 'hyper_parameters'
    if 'hyper_parameters' in checkpoint:
        model_args = checkpoint['hyper_parameters']
    else:
        raise KeyError("'hyper_parameters' not found in checkpoint. Cannot initialize model.")

    # Map parameters for FasNetTAC
    fasnet_args = {
        'enc_dim': model_args['net_enc_dim'],
        'chunk_size': model_args['net_chunk_size'],
        'hop_size': model_args['net_hop_size'],
        'feature_dim': model_args['net_feature_dim'],
        'hidden_dim': model_args['net_hidden_dim'],
        'n_layers': model_args['net_n_layers'],
        'n_src': model_args['net_n_src'],
        'window_ms': model_args['net_window_ms'],
        'context_ms': model_args['net_context_ms'],
    }

    # Initialize the model
    model = FasNetTAC(sample_rate=sample_rate, **fasnet_args)

    # Remove 'model.' prefix from state_dict keys
    state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}

    # Load the state_dict with strict=False
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

def load_data(input_json, max_mics=4):
    """Load data for inference"""
    testset = TACDataset(input_json, max_mics=max_mics, train=False)
    return testset

def infer(model, inputs, valid_channels, device='cpu'):
    """Perform inference on the given inputs."""
    inputs = inputs.to(device)
    valid_channels = valid_channels.to(device)
    with torch.no_grad():
        outputs = model(inputs, valid_channels)
    
    # TODO: verify the question of permutations
    return outputs

def save_as_wav(output_tensor, sample_rate, output_path):
    """Save the output tensor as a .wav file using soundfile"""
    if isinstance(output_tensor, torch.Tensor):
        output_tensor = output_tensor.squeeze().cpu().numpy()  # Remove batch and channel dimensions and move to CPU
    else:
        output_tensor = output_tensor.squeeze()  # Handle NumPy arrays
    sf.write(output_path, output_tensor, samplerate=sample_rate, subtype="PCM_16")

def main(args):
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, sample_rate=args.sample_rate, device=args.device)
    # Load the testset
    print(f"Loading testset from {args.input_json}")
    testset = load_data(args.input_json)

    # Output folder
    if not args.output_dir:
        output_dir = PROJECT_ROOT / "experiments" / args.experiment_tag
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics files
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_sample_path = os.path.join(metrics_dir, 'metrics_by_sample.csv')
    metrics_path = os.path.join(metrics_dir, 'metrics.csv')

    # Perform inference
    print(f"Starting inference...")
    auditus = Auditus()
    results = []
    for i, (inputs, targets, valid_channels) in tqdm(enumerate(testset), total=len(testset), desc="Processing samples"):
        # print(f"Processing sample {i+1}/{len(testset)}")
        valid_channels = torch.tensor(valid_channels)
        outputs = infer(model, inputs.unsqueeze(0), valid_channels.unsqueeze(0), device=args.device) # Add batch dim

        # Save the results
        for mic in range(4):
        # for mic in range(1): # I compute the metrics only on the first channel (for now)
            sample_dir = output_dir / f"sample_{i}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # speech
            spk1_path = os.path.join(sample_dir, f"spk1_mic{mic+1}.wav")
            spk1 = targets[mic, 0, :]
            save_as_wav(output_tensor=spk1, sample_rate=args.sample_rate, output_path=spk1_path)
            
            # noise
            spk2_path = os.path.join(sample_dir, f"spk2_mic{mic+1}.wav")
            spk2 = targets[mic, 1, :]
            save_as_wav(output_tensor=spk2, sample_rate=args.sample_rate, output_path=spk2_path)

            # mixture
            mix_path = os.path.join(sample_dir, f"mixture_mic{mic+1}.wav")
            mix = inputs[mic, :]
            save_as_wav(output_tensor=mix, sample_rate=args.sample_rate, output_path=mix_path)
        
        # Estimated speech
        for spk in [0]: # We are not interested in the estimation of the noise spk=2 
            s_estim_path = os.path.join(sample_dir, f"s_estimate_spk{spk+1}.wav")
            s_estim = outputs[0, spk, :]
            save_as_wav(output_tensor=s_estim, sample_rate=args.sample_rate, output_path=s_estim_path)


        # Signals
        speech = targets[0, 0, :].unsqueeze(0).cpu()
        noise = targets[0, 1, :].unsqueeze(0).cpu()
        mixture = inputs[0, :].unsqueeze(0).cpu()
        estimated_speech = outputs[0, 0, :].unsqueeze(0).cpu()

        quality_metrics_in = auditus.se_eval(speech, noise, mixture)
        quality_metrics_out = auditus.se_eval(speech, noise, estimated_speech)

        current_results = {
            'Sample_ID': f'sample_{i}',
            'Microphone': 0,
            'SI-SIR_in': quality_metrics_in['si-sir'],
            'SI-SIR_out': quality_metrics_out['si-sir'],
            'Delta_SI-SIR': quality_metrics_out['si-sir'] - quality_metrics_in['si-sir'],
            'SI-SAR_out': quality_metrics_out['si-sar'],
            'SI-SDR_out': quality_metrics_out['si-sdr'],
            'FW-SIR_in': quality_metrics_in['fw-si-sir'],
            'FW-SIR_out': quality_metrics_out['fw-si-sir'],
            'FW-SAR_out': quality_metrics_out['fw-si-sar'],
            'FW-SDR_out': quality_metrics_out['fw-si-sdr'],
            'STOI_in': quality_metrics_in['stoi'],
            'STOI_out': quality_metrics_out['stoi'],
        }
        results.append(current_results)
    
    # Metrics by sample 
    metrics_by_sample = pd.DataFrame(results)
    metrics_by_sample.to_csv(metrics_by_sample_path, sep=",", index=False)

    # Metrics average
    metrics_to_aggregate = ['SI-SIR_in', 'SI-SIR_out', 'Delta_SI-SIR', 'SI-SAR_out', 'SI-SDR_out', 'FW-SIR_in', 'FW-SIR_out', 'FW-SAR_out', 'FW-SDR_out', 'STOI_in', 'STOI_out']
    aggregated_metrics = {} # Initialize a dictionary to store aggregated metrics
    for metric in metrics_to_aggregate: # Compute summary statistics for each metric
        aggregated_metrics[f"{metric}"] = round(metrics_by_sample[metric].mean(), 4)

    metrics = pd.DataFrame([aggregated_metrics]) # Convert the aggregated metrics to a DataFrame
    metrics.to_csv(metrics_path, sep=",", index=False)
    print(metrics[['SI-SIR_in','SI-SIR_out', 'Delta_SI-SIR', 'SI-SAR_out', 'SI-SDR_out', 'FW-SIR_in', 'FW-SIR_out', 'FW-SAR_out', 'FW-SDR_out', 'STOI_in', 'STOI_out']])
    print("Inference and saving completed successfully!")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=True, help="Experiment Tag.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate for audio processing.")
    parser.add_argument("--device", type=str, required=True, help="device CPU/GPU.")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
