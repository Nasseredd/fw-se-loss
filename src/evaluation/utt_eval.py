import os
import sys
import glob
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pesq import pesq
import soundfile as sf
from pystoi import stoi
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(os.path.join(PROJECT_ROOT, 'external_repos/asteroid'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from auditus.evaluation import Auditus

def get_paths(sample_dir, mic):
    paths = {
        'speech': os.path.join(sample_dir, f"spk1_mic{mic}.wav"),
        'noise': os.path.join(sample_dir, f"spk2_mic{mic}.wav"),
        'mixture': os.path.join(sample_dir, f"mixture_mic{mic}.wav"),
        's_estimated': os.path.join(sample_dir, f"s_estimate_spk1.wav")
    }
    return paths

def load_signals(paths, max_samples):
    speech, _ = sf.read(paths['speech'], dtype="float32")
    noise, _ = sf.read(paths['noise'], dtype="float32")
    mix, _ = sf.read(paths['mixture'], dtype="float32")
    s_estimated, _ = sf.read(paths['s_estimated'], dtype="float32")

    if speech.shape[0] > max_samples:
        signals = {'speech': speech[:max_samples], 'noise': noise[:max_samples], 'mixture': mix[:max_samples], 's_estimated': s_estimated[:max_samples]}
    else:
        signals = {'speech': speech, 'noise': noise, 'mixture': mix, 's_estimated': s_estimated}

    return signals

def normalize_audio(tensor):
        """Normalize audio tensor between -1 and 1."""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 2 - 1

def compute_sample_metrics(sample_dir, sample_rate=16000, max_dur=10):

    metrics_list = []
    auditus = Auditus()
    # for mic in [0, 1, 2, 3]:
    # TODO: include other mics in aggregation to have metrics per mic
    for mic in [1]:

        paths = get_paths(sample_dir, mic)
        signals = load_signals(paths, max_samples=int(sample_rate*10))

        # Clean and Estimated Signals in Tensors
        speech = torch.tensor(signals['speech'], dtype=torch.float32)  
        noise = torch.tensor(signals['noise'], dtype=torch.float32)  
        mixture = torch.tensor(signals['mixture'], dtype=torch.float32)
        est_speech = torch.tensor(signals['s_estimated'], dtype=torch.float32)
        
        # SIR, SAR and SDR
        quality_metrics_in = auditus.se_eval(speech, noise, mixture)
        quality_metrics_out = auditus.se_eval(speech, noise, est_speech)
        
        # Compute PESQ (narrowband mode)
        pesq_score = pesq(sample_rate, signals['speech'], signals['s_estimated'], 'wb')

        # Compute STOI
        stoi_score = stoi(signals['speech'], signals['s_estimated'], sample_rate)

        sample_id = sample_dir.split('/')[-1]

        metrics_list.append({
            'Sample_ID': sample_id,
            'Microphone': mic,
            'SI-SIR_in': quality_metrics_in['si-sir'],
            'SI-SIR_out': quality_metrics_out['si-sir'],
            'Delta_SI-SIR': quality_metrics_out['si-sir'] - quality_metrics_in['si-sir'],
            'SI-SAR_out': quality_metrics_out['si-sar'],
            'SI-SDR_out': quality_metrics_out['si-sdr'],
            'FW-SIR_out': quality_metrics_out['fw-si-sir'],
            'FW-SAR_out': quality_metrics_out['fw-si-sar'],
            'FW-SDR_out': quality_metrics_out['fw-si-sdr'],
            'PESQ': round(pesq_score, 2),
            'STOI': round(stoi_score, 2),
        })

    return metrics_list

def compute_metrics_by_sample(experiment_tag):

    samples_dirs = list(glob.glob(os.path.join(str(PROJECT_ROOT), 'experiments', experiment_tag, 'sample_*')))

    metrics_list = []
    for sample_dir in tqdm(samples_dirs):
        metrics_list_curr = compute_sample_metrics(sample_dir)
        metrics_list.extend(metrics_list_curr)

    metrics_by_sample_df = pd.DataFrame(metrics_list)
    return metrics_by_sample_df

def compute_metrics(metrics_by_sample_df):
    """
    Compute aggregated metrics across all samples and microphones.

    Parameters:
        metrics_by_sample_df (pd.DataFrame): DataFrame containing metrics for each sample.

    Returns:
        pd.DataFrame: A DataFrame summarizing aggregated metrics.
    """
    # List of metrics to aggregate
    metrics_to_aggregate = ['SI-SIR_in', 'SI-SIR_out', 'Delta_SI-SIR', 'SI-SAR_out', 'SI-SDR_out', 'FW-SIR_out', 'FW-SAR_out', 'FW-SDR_out', 'PESQ', 'STOI']
    
    # Initialize a dictionary to store aggregated metrics
    aggregated_metrics = {}

    # Compute summary statistics for each metric
    for metric in metrics_to_aggregate:
        aggregated_metrics[f"{metric}_mean"] = round(metrics_by_sample_df[metric].mean(), 2)

    # Convert the aggregated metrics to a DataFrame
    metrics_df = pd.DataFrame([aggregated_metrics])
    return metrics_df

def main(args):
    # Create a directory to save metric results
    output_dir = PROJECT_ROOT / 'experiments' / args.experiment_tag / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics by sample and save the dataframe in a csv file
    metrics_by_sample_df = compute_metrics_by_sample(args.experiment_tag)
    metrics_by_sample_path = os.path.join(output_dir, 'metrics_by_sample.csv')
    metrics_by_sample_df.to_csv(metrics_by_sample_path, sep=',', index=False)
    print('[Stage 2] Computed metrics by sample successfully.')
    print(f'[Stage 2] Exported metrics_by_sample.csv successfully in {metrics_by_sample_path}.')

    # Export metrics 
    metrics_df = compute_metrics(metrics_by_sample_df)
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, sep=',', index=False)
    print('[Stage 2] Computed metrics successfully.')
    print(f'[Stage 2] Exported metrics.csv successfully in {metrics_path}.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=True, help="Experiment Tag.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)