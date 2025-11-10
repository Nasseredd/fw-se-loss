import os
import sys
import torch
import argparse
import pandas as pd
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'auditus'))
from evaluation import Auditus

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def compute_phoneme_category_metrics(testset, loss, phoneme_cat):
    
    # Corpus path
    corpus_path = os.path.join(PROJECT_ROOT, 'experiments')

    # Import clean, noise, mixture and estimated speech signals for the given phoneme category
    clean, _ = sf.read(os.path.join(corpus_path, testset, loss, 'phoneme', 'clean', f"{phoneme_cat}.wav"))
    noise, _ = sf.read(os.path.join(corpus_path, testset, loss, 'phoneme', 'noise', f"{phoneme_cat}.wav"))
    mix, _   = sf.read(os.path.join(corpus_path, testset, loss, 'phoneme', 'mix', f"{phoneme_cat}.wav"))
    est, _   = sf.read(os.path.join(corpus_path, testset, loss, 'phoneme', 'est', f"{phoneme_cat}.wav"))

    # Compute input metrics
    auditus = Auditus()
    input_results = auditus.se_eval(
        torch.from_numpy(clean).unsqueeze(0),
        torch.from_numpy(noise).unsqueeze(0),
        torch.from_numpy(mix).unsqueeze(0),
        metrics=['si-sdr', 'si-sir', 'si-sar', 'fw-si-sdr', 'fw-si-sir', 'fw-si-sar'])

    # Compute output metrics
    auditus = Auditus()
    output_results = auditus.se_eval(
        torch.from_numpy(clean).unsqueeze(0), 
        torch.from_numpy(noise).unsqueeze(0), 
        torch.from_numpy(est).unsqueeze(0),
        metrics=['si-sdr', 'si-sir', 'si-sar', 'fw-si-sdr', 'fw-si-sir', 'fw-si-sar'])

    metrics = {
        'testset': testset,
        'loss': loss,
        'phoneme': phoneme_cat,
        'si-sir_in': round(input_results['si-sir'], 1),
        'si-sir_out': round(output_results['si-sir'], 1),
        'si-sar_out': round(output_results['si-sar'], 1),
        'si-sdr_out': round(output_results['si-sdr'], 1),
        'fw-si-sir_in': round(input_results['fw-si-sir'], 1),
        'fw-si-sir_out': round(output_results['fw-si-sir'], 1),
        'fw-si-sar_out': round(output_results['fw-si-sar'], 1),
        'fw-si-sdr_out': round(output_results['fw-si-sdr'], 1),
    }
    return metrics 

def export_metrics(testset, loss, metrics, phoneme_cat):

    csv_path = os.path.join(PROJECT_ROOT, 'experiments', f'{testset}/{loss}/metrics/metrics_by_phoneme.csv')

    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=list(metrics.keys()))
        df.to_csv(csv_path, sep=',', index=False)
        print(f'\033[36m[INFO] {testset} | {loss} | created the csv file to save metrics for all phoneme in \n       {csv_path}\033[0m')

    # Read csv file 
    df = pd.read_csv(csv_path, sep=',')

    # if phoneme category not in the csv file
    if phoneme_cat not in df['phoneme'].unique():
        metrics_df = pd.DataFrame([metrics])
        df = pd.concat([df, metrics_df], ignore_index=True)
        df.to_csv(csv_path, sep=',', index=False)
        print(f'\033[32m[INFO] {args.testset} | {args.loss} | phoneme: {phoneme_cat} Computing metrics completed!\033[0m')
    else:
        print(f'\033[31m[WARNING] {args.testset} | {args.loss} | phoneme: {phoneme_cat} is already in the csv file!\033[0m')

def main(args):

    if args.phoneme == 'all': 
        for p in ['nasal', 'plosive', 'affricate', 'sibilant', 'fricative', 'approximant', 'tap', 'lateral', 'close', 'near-close', 'close-mid', 'open-mid', 'open']:
            metrics = compute_phoneme_category_metrics(testset=args.testset, loss=args.loss, phoneme_cat=p)
            export_metrics(testset=args.testset, loss=args.loss, metrics=metrics, phoneme_cat=p)
    else:
        metrics = compute_phoneme_category_metrics(testset=args.testset, loss=args.loss, phoneme_cat=args.phoneme)
        export_metrics(testset=args.testset, loss=args.loss, metrics=metrics, phoneme_cat=args.phoneme)

def parse_arguments():
    parser = argparse.ArgumentParser(
        epilog='Example: python3 src/evaluation/ph_eval.py --testset test11 --loss xp3 --phoneme all'
    )
    parser.add_argument('--testset', type=str, required=True, choices=['test10','test11'], help='Enter a testset tag.')
    parser.add_argument('--loss', type=str, required=True, help='Enter a loss tag.')
    parser.add_argument('--phoneme', type=str, required=True,
                        choices=['all','nasal', 'plosive', 'affricate', 'sibilant', 'fricative', 'approximant', 'tap', 'lateral', 'close', 'near-close', 'close-mid', 'open-mid', 'open'],
                        help='Enter the phoneme category.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)