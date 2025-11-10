import os
import re
import glob
import argparse
import numpy as np
from tqdm import tqdm 
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

PHONEMES = {
    "consonant": ["l", "ɫ", "ɫ̩", "ʎ", "ɾ", "ɾʲ", "ɾ̃", "w", "ɹ", "j", "f", "fʲ", "fʷ", "v", "vʲ", "vʷ", "θ", "ð", "ç", "h", "s", "z", "ʃ", "ʒ", "tʃ", "dʒ", "p", "pʰ", "pʲ", "pʷ", "b", "bʲ", "t̪", "d̪", "t", "tʰ", "tʲ", "tʷ", "d", "dʲ", "ɡ", "ɡb", "ɡʷ", "c", "cʰ", "cʷ", "ɟ", "ɟʷ", "k", "kʰ", "kʷ", "kp", "g", "ʔ", "m", "mʲ", "m̩", "n", "n̩", "ŋ", "ɲ"],
    "vowel": ["æ", "ɐ", "a", "aj", "aw", "aː", "ɑ", "ɑː", "ɒ", "ɒː", "ə", "əw", "ɚ", "ɛ", "ɛː", "ɜ", "ɜː", "ɝ", "ɔ", "ɔj", "i", "iː", "ʉ", "ʉː", "u", "uː", "ɪ", "ʊ", "e", "ej", "o", "ow"],
    "nasal": ["m", "mʲ", "m̩", "n", "n̩", "ŋ", "ɲ"],
    "plosive": ["p", "pʰ", "pʲ", "pʷ", "b", "bʲ", "t̪", "d̪", "t", "tʰ", "tʲ", "tʷ", "d", "dʲ", "ɡ", "ɡb", "ɡʷ", "c", "cʰ", "cʷ", "ɟ", "ɟʷ", "k", "kʰ", "kʷ", "kp", "g", "ʔ"],
    "affricate": ["tʃ", "dʒ"],
    "sibilant": ["s", "z", "ʃ", "ʒ"],
    "fricative": ["f", "fʲ", "fʷ", "v", "vʲ", "vʷ", "θ", "ð", "ç", "h"],
    "approximant": ["w", "ɹ", "j"],
    "tap": ["ɾ", "ɾʲ", "ɾ̃"],
    "lateral": ["l", "ɫ", "ɫ̩", "ʎ"],
    "close": ["i", "iː", "ʉ", "ʉː", "u", "uː"],
    "near-close": ["ɪ", "ʊ"], 
    "close-mid": ["e", "ej", "o", "ow"],
    "open-mid": ["ə", "əw", "ɚ", "ɛ", "ɛː", "ɜ", "ɜː", "ɝ", "ɔ", "ɔj"],
    "open": ["æ", "ɐ", "a", "aj", "aw", "aː", "ɑ", "ɑː", "ɒ", "ɒː"]
}

SIGNAL_TYPE = {
    'clean': 'spk1_mic1.wav',
    'noise': 'spk2_mic1.wav',
    'mix': 'mixture_mic1.wav',
    'est': 's_estimate_spk1.wav',
}

def extract_phonemes(textgrid_str):
    # Extract the phones tier
    phones_tier_match = re.search(r'name = "phones".*?intervals: size = \d+', textgrid_str, re.DOTALL)
    if not phones_tier_match:
        raise ValueError("Phones tier not found.")
    
    phones_section_start = phones_tier_match.end()
    phones_section = textgrid_str[phones_section_start:]
    
    # Now extract intervals one by one
    interval_pattern = re.compile(
        r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', re.DOTALL
    )
    
    phoneme_tuples = []
    for match in interval_pattern.finditer(phones_section):
        xmin, xmax, text = match.groups()
        if text.strip():  # Skip empty intervals
            phoneme_tuples.append((float(xmin), float(xmax), text.strip()))
    
    return phoneme_tuples

def retrieve_phonemes(phoneme_cat, testset, signal_type, loss='Baseline'):
    
    # Get the number of clean speech in the testset
    n = len(glob.glob(os.path.join(PROJECT_ROOT, 'data/clean/test/*.wav')))

    # Define the corpus path
    corpus_path = os.path.join(PROJECT_ROOT, 'experiments', testset)

    # Select experiment tag and signal type
    signal_file = SIGNAL_TYPE[signal_type]

    signal_segments = []
    for i in tqdm(range(n), desc=f'{phoneme_cat} | {signal_type}'):
        
        # Get the reverberate signal from the Baseline data as a reference (cf. it must be the same for the other models)
        signal_path = os.path.join(corpus_path, loss, f'sample_{i}', signal_file)
        signal, sr = sf.read(signal_path)

        # Retrieve the phoneme segmentation for each i-th sample
        phseg_path = os.path.join(PROJECT_ROOT, 'data/ph_segmentation', f'sample_{i}.TextGrid')
        with open(phseg_path, 'r') as f:
            textgrid_str = f.read()
        L = extract_phonemes(textgrid_str)

        # Append the phoneme segments in signal_segments
        for start, end, phoneme in L:
            if phoneme in PHONEMES[phoneme_cat]:
                signal_segments.append(signal[int(start*sr):int(end*sr)])

    # Concatenate the phoneme category segments
    phoneme = np.concatenate(signal_segments)

    # Create the output forder if it does not exist
    output_dir = os.path.join(corpus_path, loss, 'phoneme', signal_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output the phoneme category concatenated signal
    output_file = os.path.join(output_dir, f'{phoneme_cat}.wav')
    sf.write(output_file, phoneme, sr)

def main(args):

    # Retrieve clean phoneme category (or all phoneme categories)
    if args.phoneme == 'all': 
        for p in PHONEMES.keys():
            if args.signal_type == 'all':
                for s in ['clean', 'noise', 'mix', 'est']:
                    retrieve_phonemes(phoneme_cat=p, testset=args.testset, signal_type=s, loss=args.loss)
            else:
                retrieve_phonemes(phoneme_cat=p, testset=args.testset, signal_type=args.signal_type, loss=args.loss)
    else:
        if args.signal_type == 'all':
            for s in ['clean', 'noise', 'mix', 'est']:
                retrieve_phonemes(phoneme_cat=args.phoneme, testset=args.testset, signal_type=s, loss=args.loss)
        else:
            retrieve_phonemes(phoneme_cat=args.phoneme, testset=args.testset, signal_type=args.signal_type, loss=args.loss)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to retrieve phoneme evaluation results.",
        epilog="Example: python3 src/evaluation/ph_retrieve.py --signal_type clean --testset test11 --phoneme plosive --loss XP14"
    )
    parser.add_argument("--signal_type", type=str, required=True, choices=['all', 'clean', 'noise', 'mix', 'est'], help='Enter the type of signal.')
    parser.add_argument('--loss', type=str, required=False, default=None, help="Enter the loss if signal='est' (e.g. baseline, xp4, xp13)")
    parser.add_argument('--testset', type=str, choices=['test10', 'test11'], help='Enter the testset tag.') 
    parser.add_argument('--phoneme', type=str, 
                        choices=[
                            'all', 'consonant', 'vowel', 
                            'nasal', 'plosive', 'affricate', 'sibilant', 'fricative', 'approximant', 'tap', 'lateral', 
                            'close', 'near-close', 'close-mid', 'open-mid', 'open'
                        ],
                        help='Enter the phoneme category.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)