import json
import numpy as np
from pathlib import Path

from .config import (
    get_labels_file,
    RESULTS_DIR,
    RANKINGS_DIR,
    ensure_output_dirs,
)

def evaluate_rankings(rankings_filename, train: bool = True):
    """
    Evaluate rankings against ground truth labels.
    
    Args:
        rankings_filename: Path to the rankings JSON file.
        train: If True, use training labels; otherwise use test labels.
    
    Returns:
        dict: Evaluation metrics (MRR, P@k, R@k, F1@k for k in [5, 10, 20])
    """
    label_file = get_labels_file(train)

    results = {}
    metrics = ['MRR', 'P@5', 'R@5', 'F1@5', 'P@10', 'R@10', 'F1@10', 'P@20', 'R@20', 'F1@20']
    for metric in metrics: results[metric] = []

    with open(rankings_filename, 'r') as f:
        rankings = json.load(f)

    with open(label_file, 'r') as f:
        labels = json.load(f)

    assert(set(rankings.keys()) == set(labels.keys()))

    for query_file, noted_cases in labels.items():
        retrieved_rankings = rankings[query_file]

        # MRR
        first_rank = next((r+1 for r, f in enumerate(retrieved_rankings) if f in noted_cases), None)
        mrr = 1.0 / first_rank if first_rank else 0.0
        results['MRR'].append(mrr)


        # @5
        hits_in_5 = sum(f in noted_cases for f in retrieved_rankings[:5])
        p5 = hits_in_5 / 5
        r5 = hits_in_5 / max(len(noted_cases),1)
        results['P@5'].append(p5)
        results['R@5'].append(r5)
        results['F1@5'].append(2*r5*p5/(r5+p5) if (r5+p5) else 0.0)

        # @10
        hits_in_10 = sum(f in noted_cases for f in retrieved_rankings[:10])
        p10 = hits_in_10 / 10
        r10 = hits_in_10 / max(len(noted_cases),1)
        results['P@10'].append(p10)
        results['R@10'].append(r10)
        results['F1@10'].append(2*r10*p10/(r10+p10) if (r10+p10) else 0.0)

        # @20
        hits_in_20 = sum(f in noted_cases for f in retrieved_rankings[:20])
        p20 = hits_in_20 / 20
        r20 = hits_in_20 / max(len(noted_cases),1)
        results['P@20'].append(p20)
        results['R@20'].append(r20)
        results['F1@20'].append(2*r20*p20/(r20+p20) if (r20+p20) else 0.0)

    final_results = {}

    for metric in metrics:
        final_results[metric] = {
            'mean': float(np.mean(results[metric])),
            'std': float(np.std(results[metric])),
            'median': float(np.median(results[metric])),
            'Q1': float(np.percentile(results[metric], 25)),
            'Q3': float(np.percentile(results[metric], 75)),
            'IQR': float(np.percentile(results[metric], 75) - np.percentile(results[metric], 25))
        }

    return final_results


def evaluate_all_rankings(train: bool = True, save_results: bool = True):
    """
    Evaluate all ranking files in the rankings directory.
    
    Args:
        train: If True, use training labels; otherwise use test labels.
        save_results: If True, save aggregated results to a JSON file.
    
    Returns:
        dict: All results keyed by ranking filename.
    """
    ensure_output_dirs()
    
    all_results = {}
    
    if not RANKINGS_DIR.exists():
        print(f"Rankings directory not found: {RANKINGS_DIR}")
        return all_results
    
    for file in sorted(RANKINGS_DIR.iterdir()):
        if file.suffix == '.json':
            try:
                res = evaluate_rankings(file, train=train)
                all_results[file.name] = res
                print(f"Evaluated: {file.name}")
            except Exception as e:
                print(f"Error evaluating {file.name}: {e}")
    
    if save_results and all_results:
        split_name = 'train' if train else 'test'
        output_file = RESULTS_DIR / f'evaluation_results_{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved -> {output_file}")
    
    return all_results
