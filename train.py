import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from emg_decoder.config import Config
from emg_decoder.evaluate import run_subject_pipeline
from emg_decoder.utils import aggregate_results, print_results, run_stats, plot_results, build_final_json


def main():
    parser = argparse.ArgumentParser(description='EMG Gesture Decoder - Ninapro DB2')
    parser.add_argument('--dataset', '-d', type=str, default='local', choices=['local', 'kaggle'])
    parser.add_argument('--fast', action='store_true', help='Fast mode (5 subjects, 20 epochs)')
    parser.add_argument('--subjects', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--svm-only', action='store_true', help='Tree-based models only (skip CNNs)')
    args = parser.parse_args()

    cfg = Config.kaggle() if args.dataset == 'kaggle' else Config.local()
    if args.fast:
        cfg.fast_mode = True
    if args.subjects is not None:
        cfg.fast_n_subjects = args.subjects
        cfg.fast_mode = True
    if args.epochs is not None:
        cfg.fast_n_epochs = args.epochs
        cfg.fast_mode = True

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if torch.cuda.is_available():
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'PyTorch: {torch.__version__}')

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print('\nConfiguration loaded.')
    cfg.print_config()

    all_mat_files = sorted(cfg.dataset_root.rglob('*.mat'))
    print(f'\nTotal .mat files found: {len(all_mat_files)}')

    subject_results = []
    best_subj_cache = None

    print(f'\nWithin-subject evaluation - {cfg.n_subjects_run} subjects | 4 models per subject')
    print('=' * 92)

    le = None
    for sid in tqdm(range(1, cfg.n_subjects_run + 1), desc='Subjects'):
        entry, le = run_subject_pipeline(sid, cfg, device, all_mat_files, le)
        if entry is None:
            print(f'S{sid:02d}: SKIP')
            continue
        subject_results.append(entry)
        print(f'S{sid:02d} | n={entry["n_train"]:4d} | SVM={entry["svm_acc"]:.3f} | '
              f'LGBM={entry["lgbm_acc"]:.3f} | CNN1D={entry["cnn1d_acc"]:.3f} | '
              f'CNN2D={entry["cnn2d_acc"]:.3f} | ep1d={entry["cnn1d_epochs"]} ep2d={entry["cnn2d_epochs"]}')

        if best_subj_cache is None or entry['cnn2d_acc'] > best_subj_cache['acc']:
            best_subj_cache = {'sid': sid, 'acc': entry['cnn2d_acc'], 'y_test': None, 'yp_2d': None}

    print('\n' + '=' * 92)
    print(f'Done. {len(subject_results)} subjects evaluated.')

    if not subject_results:
        print('No subjects evaluated. Check dataset path.')
        return

    agg = aggregate_results(subject_results)
    print_results(agg, cfg)

    if len(subject_results) >= 2:
        run_stats(agg['df'], agg)

    plot_results(agg['df'], agg, best_subj_cache, cfg, cfg.figures_dir)

    final_results = build_final_json(subject_results, agg, cfg, device)
    final_results['timestamp'] = datetime.now().isoformat()
    out = cfg.results_dir / 'final_results.json'
    with open(out, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f'\nResults saved -> {out}')


if __name__ == '__main__':
    main()
