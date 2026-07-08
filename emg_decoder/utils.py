from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix

from .config import Config


def aggregate_results(subject_results: List[Dict]) -> Dict:
    df = pd.DataFrame(subject_results)

    svm_mean = df['svm_acc'].mean(); svm_std = df['svm_acc'].std()
    lgb_mean = df['lgbm_acc'].mean(); lgb_std = df['lgbm_acc'].std()
    c1d_mean = df['cnn1d_acc'].mean(); c1d_std = df['cnn1d_acc'].std()
    c2d_mean = df['cnn2d_acc'].mean(); c2d_std = df['cnn2d_acc'].std()

    svm_f1m = df['svm_f1'].mean(); svm_f1s = df['svm_f1'].std()
    lgb_f1m = df['lgbm_f1'].mean(); lgb_f1s = df['lgbm_f1'].std()
    c1d_f1m = df['cnn1d_f1'].mean(); c1d_f1s = df['cnn1d_f1'].std()
    c2d_f1m = df['cnn2d_f1'].mean(); c2d_f1s = df['cnn2d_f1'].std()

    return {
        'df': df,
        'svm_mean': svm_mean, 'svm_std': svm_std,
        'lgb_mean': lgb_mean, 'lgb_std': lgb_std,
        'c1d_mean': c1d_mean, 'c1d_std': c1d_std,
        'c2d_mean': c2d_mean, 'c2d_std': c2d_std,
        'svm_f1m': svm_f1m, 'svm_f1s': svm_f1s,
        'lgb_f1m': lgb_f1m, 'lgb_f1s': lgb_f1s,
        'c1d_f1m': c1d_f1m, 'c1d_f1s': c1d_f1s,
        'c2d_f1m': c2d_f1m, 'c2d_f1s': c2d_f1s,
    }


def print_results(agg: Dict, cfg: Config):
    df = agg['df']
    print('\nPer-subject results:')
    cols = ['subject_id', 'n_train', 'n_test',
            'svm_acc', 'lgbm_acc', 'cnn1d_acc', 'cnn2d_acc',
            'svm_f1', 'lgbm_f1', 'cnn1d_f1', 'cnn2d_f1']
    print(df[cols].to_string(index=False, float_format='{:.3f}'.format))

    print('\n' + '=' * 80)
    print('  FINAL RESULTS - Within-Subject Protocol')
    print(f'  Ninapro DB2 Ex1 | {cfg.n_classes} classes | {len(df)} subjects')
    print('=' * 80)
    print(f'  SVM (TD features):')
    print(f'    Accuracy : {agg["svm_mean"]:.4f} +/- {agg["svm_std"]:.4f}')
    print(f'    Macro-F1 : {agg["svm_f1m"]:.4f} +/- {agg["svm_f1s"]:.4f}')
    print(f'  LightGBM (TD features):')
    print(f'    Accuracy : {agg["lgb_mean"]:.4f} +/- {agg["lgb_std"]:.4f}')
    print(f'    Macro-F1 : {agg["lgb_f1m"]:.4f} +/- {agg["lgb_f1s"]:.4f}')
    print(f'  TinyCNN-1D (raw signal):')
    print(f'    Accuracy : {agg["c1d_mean"]:.4f} +/- {agg["c1d_std"]:.4f}')
    print(f'    Macro-F1 : {agg["c1d_f1m"]:.4f} +/- {agg["c1d_f1s"]:.4f}')
    print(f'  TinyCNN-2D (STFT spectrogram):')
    print(f'    Accuracy : {agg["c2d_mean"]:.4f} +/- {agg["c2d_std"]:.4f}')
    print(f'    Macro-F1 : {agg["c2d_f1m"]:.4f} +/- {agg["c2d_f1s"]:.4f}')
    print(f'  Improvement CNN-2D vs CNN-1D:')
    print(f'    dAcc: {agg["c2d_mean"] - agg["c1d_mean"]:+.4f}')
    print('=' * 80)
    print('\nLiterature (Ninapro DB2 Ex1, within-subject):')
    print('  Geng 2016 (CNN)         : 78.90%')
    print('  Chen 2022 (Transfer SVM): 82.87%')
    print('  Zanghieri 2023 (TEMGNet): 82.93%')
    print(f'  This work SVM           : {agg["svm_mean"]*100:.2f}%')
    print(f'  This work LightGBM      : {agg["lgb_mean"]*100:.2f}%')
    print(f'  This work TinyCNN-1D    : {agg["c1d_mean"]*100:.2f}%')
    print(f'  This work TinyCNN-2D    : {agg["c2d_mean"]*100:.2f}%')


def run_stats(df: pd.DataFrame, agg: Dict):
    t_stat, p_val = stats.ttest_rel(df['cnn2d_acc'], df['cnn1d_acc'])
    cd = (df['cnn2d_acc'] - df['cnn1d_acc']).mean() / (df['cnn2d_acc'] - df['cnn1d_acc']).std()
    t2, p2 = stats.ttest_rel(df['lgbm_acc'], df['svm_acc'])
    t3, p3 = stats.ttest_rel(df['cnn2d_acc'], df['svm_acc'])

    print('\nPaired t-tests:')
    print(f'  CNN-2D vs CNN-1D: t={t_stat:.4f}  p={p_val:.4f}  d={cd:.4f}')
    print(f'  LightGBM vs SVM:  t={t2:.4f}  p={p2:.4f}')
    print(f'  CNN-2D vs SVM:    t={t3:.4f}  p={p3:.4f}')
    n2v1 = (df['cnn2d_acc'] > df['cnn1d_acc']).sum()
    n2vs = (df['cnn2d_acc'] > df['svm_acc']).sum()
    lvs = (df['lgbm_acc'] > df['svm_acc']).sum()
    print(f'\nWin analysis:')
    print(f'  CNN-2D > CNN-1D : {n2v1}/{len(df)} ({n2v1/len(df)*100:.1f}%)')
    print(f'  CNN-2D > SVM    : {n2vs}/{len(df)} ({n2vs/len(df)*100:.1f}%)')
    print(f'  LightGBM > SVM  : {lvs}/{len(df)} ({lvs/len(df)*100:.1f}%)')


def plot_results(df: pd.DataFrame, agg: Dict, best_subj_cache: dict,
                 cfg: Config, figures_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EMG Gesture Decoding - Ninapro DB2 (4 Models)', fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - 1.5*w, df['svm_acc'], w, label='SVM', color='#4C72B0', alpha=0.85)
    ax.bar(x - 0.5*w, df['lgbm_acc'], w, label='LightGBM', color='#C44E52', alpha=0.85)
    ax.bar(x + 0.5*w, df['cnn1d_acc'], w, label='TinyCNN-1D', color='#DD8452', alpha=0.85)
    ax.bar(x + 1.5*w, df['cnn2d_acc'], w, label='TinyCNN-2D', color='#55A868', alpha=0.85)
    for label, mean in [('SVM', agg['svm_mean']), ('LightGBM', agg['lgb_mean']),
                        ('TinyCNN-1D', agg['c1d_mean']), ('TinyCNN-2D', agg['c2d_mean'])]:
        ax.axhline(mean, ls='--', alpha=0.4)
    ticks = list(range(0, len(df), 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([df['subject_id'].iloc[i] for i in ticks])
    ax.set_xlabel('Subject ID'); ax.set_ylabel('Test Accuracy')
    ax.set_title('Per-Subject Test Accuracy (4 models)')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(df['cnn1d_acc'], df['cnn2d_acc'], c='#2ca02c', alpha=0.7, s=60, edgecolors='k', lw=0.5)
    lo = min(df['cnn1d_acc'].min(), df['cnn2d_acc'].min()) - 0.02
    hi = max(df['cnn1d_acc'].max(), df['cnn2d_acc'].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.5)
    ax.set_xlabel('TinyCNN-1D Accuracy'); ax.set_ylabel('TinyCNN-2D Accuracy')
    ax.set_title('TinyCNN-1D vs TinyCNN-2D')
    n_imp = (df['cnn2d_acc'] > df['cnn1d_acc']).sum()
    ax.text(0.05, 0.92, f'CNN-2D better in {n_imp}/{len(df)} subjects',
            transform=ax.transAxes, fontsize=9, color='#2ca02c')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    names = ['SVM\n(ours)', 'LightGBM\n(ours)', 'TinyCNN-1D\n(ours)',
             'TinyCNN-2D\n(ours)', 'Geng\n2016', 'Chen\n2022', 'Zanghieri\n2023']
    means = [agg['svm_mean']*100, agg['lgb_mean']*100, agg['c1d_mean']*100,
             agg['c2d_mean']*100, 78.90, 82.87, 82.93]
    stds = [agg['svm_std']*100, agg['lgb_std']*100, agg['c1d_std']*100,
            agg['c2d_std']*100, 0, 0, 0]
    colors = ['#4C72B0', '#C44E52', '#DD8452', '#55A868', '#8da0cb', '#8da0cb', '#8da0cb']
    bars = ax.bar(range(len(names)), means, yerr=stds, color=colors, alpha=0.85, capsize=5, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_title('Comparison with Literature')
    ax.set_ylim(40, 100); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax = axes[1, 1]
    cm = confusion_matrix(best_subj_cache['y_test'], best_subj_cache['yp_2d'])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    short = [n[:8] for n in cfg.gesture_names]
    ax.set_xticks(range(cfg.n_classes))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(cfg.n_classes))
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'TinyCNN-2D Confusion Matrix - S{best_subj_cache["sid"]:02d} '
                 f'(acc={best_subj_cache["acc"]:.3f})')

    plt.tight_layout()
    plt.savefig(figures_dir / 'results_4model.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: results_4model.png')


def build_final_json(subject_results: List[Dict], agg: Dict, cfg: Config, device: torch.device):
    from scipy import stats as _stats
    df = agg['df']
    _t, _p = _stats.ttest_rel(df['cnn2d_acc'], df['cnn1d_acc'])

    return {
        'experiment': 'EMG Gesture Decoding - Ninapro DB2 Ex1 (Within-Subject, 4-Model)',
        'evaluation_protocol': {
            'type': 'within_subject',
            'description': 'Separate model per subject. Per-channel Z-score normalisation '
                           'using training-rep statistics only. Rep-based split.',
            'n_subjects': len(subject_results),
            'train_reps': cfg.train_reps, 'val_reps': cfg.val_reps, 'test_reps': cfg.test_reps,
        },
        'dataset': {
            'name': 'Ninapro DB2', 'doi': '10.1038/sdata.2014.53',
            'kaggle': 'quddusikashaf/ninapro-db2',
            'n_subjects': cfg.n_subjects, 'exercise': cfg.exercise,
            'n_classes': cfg.n_classes, 'gesture_names': cfg.gesture_names,
            'sample_rate_hz': cfg.sample_rate, 'n_emg_channels': cfg.n_emg_channels,
            'bandpass_hz': [cfg.bandpass_low, cfg.bandpass_high],
            'window_ms': cfg.window_ms, 'step_ms': cfg.step_ms,
        },
        'svm': {
            'model': f'LinearSVC (C={cfg.svm_c}, balanced)',
            'features': 'MAV+RMS+WL+ZC+SSC x12 (60-dim)',
            'mean_accuracy': round(float(agg['svm_mean']), 4),
            'std_accuracy': round(float(agg['svm_std']), 4),
            'mean_macro_f1': round(float(agg['svm_f1m']), 4),
            'std_macro_f1': round(float(agg['svm_f1s']), 4),
        },
        'lightgbm': {
            'model': f'LGBM (n_est={cfg.lgbm_n_estimators}, depth={cfg.lgbm_max_depth})',
            'features': 'MAV+RMS+WL+ZC+SSC x12 (60-dim)',
            'mean_accuracy': round(float(agg['lgb_mean']), 4),
            'std_accuracy': round(float(agg['lgb_std']), 4),
            'mean_macro_f1': round(float(agg['lgb_f1m']), 4),
            'std_macro_f1': round(float(agg['lgb_f1s']), 4),
        },
        'tinycnn1d': {
            'model': 'TinyCNN-1D (16->32->64, 3 conv layers)',
            'input': 'raw normalised sEMG windows (12, 400)',
            'params': 51857,
            'batch_size': cfg.batch_size_1d, 'lr': cfg.lr_1d,
            'mean_accuracy': round(float(agg['c1d_mean']), 4),
            'std_accuracy': round(float(agg['c1d_std']), 4),
            'mean_macro_f1': round(float(agg['c1d_f1m']), 4),
            'std_macro_f1': round(float(agg['c1d_f1s']), 4),
        },
        'tinycnn2d': {
            'model': 'TinyCNN-2D (8->16->32, 3 conv layers)',
            'input': f'STFT log-power spectrogram (12, {cfg.spec_n_freq}, {cfg.spec_n_time})',
            'stft': {'nperseg': cfg.spec_nperseg, 'noverlap': cfg.spec_noverlap},
            'augmentation': {'type': 'SpecAugment', 'aug_prob': cfg.aug_prob,
                             'time_mask_max': cfg.aug_time_mask, 'freq_mask_max': cfg.aug_freq_mask,
                             'amp_range': [cfg.aug_amp_lo, cfg.aug_amp_hi]},
            'params': 17745,
            'batch_size': cfg.batch_size_2d, 'lr': cfg.lr_2d,
            'mean_accuracy': round(float(agg['c2d_mean']), 4),
            'std_accuracy': round(float(agg['c2d_std']), 4),
            'mean_macro_f1': round(float(agg['c2d_f1m']), 4),
            'std_macro_f1': round(float(agg['c2d_f1s']), 4),
        },
        'statistical_test': {
            'test': 'paired t-test (CNN-2D vs CNN-1D)',
            't_statistic': round(float(_t), 4),
            'p_value': round(float(_p), 6),
            'n_subjects': len(df),
        },
        'per_subject': subject_results,
        'hardware': {
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        },
        'timestamp': None,
        'random_seed': cfg.random_seed,
    }
