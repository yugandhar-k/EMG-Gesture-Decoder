from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from .config import Config
from .data import load_single_subject, per_subject_normalise, window_to_spectrogram
from .models import extract_td_features
from .train import (
    train_tinycnn1d_subject, train_tinycnn2d_subject,
    predict_tinycnn, predict_tinycnn2d,
    evaluate_svm, evaluate_lgbm,
)


def run_subject_pipeline(sid: int, cfg: Config, device: torch.device,
                         file_cache=None, le: Optional[LabelEncoder] = None
                         ) -> Tuple[Optional[Dict], Optional[LabelEncoder]]:
    if le is None:
        le = LabelEncoder()
        le.fit(np.arange(1, cfg.n_classes + 1))

    result = load_single_subject(cfg.dataset_root, sid, cfg, file_cache)
    if result is None:
        return None, le
    windows_s, labels_s, reps_s = result

    train_mask = np.isin(reps_s, cfg.train_reps)
    val_mask = np.isin(reps_s, cfg.val_reps)
    test_mask = np.isin(reps_s, cfg.test_reps)
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        return None, le

    y_all = le.transform(labels_s)
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    X_train, X_val, X_test = per_subject_normalise(
        windows_s[train_mask], windows_s[val_mask], windows_s[test_mask]
    )

    # 1. SVM
    F_train = extract_td_features(X_train, cfg.zc_thresh, cfg.ssc_thresh)
    F_test = extract_td_features(X_test, cfg.zc_thresh, cfg.ssc_thresh)
    svm_acc, svm_model = evaluate_svm(F_train, y_train, F_test, y_test, cfg)
    yp_svm = svm_model.predict(F_test)
    svm_f1 = f1_score(y_test, yp_svm, average='macro', zero_division=0)

    # 2. LightGBM
    lgbm_acc, lgbm_model = evaluate_lgbm(F_train, y_train, F_test, y_test, cfg)
    yp_lgbm = lgbm_model.predict(F_test)
    lgbm_f1 = f1_score(y_test, yp_lgbm, average='macro', zero_division=0)

    # 3. TinyCNN-1D
    model_1d, hist_1d = train_tinycnn1d_subject(
        X_train, y_train, X_val, y_val, cfg, device
    )
    yp_1d = predict_tinycnn(model_1d, X_test, device)
    acc_1d = accuracy_score(y_test, yp_1d)
    f1_1d = f1_score(y_test, yp_1d, average='macro', zero_division=0)
    del model_1d

    # 4. TinyCNN-2D
    S_train = window_to_spectrogram(X_train, cfg)
    S_val = window_to_spectrogram(X_val, cfg)
    S_test = window_to_spectrogram(X_test, cfg)
    model_2d, hist_2d = train_tinycnn2d_subject(
        S_train, y_train, S_val, y_val, cfg, device
    )
    yp_2d = predict_tinycnn2d(model_2d, S_test, cfg, device)
    acc_2d = accuracy_score(y_test, yp_2d)
    f1_2d = f1_score(y_test, yp_2d, average='macro', zero_division=0)
    del model_2d, S_train, S_val, S_test

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    entry = {
        'subject_id':   sid,
        'n_train':      int(X_train.shape[0]),
        'n_val':        int(X_val.shape[0]),
        'n_test':       int(X_test.shape[0]),
        'svm_acc':      round(float(svm_acc), 4),
        'svm_f1':       round(float(svm_f1), 4),
        'lgbm_acc':     round(float(lgbm_acc), 4),
        'lgbm_f1':      round(float(lgbm_f1), 4),
        'cnn1d_acc':    round(float(acc_1d), 4),
        'cnn1d_f1':     round(float(f1_1d), 4),
        'cnn2d_acc':    round(float(acc_2d), 4),
        'cnn2d_f1':     round(float(f1_2d), 4),
        'cnn1d_epochs': len(hist_1d['ta']),
        'cnn2d_epochs': len(hist_2d['ta']),
    }

    return entry, le
