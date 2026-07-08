import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import Config
from .models import TinyCNN1D, TinyCNN2D, extract_td_features
from .data import EMGDataset, SpecAugmentDataset


def train_tinycnn1d_subject(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             cfg: Config, device: torch.device):
    tr_ld = DataLoader(EMGDataset(X_train, y_train),
                       batch_size=cfg.batch_size_1d, shuffle=True, num_workers=0)
    vl_ld = DataLoader(EMGDataset(X_val, y_val),
                       batch_size=256, shuffle=False, num_workers=0)
    model = TinyCNN1D(n_classes=cfg.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr_1d)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.n_epochs_run)
    history = {'ta': [], 'va': []}
    best_val = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(cfg.n_epochs_run):
        model.train()
        correct, total = 0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()
        train_acc = correct / total
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in vl_ld:
                xb, yb = xb.to(device), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / vt
        history['ta'].append(train_acc)
        history['va'].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience_1d:
                break

    model.load_state_dict(best_state)
    return model, history


def train_tinycnn2d_subject(S_train: np.ndarray, y_train: np.ndarray,
                             S_val: np.ndarray, y_val: np.ndarray,
                             cfg: Config, device: torch.device):
    tr_ds = SpecAugmentDataset(S_train, y_train, augment=True, cfg=cfg)
    vl_ds = SpecAugmentDataset(S_val, y_val, augment=False, cfg=cfg)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size_2d, shuffle=True, num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TinyCNN2D(n_classes=cfg.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr_2d)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.n_epochs_run)
    history = {'ta': [], 'va': []}
    best_val = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(cfg.n_epochs_run):
        model.train()
        correct, total = 0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()
        train_acc = correct / total
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in vl_ld:
                xb, yb = xb.to(device), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / vt
        history['ta'].append(train_acc)
        history['va'].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience_2d:
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_tinycnn(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(EMGDataset(X, np.zeros(len(X), np.int64)),
                        batch_size=256, shuffle=False)
    return np.concatenate([model(xb.to(device)).argmax(1).cpu().numpy() for xb, _ in loader])


@torch.no_grad()
def predict_tinycnn2d(model: nn.Module, S: np.ndarray, cfg: Config,
                       device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        SpecAugmentDataset(S, np.zeros(len(S), np.int64), augment=False, cfg=cfg),
        batch_size=256, shuffle=False
    )
    return np.concatenate([model(xb.to(device)).argmax(1).cpu().numpy() for xb, _ in loader])


def evaluate_svm(F_train: np.ndarray, y_train: np.ndarray,
                 F_test: np.ndarray, y_test: np.ndarray,
                 cfg: Config) -> float:
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(C=cfg.svm_c, class_weight=cfg.svm_class_weight,
                          max_iter=cfg.svm_max_iter, random_state=cfg.random_seed,
                          dual='auto'))
    ])
    svm.fit(F_train, y_train)
    return svm.score(F_test, y_test), svm


def evaluate_lgbm(F_train: np.ndarray, y_train: np.ndarray,
                  F_test: np.ndarray, y_test: np.ndarray,
                  cfg: Config) -> float:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=cfg.lgbm_n_estimators,
        max_depth=cfg.lgbm_max_depth,
        learning_rate=cfg.lgbm_lr,
        num_leaves=cfg.lgbm_num_leaves,
        subsample=cfg.lgbm_subsample,
        colsample_bytree=cfg.lgbm_colsample,
        random_state=cfg.random_seed,
        verbose=-1,
        class_weight='balanced',
    )
    model.fit(F_train, y_train)
    return model.score(F_test, y_test), model
