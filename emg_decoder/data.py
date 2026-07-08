from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io
import scipy.signal
import torch
from torch.utils.data import Dataset

from .config import Config


def find_subject_file(root: Path, sid: int, ex: int, cache=None) -> Optional[Path]:
    candidates = [
        f'S{sid}_E{ex}_A1.mat', f's{sid}_e{ex}_a1.mat',
        f'S{sid:02d}_E{ex}_A1.mat',
    ]
    for name in candidates:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    pool = cache if cache is not None else list(root.rglob('*.mat'))
    for f in pool:
        stem = f.stem.upper()
        if f'S{sid}' in stem and f'E{ex}' in stem:
            return f
    return None


def load_subject_mat(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        mat = scipy.io.loadmat(str(fp), variable_names=[
            'emg', 'stimulus', 'restimulus', 'repetition', 'rerepetition'])
    except NotImplementedError:
        import h5py
        mat = {}
        with h5py.File(str(fp), 'r') as hf:
            for k in ['emg', 'stimulus', 'restimulus', 'repetition', 'rerepetition']:
                if k in hf:
                    mat[k] = np.array(hf[k]).T
    emg = mat['emg'].astype(np.float32)
    if 'restimulus' in mat and mat['restimulus'].ndim > 0:
        lbl = mat['restimulus'].flatten().astype(np.uint8)
        reps = mat['rerepetition'].flatten().astype(np.uint8)
    else:
        lbl = mat['stimulus'].flatten().astype(np.uint8)
        reps = mat['repetition'].flatten().astype(np.uint8)
    return emg, lbl, reps


def bandpass_filter(emg: np.ndarray, cfg: Config) -> np.ndarray:
    nyq = cfg.sample_rate / 2.0
    b, a = scipy.signal.butter(
        cfg.filter_order,
        [cfg.bandpass_low / nyq, cfg.bandpass_high / nyq],
        btype='band'
    )
    return scipy.signal.filtfilt(b, a, emg, axis=0).astype(np.float32)


def segment_windows(emg: np.ndarray, labels: np.ndarray, reps: np.ndarray,
                    cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = emg.shape[0]
    wins, wlabs, wreps = [], [], []
    for start in range(0, n - cfg.window_size + 1, cfg.window_step):
        end = start + cfg.window_size
        chunk = labels[start:end]
        u, c = np.unique(chunk, return_counts=True)
        mi = np.argmax(c)
        if c[mi] / cfg.window_size < cfg.majority_thresh:
            continue
        maj = u[mi]
        if maj == 0:
            continue
        wins.append(emg[start:end, :].T)
        wlabs.append(maj)
        wreps.append(reps[start + cfg.window_size // 2])
    if not wins:
        return (np.empty((0, emg.shape[1], cfg.window_size), np.float32),
                np.empty(0, np.uint8), np.empty(0, np.uint8))
    return (np.stack(wins).astype(np.float32),
            np.array(wlabs, np.uint8),
            np.array(wreps, np.uint8))


def per_subject_normalise(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sigma = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((X_train - mu) / sigma,
            (X_val - mu) / sigma,
            (X_test - mu) / sigma)


def window_to_spectrogram(windows: np.ndarray, cfg: Config) -> np.ndarray:
    n_w, n_ch, _ = windows.shape
    channel_specs = []
    for ch in range(n_ch):
        _, _, Sxx = scipy.signal.spectrogram(
            windows[:, ch, :],
            fs=cfg.sample_rate,
            nperseg=cfg.spec_nperseg,
            noverlap=cfg.spec_noverlap,
            axis=-1,
            detrend=False
        )
        channel_specs.append(np.log1p(Sxx).astype(np.float32))
    return np.stack(channel_specs, axis=1)


def load_single_subject(root: Path, sid: int, cfg: Config,
                        file_cache=None) -> Optional[Tuple]:
    fp = find_subject_file(root, sid, cfg.exercise, cache=file_cache)
    if fp is None:
        return None
    try:
        emg, lbl, rep = load_subject_mat(fp)
        emg_f = bandpass_filter(emg, cfg)
        w, wl, wr = segment_windows(emg_f, lbl, rep, cfg)
        if len(wl) == 0:
            return None
        return w, wl, wr
    except Exception as exc:
        print(f'  S{sid} error: {exc}')
        return None


class EMGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class SpecAugmentDataset(Dataset):
    def __init__(self, specs: np.ndarray, labels: np.ndarray,
                 augment: bool = True, cfg: Optional[Config] = None):
        self.specs = torch.from_numpy(specs).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        spec = self.specs[i].clone()
        if self.augment and self.cfg is not None:
            _, n_freq, n_time = spec.shape
            if torch.rand(1).item() < self.cfg.aug_prob:
                t0 = torch.randint(0, max(1, n_time - 1), (1,)).item()
                dt = torch.randint(1, self.cfg.aug_time_mask + 1, (1,)).item()
                spec[:, :, t0:min(t0 + dt, n_time)] = 0.0
            if torch.rand(1).item() < self.cfg.aug_prob:
                f0 = torch.randint(0, max(1, n_freq - 1), (1,)).item()
                df = torch.randint(1, self.cfg.aug_freq_mask + 1, (1,)).item()
                spec[:, f0:min(f0 + df, n_freq), :] = 0.0
            if torch.rand(1).item() < self.cfg.aug_prob:
                scale = self.cfg.aug_amp_lo + torch.rand(1).item() * (self.cfg.aug_amp_hi - self.cfg.aug_amp_lo)
                spec = spec * scale
        return spec, self.labels[i]
