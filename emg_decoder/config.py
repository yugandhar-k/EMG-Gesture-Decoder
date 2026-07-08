from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    dataset_root: Path = Path('/kaggle/input/datasets/quddusikashaf/ninapro-db2')
    results_dir:  Path = Path('/kaggle/working/results')
    figures_dir:  Path = Path('/kaggle/working/figures')

    sample_rate:    int   = 2000
    n_emg_channels: int   = 12
    bandpass_low:   float = 10.0
    bandpass_high:  float = 500.0
    filter_order:   int   = 4

    window_ms:       int   = 200
    step_ms:         int   = 100
    majority_thresh: float = 0.70
    zc_thresh:       float = 0.01
    ssc_thresh:      float = 0.01

    @property
    def window_size(self) -> int:
        return int(self.window_ms * self.sample_rate / 1000)

    @property
    def window_step(self) -> int:
        return int(self.step_ms * self.sample_rate / 1000)

    n_subjects:  int = 40
    exercise:    int = 1
    n_classes:   int = 17
    gesture_names: List[str] = field(default_factory=lambda: [
        'Thumb Flex', 'Index Flex', 'Middle Flex', 'Ring Flex', 'Little Flex',
        'Thumb Abd',  'Wrist Flex', 'Wrist Ext',  'Wrist Pro',  'Wrist Sup',
        'Wrist Uln',  'Wrist Rad',  'Hand Close', 'Hand Open',  'Pinch',
        'Lat. Pinch', 'Fine Pinch',
    ])

    train_reps: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    val_reps:   List[int] = field(default_factory=lambda: [5])
    test_reps:  List[int] = field(default_factory=lambda: [6])

    # TinyCNN-1D
    batch_size_1d: int   = 128
    n_epochs_1d:   int   = 100
    lr_1d:         float = 1e-3
    patience_1d:   int   = 20

    # STFT spectrogram
    spec_nperseg:  int = 64
    spec_noverlap: int = 48

    @property
    def spec_n_freq(self) -> int:
        return self.spec_nperseg // 2 + 1

    @property
    def spec_n_time(self) -> int:
        return (self.window_size - self.spec_noverlap) // (self.spec_nperseg - self.spec_noverlap)

    # TinyCNN-2D
    batch_size_2d: int   = 64
    n_epochs_2d:   int   = 100
    lr_2d:         float = 1e-3
    patience_2d:   int   = 20

    # SpecAugment
    aug_prob:      float = 0.4
    aug_time_mask: int   = 4
    aug_freq_mask: int   = 5
    aug_amp_lo:    float = 0.85
    aug_amp_hi:    float = 1.15

    # SVM
    svm_c:            float = 1.0
    svm_class_weight: str   = 'balanced'
    svm_max_iter:     int   = 10000

    # LightGBM
    lgbm_n_estimators: int    = 200
    lgbm_max_depth:    int    = 8
    lgbm_lr:           float  = 0.05
    lgbm_num_leaves:   int    = 31
    lgbm_subsample:    float  = 0.8
    lgbm_colsample:    float  = 0.8

    fast_mode:       bool  = False
    fast_n_subjects: int   = 5
    fast_n_epochs:   int   = 20

    random_seed: int = 42

    @property
    def n_subjects_run(self) -> int:
        return self.fast_n_subjects if self.fast_mode else self.n_subjects

    @property
    def n_epochs_run(self) -> int:
        return self.fast_n_epochs if self.fast_mode else self.n_epochs_1d

    @classmethod
    def local(cls) -> 'Config':
        return cls(
            dataset_root=Path(r'F:\Projects\EMG-Gesture-Decoder\dataset'),
            results_dir=Path(r'F:\Projects\EMG-Gesture-Decoder\results'),
            figures_dir=Path(r'F:\Projects\EMG-Gesture-Decoder\figures'),
        )

    @classmethod
    def kaggle(cls) -> 'Config':
        return cls()

    def print_config(self):
        print(f'  Protocol   : WITHIN-SUBJECT (separate model per subject)')
        print(f'  Subjects   : {self.n_subjects_run} of {self.n_subjects}')
        print(f'  Window     : {self.window_ms}ms = {self.window_size} samples | Step : {self.step_ms}ms')
        print(f'  Split      : train {self.train_reps} | val {self.val_reps} | test {self.test_reps}')
        print(f'  Spectrogram: nperseg={self.spec_nperseg}, noverlap={self.spec_noverlap}')
        print(f'               shape per window = (12, {self.spec_n_freq}, {self.spec_n_time})')
        print(f'  Models     : SVM | LightGBM | TinyCNN-1D | TinyCNN-2D (STFT)')
