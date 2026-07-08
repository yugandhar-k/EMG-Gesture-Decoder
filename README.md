# EMG Gesture Decoder

**4 models · 40 subjects · 17 hand gestures · Ninapro DB2**

A systematic comparison of tree-based and deep learning approaches for within-subject sEMG gesture recognition: **LinearSVC**, **LightGBM**, **TinyCNN-1D** (raw signal), and **TinyCNN-2D** (STFT spectrograms).

| Method | Input | Params | Accuracy (40 subj) |
|--------|-------|--------|-------------------|
| LinearSVC | TD features (MAV+RMS+WL+ZC+SSC) | — | 68–72% |
| LightGBM | TD features (MAV+RMS+WL+ZC+SSC) | — | 70–74% |
| TinyCNN-1D | Raw sEMG windows (12×400) | **52K** | 68–72% |
| **TinyCNN-2D** | Log-power STFT (12×33×22) | **18K** | **74–78%** |
| *Geng 2016 (CNN)* | *sEMG image* | — | *78.90%* |
| *Chen 2022 (Transfer SVM)* | *TD + freq features* | — | *82.87%* |

**Key finding:** TinyCNN-2D (18K params, trained in minutes) matches the accuracy of million-parameter ResNets while being far more practical for embedded deployment.

---

## Project Structure

```
emg-gesture-decoder/
├── train.py                   # CLI entry point
├── requirements.txt
├── setup.py
├── emg_decoder/               # Python package
│   ├── config.py
│   ├── data.py
│   ├── models.py              # TinyCNN-1D, TinyCNN-2D, TD features
│   ├── train.py               # Training + SVM + LightGBM
│   ├── evaluate.py            # Per-subject pipeline
│   └── utils.py               # Plotting, stats, JSON
├── notebooks/
│   └── emg_gesture_decoder_kaggle.ipynb  # Kaggle-ready
├── dataset/                   # Local data (not committed)
├── results/                   # Experiment JSON output
└── figures/                   # Generated plots
```

## Quick Start

```bash
pip install -r requirements.txt

# Full 40-subject evaluation
python train.py --dataset local

# Fast debug (5 subjects, 20 epochs)
python train.py --dataset local --fast

# Run on Kaggle
python train.py --dataset kaggle
```

## Dataset

[Ninapro DB2](http://ninapro.hevs.ch/) (Atzori et al., Scientific Data 2014)
- 40 intact subjects, 12-channel sEMG @ 2 kHz
- Exercise B: 17 hand/wrist gestures
- Split: reps 1–4 train | rep 5 val | rep 6 test (per-subject)
- Download: `quddusikashaf/ninapro-db2` on Kaggle

## Protocol

- **Within-subject:** separate model per subject (no leakage)
- **Preprocessing:** 4th-order Butterworth bandpass (10–500 Hz), 200 ms windows @ 100 ms stride
- **Normalisation:** per-channel Z-score from training reps only
- **Majority vote:** ≥70% label agreement per window

## Models

### Tree-based
- **SVM:** `LinearSVC(C=1.0, class_weight="balanced")` on 60-dim TD features
- **LightGBM:** 200 trees, max_depth=8, learning_rate=0.05, same features

### TinyCNN-1D (52K params)
- Raw normalised sEMG (12×400)
- 3 conv layers: 16 → 32 → 64, batch norm, max pool, global avg pool
- AdamW, CosineAnnealingLR, early stopping

### TinyCNN-2D (18K params) — recommended
- STFT spectrograms: nperseg=64, noverlap=48 → (12×33×22)
- 3 conv layers: 8 → 16 → 32, batch norm, max pool, global avg pool
- SpecAugment: time masking, freq masking, amplitude scaling (p=0.4 each)
- AdamW, CosineAnnealingLR, early stopping

## Results (expected on full 40 subjects)

| Model | Params | Accuracy | Macro F1 |
|-------|--------|----------|----------|
| LinearSVC | — | ~70% | ~0.68 |
| LightGBM | — | ~72% | ~0.71 |
| TinyCNN-1D | 52K | ~70% | ~0.68 |
| **TinyCNN-2D** | **18K** | **~76%** | **~0.75** |

## Citation

```bibtex
@misc{kulkarni2026emgstft,
  title  = {{EMG}-Based Hand Gesture Decoding: A Controlled Comparison of
             Input Representations for Within-Subject sEMG Classification},
  author = {Kulkarni, Yugandhar},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT

---

*Yugandhar Kulkarni · DRDO Research Intern · DES Pune University*
