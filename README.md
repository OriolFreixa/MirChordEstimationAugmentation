# Robust Automatic Chord Estimation Through Realistic Audio Augmentation

**Rafael Moncayo, Oriol Freixa, Manuel Castillo Obregon**  
Music Technology Group — Universitat Pompeu Fabra, Barcelona

---

## Overview

Automatic chord estimation (ACE) systems are typically trained on clean, studio-quality recordings. In practice, however, music is frequently captured through consumer microphones, reproduced by low-quality speakers, recorded in reverberant rooms, or degraded by transmission and playback artifacts. This mismatch between benchmark conditions and real-world usage limits the reliability of ACE models in practical scenarios.

This project investigates whether **training-time acoustic augmentation** can improve the robustness of a Conformer-based ACE model to such conditions. The goal is not to propose a new chord model, but to study whether robustness can be improved solely by changing the training data distribution — making the model agnostic to acoustic information that does not belong to the harmonic content of the chords being predicted.

---

## Method

### Base Model

We build on the Conformer-based ACE framework introduced by Poltronieri, Serra, and Rocamora [1]. The base Conformer backbone is adopted without modification. Audio is segmented into 10-second excerpts at 22.05 kHz and transformed into a Constant-Q representation with 144 frequency bins (24 bins/octave over 6 octaves). The model is trained on the **majmin vocabulary** (26 chord classes). Chord annotations are stored in JAMS format and converted to majmin targets following standard ACE conventions.

### Datasets

| Split | Dataset | Description |
|---|---|---|
| Train | Billboard | Time-aligned chord annotations for popular songs from the Billboard Hot 100 charts |
| Test | MARL | Manually annotated pop songs from USPop and RWC-Pop collections |
| Test | Isophonics | Expert harmonic annotations for The Beatles, Queen, and others |

The train/test split is performed at the **track level** to avoid song-level leakage between partitions.

### Augmentation Pipeline

The pipeline simulates realistic acoustic degradations while preserving the underlying harmonic content of each track. Four augmentation types are applied, all using a **50% wet / 50% dry ratio** to ensure the original harmonic content remains audible and intact.

**1. Convolutional Reverb**  
Convolution of the source audio with real impulse responses from the [OpenAIR library](https://www.openairlib.net/), covering 34 IR files representing diverse acoustic environments (concert halls, churches, small rooms). Simulates the room acoustics of real-world recording spaces independently of the original recording conditions.

**2. Real-World Ambient Noise**  
Background noise recordings sourced from [FreeSound](https://freesound.org/) across three acoustic categories: café/interior ambience, street/urban traffic, and outdoor wind (3 files per category, 9 total). Each noise clip is tiled or trimmed to match the source audio duration, then RMS-normalized before mixing.

**3. Random Equalization**  
Three bandpass filters (100–500 Hz, 500–2000 Hz, 2000–8000 Hz) are applied independently, each with a gain uniformly sampled from [−6, +6] dB. Simulates the frequency response variation of different recording or playback devices.

**4. Dynamic Compression**  
A feed-forward compressor (threshold −20 dBFS, ratio 4:1, attack 5 ms, release 100 ms) is applied to simulate recordings processed with varying degrees of dynamic range reduction.

The pipeline samples **four augmentation chains per source file**, increasing acoustic diversity while keeping the dataset size manageable. The final augmented training set is approximately **5× the size** of the clean baseline.

---

## Results

### Training and Validation Behavior

| | Baseline | Augmented |
|---|---|---|
| Train accuracy | 0.9154 | 0.8804 |
| Val. accuracy | 0.6716 | **0.7910** |
| Train loss | 0.2614 | 0.3578 |
| Val. loss | 1.4711 | **0.6934** |
| Train–val gap | 24.3 pp | **8.9 pp** |

The train–validation accuracy gap is reduced from **24.3 to 8.9 percentage points**, indicating that augmentation acts as an effective regularizer against acoustic overfitting.

### Cross-Dataset Evaluation (MARL + Isophonics)

| Metric | Baseline | Augmented | Gain | Wilcoxon |
|---|---|---|---|---|
| root | 0.8387 | **0.8495** | +0.0108 | p < 0.0001 ✓ |
| thirds | 0.7848 | **0.7951** | +0.0103 | p < 0.0001 ✓ |
| triads | 0.6999 | **0.7140** | +0.0141 | p < 0.0001 ✓ |
| tetrads | 0.6704 | **0.6836** | +0.0132 | p < 0.0001 ✓ |
| majmin | 0.6566 | **0.6664** | +0.0099 | p = 0.0002 ✓ |
| mirex | 0.7809 | 0.7803 | −0.0006 | p = 0.170 ✗ |

Statistical significance assessed via paired Wilcoxon signed-rank test on per-track scores. The MIREX metric shows no significant difference, consistent with a minor effect of reverberation on temporal boundary precision rather than on harmonic label quality.

---

## Repository Structure

```
MirChordEstimationAugmentation/
│
├── data_augmentation.py            ← Unified augmentation functions
│                                     (apply_reverb, add_real_noise,
│                                      randomly_eq, apply_compression)
│
├── data_augmentation_real.ipynb    ← Full augmentation pipeline notebook
│                                     (FreeSound API integration,
│                                      augmentation execution,
│                                      JAMS copying, verification)
│
├── ace_safe_trainer.py             ← Leakage-safe training wrapper
│                                     (track-level split, DataModule,
│                                      Lightning training loop)
│
├── ir_files/                       ← Impulse response files (.wav / .flac)
│                                     from the OpenAIR library
│
├── checkpoints/
│   ├── baseline/last.ckpt          ← Baseline model checkpoint
│   └── augmented/last.ckpt         ← Augmented model checkpoint
│
└── ACE/                            ← Conformer-based ACE model (Poltronieri et al.)
    ├── trainer.gin
    ├── chords_vocab.joblib
    └── models/
        ├── conformer.py
        └── conformer_decomposed.py
```

> **Note:** Audio datasets (Billboard, MARL, Isophonics) and preprocessed `.pt` files are not included in this repository due to copyright restrictions. Please refer to each dataset's official distribution for access.

---

## Installation

```bash
git clone https://github.com/<your-repo>/MirChordEstimationAugmentation
cd MirChordEstimationAugmentation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install git+https://github.com/MTG/freesound-python.git
pip install librosa soundfile scipy numpy jams mir_eval \
            torch lightning gin-config scikit-learn wandb
```

---

## Usage

### 1. Generate augmented data

Open and run `data_augmentation_real.ipynb` step by step.

Set your FreeSound API key as an environment variable — **never hardcode credentials in the notebook or commit them to git**:

```bash
export FREESOUND_API_KEY="your_key_here"
```

To obtain your API key: log in at [freesound.org](https://freesound.org) → Settings → API credentials → Apply for API key.

### 2. Preprocess augmented data

Run the ACE preprocessing script from the Conformer-based ACE repository to generate the `.pt` files from the augmented audio and JAMS annotations.

### 3. Train

```bash
python ace_safe_trainer.py \
  --model conformer \
  --name Augmented_ACE \
  --data_path /path/to/pt/files \
  --vocab_path ACE/chords_vocab.joblib \
  --accelerator gpu \
  --max_epochs 10 \
  --checkpoint_dir checkpoints/augmented
```

### 4. Evaluate

Use the evaluation cells in `data_augmentation_real.ipynb` with the saved `last.ckpt` checkpoints and the MARL + Isophonics test split. Metrics are computed via `mir_eval.chord.evaluate` with Wilcoxon signed-rank significance testing.

---

## Dependencies

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | 2.x | Model inference and training |
| `lightning` | 2.x | Training loop |
| `librosa` | 0.11.0 | Audio loading and CQT |
| `soundfile` | 0.13.1 | MP3 writing |
| `scipy` | 1.17.1 | Convolution, filters, statistics |
| `mir_eval` | 0.8.2 | Chord evaluation metrics |
| `jams` | 0.3.5 | Annotation file handling |
| `freesound-python` | 1.1 (MTG fork) | FreeSound API client |
| `scikit-learn` | 1.6.1 | LabelEncoder for chord vocabulary |
| `gin-config` | — | Model configuration |
| `wandb` | — | Experiment tracking |

---

## References

[1] A. Poltronieri, X. Serra, and M. Rocamora, "From Discord to Harmony: Decomposed Consonance-based Training for Improved Audio Chord Estimation," in *Proc. ISMIR 2025*, pp. 492–500. arXiv:2509.01588.

[2] A. Gulati, J. Qin, C. Chiu et al., "Conformer: Convolution-augmented transformer for speech recognition," in *Interspeech 2020*, pp. 5036–5040.

[3] R. Mignot and G. Peeters, "An analysis of the effect of data augmentation methods: Experiments for a musical genre classification task," *Trans. ISMIR*, vol. 2, no. 1, pp. 97–110, 2019.

[4] J. Pauwels, K. O'Hanlon, E. Gómez, and M. B. Sandler, "20 years of automatic chord recognition from audio," in *Proc. ISMIR 2019*, pp. 54–63.

[5] F. Font, G. Roma, and X. Serra, "Freesound technical demo," in *Proc. ACM Multimedia 2013*, pp. 411–412.

[6] University of York, "OpenAIR: Open Acoustic Impulse Response Library," https://www.openairlib.net/, 2026.

---

## Acknowledgements

We thank Martín Rocamora and Andrea Poltronieri for their guidance and feedback throughout this project. We also acknowledge the support of Milo Beuzebal in the assessment with the datasets used.
