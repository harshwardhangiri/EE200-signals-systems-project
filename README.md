# EE200: Signals, Systems and Networks — Course Project

**Department of Electrical Engineering, IIT Kanpur · Summer 2026**
**Instructor:** Dr. Tushar Sandhan
**Worth:** 25% of course grade

---

## Overview

This repository contains my complete submission for the EE200 course project, covering three problems that apply 1D and 2D signal-processing techniques to real-world data — image restoration, biomedical signals, and audio identification.

| Problem | Title | Marks | Folder |
|---|---|---|---|
| **Q1A** | Frequency Forensics — *The Ghost Signal* | 3% | [`Q1_image_processing/`](./Q1_image_processing/) |
| **Q1B** | Digital Detective — *Missing Boundaries* | 2% | [`Q1_image_processing/`](./Q1_image_processing/) |
| **Q2** | The Midnight Episode — *Catching the Arrhythmia* | 7.5% | [`Q2_arrhythmia_detection/`](./Q2_arrhythmia_detection/) |
| **Q3A** | Sonic Signatures — *Magical Mystery Tune* | 7.5% | [`Q3_audio_fingerprinting/`](./Q3_audio_fingerprinting/) |
| **Q3B** | Signals to Softwares — *Zapptain America* | 5% | [`Q3_audio_fingerprinting/`](./Q3_audio_fingerprinting/) |

---

## Q1 — Image Processing

**Q1A: Frequency-Domain Image Recovery.** A grayscale image is corrupted by periodic interference. The corruption appears as localized spikes in the 2D Fourier magnitude spectrum, far from the low-frequency content of the underlying image. By transforming to the frequency domain, identifying and zeroing the interference peaks, and applying the inverse DFT, the hidden message is recovered.

**Q1B: Edge Detection with Sobel.** A 2D convolution with the Sobel kernels (horizontal and vertical derivative approximations) extracts the gradient magnitude of the image, revealing object boundaries. The notebook explores the effect of pre-smoothing on noise vs. edge sharpness.

**Deliverables:** `Q1A_frequency_forensics.ipynb`, `Q1B_edge_detection.ipynb`, `Q1_report.pdf`

---

## Q2 — Arrhythmia Detection from ECG

A 20-second Holter-monitor ECG (5000 samples at fs = 250 Hz) contains an arrhythmic episode partway through. The detector:

1. Models a healthy beat using a 200-sample template (one full P-QRS-T period).
2. Slides the template across the recording and computes the **normalized cross-correlation** ρ(m).
3. Flags the onset at the first beat whose ρ drops below threshold (0.5).

**Result:** Onset detected at **t = 9.6 s** (sample 2400, the 13th beat) where the QRS spike inverts and ρ ≈ −0.99. Four independent methods — beat-by-beat correlation, sample-by-sample sliding correlation, spectrogram analysis, and RR-interval statistics — all converge on the same onset time.

**Deliverables:** `Q2_EE200.ipynb`, `Q2_EE200_Report.pdf`

---

## Q3 — Shazam-style Audio Fingerprinting

A from-scratch implementation of the Shazam algorithm that identifies a song from a short query clip against a database of 50 indexed tracks.

**Pipeline:**
```
audio → Hann-windowed STFT → dB-magnitude spectrogram
     → local-max peaks (constellation)
     → pair anchor peaks with next 15 peaks → (f1, f2, Δt) hashes
     → look up hashes in DB → vote on time offset per song
     → song with the tallest offset-histogram spike wins
```

**Q3A: Algorithm and experiments** — spectrogram window-length trade-offs, single peaks vs. paired hashes, noise robustness, pitch-shift / time-stretch behavior.

**Q3B: Deployed Streamlit web app** with three tabs — Library (50 indexed songs), Identify (single-clip mode with intermediate visualizations), Batch (multiple clips → `results.csv`).

🔗 **Live app:** *(coming soon — link will be added after deployment)*

**Deliverables:** `fingerprint.py`, `app.py`, `build_database.py`, `Q3_report.pdf`, deployed app link

---

## Repository structure

```
EE200-signals-systems-project/
├── README.md                          ← you are here
├── .gitignore
│
├── Q1_image_processing/
│   ├── Q1A_frequency_forensics.ipynb
│   ├── Q1B_edge_detection.ipynb
│   ├── Q1_report.pdf
│   └── inputs/                        ← provided input images
│
├── Q2_arrhythmia_detection/
│   ├── Q2_EE200.ipynb
│   ├── Q2_EE200_Report.pdf
│   └── data/                          ← patient_ecg.npy, template.npy
│
└── Q3_audio_fingerprinting/
    ├── fingerprint.py                 ← core algorithm
    ├── app.py                         ← Streamlit app
    ├── build_database.py              ← one-time indexing script
    ├── generate_figures.py            ← report figures
    ├── requirements.txt               ← Python deps
    ├── packages.txt                   ← system deps (ffmpeg, libsndfile1)
    ├── database/
    │   ├── songs/                     ← drop the 50 provided songs here
    │   └── db.pkl                     ← created by build_database.py
    ├── samples/                       ← short demo clips
    ├── figures/                       ← report figures
    └── report/                        ← LaTeX source + compiled PDF
```

---

## How to run each part locally

**Q1 and Q2** — open the `.ipynb` files in Jupyter Notebook or VS Code and run all cells. Requires `numpy`, `scipy`, `matplotlib`. For Q2 also requires the provided `.npy` data files.

**Q3 — run the Streamlit app:**
```bash
cd Q3_audio_fingerprinting
pip install -r requirements.txt
# drop the 50 provided songs into database/songs/
python build_database.py          # creates database/db.pkl
streamlit run app.py              # opens the app on http://localhost:8501
```

---

## Tech stack

- **Python 3.11+**
- **NumPy, SciPy** — DFT, convolution, peak detection, spectrograms
- **Matplotlib** — all plots
- **Librosa** — audio loading and resampling (Q3)
- **Streamlit** — web UI for the audio identifier (Q3B)
- **LaTeX** — report typesetting

---

## Author

**Harshwardhan Giri** — Department of Electrical Engineering, IIT Kanpur

*Built as the final project for EE200: Signals, Systems and Networks, Summer 2026.*
