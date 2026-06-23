# EE200: Signals, Systems and Networks — Course Project

**Department of Electrical Engineering, IIT Kanpur · Summer 2026**
**Instructor:** Dr. Tushar Sandhan
**Worth:** 25% of course grade

---

## 🚀 Quick Links

- **🎵 Live Audio Fingerprinting App (Q3B):** [ee200-shazam-harsh.streamlit.app](https://ee200-shazam-harsh.streamlit.app)
- **📦 Source repository:** [github.com/harshwardhangiri/EE200-signals-systems-project](https://github.com/harshwardhangiri/EE200-signals-systems-project)

---

## Overview

This repository contains my complete submission for the EE200 course project, covering three problems that apply 1D and 2D signal-processing techniques to real-world data — image restoration, biomedical signals, and audio identification.

| Problem | Title | Marks | Folder |
|---|---|---|---|
| **Q1** | Frequency Forensics + Digital Detective — *Ghost Signal & Missing Boundaries* | 5% | [`Q1/`](./Q1/) |
| **Q2** | The Midnight Episode — *Catching the Arrhythmia* | 7.5% | [`Q2/`](./Q2/) |
| **Q3A** | Sonic Signatures — *Magical Mystery Tune* | 7.5% | [`Q3/`](./Q3/) |
| **Q3B** | Signals to Softwares — *Zapptain America* | 5% | [`Q3/`](./Q3/) + [live app](https://ee200-shazam-harsh.streamlit.app) |

---

## Q1 — Image Processing (Q1A + Q1B together)

Both image-processing problems are solved in a single Jupyter notebook that walks through the full pipeline end-to-end.

**Q1A — Frequency-Domain Image Recovery (3%).** A grayscale image is corrupted by periodic interference. The corruption appears as localized spikes in the 2D Fourier magnitude spectrum, far from the low-frequency content of the underlying image. By transforming to the frequency domain, identifying and zeroing the interference peaks, and applying the inverse DFT, the hidden message is recovered.

**Q1B — Edge Detection with Sobel (2%).** A 2D convolution with the Sobel kernels (horizontal and vertical derivative approximations) extracts the gradient magnitude of the image, revealing object boundaries. The notebook explores the effect of pre-smoothing on noise vs. edge sharpness.

**Deliverables:** `Q1_EE200_Solution.ipynb` (covers both A and B), `Q1_EE200_Report.pdf`, `input/` (provided input images)

---

## Q2 — Arrhythmia Detection from ECG

A 20-second Holter-monitor ECG (5000 samples at fs = 250 Hz) contains an arrhythmic episode partway through. The detector:

1. Models a healthy beat using a 200-sample template (one full P-QRS-T period).
2. Slides the template across the recording and computes the **normalized cross-correlation** ρ(m).
3. Flags the onset at the first beat whose ρ drops below threshold (0.5).

**Result:** Onset detected at **t = 9.6 s** (sample 2400, the 13th beat) where the QRS spike inverts and ρ ≈ −0.99. Four independent methods — beat-by-beat correlation, sample-by-sample sliding correlation, spectrogram analysis, and RR-interval statistics — all converge on the same onset time.

**Deliverables:** `Q2_EE200.ipynb`, `Q2_EE200_Report.pdf`, `input/` (patient_ecg.npy, template.npy)

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

**Verified on a 30-second query clip:** the system correctly identified "Two Of Us" with a cluster score of 11528 versus only 11 for the runner-up — a **1048× margin**, far above any noise floor. End-to-end identification takes under 100 ms after audio decoding.

**Q3A: Algorithm and experiments** — spectrogram window-length trade-offs, single peaks vs. paired hashes, noise robustness, pitch-shift / time-stretch behavior. See [`Q3/report/`](./Q3/report/) for the full writeup.

**Q3B: Live deployed app** at [ee200-shazam-harsh.streamlit.app](https://ee200-shazam-harsh.streamlit.app) — three tabs:
- **Library** — browse all 50 indexed songs with their constellation fingerprints
- **Identify** — upload a clip, see the spectrogram, constellation, offset histogram, and matched song
- **Batch** — process multiple clips, download `results.csv` with predictions

**Deliverables:** `fingerprint.py`, `app.py`, `build_database.py`, `Q3_report.pdf`, deployed app

---

## Repository structure

```
EE200-signals-systems-project/
├── README.md                          ← you are here
├── .gitignore
├── requirements.txt                   ← root-level deps (for Streamlit Cloud)
├── packages.txt                       ← root-level system deps
│
├── Q1/
│   ├── Q1_EE200_Solution.ipynb        ← Q1A + Q1B in one notebook
│   ├── Q1_EE200_Report.pdf
│   └── input/                         ← provided input images
│
├── Q2/
│   ├── Q2_EE200.ipynb
│   ├── Q2_EE200_Report.pdf
│   └── input/                         ← patient_ecg.npy, template.npy
│
└── Q3/
    ├── fingerprint.py                 ← core algorithm
    ├── app.py                         ← Streamlit app (deployed at the URL above)
    ├── build_database.py              ← one-time indexing script
    ├── generate_figures.py            ← report figures
    ├── requirements.txt               ← Python deps
    ├── packages.txt                   ← system deps (ffmpeg, libsndfile1)
    ├── database/
    │   ├── songs/                     ← (50 provided songs go here locally; not committed)
    │   └── db.pkl                     ← pre-built fingerprint index
    ├── samples/                       ← short demo clips for the app
    ├── figures/                       ← report figures
    ├── thumbnails/                    ← constellation thumbnails for the Library tab
    └── report/                        ← LaTeX source + compiled PDF
```

---

## How to run each part locally

**Q1** — open `Q1/Q1_EE200_Solution.ipynb` in Jupyter or VS Code and run all cells. Requires `numpy`, `scipy`, `matplotlib`, and the input images in `Q1/input/`.

**Q2** — open `Q2/Q2_EE200.ipynb` in Jupyter or VS Code and run all cells. Requires `numpy`, `scipy`, `matplotlib`, and the data files in `Q2/input/`.

**Q3 — run the Streamlit app:**
```bash
cd Q3
pip install -r requirements.txt
# drop the 50 provided songs into database/songs/
python build_database.py          # creates database/db.pkl
streamlit run app.py              # opens the app on http://localhost:8501
```

Or just visit the **live deployed version** — no setup required:
👉 **[ee200-shazam-harsh.streamlit.app](https://ee200-shazam-harsh.streamlit.app)**

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

**Harshwardhan Giri Goswami** — Department of Aerospace Engineering, IIT Kanpur

*Built as the final project for EE200: Signals, Systems and Networks, Summer 2026.*
