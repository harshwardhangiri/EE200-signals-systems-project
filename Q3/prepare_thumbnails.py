"""
Renders one constellation-style thumbnail PNG per indexed song into
`thumbnails/<song_label>.png`. Run AFTER `build_database.py`.

The thumbnails are what the Streamlit "Library" tab shows for each song.
"""
import os, glob, pickle, hashlib
import numpy as np
import matplotlib.pyplot as plt
from fingerprint import load_audio, fingerprint_audio

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus')
SONGS_DIR = 'database/songs'
OUT_DIR = 'thumbnails'

# colour palette for the cards (cycles)
PALETTE = ['#5fd3bc', '#c7d758', '#7aa6ff', '#ff8aa1', '#9fe07b',
           '#ffb86c', '#c39bff', '#74e0c8', '#f8b6ff', '#8de1ff']


def thumb_for(path, out_path, color):
    y, sr = load_audio(path)
    peaks, _, f, t, _ = fingerprint_audio(y, sr=sr)
    fig, ax = plt.subplots(figsize=(4, 2.4))
    fig.patch.set_facecolor('#0a1418')
    ax.set_facecolor('#0a1418')
    if peaks:
        # subsample peaks to keep the thumbnail clean
        peaks_arr = np.array(peaks)
        if len(peaks_arr) > 1500:
            idx = np.random.default_rng(0).choice(len(peaks_arr), 1500, replace=False)
            peaks_arr = peaks_arr[idx]
        pf = [f[p[0]] for p in peaks_arr]
        pt = [t[p[1]] for p in peaks_arr]
        ax.scatter(pt, pf, s=2, color=color, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#1c3236')
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=80, facecolor=fig.get_facecolor())
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([p for p in glob.glob(os.path.join(SONGS_DIR, '*'))
                    if p.lower().endswith(AUDIO_EXTS)])
    if not files:
        print(f"No songs found in {SONGS_DIR}")
        return
    print(f"Rendering {len(files)} thumbnails ...")
    for i, path in enumerate(files):
        label = os.path.splitext(os.path.basename(path))[0]
        # deterministic-but-varied colour based on label
        h = int(hashlib.md5(label.encode()).hexdigest(), 16)
        color = PALETTE[h % len(PALETTE)]
        out = os.path.join(OUT_DIR, label + '.png')
        if os.path.exists(out):
            print(f"  [{i+1}/{len(files)}] {label}  (cached)")
            continue
        try:
            thumb_for(path, out, color)
            print(f"  [{i+1}/{len(files)}] {label}")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {label}  FAILED: {e}")
    print(f"Done. Thumbnails in {OUT_DIR}/")


if __name__ == '__main__':
    main()