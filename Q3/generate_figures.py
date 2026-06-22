"""
Generate every figure that goes into the Q3A report.

If `database/songs/` has the real provided songs in it, this uses them.
Otherwise it falls back to a few synthetic test signals so the report still
compiles - you can rerun once the songs are in place to refresh.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa

from fingerprint import (load_audio, compute_spectrogram, find_peaks,
                         peaks_to_hashes, fingerprint_audio, FingerprintDB,
                         add_noise, pitch_shift, time_stretch,
                         SR, N_FFT, HOP)

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150,
                     'font.size': 9})


# ---------- pick a "reference song" ----------
def find_reference_song():
    files = sorted(glob.glob('database/songs/*'))
    audio_exts = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus')
    files = [f for f in files if f.lower().endswith(audio_exts)]
    if files:
        return files[0], os.path.splitext(os.path.basename(files[0]))[0]
    # synthetic fallback - a 12-second clip with a few notes + a chirp
    print("[no songs found - using synthetic test signal]")
    sr = SR
    dur = 12.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = np.zeros_like(t)
    # 3 short tones
    notes = [(0.0, 1.2, 440), (1.5, 1.0, 660), (3.0, 1.0, 523),
             (4.5, 1.0, 784), (6.0, 2.0, 392)]
    for start, length, freq in notes:
        mask = (t >= start) & (t < start + length)
        env = np.exp(-3 * (t[mask] - start))
        y[mask] += env * np.sin(2 * np.pi * freq * t[mask])
    # a chirp at the end
    mask = t >= 8.0
    tt = t[mask] - 8.0
    y[mask] += 0.5 * np.sin(2 * np.pi * (200 + 60 * tt) * tt)
    # save it so other steps can load it
    import soundfile as sf
    path = 'figures/_synthetic_demo.wav'
    sf.write(path, y.astype(np.float32), sr)
    return path, 'synthetic_demo'


SONG_PATH, SONG_LABEL = find_reference_song()
print(f"Reference song: {SONG_LABEL}  ({SONG_PATH})")
y_full, sr = load_audio(SONG_PATH)


# ============================================================
# FIGURE 1 - DFT of whole song (motivation for spectrograms)
# ============================================================
print("[fig1] whole-song DFT magnitude ...")
Y = np.fft.rfft(y_full)
freqs = np.fft.rfftfreq(len(y_full), 1 / sr)
mag = 20 * np.log10(np.abs(Y) + 1e-8)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(freqs, mag, lw=0.4, color='#225')
ax.set_xlim(0, sr / 2)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('|X(f)| (dB)')
ax.set_title(f'DFT magnitude of the entire song "{SONG_LABEL}"\n'
             '(every frequency that ever appears is here, but timing is lost)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/01_whole_song_dft.png')
plt.close()


# ============================================================
# FIGURE 2 - spectrogram of the song (with chosen window)
# ============================================================
print("[fig2] spectrogram of the song ...")
f, t, S_db = compute_spectrogram(y_full, sr=sr)
fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(S_db, origin='lower', aspect='auto',
               extent=[t[0], t[-1], f[0], f[-1]], cmap='magma',
               vmin=np.median(S_db), vmax=S_db.max())
plt.colorbar(im, ax=ax, label='magnitude (dB)')
ax.set_xlabel('time (s)'); ax.set_ylabel('frequency (Hz)')
ax.set_title(f'Spectrogram of "{SONG_LABEL}"  '
             f'(n_fft={N_FFT}, hop={HOP}, sr={sr} Hz)')
plt.tight_layout()
plt.savefig(f'{OUT}/02_spectrogram.png')
plt.close()


# ============================================================
# FIGURE 3 - short window vs long window
# ============================================================
print("[fig3] short window vs long window ...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, n_fft, hop, name in [
    (axes[0],  512,  128, 'short window (n=512)  -> good time, blurry freq'),
    (axes[1], 8192, 2048, 'long window (n=8192) -> sharp freq, blurry time')]:
    f2, t2, S2 = compute_spectrogram(y_full, sr=sr, n_fft=n_fft, hop=hop)
    ax.imshow(S2, origin='lower', aspect='auto',
              extent=[t2[0], t2[-1], f2[0], f2[-1]], cmap='magma',
              vmin=np.median(S2), vmax=S2.max())
    ax.set_title(name)
    ax.set_xlabel('time (s)'); ax.set_ylabel('freq (Hz)')
plt.tight_layout()
plt.savefig(f'{OUT}/03_window_tradeoff.png')
plt.close()


# ============================================================
# FIGURE 4 - constellation map (peaks on spectrogram)
# ============================================================
print("[fig4] constellation map ...")
peaks, hashes, f, t, S_db = fingerprint_audio(y_full, sr=sr)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.imshow(S_db, origin='lower', aspect='auto',
          extent=[t[0], t[-1], f[0], f[-1]], cmap='magma',
          vmin=np.median(S_db), vmax=S_db.max(), alpha=0.85)
if peaks:
    pf = [f[p[0]] for p in peaks]
    pt = [t[p[1]] for p in peaks]
    ax.scatter(pt, pf, s=12, facecolors='none', edgecolors='cyan',
               linewidths=0.7)
ax.set_xlabel('time (s)'); ax.set_ylabel('freq (Hz)')
ax.set_title(f'Constellation map: {len(peaks)} peaks, '
             f'{len(hashes)} (f1,f2,dt) hashes')
plt.tight_layout()
plt.savefig(f'{OUT}/04_constellation.png')
plt.close()


# ============================================================
# Build a tiny DB out of whatever songs we have, for the
# offset-histogram + noise/pitch/stretch experiments.
# ============================================================
print("[mini-db] building a small index for the experiments ...")
all_files = sorted(glob.glob('database/songs/*'))
audio_exts = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus')
all_files = [a for a in all_files if a.lower().endswith(audio_exts)]
if not all_files:
    # use 3 synthetic "songs" of different note sequences
    print("  [no real songs; synthesizing 3 dummy songs]")
    import soundfile as sf
    rng = np.random.default_rng(0)
    dummy_dir = 'figures/_synth_songs'
    os.makedirs(dummy_dir, exist_ok=True)
    for k in range(3):
        sr_d = SR
        dur = 12.0
        tt = np.linspace(0, dur, int(sr_d * dur), endpoint=False)
        y = np.zeros_like(tt)
        freqs_k = rng.uniform(200, 1500, size=8)
        starts = np.linspace(0, dur - 1.2, 8)
        for s, fr in zip(starts, freqs_k):
            mask = (tt >= s) & (tt < s + 1.0)
            y[mask] += np.exp(-3 * (tt[mask] - s)) * np.sin(2 * np.pi * fr * tt[mask])
        path = f'{dummy_dir}/synth_song_{k:02d}.wav'
        sf.write(path, y.astype(np.float32), sr_d)
        all_files.append(path)
    SONG_PATH = all_files[0]
    SONG_LABEL = 'synth_song_00'
    y_full, sr = load_audio(SONG_PATH)

mini_db = FingerprintDB()
for path in all_files:
    label = os.path.splitext(os.path.basename(path))[0]
    yy, _ = load_audio(path)
    _, hh, _, _, _ = fingerprint_audio(yy, sr=sr)
    mini_db.add_song(label, hh)
print(f"  mini-DB: {len(mini_db)} songs, {mini_db.num_hashes():,} hashes")


# helper: take a 6-second clip from the middle of the song
def clip(y, sr, length=6.0, start_frac=0.4):
    n = int(length * sr)
    start = int(start_frac * len(y))
    return y[start:start + n].copy()


# ============================================================
# FIGURE 5 - clean match: offset histogram correct vs wrong
# ============================================================
print("[fig5] offset histogram (clean) ...")
q = clip(y_full, sr)
_, q_hashes, _, _, _ = fingerprint_audio(q, sr=sr)
label, score, all_scores, hist = mini_db.match(q_hashes)

# build histograms for the winner and a wrong song
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
if hist is not None:
    ax.hist(hist, bins=80, color='#2a8')
ax.set_title(f'Correct song "{label}" - sharp spike')
ax.set_xlabel('offset (frames)'); ax.set_ylabel('count')

# pick a wrong song from mini_db
wrong_sid = None
for sid, lbl in mini_db.songs.items():
    if lbl != label:
        wrong_sid = sid
        break
wrong_offsets = []
if wrong_sid is not None:
    for key, tq in q_hashes:
        for sid, t1 in mini_db.index.get(key, []):
            if sid == wrong_sid:
                wrong_offsets.append(t1 - tq)
ax = axes[1]
if wrong_offsets:
    ax.hist(wrong_offsets, bins=80, color='#c66')
    ax.set_title(f'Wrong song "{mini_db.songs[wrong_sid]}" - random scatter')
else:
    ax.text(0.5, 0.5, 'no spurious matches at all\n(even better!)',
            ha='center', va='center')
    ax.set_axis_off()
ax.set_xlabel('offset (frames)'); ax.set_ylabel('count')
plt.tight_layout()
plt.savefig(f'{OUT}/05_offset_histogram.png')
plt.close()


# ============================================================
# FIGURE 6 - robustness to additive noise
# ============================================================
print("[fig6] noise robustness sweep ...")
snr_list = [30, 20, 10, 5, 0, -5]
scores_noise = []
correct_noise = []
for snr in snr_list:
    q_noisy = add_noise(q, snr)
    _, h2, _, _, _ = fingerprint_audio(q_noisy, sr=sr)
    lbl, sc, _, _ = mini_db.match(h2)
    scores_noise.append(sc)
    correct_noise.append(lbl == SONG_LABEL)

fig, ax = plt.subplots(figsize=(7, 3.6))
colors = ['#2a8' if c else '#c33' for c in correct_noise]
ax.bar([str(s) for s in snr_list], scores_noise, color=colors)
ax.set_xlabel('SNR (dB)  - lower = more noise'); ax.set_ylabel('match score')
ax.set_title('Match score vs added white-noise SNR  (green = identifier still correct)')
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/06_noise_robustness.png')
plt.close()


# ============================================================
# FIGURE 7 - robustness to pitch shift / time stretch
# ============================================================
print("[fig7] pitch / time-stretch robustness ...")
semitones = [-3, -2, -1, 0, 1, 2, 3]
rates = [0.85, 0.92, 1.0, 1.08, 1.15]
ps_scores, ps_correct = [], []
ts_scores, ts_correct = [], []
for s in semitones:
    q2 = pitch_shift(q, sr, s)
    _, h2, _, _, _ = fingerprint_audio(q2, sr=sr)
    lbl, sc, _, _ = mini_db.match(h2)
    ps_scores.append(sc); ps_correct.append(lbl == SONG_LABEL)
for r in rates:
    q2 = time_stretch(q, r)
    _, h2, _, _, _ = fingerprint_audio(q2, sr=sr)
    lbl, sc, _, _ = mini_db.match(h2)
    ts_scores.append(sc); ts_correct.append(lbl == SONG_LABEL)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.bar([str(s) for s in semitones], ps_scores,
       color=['#2a8' if c else '#c33' for c in ps_correct])
ax.set_xlabel('pitch shift (semitones)'); ax.set_ylabel('match score')
ax.set_title('Pitch-shift robustness')
ax = axes[1]
ax.bar([f'{r:.2f}' for r in rates], ts_scores,
       color=['#2a8' if c else '#c33' for c in ts_correct])
ax.set_xlabel('time-stretch rate'); ax.set_ylabel('match score')
ax.set_title('Time-stretch robustness')
plt.tight_layout()
plt.savefig(f'{OUT}/07_pitch_time.png')
plt.close()

# stash numbers for the report
import json
stats = {
    'song_label': SONG_LABEL,
    'song_path': SONG_PATH,
    'num_songs_indexed': len(mini_db),
    'num_hashes': mini_db.num_hashes(),
    'num_peaks': len(peaks),
    'num_hashes_song': len(hashes),
    'clean_score': int(score) if score else 0,
    'snr_list': snr_list,
    'snr_scores': [int(x) for x in scores_noise],
    'snr_correct': [bool(x) for x in correct_noise],
    'pitch_semitones': semitones,
    'pitch_scores': [int(x) for x in ps_scores],
    'pitch_correct': [bool(x) for x in ps_correct],
    'stretch_rates': rates,
    'stretch_scores': [int(x) for x in ts_scores],
    'stretch_correct': [bool(x) for x in ts_correct],
}
with open(f'{OUT}/stats.json', 'w') as fp:
    json.dump(stats, fp, indent=2)

print(f"\nAll figures saved to {OUT}/")
print("Stats: ", stats)
