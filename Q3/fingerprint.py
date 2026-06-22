"""
Audio fingerprinting core - Shazam-style.
EE200 final project, Q3.

Pipeline:
  audio -> STFT spectrogram -> local-max peaks (constellation)
        -> pair peaks into (f1, f2, dt) hashes -> store in DB
  match: extract hashes from query, look up in DB, histogram offsets per song,
         predict song with the tallest histogram bin.
"""

import os
import pickle
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
import soundfile as sf
import librosa

# ---------- defaults ----------
SR = 11025               # downsample target - plenty for hashing, fast
N_FFT = 2048             # window length in samples (~186 ms at 11025 Hz)
HOP = 512                # hop length (~46 ms)
PEAK_NEIGHBOURHOOD = 20  # local-max filter window (in spectrogram bins)
AMP_MIN = 10             # min dB above floor for a bin to be a peak
FAN_VALUE = 15           # how many forward peaks each anchor pairs with
MIN_DT = 0               # min time gap between paired peaks (frames)
MAX_DT = 200             # max time gap (frames)


def load_audio(path, sr=SR):
    """Load any audio file as a mono float32 array at sample rate sr."""
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop=HOP):
    """Return (freqs, times, S_db) - magnitude spectrogram in dB."""
    f, t, Z = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          window='hann', boundary=None, padded=False)
    S = np.abs(Z)
    # log magnitude with a small floor to avoid -inf
    S_db = 20.0 * np.log10(S + 1e-8)
    return f, t, S_db


def find_peaks(S_db, neighbourhood=PEAK_NEIGHBOURHOOD, amp_min=AMP_MIN):
    """
    Find local maxima of S_db that are above an absolute amplitude threshold.
    Returns list of (freq_bin, time_bin) tuples.
    """
    # max filter: a point is a local max iff it equals the max in its neighbourhood
    local_max = maximum_filter(S_db, size=neighbourhood) == S_db
    # absolute threshold relative to the floor: amp_min above the median
    floor = np.median(S_db)
    above = S_db > (floor + amp_min)
    peaks_mask = local_max & above
    freq_idx, time_idx = np.where(peaks_mask)
    # sort by time so pairing is in temporal order
    order = np.argsort(time_idx)
    return list(zip(freq_idx[order].tolist(), time_idx[order].tolist()))


def peaks_to_hashes(peaks, fan_value=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT):
    """
    Pair each anchor peak with up to `fan_value` peaks ahead in time.
    Hash key = (f1, f2, dt). Yields (hash_key, t1) tuples.
    """
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        # forward pairing only
        for j in range(1, fan_value + 1):
            if i + j >= n:
                break
            f2, t2 = peaks[i + j]
            dt = t2 - t1
            if min_dt <= dt <= max_dt:
                yield (f1, f2, dt), t1


def fingerprint_audio(y, sr=SR, **kw):
    """Return (peaks, hashes, freqs, times, S_db). hashes is list of (key, t1)."""
    f, t, S_db = compute_spectrogram(y, sr=sr,
                                     n_fft=kw.get('n_fft', N_FFT),
                                     hop=kw.get('hop', HOP))
    peaks = find_peaks(S_db,
                       neighbourhood=kw.get('neighbourhood', PEAK_NEIGHBOURHOOD),
                       amp_min=kw.get('amp_min', AMP_MIN))
    hashes = list(peaks_to_hashes(peaks,
                                  fan_value=kw.get('fan_value', FAN_VALUE)))
    return peaks, hashes, f, t, S_db


# ---------- database ----------
class FingerprintDB:
    """
    Simple in-memory hash table:
      key (f1, f2, dt) -> list of (song_id, t1)
    plus a song_id -> filename map.
    """

    def __init__(self):
        self.index = {}             # key -> list of (song_id, t1)
        self.songs = {}             # song_id -> filename (no extension)
        self._next_id = 0

    def add_song(self, label, hashes):
        sid = self._next_id
        self._next_id += 1
        self.songs[sid] = label
        for key, t1 in hashes:
            self.index.setdefault(key, []).append((sid, t1))
        return sid

    def match(self, query_hashes):
        """
        Returns (best_label, best_score, per_song_scores, best_offset_hist).
        Algorithm: for each query hash, look up matches, collect offsets
        (t_song - t_query) per song. The song with the tallest single
        offset-bin wins. The tallest bin's count is the score.
        """
        # song_id -> list of offsets
        offsets_per_song = {}
        for key, tq in query_hashes:
            if key not in self.index:
                continue
            for sid, t1 in self.index[key]:
                offsets_per_song.setdefault(sid, []).append(t1 - tq)

        per_song_scores = {}
        best_hist_data = None
        best_label = None
        best_score = 0
        for sid, offs in offsets_per_song.items():
            if not offs:
                continue
            arr = np.array(offs)
            # most common offset
            vals, counts = np.unique(arr, return_counts=True)
            top = counts.max()
            per_song_scores[self.songs[sid]] = int(top)
            if top > best_score:
                best_score = int(top)
                best_label = self.songs[sid]
                best_hist_data = arr
        return best_label, best_score, per_song_scores, best_hist_data

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'index': self.index,
                         'songs': self.songs,
                         '_next_id': self._next_id}, f)

    @classmethod
    def load(cls, path):
        db = cls()
        with open(path, 'rb') as f:
            d = pickle.load(f)
        db.index = d['index']
        db.songs = d['songs']
        db._next_id = d['_next_id']
        return db

    def __len__(self):
        return len(self.songs)

    def num_hashes(self):
        return sum(len(v) for v in self.index.values())


# ---------- robustness helpers (used in Q3A experiments) ----------
def add_noise(y, snr_db):
    """Add white Gaussian noise at given SNR (in dB) to signal y."""
    sig_p = np.mean(y ** 2) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(y)).astype(np.float32) * np.sqrt(noise_p)
    return y + noise


def pitch_shift(y, sr, semitones):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)
