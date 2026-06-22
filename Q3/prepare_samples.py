"""
Cuts 5 random 30-second clips from songs in the library and saves them as
samples/sample1.mp3 ... sample5.mp3. Used by the Identify tab's
"OR TRY A SAMPLE" section.

Run AFTER `build_database.py`.  Re-running picks 5 new random songs.
"""
import os, glob, random
import soundfile as sf
from fingerprint import load_audio

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus')
SONGS_DIR = 'database/songs'
OUT_DIR = 'samples'
CLIP_LEN_S = 30.0
N_SAMPLES = 5
SEED = 42      # deterministic; change for different picks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([p for p in glob.glob(os.path.join(SONGS_DIR, '*'))
                    if p.lower().endswith(AUDIO_EXTS)])
    if len(files) < N_SAMPLES:
        print(f"Need at least {N_SAMPLES} songs in {SONGS_DIR}, found {len(files)}")
        return

    rng = random.Random(SEED)
    picked = rng.sample(files, N_SAMPLES)
    # also save which song each sample came from, so the app can show
    # the correct-answer if it wants to
    label_map = {}
    for i, path in enumerate(picked, 1):
        label = os.path.splitext(os.path.basename(path))[0]
        y, sr = load_audio(path)
        if len(y) < int(CLIP_LEN_S * sr):
            clip = y
        else:
            start_max = len(y) - int(CLIP_LEN_S * sr)
            start = rng.randint(0, start_max)
            clip = y[start:start + int(CLIP_LEN_S * sr)]
        out_path = os.path.join(OUT_DIR, f'sample{i}.wav')
        sf.write(out_path, clip, sr)
        label_map[f'sample{i}'] = label
        print(f"  sample{i}  ->  {label}  ({CLIP_LEN_S:.0f}s)")

    # save the truth file so the app can optionally reveal it
    with open(os.path.join(OUT_DIR, 'truth.txt'), 'w', encoding='utf-8') as fp:
        for k, v in label_map.items():
            fp.write(f'{k}\t{v}\n')
    print(f"\nSaved {N_SAMPLES} sample clips + truth.txt to {OUT_DIR}/")


if __name__ == '__main__':
    main()