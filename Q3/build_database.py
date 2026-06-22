"""
Index every audio file inside database/songs/ and dump a pickle.

Usage:
    python build_database.py
    python build_database.py --songs path/to/folder --out database/db.pkl
"""

import os
import sys
import glob
import argparse
import time

from fingerprint import FingerprintDB, load_audio, fingerprint_audio


AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--songs', default='database/songs',
                    help='folder containing the song files')
    ap.add_argument('--out', default='database/db.pkl',
                    help='output pickle path')
    args = ap.parse_args()

    files = sorted([p for p in glob.glob(os.path.join(args.songs, '*'))
                    if p.lower().endswith(AUDIO_EXTS)])
    if not files:
        print(f"No audio files in {args.songs}")
        sys.exit(1)

    db = FingerprintDB()
    print(f"Indexing {len(files)} song(s) from {args.songs} ...")
    t0 = time.time()
    for i, path in enumerate(files, 1):
        label = os.path.splitext(os.path.basename(path))[0]
        try:
            y, sr = load_audio(path)
            _, hashes, _, _, _ = fingerprint_audio(y, sr=sr)
            db.add_song(label, hashes)
            print(f"  [{i}/{len(files)}] {label}  -  {len(hashes)} hashes")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {label}  FAILED: {e}")
    dt = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    db.save(args.out)
    print(f"\nIndexed {len(db)} songs, {db.num_hashes():,} hash entries "
          f"in {dt:.1f}s. Saved -> {args.out}")


if __name__ == '__main__':
    main()
