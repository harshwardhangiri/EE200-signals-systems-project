"""
Q3A - Single peaks vs. paired hashes experiment.

Loads the pre-built database (database/db.pkl) and uses a sample clip
from the samples/ folder as the query. No re-indexing needed.

Drop this in the Q3/ folder (next to fingerprint.py) and run:
    python single_vs_pairs.py

Outputs:
    figures/08_single_vs_pairs.png
    figures/single_vs_pairs_stats.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from fingerprint import load_audio, fingerprint_audio, FingerprintDB, SR

# Always run from this script's folder (so relative paths work no matter
# where you launched python from)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 9})


# ─── 1. Load the pre-built database ───────────────────────────────────────────
DB_PATH = 'database/db.pkl'
print(f"[1/4] Loading {DB_PATH} ...")
mini_db = FingerprintDB.load(DB_PATH)
print(f"      {len(mini_db)} songs, {mini_db.num_hashes():,} pair hashes")


# ─── 2. Pick a query clip from samples/ ───────────────────────────────────────
# truth.txt:
#   sample1 -> Two Of Us
#   sample2 -> Crazy Little Thing Called Love
#   sample3 -> A Hard Day_s Night
#   sample4 -> Within You Without You
#   sample5 -> Hey Jude
QUERY_FILE  = 'samples/sample1.wav'
GROUND_TRUTH = 'Two Of Us'           # must match the song label in the DB exactly

print(f"[2/4] Loading query: {QUERY_FILE}")
y_query, sr = load_audio(QUERY_FILE)
print(f"      query length: {len(y_query)/sr:.2f}s")


# ─── 3. PAIR-HASH matching (your normal pipeline) ─────────────────────────────
print("[3/4] Pair-hash matching ...")
q_peaks, q_pair_hashes, _, _, _ = fingerprint_audio(y_query, sr=sr)

label_pair, score_pair, all_pair_scores, _ = mini_db.match(q_pair_hashes)
sorted_pair = sorted(all_pair_scores.values(), reverse=True)
runner_pair = sorted_pair[1] if len(sorted_pair) > 1 else 0
correct_pair = (label_pair == GROUND_TRUTH)
print(f"      pair winner   : {label_pair}  score={score_pair}  "
      f"runner-up={runner_pair}  correct={correct_pair}")


# ─── 4. SINGLE-PEAK matching ──────────────────────────────────────────────────
# Collapse the existing (f1,f2,dt) DB down to a single-peak DB keyed by f1.
# Each unique (sid, t1, f1) is stored only once.
print("[4/4] Single-peak matching ...")
single_index = {}                     # f1 -> list of (sid, t1)
seen = set()
for (f1, f2, dt), entries in mini_db.index.items():
    for (sid, t1) in entries:
        tag = (sid, t1, f1)
        if tag in seen:
            continue
        seen.add(tag)
        single_index.setdefault(f1, []).append((sid, t1))

# Query side: each peak's frequency bin is the lookup key
offsets_per_song_single = {}
for (f_bin, t_bin) in q_peaks:
    if f_bin not in single_index:
        continue
    for sid, t1 in single_index[f_bin]:
        offsets_per_song_single.setdefault(sid, []).append(t1 - t_bin)

all_single_scores = {}
label_single = None
score_single = 0
for sid, offs in offsets_per_song_single.items():
    if not offs:
        continue
    arr = np.array(offs)
    _, counts = np.unique(arr, return_counts=True)
    top = int(counts.max())
    all_single_scores[mini_db.songs[sid]] = top
    if top > score_single:
        score_single = top
        label_single = mini_db.songs[sid]

sorted_single = sorted(all_single_scores.values(), reverse=True)
runner_single = sorted_single[1] if len(sorted_single) > 1 else 0
correct_single = (label_single == GROUND_TRUTH)
print(f"      single winner : {label_single}  score={score_single}  "
      f"runner-up={runner_single}  correct={correct_single}")


# ─── 5. Plot the comparison figure ───────────────────────────────────────────
print("\n[plot] writing figures/08_single_vs_pairs.png ...")

# Find the song_id of the correct (ground-truth) song
sid_correct = None
for sid, lbl in mini_db.songs.items():
    if lbl == GROUND_TRUTH:
        sid_correct = sid
        break

single_correct = offsets_per_song_single.get(sid_correct, [])

pair_correct = []
for key, tq in q_pair_hashes:
    for sid, t1 in mini_db.index.get(key, []):
        if sid == sid_correct:
            pair_correct.append(t1 - tq)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: single peaks - flat / scattered
ax = axes[0]
if single_correct:
    ax.hist(single_correct, bins=80, color='#c66', edgecolor='#933', linewidth=0.3)
else:
    ax.text(0.5, 0.5, 'no matches at all',
            ha='center', va='center', transform=ax.transAxes)
ax.set_title(f'Single-peak matching - "{GROUND_TRUTH}"\n'
             f'tallest bin = {score_single}   '
             f'runner-up song = {runner_single}   '
             f'winner = {label_single}',
             fontsize=10)
ax.set_xlabel('offset (frames)')
ax.set_ylabel('count')
ax.annotate(f'Weak spike on noisy floor\n(margin only {score_single/max(runner_single,1):.0f}x runner-up)',
            xy=(0.62, 0.85), xycoords='axes fraction',
            ha='center', fontsize=9, color='#933',
            bbox=dict(boxstyle='round,pad=0.3', fc='mistyrose', ec='#c66'))

# Right: pair hashes - sharp spike
ax = axes[1]
if pair_correct:
    ax.hist(pair_correct, bins=80, color='#2a8', edgecolor='#185', linewidth=0.3)
ax.set_title(f'Pair-hash matching - "{GROUND_TRUTH}"\n'
             f'tallest bin = {score_pair}   '
             f'runner-up song = {runner_pair}   '
             f'winner = {label_pair}',
             fontsize=10)
ax.set_xlabel('offset (frames)')
ax.set_ylabel('count')
ax.annotate(f'Sharp spike at the\ntrue alignment\n({score_pair} matches)',
            xy=(0.62, 0.72), xycoords='axes fraction',
            fontsize=9, color='#185',
            bbox=dict(boxstyle='round,pad=0.3', fc='#d5f5e3', ec='#2a8'))

fig.suptitle('Single peaks vs. paired hashes - why pairs win',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/08_single_vs_pairs.png', bbox_inches='tight')
plt.close()


# ─── 6. Save stats ────────────────────────────────────────────────────────────
stats = {
    'ground_truth':    GROUND_TRUTH,
    'query_file':      QUERY_FILE,
    'query_length_s':  round(len(y_query) / sr, 2),
    'single_peak': {
        'winner':    label_single,
        'score':     score_single,
        'runner_up': runner_single,
        'correct':   correct_single,
    },
    'pair_hash': {
        'winner':    label_pair,
        'score':     score_pair,
        'runner_up': runner_pair,
        'correct':   correct_pair,
    },
    'margin_ratio_pair_vs_single':
        round(score_pair / max(score_single, 1), 1),
}
with open(f'{OUT}/single_vs_pairs_stats.json', 'w') as fp:
    json.dump(stats, fp, indent=2)

# ─── 7. Print summary table ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Method':<28}{'Score':>10}{'Runner-up':>12}{'Correct':>10}")
print("=" * 70)
print(f"{'Single peaks (f1 only)':<28}{score_single:>10}{runner_single:>12}"
      f"{str(correct_single):>10}")
print(f"{'Pair hashes (f1,f2,dt)':<28}{score_pair:>10}{runner_pair:>12}"
      f"{str(correct_pair):>10}")
print("=" * 70)
print(f"Pair-hash margin over single-peak score: "
      f"{score_pair / max(score_single, 1):.1f}x")
print(f"\nSaved: {OUT}/08_single_vs_pairs.png")
print(f"Saved: {OUT}/single_vs_pairs_stats.json")
