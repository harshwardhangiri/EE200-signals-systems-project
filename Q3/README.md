# EE200 Final Project - Q3: Audio Fingerprinting

A Shazam-style song identifier built from scratch for the EE200 final project
(Q3A + Q3B). Given a short query clip, the system identifies which of the 50
indexed songs it came from, using a spectrogram -> constellation peaks ->
combinatorial-hash pipeline.

## Layout

```
shazam_ee200/
├── fingerprint.py       core library (STFT, peak picking, hashing, matching)
├── build_database.py    index every song in database/songs/ into database/db.pkl
├── generate_figures.py  produce all the figures used in the report
├── app.py               Streamlit app (single-clip + batch modes)
├── requirements.txt     python deps for Streamlit Cloud
├── packages.txt         apt deps for Streamlit Cloud (ffmpeg, libsndfile1)
├── database/
│   ├── songs/           <-- drop the 50 provided songs here
│   └── db.pkl           (created by build_database.py)
├── figures/             figures used in the report
└── report/              LaTeX source + compiled PDF
```

## Local setup

```bash
pip install -r requirements.txt
# put the 50 songs into database/songs/  (do NOT rename them)
python build_database.py
python generate_figures.py        # if you want to regenerate report figures
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push this folder (with `database/db.pkl` committed) to a public GitHub repo.
2. On https://share.streamlit.io click "New app", point it at the repo,
   set the main file to `app.py`.
3. Streamlit Cloud auto-installs from `requirements.txt` and `packages.txt`.

The deployed app loads `database/db.pkl` on startup, so the indexed songs ship
with the deployment - no re-indexing on the server.

## results.csv format (batch mode)

Exactly as required by the assignment:

```
filename,prediction
clip_001,song_name_without_extension
clip_002,another_song_name
...
```

The `filename` column is the uploaded file's name without extension. The
`prediction` column is the matched song's filename (also without extension);
"unknown" is emitted if no match crosses the threshold.

## Algorithm in one paragraph

Each song is downsampled to 11025 Hz, run through a Hann-windowed STFT
(n_fft=2048, hop=512), and the dB-magnitude spectrogram is searched for local
maxima above a noise floor. Those peaks form a "constellation". Each anchor
peak is paired with the next 15 peaks ahead of it in time; the triple
`(f1, f2, dt)` becomes a hash key, stored against `(song_id, t1)`. At query
time the same hashes are extracted from the clip, looked up in the database,
and for every match the offset `t_song - t_query` is recorded per song. The
song whose offset histogram has the tallest single bin wins - that tall spike
is the fingerprint of a true alignment.
