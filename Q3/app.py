"""
EE200 Q3B - Streamlit app, themed to match the project-demo look:
  three tabs (LIBRARY / IDENTIFY / BATCH), constellation thumbnails
  for every indexed song, full pipeline-timing breakdown, alignment-spike
  visualisation after each identification.

Run:  streamlit run app.py
"""
import io, os, csv, time, base64
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from fingerprint import (FingerprintDB, load_audio, compute_spectrogram,
                         find_peaks, peaks_to_hashes, SR, N_FFT, HOP)

DB_PATH = os.environ.get('SHAZAM_DB', 'database/db.pkl')
THUMBS_DIR = 'thumbnails'

st.set_page_config(page_title="EE200 Audio Fingerprinting",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------
# CSS - dark teal terminal aesthetic, monospaced everywhere
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"]  { font-family: 'JetBrains Mono', monospace; }
.stApp { background: #050d10; color: #b5d4d4; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: #e8f5f1; }
hr { border-color: #15282d; }

.eyebrow { letter-spacing: 0.25em; font-size: 0.7rem; color: #5fd3bc;
           text-transform: uppercase; }
.muted   { color: #6d8a8a; font-size: 0.85rem; }
.bigtitle{ font-size: 2.6rem; font-weight: 700; color: #e8f5f1;
           letter-spacing: -0.02em; }
.tagline { color: #8aa9a9; font-size: 0.95rem; margin-top: -0.5rem; }

.card {
  background: linear-gradient(180deg,#0d1d22,#08151a);
  border: 1px solid #15303a; border-radius: 6px;
  padding: 8px 10px 6px 10px; margin-bottom: 10px;
}
.card .name { color: #d8f0ec; font-size: 0.9rem; font-weight: 500; }
.card .hash { color: #6d8a8a; font-size: 0.75rem; }

.banner {
  background: linear-gradient(180deg,#0d3027,#08221d);
  border: 1px solid #2a8068; border-radius: 8px;
  padding: 22px 26px; margin: 18px 0;
}
.banner .eyebrow { color: #5fd3bc; }
.banner h1 { font-size: 2.3rem; margin: 4px 0 8px 0; color: #e8fff7; }
.banner .score { color: #ffb86c; font-weight: 600; }

.stat {
  background: #08161a; border: 1px solid #143036;
  border-radius: 6px; padding: 10px 12px; text-align: left;
}
.stat .lbl { color: #5fd3bc; font-size: 0.65rem; letter-spacing: 0.2em; }
.stat .val { color: #e8f5f1; font-size: 1.25rem; font-weight: 700; }
.stat .sub { color: #6d8a8a; font-size: 0.7rem; }

.steptag { letter-spacing: 0.2em; font-size: 0.7rem; color: #5fd3bc;
           text-transform: uppercase; }
.steph  { font-family: 'Space Grotesk', sans-serif;
          font-size: 1.4rem; color: #e8f5f1; margin: 2px 0 8px 0; }

.barouter { background: #0c1d22; height: 14px; border-radius: 3px;
            border: 1px solid #15303a; }
.barinner { background: linear-gradient(90deg, #5fd3bc, #74e0c8);
            height: 100%; border-radius: 2px; }
.row      { display: flex; align-items: center; padding: 5px 0; gap: 12px; }
.row .nm  { width: 260px; color: #d8f0ec; font-size: 0.85rem; }
.row .sc  { width: 60px; text-align: right; color: #b5d4d4;
            font-size: 0.85rem; }
.row .bar { flex: 1; }

.section-rule { border-top: 1px solid #15282d; margin: 26px 0 14px 0; }
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid #15282d; }
.stTabs [data-baseweb="tab"] {
  padding: 8px 2px; color: #6d8a8a; font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.2em; font-size: 0.75rem; text-transform: uppercase;
}
.stTabs [aria-selected="true"] { color: #5fd3bc !important;
  border-bottom: 2px solid #5fd3bc; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# load db
# ----------------------------------------------------------------------
@st.cache_resource
def get_db():
    if not os.path.exists(DB_PATH):
        return None
    return FingerprintDB.load(DB_PATH)

db = get_db()


# ----------------------------------------------------------------------
# header
# ----------------------------------------------------------------------
c1, c2 = st.columns([1, 14])
with c1:
    st.markdown("<div style='font-size:2.6rem'>🎵</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="bigtitle">Song Detector</div>',
                unsafe_allow_html=True)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

if db is None:
    st.error(f"No database found at {DB_PATH}. "
             "Run `python build_database.py` first.")
    st.stop()


# ----------------------------------------------------------------------
# helper: do one full identification and return everything
# ----------------------------------------------------------------------
def identify_with_stages(file_bytes):
    """Run the pipeline with per-stage timing, return a dict of everything."""
    t0 = time.time()
    y, sr = load_audio(io.BytesIO(file_bytes))
    t_load = time.time() - t0

    t0 = time.time()
    f, t, S_db = compute_spectrogram(y, sr=sr)
    t_spec = time.time() - t0

    t0 = time.time()
    peaks = find_peaks(S_db)
    t_const = time.time() - t0

    t0 = time.time()
    hashes = list(peaks_to_hashes(peaks))
    t_hash = time.time() - t0

    t0 = time.time()
    # lookup phase only
    offsets_per_song = {}
    for key, tq in hashes:
        if key not in db.index:
            continue
        for sid, t1 in db.index[key]:
            offsets_per_song.setdefault(sid, []).append(t1 - tq)
    t_lookup = time.time() - t0

    t0 = time.time()
    per_song_scores = {}
    best_label, best_score = None, 0
    best_offsets = None
    runner_up = 0
    for sid, offs in offsets_per_song.items():
        arr = np.array(offs)
        _, counts = np.unique(arr, return_counts=True)
        top = int(counts.max())
        per_song_scores[db.songs[sid]] = top
        if top > best_score:
            runner_up = best_score
            best_score = top
            best_label = db.songs[sid]
            best_offsets = arr
        elif top > runner_up:
            runner_up = top
    t_score = time.time() - t0

    return dict(
        y=y, sr=sr,
        f=f, t=t, S_db=S_db, peaks=peaks, hashes=hashes,
        label=best_label, score=best_score, runner_up=runner_up,
        per_song_scores=per_song_scores, best_offsets=best_offsets,
        timings=dict(spectrogram=t_spec, constellation=t_const,
                     hashing=t_hash, lookup=t_lookup, scoring=t_score,
                     load=t_load),
        spec_shape=S_db.shape,
        n_peaks=len(peaks), n_hashes=len(hashes),
        n_tracks_hit=len(offsets_per_song),
    )


def stage_box(num, label, ms, sub):
    st.markdown(f"""
    <div class="stat">
      <div class="lbl">{num} · {label}</div>
      <div class="val">{ms} <span style='font-size:0.75rem;color:#6d8a8a'>ms</span></div>
      <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# tabs
# ----------------------------------------------------------------------
tab_lib, tab_id, tab_batch = st.tabs(["Library", "Identify", "Batch"])


# ============================================================
# LIBRARY
# ============================================================
with tab_lib:
    st.markdown('<div class="eyebrow">Library</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="muted">Song indexing is managed by the admin. '
        f'Drop a clip in the <b>Identify</b> tab to test the library.</div>',
        unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:18px"><span class="muted">'
                f'In the database: <b style="color:#e8f5f1">{len(db)} songs</b>, '
                f'{db.num_hashes():,} hash entries.</span></div>',
                unsafe_allow_html=True)

    st.write("")
    songs_sorted = sorted(db.songs.items(), key=lambda x: x[1].lower())

    # precompute hash counts once per session - this is the slow part
    @st.cache_data
    def hash_counts_per_song():
        counts = {sid: 0 for sid in db.songs}
        for entries in db.index.values():
            for sid, _ in entries:
                counts[sid] += 1
        return counts
    counts = hash_counts_per_song()

    # 4 columns, like the demo
    cols_per_row = 4
    rows = (len(songs_sorted) + cols_per_row - 1) // cols_per_row
    for r in range(rows):
        cs = st.columns(cols_per_row)
        for c in range(cols_per_row):
            i = r * cols_per_row + c
            if i >= len(songs_sorted):
                continue
            sid, label = songs_sorted[i]
            n_hashes = counts[sid]
            thumb = os.path.join(THUMBS_DIR, label + '.png')
            with cs[c]:
                if os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:140px;border:1px solid #15303a;'
                        'border-radius:4px;background:#08161a;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#3a5a5a;font-size:0.75rem">no thumbnail</div>',
                        unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card"><div class="name">{label}</div>'
                    f'<div class="hash">{n_hashes:,} hashes</div></div>',
                    unsafe_allow_html=True)


# ============================================================
# IDENTIFY
# ============================================================
with tab_id:
    st.markdown('<div class="eyebrow">Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="bigtitle" style="font-size:1.9rem">'
                'Identify a clip</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">200 MB per file · WAV, MP3, FLAC, OGG, M4A</div>',
                unsafe_allow_html=True)
    st.write("")

    up = st.file_uploader("Upload", type=['wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac'],
                          label_visibility='collapsed')

    # ---- OR TRY A SAMPLE ----
    SAMPLES_DIR = 'samples'
    sample_files = []
    if os.path.isdir(SAMPLES_DIR):
        sample_files = sorted([f for f in os.listdir(SAMPLES_DIR)
                               if f.startswith('sample') and
                               f.lower().endswith(('.wav', '.mp3'))])

    # decide what to identify: upload wins; else a sample if user clicked one
    if 'sample_choice' not in st.session_state:
        st.session_state.sample_choice = None

    bytes_to_identify = None
    name_to_identify = None
    if up is not None:
        bytes_to_identify = up.read()
        name_to_identify = up.name
        st.session_state.sample_choice = None
    elif st.session_state.sample_choice is not None:
        sp = os.path.join(SAMPLES_DIR, st.session_state.sample_choice)
        with open(sp, 'rb') as fp:
            bytes_to_identify = fp.read()
        name_to_identify = st.session_state.sample_choice

    if sample_files:
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Or try a sample</div>',
                    unsafe_allow_html=True)
        for fn in sample_files:
            sp = os.path.join(SAMPLES_DIR, fn)
            sname = os.path.splitext(fn)[0]
            c1, c2, c3 = st.columns([1, 6, 1])
            with c1:
                st.markdown(f'<div style="padding-top:14px;color:#8aa9a9">{sname}</div>',
                            unsafe_allow_html=True)
            with c2:
                st.audio(sp)
            with c3:
                if st.button("Try", key=f"try_{sname}"):
                    st.session_state.sample_choice = fn
                    st.rerun()

    if bytes_to_identify is None:
        st.markdown('<div class="muted" style="margin-top:30px">'
                    'Drop a query clip above (or click <b>Try</b> on a sample) '
                    'to run the pipeline. 5-10 seconds is plenty.'
                    '</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="muted" style="margin-top:8px">'
                    f'Identifying: <b style="color:#e8f5f1">{name_to_identify}</b>'
                    f'</div>', unsafe_allow_html=True)
        with st.spinner("Running pipeline ..."):
            res = identify_with_stages(bytes_to_identify)

        # ---- pipeline timing strip ----
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 1, 1, 1])
        tim = res['timings']
        with cols[0]:
            stage_box("①", "SPECTROGRAM", int(tim['spectrogram']*1000),
                      f"{res['spec_shape'][0]}×{res['spec_shape'][1]}")
        with cols[1]:
            stage_box("②", "CONSTELLATION", int(tim['constellation']*1000),
                      f"{res['n_peaks']:,} peaks")
        with cols[2]:
            stage_box("③", "HASHING", int(tim['hashing']*1000),
                      f"{res['n_hashes']:,} hashes")
        with cols[3]:
            stage_box("④", "DB LOOKUP", int(tim['lookup']*1000),
                      f"{res['n_tracks_hit']} tracks hit")
        with cols[4]:
            stage_box("⑤", "SCORING", int(tim['scoring']*1000),
                      f"offset {res['score']}")
        with cols[5]:
            total_ms = int(sum(tim.values()) * 1000)
            stage_box("Σ", "TOTAL", total_ms, "end-to-end")

        # ---- match banner ----
        if res['label'] is None:
            st.markdown('<div class="banner">'
                        '<div class="eyebrow">No match</div>'
                        '<h1>Unknown clip</h1>'
                        '<div class="muted">No hashes collided with the database.</div>'
                        '</div>', unsafe_allow_html=True)
        else:
            ratio = (res['score'] / res['runner_up']) if res['runner_up'] > 0 else float('inf')
            ratio_str = f"{ratio:.0f}× the runner-up" if ratio < 10000 else "no contest"
            st.markdown(f"""
            <div class="banner">
              <div class="eyebrow">Match found</div>
              <h1>{res['label']}</h1>
              <div class="muted">cluster score <span class="score">{res['score']}</span>
                · <span class="score">{ratio_str}</span></div>
            </div>
            """, unsafe_allow_html=True)

            # ---- candidate scores ----
            st.markdown('<div class="eyebrow">Candidate scores</div>',
                        unsafe_allow_html=True)
            ranking = sorted(res['per_song_scores'].items(),
                             key=lambda x: -x[1])[:6]
            top = ranking[0][1] if ranking else 1
            for name, sc in ranking:
                w = int(100 * sc / top) if top else 0
                st.markdown(f"""
                <div class="row">
                  <div class="nm">{name}</div>
                  <div class="bar"><div class="barouter">
                    <div class="barinner" style="width:{w}%"></div>
                  </div></div>
                  <div class="sc">{sc}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

            # ---- STEP 1: feature extraction ----
            st.markdown('<div class="steptag">Step 1 · Feature extraction</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="steph">From spectrogram to constellation</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="muted">The clip was converted into a time-frequency '
                f'map (left); brighter means louder at that frequency and moment. '
                f'From that rich image, only the <b style="color:#5fd3bc">'
                f'{res["n_peaks"]:,} most prominent peaks</b> were kept (right). '
                f'Discarding amplitude and phase makes the fingerprint robust to '
                f'EQ, volume changes, and noise.</div>',
                unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 3.6))
                fig.patch.set_facecolor('#050d10')
                ax.set_facecolor('#050d10')
                ax.imshow(res['S_db'], origin='lower', aspect='auto',
                          extent=[res['t'][0], res['t'][-1],
                                  res['f'][0], res['f'][-1]],
                          cmap='magma',
                          vmin=np.median(res['S_db']),
                          vmax=res['S_db'].max())
                ax.set_xlabel('time (s)', color='#8aa9a9')
                ax.set_ylabel('frequency (Hz)', color='#8aa9a9')
                ax.tick_params(colors='#6d8a8a')
                for s in ax.spines.values(): s.set_color('#15303a')
                st.pyplot(fig, clear_figure=True)
            with c2:
                fig, ax = plt.subplots(figsize=(6, 3.6))
                fig.patch.set_facecolor('#050d10')
                ax.set_facecolor('#050d10')
                if res['peaks']:
                    pf = [res['f'][p[0]] for p in res['peaks']]
                    pt = [res['t'][p[1]] for p in res['peaks']]
                    ax.scatter(pt, pf, s=4, color='#5fd3bc', alpha=0.8)
                ax.set_xlabel('time (s)', color='#8aa9a9')
                ax.set_ylabel('frequency (Hz)', color='#8aa9a9')
                ax.tick_params(colors='#6d8a8a')
                for s in ax.spines.values(): s.set_color('#15303a')
                ax.text(0.97, 0.95, f"{res['n_peaks']:,} peaks",
                        transform=ax.transAxes, ha='right', va='top',
                        color='#5fd3bc', fontsize=9)
                st.pyplot(fig, clear_figure=True)

            st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

            # ---- STEP 2: alignment spike ----
            st.markdown('<div class="steptag">Step 2 · The proof</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="steph">The alignment spike</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="muted">Every matched hash votes for a time offset '
                f'(database frame minus query frame). Chance matches scatter votes '
                f'randomly, forming a flat noise floor. A genuine match makes them '
                f'converge: <b style="color:#ffb86c">{res["score"]} hashes agreed '
                f'on a single offset</b>. That spike cannot be a coincidence.</div>',
                unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(11, 3.8))
            fig.patch.set_facecolor('#050d10')
            ax.set_facecolor('#050d10')
            if res['best_offsets'] is not None:
                ax.hist(res['best_offsets'], bins=100, color='#ffb86c',
                        edgecolor='#ffb86c')
            ax.set_xlabel('time offset (database frame − query frame)',
                          color='#8aa9a9')
            ax.set_ylabel('# hashes', color='#8aa9a9')
            ax.tick_params(colors='#6d8a8a')
            for s in ax.spines.values(): s.set_color('#15303a')
            ax.text(0.99, 0.92, 'chance matches\n(noise floor)',
                    transform=ax.transAxes, ha='right', va='top',
                    color='#5a8080', fontsize=9, alpha=0.8)
            st.pyplot(fig, clear_figure=True)


# ============================================================
# BATCH
# ============================================================
with tab_batch:
    st.markdown('<div class="eyebrow">Batch</div>', unsafe_allow_html=True)
    st.markdown('<div class="bigtitle" style="font-size:1.9rem">'
                'Identify many clips</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload as many query clips as you like. '
                'The app produces a <code>results.csv</code> with exactly two '
                'columns: <b>filename, prediction</b>.</div>',
                unsafe_allow_html=True)
    st.write("")
    ups = st.file_uploader("clips", type=['wav','mp3','flac','m4a','ogg','aac'],
                           accept_multiple_files=True, label_visibility='collapsed')
    if ups and st.button("Run batch", type='primary'):
        rows = []
        prog = st.progress(0.0)
        t0 = time.time()
        for i, u in enumerate(ups, 1):
            try:
                r = identify_with_stages(u.read())
                pred = r['label'] if r['label'] is not None else "unknown"
            except Exception:
                pred = "unknown"
            rows.append((os.path.splitext(u.name)[0], pred))
            prog.progress(i / len(ups))

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['filename', 'prediction'])
        w.writerows(rows)
        st.success(f"Done. {len(rows)} clips processed in {time.time()-t0:.1f}s.")
        st.dataframe({'filename': [r[0] for r in rows],
                      'prediction': [r[1] for r in rows]},
                     use_container_width=True)
        st.download_button("Download results.csv",
                           data=buf.getvalue().encode('utf-8'),
                           file_name="results.csv", mime="text/csv")