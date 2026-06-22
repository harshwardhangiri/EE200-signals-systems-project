# SUBMISSION CHECKLIST - what to do before submitting

Read this once, in order. Should take ~30 minutes total.

## 1.  Get the songs (5 min)

Download the 50 songs from the Drive link on the course page. Drop them
**unchanged** into:

    shazam_ee200/database/songs/

Do NOT rename them - the filename (without extension) is the label the
identifier outputs.

## 2.  Index the database (1-2 min)

    cd shazam_ee200
    pip install -r requirements.txt
    python build_database.py

This creates `database/db.pkl`.

## 3.  Refresh the report figures with the real songs (1 min)

    python generate_figures.py

This overwrites figures/01...07 using a real song from the database
(`generate_figures.py` automatically picks the first song alphabetically as
the reference). Open one of the spectrograms to sanity-check.

## 4.  Recompile the report PDF (30 sec)

    cd report
    pdflatex report.tex

Outputs `report/report.pdf`. Open it - if the link placeholders
(`YOUR-APP-NAME`, `YOUR-USERNAME`) still show, fix them in `report.tex` after
step 5/6 and recompile.

## 5.  Push to GitHub (5 min)

    git init
    git add .
    git commit -m "EE200 Q3 - audio fingerprinter"
    git remote add origin https://github.com/<your-username>/ee200-shazam.git
    git push -u origin main

**Commit `database/db.pkl`** so Streamlit Cloud doesn't have to re-index on
boot. The 50 songs themselves are NOT committed (the .gitkeep keeps the
folder structure only).

## 6.  Deploy on Streamlit Cloud (5 min)

1. Go to https://share.streamlit.io
2. Sign in with GitHub, click "New app"
3. Pick the repo, branch `main`, main file `app.py`
4. Click "Deploy"

Cloud auto-installs from `requirements.txt` and `packages.txt`. First boot
takes 3-5 min while ffmpeg installs. Subsequent loads are instant.

You'll get a URL like `https://ee200-shazam-XXXX.streamlit.app`. Test it
with one of the songs (record yourself singing 5 seconds of any of them or
just upload a slice).

## 7.  Fill in the two link placeholders in the report

Open `report/report.tex`, find the two `YOUR-...` placeholders and replace:

* `YOUR-APP-NAME.streamlit.app` -> your actual Streamlit URL
* `YOUR-USERNAME/ee200-shazam` -> your actual GitHub repo path

Then re-run `pdflatex report.tex`.

## 8.  Make the submission zip

Zip the whole `shazam_ee200/` folder EXCEPT `database/songs/*.wav` (those
were the provided files, don't redistribute them):

    cd ..
    zip -r kartikey_ee200_q3.zip shazam_ee200 -x 'shazam_ee200/database/songs/*'

## 9.  Final submission

Upload to whatever the course's submission portal is:

* `report/report.pdf`  - the PDF report (links to live app + repo inside)
* `kartikey_ee200_q3.zip`  - the code zip

Done.

---

## Sanity-check checklist

- [ ] `database/db.pkl` exists and is roughly 20-100 MB
- [ ] Report PDF opens, all 7 figures render, no overfull-hbox warnings
- [ ] Streamlit app loads and says "Database loaded: 50 songs"
- [ ] Single-clip mode identifies a known song correctly
- [ ] Batch mode produces a results.csv with exact `filename,prediction`
      header and one row per uploaded clip, predictions matching song
      filenames without extensions
- [ ] Both URLs in the report PDF are real and clickable
