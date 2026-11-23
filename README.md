# Joint evaluation of cross-dataset shift, unseen attacks and calibration for budgeted NIDS triage

Target journal: Computers & Security (Elsevier, elsarticle template).

## Layout
- `manuscript/main.tex` (+ `sections/`, `tables/`, `refs.bib`) -> `manuscript/main.pdf`
- `manuscript/highlights.txt` - journal highlights (each <= 85 characters)
- `manuscript/refs_verification.md` - how each reference was verified (DOI / arXiv / page)
- `code/` - all code; every number in the paper comes from these scripts
  - `prepare.py` - harmonise raw datasets -> `data/processed/*.parquet`
  - `nidslib.py` - models, calibration, kNN novelty, conformal triage, metrics
  - `run_experiments.py` - `python run_experiments.py os` (leave-one-family-out), `python run_experiments.py xd` (in-domain + cross-dataset)
  - `make_figures.py` - tables (`manuscript/tables/*.tex`), figures (`figures/*.pdf`), `results/numbers.json`
  - `pilot_novelty_reference.py`, `profile_runtime.py` - small pilot / timing scripts mentioned in the text
- `data/raw/SOURCES.md`, `data/raw/md5.txt` - provenance of the raw data (URLs, commit hashes, checksums)
- `results/` - raw result tables (`xd_*.csv`, `os_*.csv`), `archive_allref/` = ablation run with all-traffic novelty reference

## Reproduce
```
pip install scikit-learn==1.9.1 lightgbm==4.7.0 pandas pyarrow scipy matplotlib
cd code
python prepare.py            # ~10 min, needs data/raw
python run_experiments.py os # ~26 min on 4 CPU cores
python run_experiments.py xd # ~38 min on 4 CPU cores
python make_figures.py
cd ../manuscript && latexmk -pdf main.tex
```
Run the two experiments one after the other; running them in parallel oversubscribes LightGBM threads and is dramatically slower.

## Revision analyses
All revision scripts reuse the stored scores of `revision_dump.py` and write to `results/revision/`; `revision_macros.py` and `revision_tables.py` turn these files into the numbers and tables of the revised manuscript.
- `revision_dump.py` - stores per-run detector and novelty scores; `revision_analysis.py` - budget rules, priorities, streams
- `revision_extra.py` - temporal (T1, T2), tool-stable features, held-out groups, reference size, subsampling, cost
- `revision_adapt.py`, `revision_adapt_ctrl.py` - adaptation to a new network, feedback loop and its controls (random labels, uncertainty sampling, label noise, unlabelled flows added as benign)
- `revision_mixed.py` - held-out families arriving among known attacks; `revision_novalt.py` - alternative novelty scores
- `revision_models_wf.py` - weighted Fisher priority for all four detectors; `revision_ties.py` - ties at the p-value floor
- `revision_gamma.py` - step size of the online update; `revision_shards_het.py` - online budget on heterogeneous shards
- `revision_stream.py` - streaming throughput, shards, retraining and index rebuild; `revision_stream_profile.py` - per-stage cost
- `revision_summary*.py` - per-run summaries and paired tests

Run the scripts one at a time on a 4-core machine; several LightGBM jobs at once oversubscribe the cores.

Code repository: https://github.com/osmancancet/nids-budgeted-triage
