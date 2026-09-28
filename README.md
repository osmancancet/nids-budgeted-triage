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

## Before submission (author TODO)
- Fill in affiliations, e-mail, and check CRediT roles and acknowledgements (search for `TODO(author)`).
- Code repository: https://github.com/osmancancet/nids-budgeted-triage (URL already in the manuscript).
- Read and verify everything; adapt the generative-AI declaration to your actual use.
