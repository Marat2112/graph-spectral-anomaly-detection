# Repository and data-quality audit

## Technical summary

The source archive is suitable for publication after removing environment
artifacts and repairing its reproducibility metadata. All 14 Python sources,
72 CSV tables, 41 PNG figures, and every Experiment 1-10 result directory were
preserved. The final DOI-bearing manuscript was copied without byte changes.

The only material reproducibility defect was the original dependency file: it
was unpinned and omitted `scikit-learn`, although Experiments 7-9 import it.
This package supplies a tested, pinned Python 3.10 environment. Scientific
formulas, parameters, random seeds, fixed fitted coefficients, tables, figures,
and conclusions were not altered.

## Scope and provenance

- **Source:** `HTI_AI_Ufa_2026.zip`
- **Source size:** 137,177,097 bytes
- **Source SHA-256:** `df528d0d5c2d995764ff006c683ed5ec484b7466689d58de105fcccdefa45d88`
- **Archive entries:** 12,907
- **Uncompressed size:** 383,268,208 bytes
- **Archive path safety:** no absolute paths or parent-directory traversal
- **Intended use:** reproducible research software, archived Monte Carlo
  outputs, and a separate Zenodo dataset deposit
- **Expected grain:** each raw CSV row is an experimental condition/run record;
  summary tables are aggregated at script-defined condition levels

There are no external observational input data. Synthetic graphs, Gaussian
noise, and anomaly signals are generated in memory from fixed seeds.

## Inventory

| Component | Preserved count | Notes |
|---|---:|---|
| Python source files from archive | 14 | Three shared modules and eleven experiment/analysis programs |
| CSV result tables | 72 | 64,172 data rows in total |
| PNG figures | 41 | Every file has a valid PNG signature |
| Final manuscript | 1 | 39 pages; DOI-bearing version |
| Third-party reference article | 1 | Included with separate CC BY 4.0 notice |

Results by stage:

| Stage | CSV | PNG | Bytes |
|---|---:|---:|---:|
| Experiment 1 | 2 | 3 | 481,403 |
| Experiment 2 | 2 | 4 | 1,224,937 |
| Experiment 3 | 3 | 5 | 1,464,212 |
| Experiment 4 | 4 | 4 | 2,124,982 |
| Experiment 5 | 5 | 4 | 2,197,193 |
| Experiment 6 | 8 | 6 | 9,508,566 |
| Experiment 6b | 19 | 6 | 899,311 |
| Experiment 7 | 4 | 2 | 540,546 |
| Experiment 8 | 7 | 2 | 1,085,068 |
| Experiment 9 | 7 | 2 | 1,388,946 |
| Experiment 10 | 11 | 3 | 3,733,926 |

## Data-quality checks

The repository validator checks directory and script coverage, CSV decoding and
rectangular row shape, empty files, blank cells, non-finite markers, exact row
duplicates, PNG signatures, fixed PDF hashes, legacy output paths, common
secret formats, and Zenodo metadata linkage.

| Check | Result | Interpretation |
|---|---:|---|
| CSV files parsed | 72 / 72 | Pass |
| Empty or malformed CSV files | 0 | Pass |
| Exact duplicate data rows | 0 | Pass at file level |
| Explicit `NaN` / `Inf` markers | 0 | Pass |
| Blank cells | 9,504 | Expected conditional sparsity in Experiment 5 |
| Valid PNG signatures | 41 / 41 | Pass |
| Common credential/secret patterns | 0 | Pass; static pattern scan only |

The blank cells are confined to two Experiment 5 tables:

- `experiment5_raw.csv`: 8,640 blanks in four star-center/star-leaf columns;
- `experiment5_summary.csv`: 864 blanks in eight corresponding aggregate
  columns.

These fields apply only to the `star` topology. They are populated for 25% of
rows and intentionally blank for the three non-star topology groups. This is
conditional missingness, not loss of measurements. **Severity: low; confidence:
high.** The field semantics should remain documented rather than imputed.

No temporal freshness test was applicable: these are fixed simulation outputs,
not a time-partitioned operational feed.

## Code and reproducibility findings

| Severity | Finding | Evidence | Action |
|---|---|---|---|
| High | Original dependency list was incomplete | Experiments 7-9 import `sklearn.metrics`; archive `.venv` had no scikit-learn distribution | Added and tested `scikit-learn==1.7.2` |
| Medium | Original dependencies were unpinned | Five package names only | Pinned a working Python 3.10 set in `requirements.txt` |
| Medium | Result folders were spread across eleven top-level paths | `results`, `results_topology`, `results_experiment4`, etc. | Consolidated under `results/experiment_XX/`; changed path literals only |
| Low | Experiment 3 and 6b have file dependencies | Experiment 3 reads Experiment 2; 6b reads Experiment 6 | Documented and enforced in `run_all.py` ordering |
| Low | Long scripts expose no reduced CLI mode | Full campaigns use thousands of samples | Added smoke tests; did not alter scientific sampling parameters |

All 14 modules imported successfully with Python 3.10.6 and the pinned package
set. Two smoke tests passed: complete module import and a small graph-Laplacian
detector pipeline. `run_all.py --list` resolved every documented module command.
The full Monte Carlo suite was not rerun because doing so would overwrite the
preserved published outputs and is intentionally computationally expensive.

## PDF verification

Final manuscript:

- path: `docs/manuscript/karimov_spectral_anomaly_detection_ru_v1.0.pdf`;
- size: 915,935 bytes;
- SHA-256:
  `99926702a5d342b9425c3c4328242b4dadc7f4b579b6d9623f0d3d00227fb3c8`;
- pages: 39;
- title metadata: `Математическая теория спектрального детектирования аномалий в графовых системах`;
- author metadata: `Каримов Марат Олегович`;
- DOI `10.5281/zenodo.22648161`: present;
- Experiments 1-10, bibliography, appendices, and manuscript ending: present;
- Unicode replacement characters in extracted text: 0.

All 39 pages rendered successfully. Four contact sheets were reviewed for
whole-document coverage, and pages 1, 27, 33, and 39 were additionally inspected
at full rendered size. No clipping, overlapping content, broken tables, black
boxes, or unreadable glyphs were observed. This is rendered-page coverage plus
representative full-page inspection, not a claim of pixel-level inspection of
every page.

## Exclusions from the publication package

The original ZIP remains the preservation source. The following items were not
copied into the public repository:

| Excluded item | Count / size | Reason |
|---|---:|---|
| `.venv/` | 12,761 entries | Platform-specific environment, approximately 355 MB uncompressed, reproducible from requirements, and missing scikit-learn |
| `__pycache__/` and `.pyc` | 3 files | Interpreter cache, not source |
| Superseded manuscript without DOI | 914,180 bytes | Avoids publishing two visually similar 39-page versions; preserved in original ZIP |

The excluded manuscript has SHA-256
`e4044105dea29e3d652519cbea18997346b799a15f8bd4db8737ead62d072154`.
No source data or reported numerical result was deleted.

## Minimal repository changes

Changes were limited to packaging and reproducibility:

1. moved scripts into `experiments/` and `analysis/` without changing numerical
   algorithms;
2. changed only result input/output path literals to use `results/experiment_XX`;
3. added missing/pinned dependencies, metadata, licenses, documentation,
   integrity checks, smoke tests, and an ordered runner;
4. renamed the final and reference PDFs to stable ASCII filenames during copy;
5. retained the original numerical CSV and PNG files byte-for-byte.

## Residual limitations and recommended checks

- Full numerical regeneration remains the definitive reproducibility test and
  should be run on a fresh clone before claiming cross-platform bitwise
  reproduction. Floating-point and optimization outputs can differ slightly
  across operating systems and BLAS implementations.
- The secret scan covers common credential forms but cannot prove absence of
  every possible confidential value. Manual review found no network, database,
  credential, or personal-data inputs in the Python sources.
- The fixed dependency versions reproduce the verified Python 3.10 environment,
  not the incomplete Python 3.14 `.venv` bundled in the source ZIP.
- If the author changes the scientific scripts, regenerate `MANIFEST.sha256`,
  rerun the smoke tests and validator, and publish a new version rather than
  silently replacing v1.0.0.
