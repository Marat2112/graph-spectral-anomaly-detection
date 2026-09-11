# Graph Spectral Anomaly Detection

Reproducible Python research package for **Experiments 1-10** accompanying
Marat Olegovich Karimov's Russian-language research report *Mathematical Theory
of Spectral Anomaly Detection in Graph Systems*.

**Related manuscript DOI:**
[10.5281/zenodo.22648161](https://doi.org/10.5281/zenodo.22648161)

**DOI software dataset:** 
[10.5281/zenodo.22648387](https://doi.org/10.5281/zenodo.22648387)

## English

### What this repository contains

The project investigates anomaly detection in graph systems using the
Laplacian quadratic statistic

\[
Q(x)=x^\mathsf{T}Lx.
\]

It contains the original computational logic, preserved numerical outputs,
figures, and the final 39-page manuscript supplied in the source archive. The
experiments cover baseline calibration, topology effects, global and local
GESNR, strict validation, numerical optimization checks, spectral detection,
general Gaussian covariance, signal geometry, nullspace behavior, and a fair
comparison of four detectors.

The scope is deliberately bounded: the Laplacian-energy detector is treated as
a spectrally selective detector of graph inconsistency, not as a universal
anomaly detector. Existing formulas and reported numerical results were not
scientifically rewritten during repository preparation.

### Repository layout

```text
.
|-- README.md
|-- LICENSE                         # MIT license for Python software
|-- CITATION.cff                    # GitHub citation metadata; no software DOI yet
|-- .zenodo.json                    # Zenodo Software metadata and manuscript link
|-- requirements.txt                # Tested Python 3.10 dependency set
|-- run_all.py                      # Ordered experiment runner
|-- config.py                       # Shared Experiment 1 configuration
|-- detector.py                     # Shared detector functions
|-- graph_model.py                  # Shared graph and Laplacian functions
|-- experiments/                    # Experiment programs 1, 2, 4-10 and 6b
|-- analysis/                       # Experiment 3 and repository validation
|-- data/                           # Data provenance; no external raw dataset
|-- results/                        # Preserved CSV and PNG outputs by experiment
|-- docs/
|   |-- manuscript/                 # Final DOI-bearing manuscript, unchanged
|   |-- references/                 # Open-access third-party reference article
|   |-- EXPERIMENTS.md              # Script-to-experiment map and commands
|   |-- REPOSITORY_AUDIT.md         # Source audit and data-quality evidence
|   |-- ZENODO_GUIDE.md             # Exact GitHub and Zenodo publication fields
|   `-- THIRD_PARTY_NOTICES.md
|-- licenses/DATA-CC-BY-4.0.txt     # Data/results license declaration
|-- tests/test_smoke.py
`-- MANIFEST.sha256                 # File integrity manifest
```

### Reproducible environment

The publication package was tested with Python **3.10.6** and the exact package
versions in `requirements.txt`. Create a clean environment from the repository
root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The supplied archive contained a local `.venv`, but it was intentionally not
published: it was platform-specific, very large, and incomplete for
Experiments 7-9 because it lacked `scikit-learn`.

### Verify before running

These checks do not rerun the Monte Carlo campaigns or overwrite preserved
results:

```bash
python -m unittest discover -s tests -v
python -m analysis.validate_repository
python run_all.py --list
```

### Run the experiments

Run all stages from the repository root:

```bash
python run_all.py
```

Run an individual stage, for example:

```bash
python -m experiments.run_experiment
python -m experiments.experiment_topology
python -m analysis.analyze_gesnr
python -m experiments.experiment10_final_benchmark
```

The complete command map and input dependencies are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Full experiments use thousands of
Monte Carlo samples across multiple conditions and may take substantial time.
Running them writes into `results/` and can replace files with the same names;
retain the packaged version or use version control if you need a strict
before/after comparison.

### Data and preserved results

There is no external observational dataset. Graphs, Gaussian noise, and anomaly
signals are generated synthetically in memory from fixed seeds. The package
preserves:

- 72 CSV tables;
- 41 PNG figures;
- results for Experiments 1-10, including the Experiment 6b numerical check.

The data audit checks parseability, row shape, blank cells, non-finite markers,
image signatures, expected files, and fixed PDF checksums. See
[`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md).

### Manuscript integrity

The final manuscript was copied without editing to
`docs/manuscript/karimov_spectral_anomaly_detection_ru_v1.0.pdf`. Its SHA-256 is:

```text
99926702a5d342b9425c3c4328242b4dadc7f4b579b6d9623f0d3d00227fb3c8
```

It contains the manuscript DOI `10.5281/zenodo.22648161`. The full repository
manifest provides checksums for every published file.

### Citation

GitHub can read `CITATION.cff` and display **Cite this repository**. Until the
software record is published, cite version `1.0.0`, the author, repository
title, release date, and GitHub release URL, together with the related
manuscript DOI. Do not reuse the manuscript DOI as a software DOI.

### Licensing choice

- **Python software:** MIT License. It is short, permissive, OSI-approved, and
  allows academic and practical reuse while retaining attribution and the
  warranty disclaimer.
- **Project-generated CSV and PNG results:** CC BY 4.0. This license is designed
  for research outputs and permits reuse with attribution and disclosure of
  changes.
- **Manuscript and third-party article:** governed by their own record or
  article terms; they are explicitly excluded from the MIT grant. See
  `docs/THIRD_PARTY_NOTICES.md`.

### Publication

Follow [`docs/ZENODO_GUIDE.md`](docs/ZENODO_GUIDE.md). In brief: publish this
repository on GitHub, enable the repository in Zenodo, create GitHub release
`v1.0.0` for the Software record, and upload the separate dataset archive as a
Dataset record. Both records should link to manuscript DOI
`10.5281/zenodo.22648161` as `Is supplement to`.

---

## Русский

### Назначение репозитория

Это воспроизводимый исследовательский пакет к русскоязычной рукописи Каримова
Марата Олеговича «Математическая теория спектрального детектирования аномалий в
графовых системах». DOI основной рукописи:
[10.5281/zenodo.22648161](https://doi.org/10.5281/zenodo.22648161).

В репозитории сохранены вычислительные программы, исходные таблицы результатов,
итоговые рисунки и финальный PDF. Отдельные DOI программного обеспечения и
набора данных пока не существуют и намеренно нигде не указаны.

### Что было сохранено

- все 14 исходных Python-файлов;
- все 72 CSV-таблицы и 41 PNG-рисунок;
- результаты экспериментов 1-10 и дополнительной проверки 6b;
- финальная 39-страничная рукопись с DOI без изменения байтов;
- открытая статья-источник с отдельным уведомлением о лицензии.

Научные формулы, параметры моделирования, фиксированные коэффициенты и
численные результаты не перерабатывались. Изменены только расположение файлов,
пути каталогов результатов и список зависимостей, необходимый для фактического
запуска.

### Быстрый запуск

Создайте чистое виртуальное окружение, установите зависимости и выполните
проверки:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m analysis.validate_repository
python run_all.py --list
```

Полный пересчёт:

```powershell
python run_all.py
```

Команды отдельных экспериментов, зависимости между этапами и каталоги вывода
перечислены в [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Запуск следует
выполнять из корня репозитория.

### Данные

Внешнего набора наблюдений нет. Графы, шум и аномальные сигналы генерируются
программно с фиксированными начальными значениями. Поэтому каталог `data/`
содержит описание происхождения, а численные таблицы и рисунки находятся в
`results/`.

### Лицензии

Код распространяется по разрешительной лицензии MIT. Для созданных автором
таблиц и рисунков выбрана CC BY 4.0, поскольку она лучше подходит для открытых
исследовательских данных и требует корректной атрибуции. Эти лицензии не
переопределяют условия использования рукописи и сторонней статьи.

### Публикация

Точные поля для GitHub, Zenodo Software и Zenodo Dataset приведены в
[`docs/ZENODO_GUIDE.md`](docs/ZENODO_GUIDE.md). После публикации нужно связать
новые DOI с DOI рукописи через раздел **Related works**, но не использовать DOI
рукописи как идентификатор программного обеспечения или набора данных.
