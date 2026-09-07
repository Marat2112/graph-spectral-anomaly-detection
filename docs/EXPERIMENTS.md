# Experiments 1-10 / Эксперименты 1-10

Run every command from the repository root. The scripts use fixed random seeds
and write into the corresponding subdirectory of `results/`. Full Monte Carlo
runs can be computationally expensive. The packaged outputs are the preserved
results from the supplied project archive.

Все команды выполняются из корня репозитория. Скрипты используют фиксированные
начальные значения генераторов и записывают результаты в соответствующий
подкаталог `results/`. Полный пересчёт Монте-Карло может быть длительным.
Вложенные результаты сохранены из исходного архива без пересчёта.

| Stage | Purpose | Command | Input dependency | Output directory |
|---|---|---|---|---|
| 1 | Baseline Laplacian-energy detector under IID Gaussian noise | `python -m experiments.run_experiment` | none | `results/experiment_01/` |
| 2 | Sensitivity to graph topology and size | `python -m experiments.experiment_topology` | none | `results/experiment_02/` |
| 3 | Logistic GESNR analysis | `python -m analysis.analyze_gesnr` | `results/experiment_02/topology_summary.csv` | `results/experiment_03/` |
| 4 | Out-of-sample validation of the frozen Experiment 3 model | `python -m experiments.experiment4_validation` | coefficients embedded in script | `results/experiment_04/` |
| 5 | Local node-dependent GESNR | `python -m experiments.experiment5_local_gesnr` | coefficients embedded in script | `results/experiment_05/` |
| 6 | Strict train/validation comparison of three GESNR criteria | `python -m experiments.experiment6_strict_validation` | none | `results/experiment_06/` |
| 6b | Numerical verification of binomial maximum-likelihood fits | `python -m experiments.experiment6B_numerical_verification` | `results/experiment_06/train_grouped.csv` | `results/experiment_06b/` |
| 7 | Fully spectral prediction of detection probability | `python -m experiments.experiment7_spectral_detection` | none | `results/experiment_07/` |
| 8 | General Gaussian covariance: IID, heteroscedastic, correlated noise | `python -m experiments.experiment8_general_covariance` | none | `results/experiment_08/` |
| 9 | Signal geometry, graph energy, and nullspace behavior | `python -m experiments.experiment9_signal_geometry` | none | `results/experiment_09/` |
| 10 | Benchmark of Laplacian, Euclidean, max-absolute, and Mahalanobis detectors | `python -m experiments.experiment10_final_benchmark` | none | `results/experiment_10/` |

## Complete run / Полный запуск

```bash
python run_all.py --list
python run_all.py
```

A partial sequence is also supported:

```bash
python run_all.py --start-at 06 --stop-after 07
```

Stage `06b` must be run after Stage `06`. Stage `03` must be run after Stage
`02`. Stages `04` and `05` document that they use coefficients frozen from
Experiment 3, but they do not read its files at runtime.

Этап `06b` запускается после этапа `06`, а этап `03` - после этапа `02`.
Этапы `04` и `05` используют зафиксированные в коде коэффициенты из
эксперимента 3 и не считывают его файлы при запуске.

## Verification / Проверка

Fast checks that do not overwrite the preserved numerical results:

```bash
python -m unittest discover -s tests -v
python -m analysis.validate_repository
```

Быстрые проверки не пересчитывают кампании Монте-Карло и не изменяют
сохранённые численные результаты.
