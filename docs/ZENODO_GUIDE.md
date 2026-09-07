# GitHub and Zenodo publication guide

This guide contains publication-ready metadata but does not invent a software
or dataset DOI. Zenodo will assign those identifiers only when the respective
records are published.

## 1. GitHub repository

Create an empty public repository with these fields:

- **Repository name:** `graph-spectral-anomaly-detection`
- **Description:** `Reproducible Python experiments for spectral anomaly detection in graph systems using Laplacian-energy statistics.`
- **Visibility:** `Public`
- **Initialize with README / .gitignore / license:** leave all unchecked,
  because these files are already present.
- **Suggested topics:** `graph-signal-processing`, `anomaly-detection`,
  `graph-laplacian`, `spectral-analysis`, `monte-carlo`,
  `reproducible-research`, `python`.

Upload or push the **contents of the repository folder**, so `README.md`,
`CITATION.cff`, `.zenodo.json`, `experiments/`, `analysis/`, `data/`, `results/`,
and `docs/` appear at the repository root. Do not commit the outer ZIP, `.venv`,
`__pycache__`, local credentials, or the separate dataset ZIP.

Command-line option after creating the empty GitHub repository:

```bash
git init
git branch -M main
git add .
git commit -m "Release reproducible research package v1.0.0"
git remote add origin https://github.com/YOUR-USERNAME/graph-spectral-anomaly-detection.git
git push -u origin main
```

Replace `YOUR-USERNAME`; it has deliberately not been guessed here.

## 2. Zenodo Software record through GitHub

Use Zenodo's GitHub integration for the software record to avoid creating a
duplicate manual software upload:

1. In Zenodo, connect the GitHub account.
2. Open the Zenodo **GitHub** page, select **Sync now**, find the repository,
   and enable it.
3. On GitHub, create a release with tag `v1.0.0`, target `main`, title
   `Graph Spectral Anomaly Detection v1.0.0`, and release notes such as
   `Initial reproducible release containing Experiments 1-10, preserved CSV and PNG outputs, the final manuscript, and validation tooling.`
4. Publish the GitHub release. Zenodo should ingest it as a new Software record.
5. Before publishing in Zenodo, verify every field below. The included
   `.zenodo.json` supplies the Zenodo-specific metadata; `CITATION.cff` supplies
   GitHub's citation panel.

### Exact Software metadata

- **Resource type:** `Software`
- **Title:** `Graph Spectral Anomaly Detection: Reproducible Experiments 1-10`
- **Publication date:** `2026-09-07` (or the actual GitHub release date if later)
- **Creator name:** `Karimov, Marat Olegovich`
- **Affiliations:** `Ufa University of Science and Technology`; `Ufa College of Statistics, Informatics and Computer Engineering`
- **ORCID:** leave blank unless the author supplies a verified ORCID
- **Version:** `1.0.0`
- **Language:** `English`
- **Access:** `Open`
- **License:** `MIT License`
- **Publisher:** leave Zenodo's default value
- **Funding:** leave blank unless a real grant applies
- **Keywords:** `graph signal processing`; `anomaly detection`; `graph Laplacian`;
  `spectral detection`; `Monte Carlo simulation`; `reproducible research`;
  `Python`
- **Related work identifier:** `10.5281/zenodo.22648161`
- **Related work relation:** `Is supplement to`
- **Related work resource type:** `Publication`

**Software description:**

> Python research software and preserved Monte Carlo outputs for Experiments
> 1-10 accompanying the Russian-language research report *Mathematical Theory
> of Spectral Anomaly Detection in Graph Systems*. The package implements
> Laplacian-energy anomaly detection, graph-topology studies, global and local
> GESNR analyses, strict out-of-sample validation, numerical verification of
> binomial maximum-likelihood fits, spectral distribution calculations,
> general Gaussian covariance models, signal-geometry experiments, and a fair
> four-detector benchmark. The related research report is available at
> https://doi.org/10.5281/zenodo.22648161. No software DOI is embedded in this
> release because it is assigned only on publication.

Do not type `10.5281/zenodo.22648161` into the Software DOI field: it identifies
the manuscript, not the code. It belongs only under **Related works**.

## 3. Separate Zenodo Dataset record

Create a separate **New upload** manually and upload the prepared file
`graph-spectral-anomaly-detection-dataset-v1.0.zip`.

### Exact Dataset metadata

- **Resource type:** `Dataset`
- **Title:** `Graph Spectral Anomaly Detection: Monte Carlo Results for Experiments 1-10`
- **Publication date:** `2026-09-07` (or the actual dataset publication date)
- **Creator name:** `Karimov, Marat Olegovich`
- **Affiliations:** the same two affiliations as for Software
- **ORCID:** leave blank unless verified
- **Version:** `1.0.0`
- **Language:** `English`
- **Access:** `Open`
- **License:** `Creative Commons Attribution 4.0 International`
- **Publisher:** leave Zenodo's default value
- **Funding:** leave blank unless a real grant applies
- **Keywords:** the Software keywords plus `synthetic data`, `simulation results`,
  `graph anomaly benchmark`
- **Related work identifier:** `10.5281/zenodo.22648161`
- **Related work relation:** `Is supplement to`
- **Related work resource type:** `Publication`

**Dataset description:**

> Preserved tabular and graphical outputs from ten computational experiments
> on spectral anomaly detection in graph systems. The dataset contains 72 CSV
> tables and 41 PNG figures produced from synthetic graphs, Gaussian noise,
> and anomaly signals generated with fixed random seeds. No personal,
> confidential, or externally collected observational data are included. The
> experiments cover baseline calibration, topology sensitivity, GESNR models,
> out-of-sample validation, local criteria, general covariance, signal geometry,
> nullspace behavior, and detector benchmarking. The related research report
> is https://doi.org/10.5281/zenodo.22648161. The software DOI will be added as
> a related work after the Software record is published.

After the Software DOI exists, add it to the Dataset **Related works** with the
relation `Is derived from` and resource type `Software`. Do not guess this DOI.

## 4. Close the citation links after publication

After both new records are public:

1. Edit the manuscript record `10.5281/zenodo.22648161` metadata.
2. Add the Software DOI and Dataset DOI under **Related works**, each with
   relation `Is supplemented by` and the matching resource type.
3. Add the Dataset DOI to the Software record as `Is supplemented by` if you
   want the reciprocal link.
4. In a later repository release, replace the README placeholders with the real
   DOI badges and add the Software DOI to `CITATION.cff`. Keep `v1.0.0`
   immutable; make that metadata update as a new patch release if needed.

## Official documentation

- Zenodo record metadata: https://help.zenodo.org/docs/deposit/describe-records/
- Zenodo resource types: https://help.zenodo.org/docs/deposit/describe-records/resource-type/
- Zenodo licenses: https://help.zenodo.org/docs/deposit/describe-records/licenses/
- Zenodo GitHub integration: https://help.zenodo.org/docs/github/enable-repository/
- Zenodo software metadata: https://help.zenodo.org/docs/github/describe-software/
- GitHub import instructions: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- GitHub releases: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
