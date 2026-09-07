"""Validate repository structure and preserved research artifacts.

Run from the repository root with::

    python -m analysis.validate_repository
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIRS = ("analysis", "data", "docs", "experiments", "results")
EXPECTED_MODULES = (
    "experiments/run_experiment.py",
    "experiments/experiment_topology.py",
    "analysis/analyze_gesnr.py",
    "experiments/experiment4_validation.py",
    "experiments/experiment5_local_gesnr.py",
    "experiments/experiment6_strict_validation.py",
    "experiments/experiment6B_numerical_verification.py",
    "experiments/experiment7_spectral_detection.py",
    "experiments/experiment8_general_covariance.py",
    "experiments/experiment9_signal_geometry.py",
    "experiments/experiment10_final_benchmark.py",
)
EXPECTED_PDF_HASHES = {
    "docs/manuscript/karimov_spectral_anomaly_detection_ru_v1.0.pdf":
        "99926702a5d342b9425c3c4328242b4dadc7f4b579b6d9623f0d3d00227fb3c8",
    "docs/references/verdoja_grangetto_2020_graph_laplacian_image_anomaly_detection.pdf":
        "4ac060e7fbaebe1eca0cb6f812eaf914b93b0f3dadb5408fe2bac075f317a0d7",
}
LEGACY_PATHS = (
    "results_topology",
    "results_gesnr",
    "results_experiment4",
    "results_experiment5",
    "results_experiment6\"",
    "results_experiment6B",
    "results_experiment7",
    "results_experiment8",
    "results_experiment9",
    "results_experiment10",
)
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_csv(path: Path) -> dict[str, int | str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)
    if not rows:
        raise ValueError("empty CSV")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("inconsistent row width")
    body = rows[1:]
    blank_cells = sum(cell.strip() == "" for row in body for cell in row)
    nonfinite = 0
    for row in body:
        for cell in row:
            value = cell.strip().lower()
            if value in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
                nonfinite += 1
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "rows": len(body),
        "columns": width,
        "blank_cells": blank_cells,
        "nonfinite_cells": nonfinite,
        "duplicate_rows": len(body) - len({tuple(row) for row in body}),
    }


def main() -> int:
    errors: list[str] = []
    for directory in EXPECTED_DIRS:
        if not (ROOT / directory).is_dir():
            errors.append(f"missing directory: {directory}")
    for module in EXPECTED_MODULES:
        if not (ROOT / module).is_file():
            errors.append(f"missing module: {module}")

    for relative, expected in EXPECTED_PDF_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing PDF: {relative}")
        elif sha256(path) != expected:
            errors.append(f"PDF checksum mismatch: {relative}")

    csv_profiles = []
    for path in sorted((ROOT / "results").rglob("*.csv")):
        try:
            csv_profiles.append(profile_csv(path))
        except Exception as exc:  # report every malformed file in one pass
            errors.append(f"invalid CSV {path.relative_to(ROOT)}: {exc}")

    for path in ROOT.rglob("*.png"):
        if path.stat().st_size < 8 or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"invalid PNG signature: {path.relative_to(ROOT)}")

    text_suffixes = {".py", ".md", ".txt", ".json", ".cff"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py" and path.resolve() != Path(__file__).resolve():
            for legacy in LEGACY_PATHS:
                if legacy in text:
                    errors.append(f"legacy result path in {path.relative_to(ROOT)}: {legacy}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible secret in {path.relative_to(ROOT)}")

    required = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    if required.get("upload_type") != "software":
        errors.append(".zenodo.json upload_type must be software")
    if "10.5281/zenodo.22648161" not in json.dumps(required):
        errors.append("manuscript DOI missing from .zenodo.json")

    manifest_path = ROOT / "MANIFEST.sha256"
    if manifest_path.is_file():
        manifest_entries = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            manifest_entries[relative] = digest
        excluded = {".git", ".venv", "__pycache__"}
        actual_files = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file()
            and path != manifest_path
            and not excluded.intersection(path.relative_to(ROOT).parts)
        }
        if actual_files != set(manifest_entries):
            errors.append("MANIFEST.sha256 file list is stale")
        for relative, expected in manifest_entries.items():
            path = ROOT / relative
            if path.is_file() and sha256(path) != expected:
                errors.append(f"MANIFEST.sha256 mismatch: {relative}")

    summary = {
        "csv_files": len(csv_profiles),
        "csv_rows": sum(int(item["rows"]) for item in csv_profiles),
        "csv_blank_cells": sum(int(item["blank_cells"]) for item in csv_profiles),
        "csv_nonfinite_cells": sum(int(item["nonfinite_cells"]) for item in csv_profiles),
        "csv_duplicate_rows": sum(int(item["duplicate_rows"]) for item in csv_profiles),
        "files_with_blank_cells": [
            item for item in csv_profiles if int(item["blank_cells"]) > 0
        ],
        "files_with_duplicate_rows": [
            item for item in csv_profiles if int(item["duplicate_rows"]) > 0
        ],
        "png_files": len(list((ROOT / "results").rglob("*.png"))),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
