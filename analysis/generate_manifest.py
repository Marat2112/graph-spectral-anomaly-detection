"""Generate MANIFEST.sha256 for all publishable repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1]
)
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    ]
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} entries to {OUTPUT.name}")


if __name__ == "__main__":
    main()
