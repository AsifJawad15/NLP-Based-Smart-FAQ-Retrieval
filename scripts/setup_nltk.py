"""Download the NLTK resources this project needs into a project-local folder.

Run this once after installing the requirements. Everything afterwards works
without an internet connection.

Example:
    python scripts/setup_nltk.py
"""

from __future__ import annotations

from pathlib import Path

import nltk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLTK_DIR = PROJECT_ROOT / ".nltk_data"

# "stopwords" powers the stop-word configuration and "wordnet"/"omw-1.4" power
# the lemmatised configuration that the University corpus selects.
REQUIRED_RESOURCES = ["stopwords", "wordnet", "omw-1.4"]


def main() -> None:
    """Download each required corpus into the project-local NLTK folder."""

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    for resource in REQUIRED_RESOURCES:
        print(f"Downloading {resource} into {NLTK_DIR} ...")
        if not nltk.download(resource, download_dir=str(NLTK_DIR), quiet=True):
            raise SystemExit(f"Failed to download NLTK resource: {resource}")
    print("NLTK resources are ready. The project now runs offline.")


if __name__ == "__main__":
    main()
