# NLP-Based Smart FAQ Retrieval and Question Answering System

This project is a corpus-configurable FAQ retrieval engine for the CSE 4122
Natural Language Processing Laboratory course. Phase 1 uses TF-IDF vectors and
cosine similarity to retrieve an existing answer; it does not generate answers.

The final offline demonstration will support independently indexed University
and E-commerce FAQ corpora. Work is being added in verified checkpoints so the
TF-IDF baseline remains understandable and reproducible.

## Current checkpoint

The reusable TF-IDF baseline works with small University and E-commerce
fixtures. It includes corpus discovery, schema validation, configurable
preprocessing, cosine-similarity top-k retrieval, threshold rejection, automated
tests, and a terminal demonstration. Full source-based corpora and formal
evaluation are added in subsequent checkpoints.

Run the small baseline with:

```powershell
python main.py
```

## Development environment

- Python 3.12
- Project-local `.venv`
- Windows PowerShell and VS Code
