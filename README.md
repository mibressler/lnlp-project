# LNLP Seminar

This repository contains various experiments and utilities used for our approach for the LNLP course. The most relevant directories are:

* `dataset/` – helpers for loading the Claudette dataset.
* `utils/` – generic helper functions and prompt templates.
* `baseline_implementations/` – several baseline models (BERT, TF‑IDF and
  heuristic approaches).  Each baseline has its own subfolder and uses the
  `.tsv` files located in the repository-level `dataset/` directory.
* `config.py` – central constants like `DATASET_DIR` for locating data files.

To install the project in editable mode run:

```bash
pip install -r requirements.txt
pip install -e .
```

Some scripts require API keys which should be supplied via environment
variables.  For the LLM utilities, set `OPENROUTER_API_KEY` with your
OpenRouter token.  A `.env.example` file is provided—copy it to `.env` and
add your key. The utilities load this file automatically.
All code should import `DATASET_DIR` from `config.py` when accessing the dataset.
