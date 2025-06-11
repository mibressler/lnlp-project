# OPTS on Claudette

This folder contains a minimal adaptation of the OPTS prompt-optimization approach for the Claudette dataset. It relies on the project utilities `utils.llm` and `utils.dataset` and keeps only the small subset of code required to demonstrate the method.

Run the example optimisation loop with:

```bash
python main.py
```

The script samples a small validation subset from the dataset, proposes new instructions with the APET template and keeps the highest scoring one based on binary accuracy.
