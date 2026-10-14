# Benchmark Data Directory

- **Synthetic Experiments (Tables 2, 3, 4, 5, 7, A1, B1):** Generated on the fly by `src/synthetic_regimes.py`. No external data files are required.
- **Semi-Synthetic `IHDP-X` Benchmark (Table 6 & Fig. 3 Right):** Uses the 100-replication IHDP covariates (`ihdp_npci_1-100.train.npz`). Run `python scripts/download_data.py` from the repository root to download them into this folder.
