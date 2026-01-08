# Guyon et al., Result analysis of the NIPS 2003 feature selection challenge, NIPS 2004
# Dua & Graff, UCI Machine Learning Repository, 2019

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import lil_matrix

BASE_DIR = Path("datasets_uci")
splits = ["train", "valid", "test"]
datasets = ["arcene", "gisette", "madelon", "dexter", "dorothea"]

for name in datasets:
    print("\nProcesando", name)
    for split in splits:
        data_file = BASE_DIR / f"{name}_{split}.data"
        label_file = BASE_DIR / f"{name}_{split}.labels"

        if not data_file.exists():
            continue

        # =========================
        # Leer features
        # =========================
        if name in ["arcene", "gisette", "madelon"]:
            X = np.loadtxt(data_file)
            df = pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])])
        elif name == "dexter":
            lines = open(data_file).readlines()
            X = lil_matrix((len(lines), 20000))
            for i, line in enumerate(lines):
                for pair in line.split():
                    idx, val = pair.split(":")
                    X[i, int(idx)-1] = float(val)
            df = pd.DataFrame(X.toarray(), columns=[f"V{i}" for i in range(X.shape[1])])
        elif name == "dorothea":
            lines = open(data_file).readlines()
            X = lil_matrix((len(lines), 100000))
            for i, line in enumerate(lines):
                for idx in line.split():
                    X[i, int(idx)-1] = 1
            df = pd.DataFrame(X.toarray(), columns=[f"V{i}" for i in range(X.shape[1])])

        # =========================
        # Leer etiquetas solo para train y valid
        # =========================
        if split in ["train", "valid"] and label_file.exists():
            y = pd.read_csv(label_file, header=None)[0].values
            df["y"] = y
        else:
            df["target"] = pd.NA  # Test o labels inexistentes

        # =========================
        # Guardado
        # =========================
        if name in ["dexter", "dorothea"]:
            out_file = BASE_DIR / f"{name}_{split}.parquet"
            df.to_parquet(out_file, index=False)
        else:
            out_file = BASE_DIR / f"{name}_{split}.csv"
            df.to_csv(out_file, index=False)

        print("   guardado:", out_file.name, df.shape)
