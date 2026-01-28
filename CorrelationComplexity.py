## En este script vaamos a calcular la correlación entre la complejidad a nivel instancia que arroja cada  medida de compleejidad de forma univariante
## y lueego filtrar aquellas con correlación alta
## La idea es luego ver con qué variables nos quedamos, calcular su complejidad y la performancee

# 28/01/2026


import numpy as np
import pandas as pd
import random
import re

from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score


from All_measures import *
import matplotlib.pyplot as plt
import seaborn as sns

import os
import glob


## Lo voy a ejecutar con la selección de datos artificiales y algunos reales no enormes



# Función para generar datos sintéticos
def generate_synthetic_dataset(n_samples, n_informative, n_noise,n_redundant_linear, n_redundant_nonlinear,
                                flip_y, class_sep, n_clusters_per_class, weights, random_state=42, noise_std=0.05):
    rng = np.random.RandomState(random_state)

    # Generamos solo informativas + ruido
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        flip_y = flip_y,
        class_sep = class_sep,
        n_clusters_per_class = n_clusters_per_class,
        weights =weights,
        shuffle=False,
        random_state=random_state
    )

    # X = preprocessing.scale(X)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    formulas = {}
    formulas_nonlinear = {}

    # Redundantes lineales
    for j in range(n_redundant_linear):
        idx1, idx2 = rng.choice(n_informative, size=2, replace=False)
        coef1, coef2 = rng.uniform(-2, 2, size=2)
        new_name = f"f{df.shape[1]}"
        new_feature = coef1*df[f"f{idx1}"] + coef2*df[f"f{idx2}"]
        if noise_std > 0:
            new_feature += rng.normal(0, noise_std, size=n_samples)
        df[new_name] = new_feature
        formulas[new_name] = f"{coef1:.2f}*f{idx1} + {coef2:.2f}*f{idx2}" + ("" if noise_std==0 else " + ruido")

    # Redundantes no lineales
    for j in range(n_redundant_nonlinear):
        idx = rng.choice(n_informative, size=2, replace=False)
        func = rng.choice([np.sin, np.cos, np.square, np.exp])
        new_name = f"f{df.shape[1]}"
        new_feature = func(df[f"f{idx[0]}"]) + df[f"f{idx[1]}"]
        if noise_std > 0:
            new_feature += rng.normal(0, noise_std, size=n_samples)
        df[new_name] = new_feature
        formulas_nonlinear[new_name] = f"{func.__name__}(f{idx[0]}) + f{idx[1]}" + ("" if noise_std==0 else " + ruido")

    dict_info_feature = {
        "informative": [f"f{i}" for i in range(n_informative)],
        "noise": [f"f{i}" for i in range(n_informative, n_informative + n_noise)],
        "redundant_linear": list(formulas.keys()),
        "redundant_nonlinear": list(formulas_nonlinear.keys()),
        "formulas_linear": formulas,
        "formulas_nonlinear": formulas_nonlinear
    }

    df[df.columns] = StandardScaler(with_mean=True, with_std=True).fit_transform(df)

    return df, y, dict_info_feature


def compute_gps(y_true, y_pred):
    """
    Calcula GPS para un problema binario.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    TN, FP, FN, TP = cm.ravel()

    # métricas base
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    # F1+ y F1-
    F1_pos = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else 0
    F1_neg = 2 * (NPV * TNR) / (NPV + TNR) if (NPV + TNR) > 0 else 0

    # GPS
    GPS = 2 * (F1_pos * F1_neg) / (F1_pos + F1_neg) if (F1_pos + F1_neg) > 0 else 0
    return GPS




def univariate_instance_complexity(X, y, measure):
    """
    Devuelve DataFrame:
    filas    → instancias
    columnas → variables
    """
    comp_matrix = []

    for col in X.columns:
        datos = pd.DataFrame({col: X[col], "y": y})

        compl_instance, _, _ = all_measures_FS(
            datos, save_csv=False, path_to_save=None, name_data=None
        )

        comp_matrix.append(compl_instance[measure].rename(col))

    return pd.concat(comp_matrix, axis=1)

C_kdn = univariate_instance_complexity(X, y, "kDN")

corr_spearman = C_kdn.corr(method="spearman")
corr_pearson  = C_kdn.corr(method="pearson")


def filter_corr_matrix(C, method="spearman", th=0.9):
    corr = C.corr(method=method).abs()

    selected, removed = [], set()

    for i, col in enumerate(corr.columns):
        if col in removed:
            continue
        selected.append(col)
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > th:
                removed.add(corr.columns[j])

    return selected

feats_kdn = filter_corr_matrix(univariate_instance_complexity(X, y, "kDN"),method="spearman", th=0.7)



def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "acc_train": accuracy_score(y_train, y_pred_train),
        "gps_train": compute_gps(y_train, y_pred_train),
        "acc_test": accuracy_score(y_test, y_pred_test),
        "gps_test": compute_gps(y_test, y_pred_test)
    }


model = KNeighborsClassifier()

def evaluate_complexity_corr_cv(X, y, model,measures=["kDN", "N1", "Hostility"],
                                cv_splits=5,random_state=0,corr_th=0.7):
    """
    CV estratificada donde:
      - el baseline usa todas las variables
      - cada medida de complejidad define su propio filtrado
      - Pearson y Spearman se evalúan por separado
    """

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    records = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n=== FOLD {fold}/{cv_splits} ===")

        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y[tr], y[te]

        feats_all = X.columns.tolist()

        # ======================================================
        # BASELINE: todas las variables
        # ======================================================
        datos_all = X_train.copy()
        datos_all["y"] = y_train
        _, df_classes_all, _ = all_measures_FS(datos_all, save_csv=False, path_to_save=None, name_data=None)
        comp_all = df_classes_all.loc["dataset", measures].to_dict()

        perf_all = evaluate_model(model,X_train[feats_all], y_train,X_test[feats_all], y_test)

        records.append({
            "fold": fold,
            "measure": "ALL",
            "corr_type": "none",
            "n_features": len(feats_all),
            "features": feats_all,
            **perf_all,
            **{f"complexity_all_{m}": comp_all[m] for m in measures},
        })

        # ======================================================
        # POR CADA MEDIDA DE COMPLEJIDAD
        # ======================================================
        for m in measures:
            # 1) complejidad univariante instancia × variables
            C = univariate_instance_complexity(X_train, y_train, m)

            for corr_type in ["pearson", "spearman"]:
                feats_sel = filter_corr_matrix(C, method=corr_type, th=corr_th)

                # --- complejidad dataset (TRAIN) ---
                datos_sel = X_train[feats_sel].copy()
                datos_sel["y"] = y_train
                _, df_classes_sel, _ = all_measures_FS(datos_sel, save_csv=False, path_to_save=None, name_data=None)
                comp_sel = df_classes_sel.loc["dataset", measures].to_dict()

                # --- performance con selección---
                perf_sel = evaluate_model(model,X_train[feats_sel], y_train,X_test[feats_sel], y_test)

                records.append({
                    "fold": fold,
                    "measure": m,
                    "corr_type": corr_type,
                    "n_features": len(feats_sel),
                    "features": feats_sel,
                    **perf_sel,
                    **{f"complexity_sel_{mm}": comp_sel[mm] for mm in measures},
                })

    # ==========================================================
    # Resultados finales
    # ==========================================================
    df_folds = pd.DataFrame(records)

    # agregamos solo las numéricas
    num_cols = df_folds.select_dtypes(include="number").columns
    num_cols = num_cols[1:]
    agg = (df_folds.groupby(["measure", "corr_type"])[num_cols].agg(["mean", "median", "std"]).reset_index())

    return df_folds, agg





### Dataset 2
dataset_name = 'ArtificialDataset2'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                    flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)

datos = pd.DataFrame(X)
datos['y'] = y


#### Dataset 7
dataset_name = 'ArtificialDataset7'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
                                         n_redundant_linear=10,n_redundant_nonlinear=10,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.05)





#### Dataset 12
dataset_name = 'ArtificialDataset12'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=987,noise_std=0.5)





#### Dataset 14
dataset_name = 'ArtificialDataset14'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=95,noise_std=0.5)



#### Dataset 18
dataset_name = 'ArtificialDataset18'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=9462,noise_std=0.5)


#### Dataset 20
dataset_name = 'ArtificialDataset20'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
                                         n_redundant_linear=60,n_redundant_nonlinear=60,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=4556,noise_std=0.5)

list_datasets = ['Australian.csv', 'bands.csv', 'credit-g.csv',
                 'plasma_retinol.csv',
                 'pollution.csv', 'vehicle2.csv', 'diabetic_retinopathy.csv', 'parkinsons.csv']