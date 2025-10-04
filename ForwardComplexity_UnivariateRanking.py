
# 03/10/2025
# El ranking univariante usando medidas de complejidad funciona fatal
# Vamos a hacer una espeecie de forward selection siguiendo ese ranking y evaluando la evolución de la complejidad de forma multivariante
# La idea es parar cuando la disminución de la complejidad no merezca la pena


import numpy as np
import pandas as pd
import re

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

from All_measures import *
import matplotlib.pyplot as plt
import seaborn as sns

import os



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

    X = preprocessing.scale(X)
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

    return df, y, dict_info_feature




# dataset_name = 'rpueba'
# save_csv = True
def univariate_complexity(X, y, measures=["Hostility", "N1", "kDN"],
                          save_csv=False, path="Results_UnivariateRanking_CM",
                          dataset_name=None):
    """
    Calcula la complejidad de forma univariante para cada feature, tanto a nivel dataset como instancia.

    Returns
    -------
    df_results : DataFrame con complejidad por feature y nivel (dataset, class_x, ...)
    dataset_vals : DataFrame con complejidad dataset-level (ranking global)
    instance_vals : DataFrame con complejidad instancia-level por feature
    """
    results = []
    inst_results = []

    for feature in X.columns:
        datos = pd.DataFrame({feature: X[feature], "y": y})
        df_measures, df_classes, extras = all_measures(datos, save_csv=False, path_to_save=None, name_data=feature)

        # --- dataset / class level ---
        df_subset = df_classes.loc[:, measures].copy()
        df_subset["feature"] = feature
        df_subset["level"] = df_subset.index  # dataset, class_0, class_1, ...
        results.append(df_subset)

        # --- instance level ---
        df_inst = df_measures[measures].copy()
        df_inst["feature"] = feature
        df_inst["instance_id"] = df_inst.index
        inst_results.append(df_inst)

    # juntar resultados dataset/class level
    df_results = pd.concat(results).set_index(["feature", "level"]).sort_index()
    dataset_vals = df_results.xs("dataset", level="level")[measures]

    # juntar instancia-level
    instance_vals = pd.concat(inst_results) if inst_results else None

    # guardar si procede
    if save_csv and dataset_name:
        fname = f"{path}/{dataset_name}_featuresComplexityRanking.csv"
        df_results.to_csv(fname)
        if instance_vals is not None:
            fname2 = f"{path}/{dataset_name}_featuresComplexityInstances.csv"
            instance_vals.to_csv(fname2, index=False)

    return df_results, dataset_vals, instance_vals

# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
#                                                      n_redundant_nonlinear=2,
#                                 flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)
#
# k = len(dict_info_feature["informative"])
# feature_names = X.columns.tolist()


def forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=False, path="Results_ForwardComplexity",
                                dataset_name=None):
    """
    Evalúa complejidad univariante y acumulada (forward) tanto a nivel dataset como instancia.
    """
    # Ranking univariante
    df_results, dataset_vals = univariate_complexity(X, y, measures=measures, save_csv=False, dataset_name=dataset_name)

    # Ranking global (ordenar por media de complejidad o medida específica)
    ranking = dataset_vals.mean(axis=1).sort_values().index.tolist()


    # Complejidad instancia-level acumulada (forward)
    instance_forward = []
    for k in range(1, len(ranking ) +1):
        subset = ranking[:k]
        datos = pd.concat([X[subset], pd.Series(y, name="y")], axis=1)
        _, df_classes, extras = all_measures(datos, save_csv=False, path_to_save=None, name_data=f"subset_{k}")
        if "instances" in extras:
            df_inst = extras["instances"][measures].copy()
            df_inst["subset_k"] = k
            df_inst["variables_incluidas"] = ",".join(subset)
            df_inst["instance_id"] = df_inst.index
            instance_forward.append(df_inst)
    instance_forward = pd.concat(instance_forward) if instance_forward else None

    # Guardar CSVs
    if save_csv and dataset_name:
        if instance_forward is not None:
            instance_forward.to_csv(f"{path}/{dataset_name}_forward_instancias.csv", index=False)
        dataset_vals.to_csv(f"{path}/{dataset_name}_ranking_dataset.csv")

    return dataset_vals, instance_forward
