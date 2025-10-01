#####################################################################################
####   RANKING DE VARIABLES EVALUANDO LA COMPLEJIDAD DE FORMA UNIVARIANTE      ######
#####################################################################################

# 01/10/2025

# Hemos visto en FS_ComplexityExploration.ipynb que las medidas de complejidad Hostility, N1 y kDN
# son capaces de, dados distintos subsets de features de un dataset, discernir cuál es el subset de variables informativas
# dado quee le otorgan menor complejidad

# Ahora queremos ser capaces de identificarlas
# Para ello, comenzamos evaluando la complejidad de cada feature univariantemente y estableciendo un ranking


import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

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
def univariate_complexity(X, y, measures=["Hostility", "N1", "kDN"], save_csv=False, path="Results_UnivariateComplexity", dataset_name=None):
    """
    Calcula la complejidad de forma univariante para cada feature

    Parameters
    ----------
    X : DataFrame
        Variables.
    y : array-like
        Etiquetas.
    measures : list
        Lista de medidas de complejidad a considerar.
    save_csv : bool
        Si True, guarda resultados en un CSV.
    path : str
        Carpeta donde guardar el CSV.
    dataset_name : str
        Nombre del dataset (para el CSV).

    Returns
    -------
    df_results : DataFrame con complejidad por feature y ranking.
    """

    results = []
    # feature = 'f0'
    for feature in X.columns:
        datos = pd.DataFrame({feature: X[feature], "y": y})
        df_measures, df_classes, extras = all_measures(datos, save_csv=False, path_to_save=None, name_data=feature)

        # nos quedamos con las medidas seleccionadas
        df_subset = df_classes.loc[:, measures].copy()
        df_subset["feature"] = feature
        df_subset["level"] = df_subset.index  # dataset, class_0, class_1, ...
        results.append(df_subset)

    # juntamos
    df_results = pd.concat(results)
    df_results = df_results.set_index(["feature", "level"]).sort_index()

    dataset_vals = df_results.xs("dataset", level="level")[measures]

    if save_csv and dataset_name:
        fname = f"{path}/{dataset_name}_featuresComplexityRanking.csv"
        df_results.to_csv(fname)

    return df_results, dataset_vals


X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
                                                     n_redundant_nonlinear=2,
                                flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)

# Número de features informativas como k
k = len(dict_info_feature["informative"])
feature_names = X.columns.tolist()

