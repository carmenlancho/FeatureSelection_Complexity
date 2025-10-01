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
import re

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


# Relaciones redundantes entre variables
def get_redundant_feature_relation(dict_info_feature):
    """
    Construye un diccionario {redundante: [informativas de origen]} a partir de las fórmulas.
    """
    redundant_sources = {}

    # lineales
    for r, formula in dict_info_feature.get("formulas_linear", {}).items():
        parents = re.findall(r"f\d+", formula)
        redundant_sources[r] = parents

    # no lineales
    for r, formula in dict_info_feature.get("formulas_nonlinear", {}).items():
        parents = re.findall(r"f\d+", formula)
        redundant_sources[r] = parents

    return redundant_sources

# redundant_sources = get_redundant_feature_relation(dict_info_feature)



def evaluate_univariate_ranking(dataset_vals, dict_info_feature,redundant_sources):
    """
    Evalúa el ranking univariante de cada medida de complejidad respecto
    a las variables informativas conocidas.

    Parameters
    ----------
    dataset_vals : DataFrame
        Subtabla de univariate_complexity a nivel 'dataset'.
    dict_info_feature : dict
        Diccionario de generate_synthetic_dataset con keys: "informative", ...

    Returns
    -------
    summary : DataFrame con métricas de recall por medida.
    """

    measures = dataset_vals.columns.tolist()
    # Tipos de features
    informative = set(dict_info_feature.get("informative", []))
    redundant_lin = set(dict_info_feature.get("redundant_linear", []))
    redundant_nonlin = set(dict_info_feature.get("redundant_nonlinear", []))
    noise = set(dict_info_feature.get("noise", []))
    k = len(informative)

    analysis = []
    for m in measures:
        # ranking ascendente: menor complejidad = mejor
        ranking = dataset_vals[m].sort_values(ascending=True)
        topk = set(ranking.index[:k])

        # Lo que pilla en el top
        caught_info = len(informative & topk)
        caught_lin = len(redundant_lin & topk)
        caught_nonlin = len(redundant_nonlin & topk)
        caught_noise = len(noise & topk)

        # Cobertura teniendo en cuenta redundancias
        # Si no está la variable original pero está alguna redundante que la contiene, la añadimos a la lista
        # de pilladas
        covered_info = set(informative & topk)  # informativas directas
        for r in (topk & (redundant_lin | redundant_nonlin)):
            if r in redundant_sources:
                covered_info = covered_info | set(redundant_sources[r])  # añadimos sus "padres" informativos

        extended_recall = len(covered_info) / len(informative) if informative else 0

        redundancy_info = {f: False for f in topk}

        # Iteramos sobre los features del top que son redundantes
        # Redundantes en el sentido de combinación de otras que ya están en el top seleccionado
        for r in topk:
            if r in redundant_sources:
                r = 'f12'
                # si todas las f ya están en el top, entonces es redundante
                if all(src in topk for src in redundant_sources[r]):
                    redundancy_info[r] = True
        # OJO QUE ES NORMAL QUE SALGAN REDUNDANTES PORQUE ESTO ES UNIVARIANTE


        analysis.append({
            "measure": m,
            "informative_total": len(informative),
            "caught_informative": caught_info,
            "recall_informative": caught_info / len(informative) if informative else 0,
            "extended_recall_informative": extended_recall,
            "caught_redundant_linear": caught_lin,
            "caught_redundant_nonlinear": caught_nonlin,
            "caught_noise": caught_noise,
            'redundant_f': sum(redundancy_info.values()),
            'redundant_f_%': sum(redundancy_info.values())/len(redundancy_info.values())
        })

    summary = pd.DataFrame(analysis).set_index("measure")
    return summary






X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
                                                     n_redundant_nonlinear=2,
                                flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)

# Calcular complejidad univariante
df_results, dataset_vals = univariate_complexity(X, y, measures=["Hostility", "N1", "kDN"])
# info de las redundantes
redundant_sources = get_redundant_feature_relation(dict_info_feature)
# Obtener ranking de informativas
summary = evaluate_univariate_ranking(dataset_vals, dict_info_feature,redundant_sources)

