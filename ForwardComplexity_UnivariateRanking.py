
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
    instance_vals = pd.concat(inst_results)

    # guardar si procede
    if save_csv and dataset_name:
        # Lo pongo en comentarios porque ya está ejecutado
        # fname = f"{path}/{dataset_name}_featuresComplexityRanking.csv"
        # df_results.to_csv(fname)
        fname2 = f"{path}/{dataset_name}_featuresComplexityInstances.csv"
        instance_vals.to_csv(fname2, index=False)

    return df_results, dataset_vals, instance_vals

# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
#                                                      n_redundant_nonlinear=2,
#                                 flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)
#
# k = len(dict_info_feature["informative"])
# feature_names = X.columns.tolist()

# save_csv = True
# dataset_name = 'prueba'
def forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=False, path="Results_ForwardComplexity",
                                dataset_name=None):
    """
    Evalúa complejidad univariante y acumulada (forward) tanto a nivel dataset como instancia.
    """
    # Ranking univariante
    df_results, dataset_vals, instance_vals = univariate_complexity(X, y, measures=measures, save_csv=True, dataset_name=dataset_name)

    all_forward_inst = []
    all_forward_classes = []

    # Ranking por medida de complejidad
    # m = 'Hostility'
    for m in measures:
        # ranking ascendente: menor complejidad = mejor
        ranking = dataset_vals[m].sort_values(ascending=True)

        # k = 1
        for k in range(1, len(ranking) +1):
            subset = ranking[:k].index
            datos = pd.concat([X[subset], pd.Series(y, name="y")], axis=1)
            df_measures, df_classes, extras = all_measures(datos, save_csv=False, path_to_save=None, name_data=f"subset_{k}")
            # instancia
            df_inst = df_measures[[m]].copy()
            df_inst = df_inst.rename(columns={m: "complexity"})
            df_inst["subset_k"] = k
            df_inst["variables_incluidas"] = ",".join(subset)
            df_inst["instance_id"] = df_inst.index
            df_inst["measure"] = m
            all_forward_inst.append(df_inst)

            # clase y dataset
            df_c = df_classes[measures].copy()
            df_c["subset_k"] = k
            df_c["variables_incluidas"] = ",".join(subset)
            df_c["measure"] = m
            df_c["level"] = df_c.index  # dataset, class_0, class_1
            all_forward_classes.append(df_c)


    all_forward_inst_df = pd.concat(all_forward_inst, ignore_index=True)
    all_forward_classes_df = pd.concat(all_forward_classes, ignore_index=False)

    # Guardar CSVs
    if save_csv and dataset_name:
        all_forward_inst_df.to_csv(f"{path}/{dataset_name}_forward_instancias.csv", index=False)
        all_forward_classes_df.to_csv(f"{path}/{dataset_name}_forward_dataset_classes.csv")

    return all_forward_inst_df, all_forward_classes_df



### Dataset 1
dataset_name = 'ArtificialDataset1'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                        flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)




### Dataset 2
dataset_name = 'ArtificialDataset2'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                    flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

### Dataset 3
dataset_name = 'ArtificialDataset3'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=25,n_noise=5,
                                         n_redundant_linear=7,n_redundant_nonlinear=8,
                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=0,noise_std=0.05)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

### Dataset 4
dataset_name = 'ArtificialDataset4'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=15,n_noise=15,
                                         n_redundant_linear=4,n_redundant_nonlinear=5,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=10,noise_std=0.01)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

### Dataset5
dataset_name = 'ArtificialDataset5'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=25,n_noise=15,
                                         n_redundant_linear=8,n_redundant_nonlinear=7,
                                     flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=10,noise_std=0.05)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

#### Dataseet 6
dataset_name = 'ArtificialDataset6'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=10000,n_informative=8,n_noise=15,
                                         n_redundant_linear=4,n_redundant_nonlinear=5,
                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.01)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)


#### Dataset 7
dataset_name = 'ArtificialDataset7'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
                                         n_redundant_linear=10,n_redundant_nonlinear=10,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.05)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

#### Dataset 8
dataset_name = 'ArtificialDataset8'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=40,n_noise=15,
                                         n_redundant_linear=15,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=86785,noise_std=0.1)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

#### Dataset 9
dataset_name = 'ArtificialDataset9'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=10,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=20,
                                        flip_y=0, class_sep=0.7, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=959,noise_std=0.3)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

#### Dataset 10
dataset_name = 'ArtificialDataset10'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=2000,n_informative=6,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=0.8, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=959,noise_std=0.3)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

#### Dataset 11
dataset_name = 'ArtificialDataset11'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=20,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=0.6, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=959,noise_std=0.1)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)

#### Dataset 12
dataset_name = 'ArtificialDataset12'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=987,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)



#### Dataset 13
dataset_name = 'ArtificialDataset13'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.4],
                                                     random_state=95,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)


#### Dataset 14
dataset_name = 'ArtificialDataset14'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=95,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)


#### Dataset 15
dataset_name = 'ArtificialDataset15'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=40,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=78,noise_std=0.1)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)


#### Dataset 16
dataset_name = 'ArtificialDataset16'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=50,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.2],
                                                     random_state=756,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)



#### Dataset 17
dataset_name = 'ArtificialDataset17'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.6, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=756,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)



#### Dataset 18
dataset_name = 'ArtificialDataset18'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=9462,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)




#### Dataset 19
dataset_name = 'ArtificialDataset19'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=150,n_noise=50,
                                         n_redundant_linear=50,n_redundant_nonlinear=50,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=655,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)



#### Dataset 20
dataset_name = 'ArtificialDataset20'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
                                         n_redundant_linear=60,n_redundant_nonlinear=60,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=4556,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)



#### Dataset 21
dataset_name = 'ArtificialDataset21'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
                                         n_redundant_linear=100,n_redundant_nonlinear=100,
                                        flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
                                                     random_state=996,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)




#### Dataset 22
dataset_name = 'ArtificialDataset22'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=500,n_noise=150,
                                         n_redundant_linear=150,n_redundant_nonlinear=150,
                                        flip_y=0.2, class_sep=0.7, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=996,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)



#### Dataset 23
dataset_name = 'ArtificialDataset23'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=5000,n_noise=1500,
                                         n_redundant_linear=1500,n_redundant_nonlinear=1500,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=996,noise_std=0.5)
forward_complexity_analysis(X, y, measures=["Hostility" ,"N1" ,"kDN"],
                                save_csv=True, path="Results_ForwardComplexity",
                                dataset_name=dataset_name)


