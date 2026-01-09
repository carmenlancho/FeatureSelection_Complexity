# 29/12/2025

###################################################################################################
#####                        FS COMPLEXITY BASED, DISTRIBUTED VERSION, CV                         #####
###################################################################################################



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





########################################################################################################################
########                VERSION DISTRIBUTED QUITANDO NEGATIVAS Y FIJANDO LAS DE COMPLEJIDAD ALTA                ########
########################################################################################################################

# n_replicas=2

def distributed_complexity_random_neg_out_high_fixed(X, y, dataset_name, n_replicas, m_vars,
                                                measures=["Hostility", "N1", "kDN"],
                                                filter_corr=True, corr_th=0.9, corr_method="pearson",
                                                random_state=0, save_csv=False, path='Results_FS_Distributed_CV',
                                                tau=0.01): # Umbral para fijar variables por importancia alta

    np.random.seed(random_state)
    random.seed(random_state)

    # Filtro por correlación
    if filter_corr:
        corr = X.corr(method=corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        X = X.drop(columns=to_drop)

    variables = X.columns.tolist()

    # Inicialización
    importances = {m: pd.Series(0.0, index=variables) for m in measures}
    importances_norm = {m: pd.Series(0.0, index=variables) for m in measures}
    count_vars = pd.Series(0.0, index=variables)
    removed_vars = {m: pd.Series(0.0, index=variables) for m in measures}
    fixed_vars = {m: pd.Series(0.0, index=variables) for m in measures}

    # Conjuntos activos y de control
    active_vars = {m: set(variables) for m in measures}
    permanently_removed = {m: set() for m in measures}
    permanently_fixed = {m: set() for m in measures}

    # Bucle principal
    for rep in range(n_replicas):
        for m in measures:
            print(f"[{m}] Réplica {rep + 1}/{n_replicas}")


            # Variables disponibles (ni eliminadas ni fijadas)
            available_vars = list(active_vars[m] - permanently_removed[m])
            # print('Available variables:')
            # print(available_vars)

            # if len(available_vars) < 2:
            #     continue

            p = len(available_vars)
            m_vars = int(np.floor(np.sqrt(p)))
            subset_vars = random.sample(available_vars, k=m_vars)

            # Añadimos las variables fijas
            subset_vars = list(set(subset_vars) | permanently_fixed[m])
            # print('Muestra bootrstap:')
            # print(subset_vars)

            Xsub = X[subset_vars]
            datos = pd.DataFrame(Xsub)
            datos['y'] = y

            _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
            base_complexity = df_classes.loc['dataset', m]

            current_vars = subset_vars.copy()

            while len(current_vars) > 1:
                count_vars[current_vars] += 1
                to_remove = random.choice(current_vars)
                current_vars.remove(to_remove)

                Xtemp = X[current_vars]
                datos_temp = pd.DataFrame(Xtemp)
                datos_temp['y'] = y
                _, df_classes_temp, _ = all_measures_FS(datos_temp, save_csv=False, path_to_save=None, name_data=None)
                # print(df_classes_temp)
                # print(current_vars)
                new_complexity = df_classes_temp.loc['dataset', m]

                # cambio de complejidad
                delta = new_complexity - base_complexity
                importances[m][to_remove] += delta

                # Si baja complejidad (delta < 0): eliminar permanentemente
                if delta <= -0.01:
                    # print(delta)
                    # print('Se elimina:')
                    # print(to_remove)
                    permanently_removed[m].add(to_remove)
                    active_vars[m].discard(to_remove)
                    removed_vars[m][to_remove] += 1
                    # Si estaba en fixed, la sacamos porque preferimos darle más fuerza a lo negativo y cargárnosla
                    if to_remove in permanently_fixed[m]:
                        permanently_fixed[m].remove(to_remove)
                        fixed_vars[m][to_remove] -= 1  # guardamos quee la quitamos

                # Si delta > tau: fijar variable. Exigimos tb que no esté en las eliminadas por delta negativo
                elif ((delta >= tau) and (to_remove not in permanently_removed[m])):
                    # print('Se fija:')
                    # print(to_remove)
                    # print('Delta de la que se fija:')
                    # print(delta)
                    permanently_fixed[m].add(to_remove)
                    fixed_vars[m][to_remove] += 1

                base_complexity = new_complexity
            permanently_fixed[m] -= permanently_removed[m]  # por si acaso

    # Normalización
    count_vars = count_vars.replace(0, np.nan)
    for m in measures:
        importances_norm[m] = importances[m] / count_vars
        importances_norm[m].sort_values(ascending=False, inplace=True)

    # Formato resultados
    results_norm = pd.DataFrame.from_dict(importances_norm).add_suffix('_importances_norm')
    results = pd.DataFrame.from_dict(importances).add_suffix('_importances')
    results_count = pd.DataFrame(count_vars, columns=['count_vars'])
    removed_df = pd.DataFrame.from_dict(removed_vars).add_suffix('_removed_count')
    fixed_df = pd.DataFrame.from_dict(fixed_vars).add_suffix('_fixed_count')

    results_complete = pd.concat([results, results_norm, results_count, removed_df, fixed_df], axis=1)

    if filter_corr:
        results_complete = results_complete.reindex(results_complete.index.union(to_drop))
        results_complete.loc[to_drop, :] = np.nan

    if save_csv:
        name_csv = f"{path}/{dataset_name}_ComplexityRandomDistributed_NegOut_HighFixed.csv"
        results_complete.to_csv(name_csv, index=True)

    # Devuelve también los conjuntos fijos y eliminados
    return (importances_norm, importances, count_vars,removed_vars,fixed_vars,
        permanently_removed,permanently_fixed,results_complete)


# dataset_name = 'ArtificialDataset12'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=987,noise_std=0.5)


# measures = ['kDN']
# dataset_name = 'prueba'
# n_replicas=2
def evaluate_distributed_fs_cv2(X, y, k, model, dataset_name,measures=["Hostility", "N1", "kDN"],
                                cv_splits=5, random_state=0, n_replicas=200,tau=0.01):
    """
    Realiza CV evaluando el métod distributed:
      - selecciona top-k variables según cada medida
      - entrena y evalúa modelo en cada fold
      - registra variables eliminadas (por importancia negativa)
        y variables fijadas (por importancia positiva > tau)
    """

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    importances_records = []
    performance_records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== FOLD {fold}/{cv_splits} ===")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        p = X.shape[1]
        m_vars = np.floor(np.sqrt(p))

        (importances_norm, importances, count_vars,removed_vars, fixed_vars,
            permanently_removed, permanently_fixed,imp_df) = distributed_complexity_random_neg_out_high_fixed(
            X_train, y_train, dataset_name, n_replicas, m_vars,
            measures=measures, filter_corr=True, corr_th=0.9,
            corr_method="pearson", random_state=random_state,
            save_csv=False, tau=tau)

        # Añadimos info del fold
        imp_df = imp_df.reset_index().rename(columns={"index": "feature"})
        imp_df["fold"] = fold
        importances_records.append(imp_df)



        # Info de variables eliminadas y fijas
        removed_fixed_record = {m: {"removed": list(permanently_removed[m]),
                                    "fixed": list(permanently_fixed[m])} for m in measures}
        imp_df.sort_values(by=["kDN_importances_norm"], ascending=False)
        imp_df.set_index("feature", inplace=False, drop=False)
        # top-k variables por medida
        measures_norm = [s + '_importances_norm' for s in measures]
        for measure, m_name in zip(measures_norm, measures):
            imp_m = imp_df[measure]
            top_feats = (imp_m.sort_values(ascending=False).index.tolist())[:int(k)]
            feats = imp_df.loc[top_feats, 'feature']

            # Entrenamos modelo
            X_train_sel = X_train[feats]
            X_test_sel = X_test[feats]

            datos = pd.DataFrame(X_train_sel)
            datos['y'] = y_train
            _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
            subset_complexity = df_classes.loc["dataset", measures].to_dict()

            model.fit(X_train_sel, y_train)

            # Métricas train/test
            y_pred_train = model.predict(X_train_sel)
            acc_train = accuracy_score(y_train, y_pred_train)
            gps_train = compute_gps(y_train, y_pred_train)

            y_pred_test = model.predict(X_test_sel)
            acc_test = accuracy_score(y_test, y_pred_test)
            gps_test = compute_gps(y_test, y_pred_test)

            performance_records.append({
                "fold": fold,
                "measure": measure,
                "n_features": k,
                "top_features": list(feats),
                "acc_train": acc_train,
                "gps_train": gps_train,
                "acc_test": acc_test,
                "gps_test": gps_test,
                **{f"complexity_{m}": subset_complexity[m] for m in measures},
                # Nuevas columnas para rastrear
                "removed_features": removed_fixed_record[m_name]["removed"],
                "fixed_features": removed_fixed_record[m_name]["fixed"]
            })

    importances_df = pd.concat(importances_records, ignore_index=True)
    performance_df = pd.DataFrame(performance_records)

    return importances_df, performance_df

# model = KNeighborsClassifier()
# k=3
# importances_df, performance_df= evaluate_distributed_fs_cv2(X, y, k, model, dataset_name,measures=["Hostility", "N1", "kDN"],
#                                 cv_splits=3, random_state=0, n_replicas=2,tau=0.01)
#




def run_distributed_cv_multiple_models2_real(X, y, dataset_name, models_dict,
    measures=["Hostility", "N1", "kDN"],cv_splits=5, n_replicas=200, random_state=0,
    tau=0.01,path="Results_FS_Distributed_CV", save_csv=False):

    # k = len(dict_info_feature["informative"])  # nº de variables informativas
    p = X.shape[1]
    k = p/3
    if (k>=1000):
        k = np.ceil(np.sqrt(p))

    all_importances = []
    all_performance = []

    for model_name, model in models_dict.items():
        print(f"\n Model: {model_name}")

        imp_df, perf_df = evaluate_distributed_fs_cv2(
            X=X, y=y, k=k, model=model,
            dataset_name=dataset_name, measures=measures,
            cv_splits=cv_splits, random_state=random_state,
            n_replicas=n_replicas, tau=tau
        )

        # Añadir nombre del modelo
        imp_df["model"] = model_name
        perf_df["model"] = model_name

        all_importances.append(imp_df)
        all_performance.append(perf_df)

    # Concatenar todos los resultados
    importances_all = pd.concat(all_importances, ignore_index=True)
    performance_all = pd.concat(all_performance, ignore_index=True)

    # Resumen por modelo y medida
    summary = (
        performance_all
        .groupby(["model", "measure"])
        .agg({
            "acc_train": ["mean", "std", "max"],
            "gps_train": ["mean", "std", "max"],
            "acc_test": ["mean", "std", "max"],
            "gps_test": ["mean", "std", "max"],
            # "complexity_Hostility": ["mean", "std"],
            # "complexity_N1": ["mean", "std"],
            "complexity_kDN": ["mean", "std"]
        })
    )

    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Guardado opcional
    if save_csv:
        os.makedirs(path, exist_ok=True)
        name_csv1 = f"{path}/{dataset_name}_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv"
        importances_all.to_csv(name_csv1, index=False)
        name_csv2 = f"{path}/{dataset_name}_DistributedCVRandom_OutHigh_Performance_Folds.csv"
        performance_all.to_csv(name_csv2, index=False)
        name_csv3 = f"{path}/{dataset_name}_DistributedCVRandom_OutHigh_SummaryResults.csv"
        summary.to_csv(name_csv3, index=False)

    return importances_all, performance_all, summary






###############################################################################################################
#####                          PLOT COMPLEXITY RESULTADOS VERSION CV                                      #####
###############################################################################################################
# Hacemos un plot tipo importancia de variables en RF

# df = pd.read_csv("Results_FS_Distributed_CV/ArtificialDataset2_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv", index_col=0)

def plot_complexity_importances_by_model(df, dataset_name="Dataset", save_path=None):
    """
    Produce para cada modelo:
    1) Gráfico de barras de kDN_importances_norm agregadas por feature (mean ± std)
    2) Gráfico de frecuencias (count_vars sumado)
    """

    sns.set(style="whitegrid", font_scale=1.1)
    modelos = df["model"].unique()
    n_models = len(modelos)
    modelo = modelos[0]

    # GRID: dos columnas (importancia, frecuencias)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes = np.array([axes])
    df_model = df[df["model"] == modelo]

    # agregamos por fold
    agg_df = (
        df_model.groupby("feature")
                    .agg(
                         kdn_mean=("kDN_importances_norm", "mean"),
                         kdn_median=("kDN_importances_norm", "median"),
                         kdn_std=("kDN_importances_norm", "std"),
                         count_total=("count_vars", "sum")
                    )
                    .sort_values("kdn_mean", ascending=False)
    )
    agg_df["kdn_std"] = agg_df["kdn_std"].fillna(0) # Evitar NaN en std cuando solo hay 1 fold

    # IMPORTANCIAS (mean ± std)
    ax1 = axes[0, 0]
    ax1.bar(
        agg_df.index,
        agg_df["kdn_mean"],
        yerr=agg_df["kdn_std"],
        capsize=4
    )
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_title(f"kDN importance (mean ± std)")
    ax1.set_xlabel("Variable")
    ax1.set_ylabel("Importance")
    ax1.tick_params(axis="x", rotation=45)

    # FRECUENCIA de aparición
    ax2 = axes[0, 1]
    ax2.bar(agg_df.index, agg_df["count_total"])
    ax2.set_title(f"Variable occurrence count")
    ax2.set_xlabel("Variable")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)

    plt.suptitle(f"{dataset_name} — Complexity Random FS", fontsize=16, y=1.02)
    plt.tight_layout()

    # if save_path:
    #     plt.savefig(f"{save_path}/{dataset_name}_Complexity_Grid.png",
    #                 dpi=300, bbox_inches="tight")

    plt.show()





# df_random = pd.read_csv("Results_FS_Distributed_CV/ArtificialDataset2_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv", index_col=0)
# # df = df_random
#
# plot_complexity_importances_by_model(df_random, dataset_name="Dataset2")
#

# plot_complexity_importances_by_model(df, dataset_name="Dataset2")





models_dict = {#"LogReg": LogisticRegression(max_iter=1000, random_state=0),
    # "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
    "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
    # "RandomForest": RandomForestClassifier(random_state=0),
    "KNN": KNeighborsClassifier()
    # "NaiveBayes": GaussianNB(),
    # "DecisionTree": DecisionTreeClassifier(random_state=0),
    # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
    }


# n_replicas = 100
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=200, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)
#
#
#
#


# os.chdir("datasets")
# for file in glob.glob("*.csv"):
#     print(file)
# file = 'ionosphere.csv'
list_datasets = ['spambase.csv']#,'ionosphere.csv', 'sonar.csv',
                # 'parkinsons.csv']
                # 'wdbc.csv',
                #  'musk2.csv','parkinsons.csv',
                #  'ozone.csv','sonar.csv','spambase.csv',
                #  'Colon.csv','arcene_train.csv','gisette_train.csv']

def format_labels(y):
    """
    Ensure labels y are integers starting at 0: 0,1,...,n_classes-1.
    Works with categorical (str) or numeric labels.
    """
    y = pd.Series(y)  # ensure consistent type

    # Case 1: non-numeric labels (strings, categories, mixed)
    if not np.issubdtype(y.dtype, np.number):
        categories = sorted(y.unique())  # alphabetic order
        mapping = {cat: i for i, cat in enumerate(categories)}
        y = y.map(mapping)

    # Case 2: numeric but not starting at 0
    else:
        unique_vals = sorted(y.unique())
        if unique_vals[0] != 0 or unique_vals != list(range(len(unique_vals))):
            mapping = {val: i for i, val in enumerate(unique_vals)}
            y = y.map(mapping)

    return y.to_numpy()

# path2 = "datasets"
# for file in list_datasets:
#     # os.makedirs(path2, exist_ok=True)
#     read_csv = f"{path2}/{file}"
#     df = pd.read_csv(read_csv)
#     # print(df)
#     print(read_csv)
#     y = format_labels(df['y'])
#     cols = df.drop('y', axis=1).columns
#     X = df.drop('y', axis=1)
#     X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
#     # df[cols] = StandardScaler(with_mean=True, with_std=True).fit_transform(df)
#     X = pd.DataFrame(X)
#     X.columns = cols
#     dataset_name = file.split(".")[0]
#     print(dataset_name)
#     n_replicas = 250
#
#     run_distributed_cv_multiple_models2_real(X, y, dataset_name, models_dict,
#         measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#         tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)



############################################################################
######        PERFORMANCE POR FOLD SIGUIENDO RANKING out y high       ######
############################################################################
# Ya tenemos el ranking con kdn con neg out y high fixed por fold
# Ahora vamos a sacar la performance de los modelo según vamos añadiendo variables


# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset12_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'

def load_importances_per_fold(path_csv, measures=["kDN_importances_norm"]):
    """
    Lee el archivo CSV con importancias por fold.
    Devuelve: dict[fold][measure] = DataFrame con columnas (feature, importance)
    Solo incluye las variables que tengan valor (no NaN).
    """
    df = pd.read_csv(path_csv)

    df = df.loc[df.model == 'KNN',:] # aquí los modelos no interactuan, realmente esa columna sobra en los csvs

    result = {}

    for fold in sorted(df["fold"].unique()):
        df_f = df[df["fold"] == fold]

        result[fold] = {}

        for m in measures:

            df_m = df_f[["feature", m]].dropna(subset=[m])  # elimina variables que quitamos por alta correlación

            # Orden descendente: mayor importancia primero
            df_m = df_m.sort_values(m, ascending=False)

            result[fold][m] = df_m.reset_index(drop=True)

    return result

# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset2_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# dfs = []
# for fold, v in importances_dict.items():
#     df = v["kDN_importances_norm"].copy()
#     df["fold"] = fold
#     dfs.append(df)
#
# importances_all = pd.concat(dfs, ignore_index=True)


# Función para evaluar rendimiento metiendo variables modo forward siguiendo el ranking
# establecido por la complejidad modo neg out high fixed
def evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0):
    """
    Para cada fold, measure y modelo:
      evalúa rendimiento con k = 1..K variables ordenadas por importancia.
    """

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        X_train_base, X_test_base = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for measure, df_m in importances_dict[fold_id].items():

            features_ranked = df_m["feature"].tolist()
            K = len(features_ranked)

            for k in range(1, K + 1):

                selected = features_ranked[:k]

                Xt = X_train_base[selected]
                Xs = X_test_base[selected]

                for model_name, model in models.items():

                    clf = model
                    clf.fit(Xt, y_train)
                    pred = clf.predict(Xs)

                    acc = accuracy_score(y_test, pred)
                    gps = compute_gps(y_test, pred)

                    rows.append({
                        "dataset": dataset_name,
                        "fold": fold_id,
                        "measure": measure,
                        "k": k,
                        "n_available_features": K,
                        "model": model_name,
                        "acc_test": acc,
                        "gps_test": gps})

    perf_final = pd.DataFrame(rows)

    return perf_final

models = {#"LogReg": LogisticRegression(max_iter=1000, random_state=0),
    # "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
    "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
    # "RandomForest": RandomForestClassifier(random_state=0),
    "KNN": KNeighborsClassifier()
    # "NaiveBayes": GaussianNB(),
    # "DecisionTree": DecisionTreeClassifier(random_state=0),
    # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
    }





# os.chdir("datasets")
# for file in glob.glob("*.csv"):
#     print(file)
# file = 'ionosphere.csv'
list_datasets = ['spambase.csv']#,'ionosphere.csv', 'sonar.csv', 'spambase.csv'
                # 'parkinsons.csv']
                # 'wdbc.csv',
                #  'musk2.csv','parkinsons.csv',
                #  'ozone.csv','sonar.csv','spambase.csv',
                #  'Colon.csv','arcene_train.csv','gisette_train.csv']





path2 = "datasets"
for file in list_datasets:
    # os.makedirs(path2, exist_ok=True)
    read_csv = f"{path2}/{file}"
    df = pd.read_csv(read_csv)
    # print(df)
    print(read_csv)
    y = format_labels(df['y'])
    cols = df.drop('y', axis=1).columns
    X = df.drop('y', axis=1)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    # df[cols] = StandardScaler(with_mean=True, with_std=True).fit_transform(df)
    X = pd.DataFrame(X)
    X.columns = cols
    dataset_name = file.split(".")[0]
    print(dataset_name)
    path_csv = 'Results_FS_Distributed_CV/'+str(dataset_name)+'_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
    importances_dict = load_importances_per_fold(path_csv)

    # name_csv = 'Results_FS_Distributed_CV/'+str(dataset_name)+'_OutHigh_EvolutivePerformance.csv'
    # perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
    # perf_final.to_csv(name_csv,index=False)


def plot_incremental_performance(performance_df, importances_df, dataset, measure="acc_test"):
    """
    measure = 'acc_test' o 'gps_test'
    """

    df = performance_df.copy()
    df = df[df["dataset"] == dataset]
    models = df["model"].unique()
    folds = df["fold"].unique()

    # Colores por modelo
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color = {m: c for m, c in zip(models, colors)}

    # Estilos por fold
    line_styles = ["solid", "dashed", "dotted", "dashdot"]
    fold_style = {f: line_styles[(f-1) % len(line_styles)] for f in folds}

    # mapa de negativo por fold
    negative_k = {}
    for fold in folds:
        imp_f = importances_df[importances_df["fold"] == fold]
        imp_sorted = imp_f.sort_values("kDN_importances_norm", ascending=False)
        neg_idx = np.where(imp_sorted["kDN_importances_norm"].values < 0)[0]
        negative_k[fold] = neg_idx[0] + 1 if len(neg_idx) > 0 else None

    plt.figure(figsize=(13, 8))

    for model in models:
        df_m = df[df["model"] == model]

        for fold in folds:
            df_f = df_m[df_m["fold"] == fold].sort_values("k")
            k = df_f["k"].values
            y = df_f[measure].values

            k_neg = negative_k[fold]

            # colores/estilos
            col = model_color[model]
            ls = fold_style[fold]

            # trozo principal (antes de k negativo)
            if k_neg is None:
                # Nada negativo: toda la curva normal
                plt.plot(k, y, marker="o", color=col, linestyle=ls,
                         alpha=0.8, label=f"{model} – fold {fold}")
            else:
                # tramo coloreado (hasta k_neg)
                idx_color = k <= k_neg
                plt.plot(k[idx_color], y[idx_color], marker="o",
                         color=col, linestyle=ls, alpha=0.8,
                         label=f"{model} – fold {fold}")

                # tramo gris
                idx_grey = k >= k_neg
                if np.any(idx_grey):
                    plt.plot(k[idx_grey], y[idx_grey], marker="o",
                             color="grey", linestyle=ls, alpha=0.6)

            # máximo
            max_row = df_f.loc[df_f[measure].idxmax()]
            plt.scatter(max_row["k"], max_row[measure],
                        s=80, edgecolor="black", facecolor="yellow", zorder=5)

            plt.text(max_row["k"], max_row[measure],
                     f"{max_row[measure]:.3f}",
                     fontsize=9, verticalalignment="bottom")

    plt.title(f"{dataset} – Evolución de {measure.replace('_',' ').upper()} con nº de variables")
    plt.xlabel("Número de variables usadas (k)")
    plt.ylabel(measure.replace("_"," ").upper())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()




# path_csv = 'Results_FS_Distributed_CV/ionosphere_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
# importances_all = pd.concat([v["kDN_importances_norm"] for v in importances_dict.values()],ignore_index=True)
#
# dfs = []
# for fold, v in importances_dict.items():
#     df = v["kDN_importances_norm"].copy()
#     df["fold"] = fold
#     dfs.append(df)
#
# importances_all = pd.concat(dfs, ignore_index=True)
# perf = pd.read_csv('Results_FS_Distributed_CV/ionosphere_OutHigh_EvolutivePerformance.csv')
#
# plot_incremental_performance(perf, importances_all,dataset="ionosphere",measure="acc_test")
#





# importances_dict = load_importances_per_fold(path_csv)
# importances_all = pd.concat([v["kDN_importances_norm"] for v in importances_dict.values()],ignore_index=True)
# dfs = []
# for fold, v in importances_dict.items():
#     df = v["kDN_importances_norm"].copy()
#     df["fold"] = fold
#     dfs.append(df)
#
# importances_all = pd.concat(dfs, ignore_index=True)
#
# importances_df = importances_all
# performance_df = perf_final
#
# plot_incremental_performance(performance_df, importances_df,
#                              dataset="ArtificialDataset2",
#                              measure="acc_test")



