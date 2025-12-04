## 28/10/2025

#### Script para comparar los resultados que obtenemos con la versión Distributed y los que logran
#### los métodos del SOTA

# La mayor parte de las funciones las cogemos de FeatureSelectionComplexityEvaluation.py

### Comenzamos comparando los resultados eligiendo un número fijo de variables
# que será el número de variables informativas dado que todavía estamos con casos artificiales

from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
import xgboost as xgb
import pandas as pd
import numpy as np
from All_measures import *
from sklearn.datasets import make_classification

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
        random_state=random_state)

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





## Función para ejecutar diversos métodos de FS tipo filtro del SOTA
def select_features_by_filters_and_corr(X, y, feature_names,k=None,methods=None,random_state=0,
                                        filter_corr=False,corr_method="pearson",corr_th=0.9):
    """
    Aplica varios métodos de filtro y devuelve:
        selections: dict {method_name: {"scores": pd.Series(index=feature_names), "selected": [names...] }}

    - X: np.ndarray or DataFrame
    - y: array-like
    - feature_names: list of names (length = X.shape[1])
    - k: número de features a seleccionar (si None -> k = n_informative_guess ~ sqrt(n_features) fallback)
    - methods: lista de strings entre {"mutual_info","f_classif","rf","relief",'xgboost'}
    """
    if methods is None:
        methods = ["mutual_info", "f_classif", "rf", "relief","xgboost"]

    X_df = X.copy()# if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    # Filtro previo opcional de eliminación de variables con correelación > corr_th
    if filter_corr:
        corr = X_df.corr(method=corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        if len(to_drop) > 0:
            X_df = X_df.drop(columns=to_drop)
            feature_names = [f for f in feature_names if f not in to_drop]


    # Xarr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    Xarr = X_df.values
    n_features = Xarr.shape[1]
    if k is None:
        k = max(1, int(np.sqrt(n_features)))  # heuristic value

    results = {}


    # mutual information
    if "mutual_info" in methods:
        mi = mutual_info_classif(Xarr, y, random_state=random_state)
        s = pd.Series(mi, index=feature_names).sort_values(ascending=False)
        results["mutual_info"] = {"scores": s, "selected": list(s.index[:k])}

    # ANOVA F (f_classif)
    if "f_classif" in methods:
        F, p = f_classif(Xarr, y)
        s = pd.Series(F, index=feature_names).sort_values(ascending=False)
        results["f_classif"] = {"scores": s, "selected": list(s.index[:k])}

    # Random Forest importance
    if "rf" in methods:
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        rf.fit(Xarr, y)
        imp = rf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["rf"] = {"scores": s, "selected": list(s.index[:k])}

    # ReliefF
    if "relief" in methods:
        rf_sel = ReliefF(n_features_to_select=Xarr.shape[1]) # n_neighbors usamos el valor por defecto de la librería
        rf_sel.fit(Xarr, y)
        scores = rf_sel.feature_importances_
        s = pd.Series(scores, index=feature_names).sort_values(ascending=False)
        results["relief"] = {"scores": s, "selected": list(s.index[:k])}

    # XGBoost
    if "xgboost" in methods:
        xgb_clf = xgb.XGBClassifier(eval_metric="logloss",random_state=random_state)
        xgb_clf.fit(Xarr, y)
        imp = xgb_clf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["xgb"] = {"scores": s, "selected": list(s.index[:k])}


    return results


# Función para generar los subconjuntos de interés para cada dataset
def build_subsets_for_complexity(feature_names, feature_types, fs_selections,k_random=3, random_state=0):
    rng = np.random.RandomState(random_state)
    subsets = {}

    subsets['all'] = list(feature_names)
    inform = [f for f, t in feature_types.items() if t == 'informative']
    noise = [f for f, t in feature_types.items() if t == 'noise']
    redun = [f for f, t in feature_types.items() if t == 'redundant_linear']
    redun_nonlineal = [f for f, t in feature_types.items() if t == 'redundant_nonlinear']

    subsets['informative'] = inform
    subsets['informative+redundant'] = inform + redun
    subsets['informative+redundant_nonLinear'] = inform + redun_nonlineal
    subsets['informative+noise'] = inform + noise

    # selección aleatoria (informativas + ruido/redundantes al azar)
    pool_extra = noise + redun + redun_nonlineal
    if pool_extra and k_random > 0:
        ksel = min(k_random, len(pool_extra))
        rand_pick = rng.choice(pool_extra, size=ksel, replace=False).tolist()
        subsets['informative+rand_extra'] = inform + rand_pick

    # subsets según métodos de FS
    for method, info in fs_selections.items():
        if 'selected' in info:
            sel = info['selected']
            name = f"{method}_top{len(sel)}"
            subsets[name] = sel

    return subsets

# csv_path = 'Results_FS_Distributed/ArtificialDataset2_ComplexityRandomDistributed.csv'
# Función para generar los subconjuntos de interés para cada dataset en base a los resultados de distributed
def build_distributed_subsets_from_csv(csv_path, k, prefix="guided"):
    subsets = None
    if csv_path:
        df = pd.read_csv(csv_path, index_col=0)
        subsets = {}

        # Medidas
        measures = ["Hostility_importances_norm", "N1_importances_norm", "kDN_importances_norm"]
        # m = "Hostility_importances_norm"
        for m in measures:
            df_m = df[[m]].dropna().sort_values(m, ascending=False)
            top_features = df_m.index[:k].tolist()
            subset_name = f"{prefix}_{m.split('_')[0]}_top{k}"
            subsets[subset_name] = top_features

    return subsets




def evaluate_complexity_across_subsets(X, y, subsets, save_csv=False, path_to_save=None):
    """
    Aplica all_measures a cada subset de features y organiza los resultados.

    Parameters
    ----------
    X : DataFrame
        Dataset completo con todas las features.
    y : array-like
        Etiquetas.
    subsets : dict
        Diccionario {subset_name: list_of_features}.
    save_csv, path_to_save : para pasar a all_measures.

    Returns
    -------
    results_total : DataFrame
        Filas = subset_name, Columnas = medidas de complejidad (dataset total).
    results_classes : dict
        {subset_name: df_classes_dataset} (una fila por clase + dataset).
    extras_host : dict
        {subset_name: extra_results_host}.
    """
    results_total = []
    results_classes = {}
    extras_host = {}
    selected_measures = ["Hostility", "N1", "N2", "kDN", "LSC", "CLD", "TD_U", "DCP", "F1", "L1"]

    for subset_name, features in subsets.items():
        # Xsub = preprocessing.scale(X[features])
        Xsub = X[features]
        datos = pd.DataFrame(Xsub, columns=features)
        datos['y'] = y
        df_measures, df_classes, extra_results = all_measures(datos, save_csv, path_to_save, subset_name)

        # Nos quedamos solo con las medidas seleccionadas
        df_classes = df_classes.loc[:, df_classes.columns.intersection(selected_measures)]

        # Guardar fila resumen (total del dataset)
        total_row = df_classes.loc["dataset"].copy()
        total_row.name = subset_name
        total_row["n_features"] = len(features)  # extra info
        results_total.append(total_row)

        results_classes[subset_name] = df_classes
        extras_host[subset_name] = {
            "extra_results": extra_results,
            "instance_measures": df_measures[selected_measures + ["y"]].copy()  # incluye etiquetas
        }

    results_total = pd.DataFrame(results_total)

    return results_total, results_classes, extras_host



def build_comparison_table(results_per_dataset):
    """
    results_per_dataset: dict
        {dataset_name: results_total (DataFrame con subsets x medidas)}

    Devuelve un DataFrame multi-índice (dataset, subset).
    """
    df_list = []
    for dname, res in results_per_dataset.items():
        res = res.copy()
        res["dataset_name"] = dname
        df_list.append(res)

    combined = pd.concat(df_list)
    combined = combined.set_index(["dataset_name", combined.index])
    combined.index.names = ["Dataset", "Subset"]
    return combined



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



def evaluate_models_across_subsets(X, y, subsets, cv_splits=10, random_state=0):
    """
    Evalúa modelos en los subsets de features.

    Modelos: Logistic Regression, SVM linear, SVM rbf, Random Forest,
             KNN, Naive Bayes, Decision Tree, XGBoost.

    Returns:
    --------
    results_df : DataFrame con [subset, best_model, best_acc, best_gps]
    detailed_results : dict {subset: {model_name: {"acc":..., "gps":..., "acc_per_class": {...}}}}
    """
    models = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM-linear": SVC(kernel="linear", probability=True, random_state=random_state),
        "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=random_state)
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results_records = []  # registro fold a fold
    summary_records = []  # medias y std
    detailed_results = {}

    classes = np.unique(y)

    for subset_name, features in subsets.items():
        Xsub = X[features].values
        subset_scores = {}

        for model_name, model in models.items():
            fold_acc = []
            fold_gps = []
            fold_acc_class = {int(c): [] for c in classes} # accs por clase

            for fold, (train_idx, test_idx) in enumerate(skf.split(Xsub, y), 1):
                X_train, X_test = Xsub[train_idx], Xsub[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                gps = compute_gps(y_test, y_pred)

                # accuracy por clase en este fold
                acc_per_class = {}
                for c in classes:
                    idx = (y_test == c)
                    acc_c = accuracy_score(y_test[idx], y_pred[idx])
                    acc_per_class[int(c)] = acc_c
                    fold_acc_class[int(c)].append(acc_c)

                # guardar registro por fold
                record = {
                    "subset": subset_name,
                    "model": model_name,
                    "fold": fold,
                    "acc": acc,
                    "gps": gps,
                }
                for cls, val in acc_per_class.items():
                    record[f"acc_class_{cls}"] = val

                results_records.append(record)

                # acumular para medias
                fold_acc.append(acc)
                fold_gps.append(gps)

            # resumen media y std por modelo y subset
            mean_acc = np.mean(fold_acc)
            mean_gps = np.mean(fold_gps)
            std_acc = np.std(fold_acc)
            std_gps = np.std(fold_gps)

            # por clase
            mean_acc_class = {cls: np.mean(vals) for cls, vals in fold_acc_class.items()}
            std_acc_class = {cls: np.std(vals) for cls, vals in fold_acc_class.items()}

            resumen = {
                "subset": subset_name,
                "model": model_name,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_gps": mean_gps,
                "std_gps": std_gps,
            }

            for cls in classes:
                resumen[f"mean_acc_class_{int(cls)}"] = mean_acc_class[int(cls)]
                resumen[f"std_acc_class_{int(cls)}"] = std_acc_class[int(cls)]

            summary_records.append(resumen)

            subset_scores[model_name] = {
                "fold_acc": fold_acc,
                "fold_gps": fold_gps,
                "fold_acc_class": fold_acc_class,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_gps": mean_gps,
                "std_gps": std_gps,
                "mean_acc_class": mean_acc_class,
                "std_acc_class": std_acc_class,
            }


        detailed_results[subset_name] = subset_scores # dict con tod

    # a DataFrames
    results_df = pd.DataFrame(results_records).set_index(["subset", "model", "fold"]) # resultados por fold
    summary_df = pd.DataFrame(summary_records).set_index(["subset", "model"]) # resultados promedio

    return results_df, summary_df, detailed_results




def save_complexity_csv(dataset_name, subset_name, results_classes, extras_host,
                        path="Results_ComparisonDistributed_SOTA"):
    inst = extras_host[subset_name]["instance_measures"].reset_index(drop=True)
    classes = results_classes[subset_name].reset_index()

    # Añadimos columnas auxiliares
    inst["level"] = "instance"
    classes["level"] = "class"

    inst["subset"] = subset_name
    classes["subset"] = subset_name

    # Unimos
    final = pd.concat([classes, inst], axis=0, ignore_index=True)
    fname = f"{path}/{dataset_name}_{subset_name}_complexity.csv"
    final.to_csv(fname, index=False)

    return final



def save_models_csv(dataset_name, results_models, path="Results_ComparisonDistributed_SOTA"):
    """
    Guarda en CSV el rendimiento de TODOS los modelos en TODOS los subsets para un dataset.
    results_models debe ser un DataFrame con índices [subset, model] y columnas [acc, gps, acc_class_*].
    """
    # Reset index para que subset y model queden como columnas normales
    final = results_models.reset_index()
    final.insert(0, "dataset", dataset_name)  # añadimos dataset como primera columna

    fname = f"{path}/{dataset_name}_modelsPerformance_subsets.csv"
    final.to_csv(fname, index=False)
    return final




# dataset_name = 'prueba'
def FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
                                                csv_guided_path, csv_random_path,
                                              path_to_save="Results_ComparisonDistributed_SOTA"):
    # Número de features informativas como k
    k = len(dict_info_feature["informative"])
    feature_names = X.columns.tolist()

    # Ejecutamos los métodos de FS con y sin corr
    fs_results_corr = select_features_by_filters_and_corr(X, y, feature_names, k=k, filter_corr=True)
    fs_results = select_features_by_filters_and_corr(X, y, feature_names, k=k, filter_corr=False)
    # cambiamos nombre para distinguir con filtro corr y sin corr
    fs_results_corr_renamed = {f"{name}_corr": info for name, info in fs_results_corr.items()}
    fs_results_combined = {**fs_results, **fs_results_corr_renamed}

    # Construir subconjuntos
    feature_types = {}
    for f in dict_info_feature["informative"]: feature_types[f] = "informative"
    for f in dict_info_feature["noise"]: feature_types[f] = "noise"
    for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
    for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
    subsets = build_subsets_for_complexity(feature_names, feature_types, fs_results_combined)

    # Añadimos subsets de k variables escogidas por el métod distribuido
    subsets_guided = build_distributed_subsets_from_csv(csv_guided_path, k, prefix="guided")
    if subsets_guided:
        subsets.update(subsets_guided)
    subsets_random = build_distributed_subsets_from_csv(csv_random_path, k, prefix="random")
    subsets.update(subsets_random)

    # Evaluación de complejidad
    results_total, results_classes, extras_host = evaluate_complexity_across_subsets(X, y, subsets)

    # Evaluación de modelos
    # results_models, detailed_models = evaluate_models_across_subsets(X, y, subsets)
    results_models_folds, summary_df_models, detailed_results = evaluate_models_across_subsets(X, y, subsets)

    # Guardar csvs de complejidad por subset
    for subset_name in subsets.keys():
        save_complexity_csv(dataset_name, subset_name, results_classes, extras_host, path_to_save)

    # Guardar csv de modelos por dataset
    save_models_csv(dataset_name, summary_df_models, path_to_save)



    # --- TABLA DE COMPARACIÓN ---
    # results_models tiene MultiIndex (subset, model),
    # hacemos un resumen (medias por subset)
    # nos quedamos con las columnas que empiezan por "mean_"
    perf_cols = [c for c in summary_df_models.columns if c.startswith("mean_")]
    # media, máximo y desviación estándar por subset
    summary_models = summary_df_models.groupby(level="subset")[perf_cols].agg(["mean", "max", "std"])
    # nombres columnas
    summary_models.columns = [f"{m}_{stat}" for m, stat in summary_models.columns]


    # Juntamos en una sola tabla
    results_all = results_total.join(summary_models, how="left")

    # Nombres y tal
    results_all["dataset_name"] = dataset_name
    comparison_table = results_all.set_index(["dataset_name", results_all.index])
    comparison_table.index.names = ["Dataset", "Subset"]

    fname = f"{path_to_save}/{dataset_name}_comparisonTable.csv"
    comparison_table.to_csv(fname)

    return comparison_table, results_classes, results_models_folds


#
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset2_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset2_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
# ### Dataset 3
# dataset_name = 'ArtificialDataset3'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=25,n_noise=5,
#                                          n_redundant_linear=7,n_redundant_nonlinear=8,
#                                          flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=0,noise_std=0.05)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset3_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset3_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
# #### Dataset 7
# dataset_name = 'ArtificialDataset7'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=589,noise_std=0.05)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset7_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset7_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
# #### Dataset 8
# dataset_name = 'ArtificialDataset8'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=40,n_noise=15,
#                                          n_redundant_linear=15,n_redundant_nonlinear=15,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=86785,noise_std=0.1)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset8_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset8_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
# #### Dataset 10
# dataset_name = 'ArtificialDataset10'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=2000,n_informative=6,n_noise=20,
#                                          n_redundant_linear=20,n_redundant_nonlinear=15,
#                                         flip_y=0, class_sep=0.8, n_clusters_per_class=2, weights=[0.3],
#                                                      random_state=959,noise_std=0.3)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset10_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset10_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
# #### Dataset 11
# dataset_name = 'ArtificialDataset11'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=20,n_noise=20,
#                                          n_redundant_linear=20,n_redundant_nonlinear=15,
#                                         flip_y=0, class_sep=0.6, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=959,noise_std=0.1)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset11_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset11_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
# #### Dataset 12
# dataset_name = 'ArtificialDataset12'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=987,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset12_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset12_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
#
# #### Dataset 14
# dataset_name = 'ArtificialDataset14'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
#                                          n_redundant_linear=30,n_redundant_nonlinear=40,
#                                         flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
#                                                      random_state=95,noise_std=0.5)
#
# # csv_guided_path = 'Results_FS_Distributed/ArtificialDataset14_ComplexityGuidedDistributed.csv'
# csv_guided_path = None
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset14_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
#
# #### Dataset 16
# dataset_name = 'ArtificialDataset16'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=50,n_noise=40,
#                                          n_redundant_linear=30,n_redundant_nonlinear=40,
#                                         flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.2],
#                                                      random_state=756,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset16_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset16_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
# #### Dataset 17
# dataset_name = 'ArtificialDataset17'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.3, class_sep=0.6, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=756,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset17_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset17_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
# #### Dataset 18
# dataset_name = 'ArtificialDataset18'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset18_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset18_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
# #### Dataset 19
# dataset_name = 'ArtificialDataset19'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=150,n_noise=50,
#                                          n_redundant_linear=50,n_redundant_nonlinear=50,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=655,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset19_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset19_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
# #### Dataset 20
# dataset_name = 'ArtificialDataset20'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
#                                          n_redundant_linear=60,n_redundant_nonlinear=60,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset20_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset20_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")
#
#
#
# #### Dataset 21
# dataset_name = 'ArtificialDataset21'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
#                                          n_redundant_linear=100,n_redundant_nonlinear=100,
#                                         flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
#                                                      random_state=996,noise_std=0.5)
#
# csv_guided_path = 'Results_FS_Distributed/ArtificialDataset21_ComplexityGuidedDistributed.csv'
# csv_random_path = 'Results_FS_Distributed/ArtificialDataset21_ComplexityRandomDistributed.csv'
# FS_complexity_experiment_with_distributed(X, y, dict_info_feature, dataset_name,
#                                                 csv_guided_path, csv_random_path,
#                                               path_to_save="Results_ComparisonDistributed_SOTA")





########################################################################################################################
####################                            VERSION CV                                          ####################
########################################################################################################################


### Comenzamos ejecutando los subsets con CV

# Función para generar los subconjuntos de interés para cada dataset
def build_subsets(feature_names, feature_types,k_random=3, random_state=0):
    rng = np.random.RandomState(random_state)
    subsets = {}

    subsets['all'] = list(feature_names)
    inform = [f for f, t in feature_types.items() if t == 'informative']
    noise = [f for f, t in feature_types.items() if t == 'noise']
    redun = [f for f, t in feature_types.items() if t == 'redundant_linear']
    redun_nonlineal = [f for f, t in feature_types.items() if t == 'redundant_nonlinear']

    subsets['informative'] = inform
    subsets['informative+redundant'] = inform + redun
    subsets['informative+redundant_nonLinear'] = inform + redun_nonlineal
    subsets['informative+noise'] = inform + noise

    # selección aleatoria (informativas + ruido/redundantes al azar)
    pool_extra = noise + redun + redun_nonlineal
    if pool_extra and k_random > 0:
        ksel = min(k_random, len(pool_extra))
        rand_pick = rng.choice(pool_extra, size=ksel, replace=False).tolist()
        subsets['informative+rand_extra'] = inform + rand_pick


    return subsets



def evaluate_subsets_cv(X, y, subsets, models, dataset_name,n_splits=5, random_state=0):
    """
    Calcula accuracy y GPS para TODOS los subsets y modelos usando CV.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []

    for subset_name, feat_list in subsets.items():
        cols = [c for c in feat_list if c in X.columns]

        for model_name, model in models.items():
            fold_idx = 0

            for train_idx, test_idx in skf.split(X, y):
                X_train = X.iloc[train_idx][cols]
                X_test  = X.iloc[test_idx][cols]
                y_train = y[train_idx]
                y_test  = y[test_idx]

                # Entrenar
                clf = model
                clf.fit(X_train, y_train)

                # Predicciones
                pred_train = clf.predict(X_train)
                pred_test = clf.predict(X_test)

                # Métricas
                acc_train = accuracy_score(y_train, pred_train)
                acc_test  = accuracy_score(y_test, pred_test)
                gps_train = compute_gps(y_train, pred_train)
                gps_test  = compute_gps(y_test, pred_test)

                # COMPLEJIDAd DEL SUBSET
                data_comp = pd.DataFrame(X_train, columns=cols)
                data_comp["y"] = y_train

                _, df_classes, _ = all_measures_FS(data_comp, save_csv=False, path_to_save=None, name_data=None)

                Hostility = df_classes.loc["dataset", "Hostility"]
                kDN = df_classes.loc["dataset", "kDN"]
                N1 = df_classes.loc["dataset", "N1"]

                # Guardamos registro del fold
                results.append({
                    "dataset": dataset_name,
                    "subset": subset_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "n_features": len(cols),

                    # performance
                    "acc_train": acc_train,
                    "gps_train": gps_train,
                    "acc_test": acc_test,
                    "gps_test": gps_test,

                    # complexity
                    "Hostility": Hostility,
                    "kDN": kDN,
                    "N1": N1
                })

                fold_idx += 1

    results_df = pd.DataFrame(results)

    return results_df




# df_folds = df_perf

def summarize_subset_results(df_folds, save_csv=False, name_data=None, path_to_save="Results_Subsets_Artificial_CV"):
    """
    Agrupa y promedia:
    acc_train, acc_test, gps_train, gps_test, Hostility, kDN, N1
    """

    metrics = ["acc_train", "acc_test", "gps_train", "gps_test",
        "Hostility", "kDN", "N1"]

    df_summary = df_folds.groupby(["dataset", "subset", "model", "n_features"]
    )[metrics].agg(["mean", "std"])

    # Quitamos MultiIndex
    df_summary.columns = [f"{metric}_{stat}" for metric, stat in df_summary.columns]
    df_summary.reset_index(inplace=True)

    if save_csv:
        folds_path = f"{path_to_save}/{name_data}_Subsets_Folds_Performance_Complexity.csv"
        summ_path  = f"{path_to_save}/{name_data}_Subsets_Summary_Performance_Complexity.csv"
        df_folds.to_csv(folds_path, index=False)
        df_summary.to_csv(summ_path, index=False)

    return df_summary


random_state = 0
models = {
    # "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
    # "SVM-linear": SVC(kernel="linear", probability=True, random_state=random_state),
    "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
    # "RandomForest": RandomForestClassifier(random_state=random_state),
    "KNN": KNeighborsClassifier(),
    # "NaiveBayes": GaussianNB(),
    # "DecisionTree": DecisionTreeClassifier(random_state=random_state),
    # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=random_state)
}


# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
#
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
# #### Dataset 7
# dataset_name = 'ArtificialDataset7'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=589,noise_std=0.05)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
#
# #### Dataset 12
# dataset_name = 'ArtificialDataset12'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=987,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
# #### Dataset 14
# dataset_name = 'ArtificialDataset14'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
#                                          n_redundant_linear=30,n_redundant_nonlinear=40,
#                                         flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
#                                                      random_state=95,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
#
# #### Dataset 18
# dataset_name = 'ArtificialDataset18'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
#
# #### Dataset 20
# dataset_name = 'ArtificialDataset20'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
#                                          n_redundant_linear=60,n_redundant_nonlinear=60,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
# #### Dataset 21
# dataset_name = 'ArtificialDataset21'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
#                                          n_redundant_linear=100,n_redundant_nonlinear=100,
#                                         flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
#                                                      random_state=996,noise_std=0.5)
#
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#

#
#
# #### Dataset 18a
# dataset_name = 'ArtificialDataset18a'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=50,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
# #### Dataset 18b
# dataset_name = 'ArtificialDataset18b'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=40,n_noise=20,
#                                          n_redundant_linear=20,n_redundant_nonlinear=20,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
#
# #### Dataset 18c
# dataset_name = 'ArtificialDataset18c'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
#
# #### Dataset 20a
# dataset_name = 'ArtificialDataset20a'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=150,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
# #### Dataset 20b
# dataset_name = 'ArtificialDataset20b'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=60,n_noise=20,
#                                          n_redundant_linear=20,n_redundant_nonlinear=20,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#
#
#
# #### Dataset 20c
# dataset_name = 'ArtificialDataset20c'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=30,n_noise=15,
#                                          n_redundant_linear=15,n_redundant_nonlinear=15,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# feature_names = X.columns.tolist()
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# feature_names = X.columns.tolist()
# fs_list = None
#
# subsets = build_subsets(feature_names, feature_types)
#
#
# df_perf = evaluate_subsets_cv(X, y,subsets=subsets,models=models,dataset_name=dataset_name,
#     n_splits=5,random_state=0)
# summarize_subset_results(df_perf, save_csv=True, name_data=dataset_name, path_to_save="Results_Subsets_Artificial_CV")
#






import glob

## Leemos los subsets de orientación que están en Results_ComparisonDistributed_SOTA en los archivos ComparisonTable

# files = glob.glob("Results_ComparisonDistributed_SOTA/*_comparisonTable.csv")
# files = ['Results_ComparisonDistributed_SOTA/ArtificialDataset12_comparisonTable.csv',
#  'Results_ComparisonDistributed_SOTA/ArtificialDataset2_comparisonTable.csv',
#  'Results_ComparisonDistributed_SOTA/ArtificialDataset14_comparisonTable.csv',
#          'Results_ComparisonDistributed_SOTA/ArtificialDataset18_comparisonTable.csv',
#          'Results_ComparisonDistributed_SOTA/ArtificialDataset20_comparisonTable.csv',
#          'Results_ComparisonDistributed_SOTA/ArtificialDataset21_comparisonTable.csv',
#  'Results_ComparisonDistributed_SOTA/ArtificialDataset7_comparisonTable.csv']
#
# file=files[0]
#
# folder_path = 'Results_FS_ComplexityEvaluation'


### Esta función era para leer los resultados de performance y de complexity de los subsets de
### interés que creamos nosotros (informative, informative+redundant, etc)
### cuando solo se habían ejecutado 1 vez
# Se queda obsoleta porque ahora hacemos cross validation
# def load_custom_results_from_comparison_table():
#     """
#     Carga los resultados de subsets personalizados desde el ComparisonTable existente
#     y los adapta al formato estándar usado en la tabla combinada final.
#     """
#
#     # all_files = glob.glob(f"{folder_path}/*_comparisonTable.csv")
#     all_files = ['Results_FS_ComplexityEvaluation/ArtificialDataset12_comparisonTable.csv',
#     'Results_FS_ComplexityEvaluation/ArtificialDataset2_comparisonTable.csv',
#     'Results_FS_ComplexityEvaluation/ArtificialDataset14_comparisonTable.csv',
#          'Results_FS_ComplexityEvaluation/ArtificialDataset18_comparisonTable.csv',
#          'Results_FS_ComplexityEvaluation/ArtificialDataset20_comparisonTable.csv',
#          'Results_FS_ComplexityEvaluation/ArtificialDataset21_comparisonTable.csv',
#     'Results_FS_ComplexityEvaluation/ArtificialDataset7_comparisonTable.csv']
#     dfs = []
#     for f in all_files:
#         df = pd.read_csv(f)
#         dfs.append(df)
#     df_subsets = pd.concat(dfs, ignore_index=True)
#
#     # Renombrar columnas clave
#     df_subsets = df_subsets.rename(columns={
#         "Dataset": "dataset",
#         "Subset": "method",
#         "mean_acc_mean": "acc_test_mean",
#         "mean_gps_mean": "gps_test_mean",
#         "n_features": "n_features",
#         "Hostility": "Hostility_mean",
#         "kDN": "kDN_mean",
#         "N1": "N1_mean"
#     })
#
#     # Añadir columnas para consistencia con los otros conjuntos
#     df_subsets["model"] = "mean_model"  # porque los resultados ya están promediados
#     df_subsets["type"] = "Custom"
#     df_subsets["filter_corr"] = True
#     df_subsets["acc_train_mean"] = np.nan
#     df_subsets["gps_train_mean"] = np.nan
#     df_subsets["acc_train_std"] = np.nan
#     df_subsets["gps_train_std"] = np.nan
#     df_subsets["acc_test_std"] = 0
#     df_subsets["gps_test_std"] = 0
#     df_subsets["Hostility_std"] = np.nan
#     df_subsets["kDN_std"] = np.nan
#     df_subsets["N1_std"] = np.nan
#
#     #Seleeccionamos columnas
#     df_final = df_subsets[['dataset',
#         "model", "method", "filter_corr", "n_features",
#         "acc_train_mean", "gps_train_mean", "acc_train_std", "gps_train_std",
#         "acc_test_mean", "gps_test_mean","acc_test_std", "gps_test_std",
#         "Hostility_mean","Hostility_std", "N1_mean", "N1_std","kDN_mean","kDN_std", "type"
#     ]].copy()
#
#     valid_methods = ['all','informative','informative+redundant',
#                                    'informative+redundant_nonLinear',
#                                    'informative+noise',
#                                    'informative+rand_extra']
#     df_final = df_final[df_final['method'].isin(valid_methods)]
#
#     return df_final


# folder_path = 'Results_FS_SOTA_CV'
def load_sota_results(folder_path):
    """
    Carga todos los Performance_Folds_*.csv del SOTA,
    los promedia por modelo y métod, y devuelve un DataFrame.
    """
    all_files = glob.glob(f"{folder_path}/*_Performance_Folds.csv")
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dataset_name = os.path.basename(f).split('_SOTA_CV_Performance_Fold')[0]
        df["dataset"] = dataset_name
        dfs.append(df)
    df_sota = pd.concat(dfs, ignore_index=True)

    # Promediar por modelo + métod
    group_cols = ["model", "method", "filter_corr", "n_features",'dataset']
    metrics = ["acc_train", "gps_train", "acc_test", "gps_test", "Hostility", "kDN", "N1"]
    df_summary = (df_sota.groupby(group_cols)[metrics].agg(['mean', 'std']))

    # Aplana el MultiIndex de columnas
    df_summary.columns = [f"{col}_{stat}" for col, stat in df_summary.columns]

    # Devuelve el índice a columnas normales
    df_summary = df_summary.reset_index()

    df_summary["type"] = "SOTA"
    return df_summary



# folder_path = 'Results_FS_Distributed_CV'
def load_distributed_results(folder_path):
    """
    Carga los summaryResults_*.csv del métod Distributed,
    y genera un DataFrame comparable con el del SOTA.
    """
    all_files = glob.glob(f"{folder_path}/*_DistributedCVRandom_OutHigh_SummaryResults.csv")
    dfs = []

    for f in all_files:
        df = pd.read_csv(f)
        dataset_name = os.path.basename(f).split('_DistributedCVRandom_OutHigh_SummaryResults')[0]
        df["dataset"] = dataset_name
        dfs.append(df)
    df_dist = pd.concat(dfs, ignore_index=True)

    # Nos quedamos solo con los modelos que hemos ejecutado para todos
    valid_models = ['SVM-rbf','KNN']
    df_dist = df_dist[df_dist['model'].isin(valid_models)]

    # Cambiamos nombres
    df_dist = df_dist.rename(columns={
        "measure": "method",
        "complexity_Hostility_mean": "Hostility_mean",
        "complexity_N1_mean": "N1_mean",
        "complexity_kDN_mean": "kDN_mean",
        "complexity_Hostility_std": "Hostility_std",
        "complexity_N1_std": "N1_std",
        "complexity_kDN_std": "kDN_std"
    })
    df_dist["type"] = "Distributed"
    df_dist["filter_corr"] = True  # igual que sota

    df_dist.drop(['acc_train_max','acc_test_max','gps_train_max','gps_test_max'],axis=1,inplace=True)

    return df_dist

# df_dist.columns

# # Unimos tod
# folder_sota = 'Results_FS_SOTA_CV'
# sota = load_sota_results(folder_sota)
# folder_dist = 'Results_FS_Distributed_CV'
# dist = load_distributed_results(folder_dist)
# subsets_reference = load_custom_results_from_comparison_table()
#
# dfs = [sota, dist, subsets_reference]
# all_cols = sorted(set().union(*[df.columns for df in dfs]))
#
# for i, df in enumerate(dfs):
#     dfs[i] = df.reindex(columns=all_cols)
#
# df_all = pd.concat(dfs, ignore_index=True)
# df_all_order = df_all[['dataset','method', 'model','acc_train_mean', 'acc_train_std',
#                        'acc_test_mean','acc_test_std','gps_train_mean', 'gps_train_std',
#                         'gps_test_mean', 'gps_test_std',
#                     'Hostility_mean', 'Hostility_std', 'N1_mean', 'N1_std',
#                     'kDN_mean', 'kDN_std','n_features','type','filter_corr']]
# df_all_order.sort_values(by='dataset',inplace=True)
# # out_path = 'Results_ComparisonDistributed_SOTA'
# # df_all_order.to_csv('Results_ComparisonDistributed_SOTA/AllArtificialDatasets_ComparisonTable_CV.csv', index=False)
# # #
# # sota.columns
# # dist.columns
# # subsets_reference.columns




def load_subsets_results(folder_path):
    """
    Carga los <dataset>_Subsets_Summary.csv generados con evaluate_subsets_cv(),
    normaliza nombres de columnas y devuelve un DF comparable a SOTA y Distributed.
    """
    all_files = glob.glob(f"{folder_path}/*_Subsets_Summary_Performance_Complexity.csv")
    dfs = []

    for f in all_files:
        df = pd.read_csv(f)

        # dataset name
        dataset_name = os.path.basename(f).replace("_Subsets_Summary_Performance_Complexity.csv", "")
        df["dataset"] = dataset_name

        dfs.append(df)

    df_sub = pd.concat(dfs, ignore_index=True)

    # Aseguramos que existen columnas comunes
    df_sub["type"] = "Subsets"
    df_sub["filter_corr"] = np.nan

    df_sub = df_sub.rename(columns={"subset": "method"})

    return df_sub






# Unimos tod
folder_path = 'Results_Subsets_Artificial_CV'
subsets_reference = load_subsets_results(folder_path)
folder_sota = 'Results_FS_SOTA_CV'
sota = load_sota_results(folder_sota)
folder_dist = 'Results_FS_Distributed_CV'
distributed = load_distributed_results(folder_dist)

dfs = [sota, distributed, subsets_reference]
all_cols = sorted(set().union(*[df.columns for df in dfs]))

for i, df in enumerate(dfs):
    dfs[i] = df.reindex(columns=all_cols)

df_all = pd.concat(dfs, ignore_index=True)

df_all_order = df_all[[
    'dataset','method','model',
    'acc_train_mean','acc_train_std',
    'acc_test_mean','acc_test_std',
    'gps_train_mean','gps_train_std',
    'gps_test_mean','gps_test_std',
    'Hostility_mean','Hostility_std',
    'N1_mean','N1_std',
    'kDN_mean','kDN_std',
    'n_features','type','filter_corr'
]]

# Orden
method_order = ["all","informative","informative+redundant","informative+redundant_nonLinear",
    "informative+rand_extra","informative+noise","f_classif", "mutual_info", "relief", "rf", "xgb",
     "Hostility_importances_norm","N1_importances_norm","kDN_importances_norm"]

# categoría ordenada
df_all_order["method"] = pd.Categorical(df_all_order["method"], categories=method_order, ordered=True)

# Orden: dataset → model → method
df_all_order = df_all_order.sort_values(by=["dataset", "model", "method"]).reset_index(drop=True)

# df_all_order.sort_values(by='dataset', inplace=True)

out_path = 'Results_ComparisonDistributed_SOTA'
df_all_order.to_csv(f'{out_path}/AllArtificialDatasets_ComparisonTable_CV_neg_high.csv', index=False)
# df_all_order.to_csv(f'{out_path}/AllArtificialDatasets_ComparisonTable_CV_neg_high_DatasetVersions.csv', index=False)
# este cuando ejecutamos los resultados para distintas versiones de los datasets 18 y 20 para ver
# si los resultados se deben a la complejidad o al métod que estamos construyendo
