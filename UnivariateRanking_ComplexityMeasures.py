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
def univariate_complexity(X, y, measures=["Hostility", "N1", "kDN"], save_csv=False, path="Results_UnivariateRanking_CM", dataset_name=None):
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





#
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
#                                                      n_redundant_nonlinear=2,
#                                 flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)
#
# # Calcular complejidad univariante
# df_results, dataset_vals = univariate_complexity(X, y, measures=["Hostility", "N1", "kDN"])
# # info de las redundantes
# redundant_sources = get_redundant_feature_relation(dict_info_feature)
# # Obtener ranking de informativas
# summary = evaluate_univariate_ranking(dataset_vals, dict_info_feature,redundant_sources)




## Función para ejecutar diversos métodos de FS tipo filtro del SOTA y tb los nuestros
def select_features_by_filters_and_complexity(X, y, feature_names,dataset_name,dict_info_feature,
                                              k=None,methods=None,
                                              complexity_measures = ["Hostility", "N1", "kDN"],
                                              random_state=0):
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
        methods = ["mutual_info", "f_classif", "rf", "relief","xgboost",
                   'complexity']

    Xarr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    n_features = Xarr.shape[1]
    if k is None:
        k = max(1, int(np.sqrt(n_features)))  # heuristic fallback

    results = {}
    summary_rows = []  # resúmenes para el CSV final

    # # standardize for methods that need it
    # scaler = StandardScaler()
    # Xs = scaler.fit_transform(Xarr)
    # ya hemos preprocesado previamente

    # info de las redundantes
    redundant_sources = get_redundant_feature_relation(dict_info_feature)

    # mutual information
    if "mutual_info" in methods:
        mi = mutual_info_classif(Xarr, y, random_state=random_state)
        s = pd.Series(mi, index=feature_names).sort_values(ascending=False)
        results["mutual_info"] = {"scores": s, "selected": list(s.index[:k])}
        summary_mi = evaluate_univariate_ranking(pd.DataFrame({"score": s}), dict_info_feature, redundant_sources)
        row = summary_mi.iloc[0].copy()
        row["method"] = "mutual_info"
        summary_rows.append(row)

    # ANOVA F (f_classif)
    if "f_classif" in methods:
        F, p = f_classif(Xarr, y)
        s = pd.Series(F, index=feature_names).sort_values(ascending=False)
        results["f_classif"] = {"scores": s, "selected": list(s.index[:k])}
        summary_f = evaluate_univariate_ranking(pd.DataFrame({"score": s}), dict_info_feature, redundant_sources)
        row = summary_f.iloc[0].copy()
        row["method"] = "f_classif"
        summary_rows.append(row)

    # Random Forest importance
    if "rf" in methods:
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        rf.fit(Xarr, y)
        imp = rf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["rf"] = {"scores": s, "selected": list(s.index[:k])}
        summary_rf = evaluate_univariate_ranking(pd.DataFrame({"score": s}), dict_info_feature, redundant_sources)
        row = summary_rf.iloc[0].copy()
        row["method"] = "rf"
        summary_rows.append(row)

    # ReliefF
    if "relief" in methods:
        rf_sel = ReliefF(n_features_to_select=Xarr.shape[1]) # n_neighbors usamos el valor por defecto de la librería
        rf_sel.fit(Xarr, y)
        scores = rf_sel.feature_importances_
        s = pd.Series(scores, index=feature_names).sort_values(ascending=False)
        results["relief"] = {"scores": s, "selected": list(s.index[:k])}
        summary_relief = evaluate_univariate_ranking(pd.DataFrame({"score": s}), dict_info_feature, redundant_sources)
        row = summary_relief.iloc[0].copy()
        row["method"] = "relief"
        summary_rows.append(row)

    # XGBoost
    if "xgboost" in methods:
        xgb_clf = xgb.XGBClassifier(eval_metric="logloss",random_state=random_state)
        xgb_clf.fit(Xarr, y)
        imp = xgb_clf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["xgb"] = {"scores": s, "selected": list(s.index[:k])}
        summary_xgb = evaluate_univariate_ranking(pd.DataFrame({"score": s}), dict_info_feature, redundant_sources)
        row = summary_xgb.iloc[0].copy()
        row["method"] = "xgb"
        summary_rows.append(row)

    # COMPLEXITY (univariate study)
    if 'complexity' in methods:
        # complexity_measures = ["Hostility", "N1", "kDN"]
        df_results, dataset_vals = univariate_complexity(X, y, measures=complexity_measures,save_csv = True,
                                                         path = "Results_UnivariateRanking_CM", dataset_name = dataset_name)
        # no ponemos Xarr porque necesita columns

        for m in complexity_measures:
            s = dataset_vals[m].sort_values(ascending=True)  # menor es mejor
            results[f"complexity_{m}"] = {"scores": s, "selected": list(s.index[:k])}
            summary_m = evaluate_univariate_ranking(pd.DataFrame({"score": s}), dict_info_feature, redundant_sources)
            row = summary_m.iloc[0].copy()
            row["method"] = f"complexity_{m}"
            summary_rows.append(row)


    # ----------- Guardar resumen -----------
    summary_df = pd.DataFrame(summary_rows).set_index("method")
    fname = f"Results_UnivariateRanking_CM/TopFeaturesSummary_AllMethods_{dataset_name}.csv"
    summary_df.to_csv(fname, index=True)

    return results, summary_df
# Para estas primeras pruebas, para no tener que elegir manualmente el k, podemos
# escoger k como el número de features realmente informativas





# Función para generar los subconjuntos de interés para cada dataset
# Copiada de FeatureSelectionComplexityEvaluation
def build_subsets_for_complexity(feature_names, feature_types, fs_selections,
        k_random=3, random_state=0):
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
    selected_measures = ["Hostility", "N1", "N2", "kDN", "LSC", "CLD", "TD_U", "DCP", "F1","F2","F3","F4", "L1"]

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
    results_summary = []
    detailed_results = {}

    classes = np.unique(y)

    for subset_name, features in subsets.items():
        Xsub = X[features].values
        subset_scores = {}

        for model_name, model in models.items():
            y_pred = cross_val_predict(model, Xsub, y, cv=skf)

            acc = accuracy_score(y, y_pred)
            gps = compute_gps(y, y_pred)

            # Accuracy por clase
            acc_per_class = {}
            for c in classes:
                idx = (y == c)
                acc_per_class[int(c)] = accuracy_score(y[idx], y_pred[idx])

            subset_scores[model_name] = {
                "acc": acc,
                "gps": gps,
                "acc_per_class": acc_per_class
            }

            registro = {"subset": subset_name,"model": model_name,"acc": acc,"gps": gps}
            for cls, val in acc_per_class.items():
                registro[f"acc_class_{cls}"] = val # accuracy por clase
            results_summary.append(registro)

        detailed_results[subset_name] = subset_scores

    results_df = pd.DataFrame(results_summary).set_index(["subset", "model"])

    return results_df, detailed_results




def save_complexity_csv(dataset_name, subset_name, results_classes, extras_host,
                        path="Results_FS_ComplexityEvaluation_WithUnivariate"):
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




def save_models_csv(dataset_name, results_models, path="Results_FS_ComplexityEvaluation_WithUnivariate"):
    """
    Guarda en CSV el rendimiento de TODOS los modelos en TODOS los subsets para un dataset.
    results_models debe ser un DataFrame con índices [subset, model] y columnas [acc, gps, acc_class_*].
    """
    # Reset index para que subset y model queden como columnas normales
    final = results_models.reset_index()
    final.insert(0, "dataset", dataset_name)  # añadimos dataset como primera columna

    fname = f"{path}/{dataset_name}_modelsPerformance.csv"
    final.to_csv(fname, index=False)
    return final



# dataset_name = 'prueba'
def FS_complexity_experiment_uni(X, y, dict_info_feature, dataset_name,path_to_save="Results_FS_ComplexityEvaluation_WithUnivariate"):
    # Número de features informativas como k
    k = len(dict_info_feature["informative"])
    feature_names = X.columns.tolist()

    fs_results,_ = select_features_by_filters_and_complexity(X, y, feature_names,dataset_name,
                                                            dict_info_feature, k=k)
    # Construir subconjuntos
    feature_types = {}
    for f in dict_info_feature["informative"]: feature_types[f] = "informative"
    for f in dict_info_feature["noise"]: feature_types[f] = "noise"
    for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
    for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
    subsets = build_subsets_for_complexity(feature_names, feature_types, fs_results)

    # Evaluación de complejidad
    results_total, results_classes, extras_host = evaluate_complexity_across_subsets(X, y, subsets)

    # Evaluación de modelos
    results_models, detailed_models = evaluate_models_across_subsets(X, y, subsets)

    # Guardar csvs de complejidad por subset
    for subset_name in subsets.keys():
        save_complexity_csv(dataset_name, subset_name, results_classes, extras_host, path_to_save)

    # Guardar csv de modelos por dataset
    save_models_csv(dataset_name, results_models, path_to_save)

    # --- TABLA DE COMPARACIÓN ---
    # results_models tiene MultiIndex (subset, model),
    # hacemos un resumen (medias por subset)
    summary_models = results_models.groupby(level="subset")[["acc", "gps"]].agg(["mean", "max", "std"])
    # Formato nombres columnas
    summary_models.columns = [f"{m}_{stat}" for m, stat in summary_models.columns]


    # Juntamos en una sola tabla
    results_all = results_total.join(summary_models, how="left")

    # Nombres y tal
    results_all["dataset_name"] = dataset_name
    comparison_table = results_all.set_index(["dataset_name", results_all.index])
    comparison_table.index.names = ["Dataset", "Subset"]

    fname = f"{path_to_save}/{dataset_name}_comparisonTable.csv"
    comparison_table.to_csv(fname)

    return comparison_table, results_classes, detailed_models




#
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
#                                                      n_redundant_nonlinear=2,
#                                 flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)
#
# k = len(dict_info_feature["informative"])
# feature_names = X.columns.tolist()
#
# # Ejecutamos los métodos de FS
# fs_results = select_features_by_filters_and_complexity(X, y, feature_names,k=k,
#                         methods=["mutual_info", "rf", "xgboost", "complexity"],
#                                 complexity_measures=["Hostility", "N1",'kDN'])
#
# # fs_results["complexity_Hostility"]
# # fs_results["complexity_N1"]
#
#
#
# # Construir subconjuntos
# feature_types = {}
# for f in dict_info_feature["informative"]: feature_types[f] = "informative"
# for f in dict_info_feature["noise"]: feature_types[f] = "noise"
# for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
# for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
# subsets = build_subsets_for_complexity(feature_names, feature_types, fs_results)
#

# # Resultados específicos del ranking univariante
#
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4,
#                                                      n_redundant_nonlinear=2,
#                                 flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5], random_state=0, noise_std=0.01)
#
# # Calcular complejidad univariante
# df_results, dataset_vals = univariate_complexity(X, y, measures=["Hostility", "N1", "kDN"])
# # info de las redundantes
# redundant_sources = get_redundant_feature_relation(dict_info_feature)
# # Obtener ranking de informativas
# summary = evaluate_univariate_ranking(dataset_vals, dict_info_feature,redundant_sources)
#



path_to_save = "Results_FS_ComplexityEvaluation_WithUnivariate"
### Dataset 1
dataset_name = 'ArtificialDataset1'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                        flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)




### Dataset 2
dataset_name = 'ArtificialDataset2'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                    flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

### Dataset 3
dataset_name = 'ArtificialDataset3'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=25,n_noise=5,
                                         n_redundant_linear=7,n_redundant_nonlinear=8,
                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=0,noise_std=0.05)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

### Dataset 4
dataset_name = 'ArtificialDataset4'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=15,n_noise=15,
                                         n_redundant_linear=4,n_redundant_nonlinear=5,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=10,noise_std=0.01)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

### Dataset5
dataset_name = 'ArtificialDataset5'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=25,n_noise=15,
                                         n_redundant_linear=8,n_redundant_nonlinear=7,
                                     flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=10,noise_std=0.05)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

#### Dataseet 6
dataset_name = 'ArtificialDataset6'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=10000,n_informative=8,n_noise=15,
                                         n_redundant_linear=4,n_redundant_nonlinear=5,
                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.01)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)
# ha tardado como unas 7-8 horas

#### Dataset 7
dataset_name = 'ArtificialDataset7'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
                                         n_redundant_linear=10,n_redundant_nonlinear=10,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.05)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

#### Dataset 8
dataset_name = 'ArtificialDataset8'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=40,n_noise=15,
                                         n_redundant_linear=15,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=86785,noise_std=0.1)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

#### Dataset 9
dataset_name = 'ArtificialDataset9'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=10,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=20,
                                        flip_y=0, class_sep=0.7, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=959,noise_std=0.3)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

#### Dataset 10
dataset_name = 'ArtificialDataset10'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=2000,n_informative=6,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=0.8, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=959,noise_std=0.3)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

#### Dataset 11
dataset_name = 'ArtificialDataset11'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=20,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=0.6, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=959,noise_std=0.1)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)

#### Dataset 12
dataset_name = 'ArtificialDataset12'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=987,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)



#### Dataset 13
dataset_name = 'ArtificialDataset13'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.4],
                                                     random_state=95,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)


#### Dataset 14
dataset_name = 'ArtificialDataset14'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=95,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)


#### Dataset 15
dataset_name = 'ArtificialDataset15'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=40,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=78,noise_std=0.1)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)


#### Dataset 16
dataset_name = 'ArtificialDataset16'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=50,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.2],
                                                     random_state=756,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)



#### Dataset 17
dataset_name = 'ArtificialDataset17'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.6, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=756,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)



#### Dataset 18
dataset_name = 'ArtificialDataset18'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=9462,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)




#### Dataset 19
dataset_name = 'ArtificialDataset19'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=150,n_noise=50,
                                         n_redundant_linear=50,n_redundant_nonlinear=50,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=655,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)



#### Dataset 20
dataset_name = 'ArtificialDataset20'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
                                         n_redundant_linear=60,n_redundant_nonlinear=60,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=4556,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)



#### Dataset 21
dataset_name = 'ArtificialDataset21'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
                                         n_redundant_linear=100,n_redundant_nonlinear=100,
                                        flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
                                                     random_state=996,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)




#### Dataset 22
dataset_name = 'ArtificialDataset22'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=500,n_noise=150,
                                         n_redundant_linear=150,n_redundant_nonlinear=150,
                                        flip_y=0.2, class_sep=0.7, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=996,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)



#### Dataset 23
dataset_name = 'ArtificialDataset23'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=5000,n_noise=1500,
                                         n_redundant_linear=1500,n_redundant_nonlinear=1500,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=996,noise_std=0.5)
comparison_table, results_classes, detailed_models = FS_complexity_experiment_uni(X, y, dict_info_feature,dataset_name)


