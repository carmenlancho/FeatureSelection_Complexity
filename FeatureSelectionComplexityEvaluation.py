##### 24/09/2025
#### En este script vamos a calcular (hacemos las funciones para) la complejidad de distintos conjuntos de datos
#### en diferentes versiones: todas las variables, solo las informativas, solo las redundantes o ruidosa,
### un mix de ellas, las que seleccionan algunos métodos de filtro de FS del estado del artee
### La idea es ver cómo cambia el comportamiento de las medidas de complejidad en esas circunstancias
### para evaluar, un poco a priori, si las podemos utilizar para crear un method de FS basado en
#### medidas de complejidad. en el notebook TrackingCentroides_Hostility hemos visto que las variables que son
### literalmente la copia de otras, son fáciles de pillar porque muestran el mismo comportamiento.
### Sin embargo, las que son redundantes por ser combinación lineal de otras, ya no se ve claramente cómo pillarlas
### Leyendo el SOTA veo que ese es uno de los principales problemas de los métodos de filtro.


import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from skrebate import ReliefF
from All_measures import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os



# https://epistasislab.github.io/scikit-rebate/using/
# Para pymrmr hace falta instalar antes pip install numpy Cython
import pymrmr

# flip_y Proporción de etiquetas que se invierte aleatoriamente
# class_sep  separación entre clases en el espacio de características, 1 es el máximo de separabilidad
# n_clusters_per_class  Número de clusters gausianos por clase. Más clusters → más difícil.
#  weights controla desbalance de clases.
# More than n_samples samples may be returned if the sum of weights exceeds 1.
# Note that the actual class proportions will not exactly match weights when flip_y isn’t 0

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


#
#
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                                      random_state=0,noise_std=0.01)





## Función para ejecutar diversos métodos de FS tipo filtro del SOTA
def select_features_by_filters(X, y, feature_names,k=None,methods=None,random_state=0):
    """
    Aplica varios métodos de filtro y devuelve:
        selections: dict {method_name: {"scores": pd.Series(index=feature_names), "selected": [names...] }}

    - X: np.ndarray or DataFrame
    - y: array-like
    - feature_names: list of names (length = X.shape[1])
    - k: número de features a seleccionar (si None -> k = n_informative_guess ~ sqrt(n_features) fallback)
    - methods: lista de strings entre {"mutual_info","f_classif","rf","relief"}
    """
    if methods is None:
        methods = ["mutual_info", "f_classif", "rf", "relief"]

    Xarr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    n_features = Xarr.shape[1]
    if k is None:
        k = max(1, int(np.sqrt(n_features)))  # heuristic fallback

    results = {}

    # standardize for methods that need it
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xarr)

    # mutual information
    if "mutual_info" in methods:
        mi = mutual_info_classif(Xs, y, random_state=random_state)
        s = pd.Series(mi, index=feature_names).sort_values(ascending=False)
        results["mutual_info"] = {"scores": s, "selected": list(s.index[:k])}

    # ANOVA F (f_classif)
    if "f_classif" in methods:
        F, p = f_classif(Xs, y)
        s = pd.Series(F, index=feature_names).sort_values(ascending=False)
        results["f_classif"] = {"scores": s, "selected": list(s.index[:k])}

    # Random Forest importance
    if "rf" in methods:
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        rf.fit(Xs, y)
        imp = rf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["rf"] = {"scores": s, "selected": list(s.index[:k])}

    # ReliefF
    if "relief" in methods:
        rf_sel = ReliefF(n_features_to_select=Xs.shape[1]) # n_neighbors usamos el valor por defecto de la librería
        rf_sel.fit(Xs, y)
        scores = rf_sel.feature_importances_
        s = pd.Series(scores, index=feature_names).sort_values(ascending=False)
        results["relief"] = {"scores": s, "selected": list(s.index[:k])}


    return results
# Para estas primeras pruebas, para no tener que elegir manualmente el k, podemos
# escoger k como el número de features realmente informativas



# Función para generar los subconjuntos de interés para cada dataset
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
    selected_measures = ["Hostility", "N1", "N2", "kDN", "LSC", "CLD", "TD_U", "DCP", "F1", "L1"]

    for subset_name, features in subsets.items():
        Xsub = preprocessing.scale(X[features])
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





# -----------------------------
# Plot 1: complejidad total por subset de un dataset
# -----------------------------
def plot_complexity_totals(results_total, dataset_name):
    """
    results_total: DataFrame con filas=subsets, columnas=medidas (ej. output[0] de evaluate_complexity_across_subsets)
    dataset_name: str
    """

    results_total = results_total.iloc[:, :-1] #  quitamos n_features
    df = results_total.reset_index().melt(
        id_vars=["index"], var_name="Measure", value_name="Value"
    )

    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="index", y="Value", hue="Measure")
    plt.title(f"Complejidad total por subset – {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plot 2: complejidad por clase para un subset concreto
# -----------------------------
def plot_class_complexity(df_classes, subset_name, dataset_name):
    """
    df_classes: DataFrame con filas=clases+dataset, columnas=medidas (ej. results_classes[subset_name])
    subset_name: str
    dataset_name: str
    """
    plt.figure(figsize=(8,5))
    sns.heatmap(df_classes, annot=True, fmt=".2f", cmap="mako")
    plt.title(f"Complejidad por clase – {dataset_name}, subset {subset_name}")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plot 3: comparación entre datasets para una medida
# -----------------------------
# def plot_across_datasets(comparison_table, measure):
#     """
#     comparison_table: DataFrame con índice (Dataset, Subset), columnas=medidas
#     measure: str, nombre de la medida a comparar
#     """
#     df = comparison_table.reset_index()
#     plt.figure(figsize=(10,6))
#     sns.barplot(data=df, x="Subset", y=measure, hue="Dataset")
#     plt.title(f"Comparación de {measure} entre datasets")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()



def plot_across_datasets(results_total, results_classes, measure, dataset_name="Dataset"):
    """
    results_total: DataFrame (índice = subsets, columnas = medidas)
    results_classes: dict {subset: df_classes}
    measure: str, nombre de la medida
    dataset_name: nombre del dataset (para el título)
    """

    # --- Unificar dataset y clases ---
    rows = []
    for subset in results_total.index:
        # valor total
        rows.append({
            "Subset": subset,
            "Grupo": "dataset",
            measure: results_total.loc[subset, measure]
        })

        # valores por clase
        df_cls = results_classes[subset]
        for cls in df_cls.index:
            if cls != "dataset":
                rows.append({
                    "Subset": subset,
                    "Grupo": f"Clase {cls}",
                    measure: df_cls.loc[cls, measure]
                })

    df_long = pd.DataFrame(rows)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_long, x="Subset", y=measure, hue="Grupo")
    plt.title(f"{measure} por subset y clase ({dataset_name})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# # -----------------------------
# # Ejemplo con 2 datasets
# # -----------------------------
# # Dataset 1
# X1, y1 = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=0)
# X1 = pd.DataFrame(X1, columns=[f"f{i}" for i in range(10)])
# subsets1 = {
#     "all": list(X1.columns),
#     "informative": [f"f{i}" for i in range(5)],
#     "random3": np.random.choice(X1.columns, 3, replace=False).tolist()
# }
# res1, res1_classes, _ = evaluate_complexity_across_subsets(X1, y1, subsets1)
#
# # Dataset 2
# X2, y2 = make_classification(n_samples=300, n_features=12, n_informative=4, n_redundant=4, random_state=1)
# X2 = pd.DataFrame(X2, columns=[f"f{i}" for i in range(12)])
# subsets2 = {
#     "all": list(X2.columns),
#     "informative": [f"f{i}" for i in range(4)],
#     "random4": np.random.choice(X2.columns, 4, replace=False).tolist()
# }
# res2, res2_classes, _ = evaluate_complexity_across_subsets(X2, y2, subsets2)
#
# # Comparación
# results_all = {"Dataset1": res1, "Dataset2": res2}
# comparison_table = build_comparison_table(results_all)
#
# # Mostrar tabla coloreada
# comparison_table.style.background_gradient(cmap="viridis")
#
#
# # Para un dataset
# plot_complexity_totals(res1, "Dataset1")
#
# # Para ver detalle de un subset concreto (ej. "informative")
# plot_class_complexity(res1_classes["informative"], "informative", "Dataset1")
#
# # Comparar entre datasets en una medida concreta (ej. "Hostility")
# plot_across_datasets(results_total, results_classes, measure="Hostility", dataset_name="synthetic1")



#### Vamos a obtener resultados de performance de modelos también
## Los evaluamos con accuracy y con GPS


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

        # Mejor modelo por GPS (puedes cambiar a "acc" si prefieres)
        best_model = max(subset_scores.items(), key=lambda x: x[1]["gps"])
        best_model_name, best_scores = best_model

        results_summary.append({
            "subset": subset_name,
            "best_model": best_model_name,
            "best_acc": best_scores["acc"],
            "best_gps": best_scores["gps"]
        })

        detailed_results[subset_name] = subset_scores

    results_df = pd.DataFrame(results_summary).set_index("subset")

    return results_df, detailed_results



# results_df, detailed_results = evaluate_models_across_subsets(X, y, subsets)
#
# results_df
# detailed_results


# Para guardar los resultados en csv
def save_subset_results_to_csv(dataset_name, subset_name,
                               complex_classes_total, complex_instance,
                               models_results, best_model,
                               output_dir="Results_FS_ComplexityEvaluation"):
    """
    CSV con toda la información de un subset:
    - Complejidad por clase + total
    - Complejidad a nivel instancia
    - Resultados de todos los modelos (acc, gps, acc_per_class)
    - Indica mejor modelo

    Parameters
    ----------
    dataset_name : str
    subset_name : str
    complex_classes_total : DataFrame (complexity por clase y total)
    complex_instance : DataFrame (complexity por instancia)
    models_results : dict {model_name: {"acc", "gps", "acc_per_class": {...}}}
    best_model : str (nombre del modelo ganador)
    output_dir : str, carpeta donde guardar los CSVs
    """

    os.makedirs(output_dir, exist_ok=True)
    filename = f"ComplexityEvaluation_FS_{dataset_name}_{subset_name}.csv"
    filepath = os.path.join(output_dir, filename)

    # Parte 1: complejidad por clase y dataset
    part1 = complex_classes_total.copy()
    part1["type"] = "complexity_class_total"

    # Parte 2: complejidad a nivel instancia
    part2 = complex_instance.copy()
    part2["type"] = "complexity_instance"

    # Parte 3: resultados de modelos
    rows = []
    for model, res in models_results.items():
        row = {
            "model": model,
            "acc": res["acc"],
            "gps": res["gps"],
            "is_best": (model == best_model),
            "type": "model"
        }
        # accuracy por clase
        for c, acc_c in res["acc_per_class"].items():
            row[f"acc_class_{c}"] = acc_c
        rows.append(row)

    part3 = pd.DataFrame(rows)

    # Unir tod
    final = pd.concat([part1.reset_index(), part2.reset_index(), part3],
                      axis=0, ignore_index=True)

    # Guardar CSV
    final.to_csv(filepath, index=False)
    return final



def FS_complexity_experiment(X, y, dict_info_feature, dataset_name,output_dir="Results_FS_ComplexityEvaluation"):
    # Número de features informativas como k
    k = len(dict_info_feature["informative"])
    feature_names = X.columns.tolist()

    # Ejecutamos los métodos de FS
    fs_results = select_features_by_filters(X, y, feature_names, k=k)

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

    # Guardamos resultados de cada subset
    for subset_name in subsets.keys():
        complex_classes_total = results_classes[subset_name]
        complex_instance = extras_host[subset_name]["instance_measures"]
        models_results = detailed_models[subset_name]
        best_model = results_models.loc[subset_name, "best_model"]

        save_subset_results_to_csv(dataset_name, subset_name,
                                   complex_classes_total, complex_instance,
                                   models_results, best_model,
                                   output_dir=output_dir)

    # Juntamos en una sola tabla
    results_all = results_total.join(results_models, how="left")

    # Para comparación multi-dataset
    results_all["dataset_name"] = dataset_name
    comparison_table = results_all.set_index(["dataset_name", results_all.index])
    comparison_table.index.names = ["Dataset", "Subset"]

    return comparison_table, results_classes, detailed_models






#########################################################################################################3
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=15,n_noise=2,
                                         n_redundant_linear=2,n_redundant_nonlinear=2,
                                                     random_state=86785,noise_std=0.1)
generate_synthetic_dataset(n_samples=1000, n_informative=10, n_noise=2,n_redundant_linear=4, n_redundant_nonlinear=2,
                                flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=0.5, random_state=0, noise_std=0.01)

# Número de features informativas como k
k = len(dict_info_feature["informative"])
feature_names = X.columns.tolist()

# Ejecutamos los métodos de FS
fs_results = select_features_by_filters(X, y, feature_names, k=k)

# construir subconjuntos
feature_types = {}
for f in dict_info_feature["informative"]: feature_types[f] = "informative"
for f in dict_info_feature["noise"]: feature_types[f] = "noise"
for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"
for f in dict_info_feature["redundant_nonlinear"]: feature_types[f] = "redundant_nonlinear"
subsets = build_subsets_for_complexity(feature_names, feature_types, fs_results)


results_total, results_classes, extras_host = evaluate_complexity_across_subsets(X, y, subsets)


# Para un dataset
plot_complexity_totals(results_total, "Dataset1")
# Para ver detalle de un subset concreto (ej. "informative")
plot_class_complexity(results_classes["informative"], "informative", "Dataset1")
# Para una medida concreta
plot_across_datasets(results_total, results_classes, measure="Hostility", dataset_name="synthetic1")

results_all = {
    "dataset1": results_total}

# Comparación
results_all = {"Dataset1": results_total}
comparison_table = build_comparison_table(results_all)
#
#




# comparison_table = build_comparison_table(results_all)
# display(comparison_table.style.background_gradient(cmap="viridis"))


X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=15,n_noise=2,
                                         n_redundant_linear=2,n_redundant_nonlinear=2,
                                                     random_state=86785,noise_std=0.1)

dataset_name = 'dataset_prueba'
comparison_table, results_classes, detailed_models = FS_complexity_experiment(X, y, dict_info_feature,dataset_name)
comparison_table


