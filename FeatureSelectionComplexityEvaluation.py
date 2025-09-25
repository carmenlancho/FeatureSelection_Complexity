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
# https://epistasislab.github.io/scikit-rebate/using/
# Para pymrmr hace falta instalar antes pip install numpy Cython
import pymrmr


# Función para generar datos sintéticos a nuestro gusto
def generate_synthetic_dataset(n_samples=200, n_informative=5, n_noise=5,
                        n_redundant_linear=2, random_state=42, noise_std=0.05):
    rng = np.random.RandomState(random_state)

    # Generamos solo informativas + ruido
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,   # no usamos sklearn redundantes
        n_repeated=0,
        shuffle=False,
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    formulas = {}

    # Añadimos redundantes lineales a mano para controlar cómo se han creado
    for j in range(n_redundant_linear):
        # elijo al azar 2 features informativas para combinarlas
        idx1, idx2 = rng.choice(n_informative, size=2, replace=False)
        coef1, coef2 = rng.uniform(-2, 2, size=2)

        if (noise_std==0):
            new_feature = (coef1 * df[f"f{idx1}"] +
                           coef2 * df[f"f{idx2}"])
            new_name = f"f{df.shape[1]}"
            df[new_name] = new_feature

            formulas[new_name] = f"{coef1:.2f}*f{idx1} + {coef2:.2f}*f{idx2}"
        else:
            new_feature = (coef1 * df[f"f{idx1}"] +
                           coef2 * df[f"f{idx2}"] +
                           rng.normal(0, noise_std, size=n_samples))
            new_name = f"f{df.shape[1]}"
            df[new_name] = new_feature

            formulas[new_name] = f"{coef1:.2f}*f{idx1} + {coef2:.2f}*f{idx2} + ruido"

    dict_info_feature = {
        "informative": [f"f{i}" for i in range(n_informative)],
        "noise": [f"f{i}" for i in range(n_informative, n_informative + n_noise)],
        "redundant_linear": list(formulas.keys()),
        "formulas": formulas
    }

    return df, y, dict_info_feature



X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,random_state=0,noise_std=0.01)

print("Informativas:", dict_info_feature["informative"])
print("Ruidosas:", dict_info_feature["noise"])
print("Redundantes lineales:", dict_info_feature["redundant_linear"])
print("Fórmulas:")
for k, v in dict_info_feature["formulas"].items():
    print(f"  {k} = {v}")



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



# Número de features informativas como k
k = len(dict_info_feature["informative"])
feature_names = X.columns.tolist()

# Ejecutamos los métodos de FS
fs_results = select_features_by_filters(X, y, feature_names, k=k)

# Mostrar resultados
for method, info in fs_results.items():
    print(f"\n--------- Método: {method} ---------")
    print("Top-k seleccionadas:", info["selected"])
    print("Scores (top 10):")
    print(info["scores"].head(10))


# Función para generar los subconjuntos de interés para cada dataset
def build_subsets_for_complexity(feature_names, feature_types, fs_selections,
        k_random=3, random_state=0):
    rng = np.random.RandomState(random_state)
    subsets = {}

    subsets['all'] = list(feature_names)
    inform = [f for f, t in feature_types.items() if t == 'informative']
    noise = [f for f, t in feature_types.items() if t == 'noise']
    redun = [f for f, t in feature_types.items() if 'redundant' in t]

    subsets['informative'] = inform
    subsets['informative+redundant'] = inform + redun
    subsets['informative+noise'] = inform + noise

    # selección aleatoria (informativas + ruido/redundantes al azar)
    pool_extra = noise + redun
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


# mapeo a tipos
feature_types = {}
for f in dict_info_feature["informative"]: feature_types[f] = "informative"
for f in dict_info_feature["noise"]: feature_types[f] = "noise"
for f in dict_info_feature["redundant_linear"]: feature_types[f] = "redundant_linear"

# construir subconjuntos
subsets = build_subsets_for_complexity(feature_names, feature_types, fs_results)

for name, feats in subsets.items():
    print(name, "->", feats)



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

    for subset_name, features in subsets.items():
        Xsub = preprocessing.scale(X[features])
        datos = pd.DataFrame(Xsub, columns=features)
        datos['y'] = y
        df_measures, df_classes, extra_results = all_measures(datos, save_csv, path_to_save, subset_name)

        # Guardar fila resumen (total del dataset)
        total_row = df_classes.loc["dataset"].copy()
        total_row.name = subset_name
        total_row["n_features"] = len(features)  # extra info
        results_total.append(total_row)

        results_classes[subset_name] = df_classes
        extras_host[subset_name] = extra_results

    results_total = pd.DataFrame(results_total)

    return results_total, results_classes, extras_host





# 1. Dataset sintético
df, y, dict_info = generate_synthetic_dataset(n_samples=5000, n_informative=10, n_noise=4, n_redundant_linear=5)

# 2. Mapear tipos
feature_types = {}
for f in dict_info["informative"]: feature_types[f] = "informative"
for f in dict_info["noise"]: feature_types[f] = "noise"
for f in dict_info["redundant_linear"]: feature_types[f] = "redundant_linear"

# 3. FS clásico
# fs_results = select_features_by_filters(df, y, df.columns.tolist())

# Número de features informativas como k
k = len(dict_info_feature["informative"])
feature_names = df.columns.tolist()

# Ejecutamos los métodos de FS
fs_results = select_features_by_filters(df, y, feature_names, k=k)

# 4. Subsets
subsets = build_subsets_for_complexity(df.columns, feature_types, fs_results)

# 5. Evaluación de complejidad
results_total, results_classes, extras_host = evaluate_complexity_across_subsets(df, y, subsets)

print("=== Complejidad total por subset ===")
print(results_total)

print("\n=== Complejidad por clases en subset 'informative' ===")
print(results_classes["informative"])



