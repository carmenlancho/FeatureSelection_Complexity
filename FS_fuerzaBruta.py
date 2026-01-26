## 22/01/2026
## Con los datasets artificiales ya hemos visto que las medidas de complejidad detectan cuáles son las variables
## realmente informativas. Ahora vamos a hacer una pequeña evaluación con datos reales.
## Para ello, hacemos FS por fuerza bruta, es decir, calculamos la performance con todos los subocnjuntos posibles de variables

import numpy as np
import pandas as pd
import os
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone
from tqdm import tqdm # barra de progreso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed



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



## FS por fuerza bruta
def brute_force_evaluation(X, y, models_dict, k_folds=5, verbose=True):
    """
    Evalúa todas las combinaciones posibles de variables para múltiples modelos.

    Args:
        X (pd.DataFrame): Variables predictoras.
        y (pd.Series): Variable objetivo binaria.
        models_dict (dict): Diccionario {nombre_modelo: instancia_modelo}.
        k_folds (int): Número de splits para Cross Validation.

    Returns:
        pd.DataFrame: DataFrame con los resultados detallados por fold, modelo y combinación.
    """
    features = X.columns.tolist()
    n_features = len(features)
    results = []

    # Configuramos Cross-Validation estratificado para mantener balance de clases
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Calculamos el número total de iteraciones para la barra de progreso
    # Suma de combinaciones C(n, k) para k=1 hasta n
    total_combinations = (2 ** n_features) - 1

    if verbose:
        print(f"--- Iniciando Fuerza Bruta ---")
        print(f"Total variables: {n_features}")
        print(f"Total combinaciones a evaluar: {total_combinations}")
        print(f"Modelos a evaluar: {list(models_dict.keys())}")
        print("-" * 30)

    # Iterador con barra de progreso
    pbar = tqdm(total=total_combinations) if verbose else None

    # 1. Bucle por tamaño del subconjunto (de 1 a N)
    for r in range(1, n_features + 1):
        # 2. Bucle por cada combinación de tamaño r
        for combo in combinations(features, r):
            features_subset = list(combo)
            X_subset = X[features_subset].values  # Usamos numpy array para velocidad

            # 3. Bucle por Modelo
            for model_name, model_instance in models_dict.items():

                # 4. Bucle por Folds (Cross Validation)
                fold_idx = 1
                for train_index, test_index in skf.split(X_subset, y):
                    X_train, X_test = X_subset[train_index], X_subset[test_index]
                    y_train, y_test = y[train_index], y[test_index]


                    # Clonamos el modelo para asegurar un entrenamiento limpio en cada fold
                    clf = clone(model_instance)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    # Cálculo de métricas
                    acc = accuracy_score(y_test, y_pred)
                    gps_score = compute_gps(y_test, y_pred)

                    # Guardar resultados
                    results.append({
                        'model': model_name,
                        'n_features': r,
                        'feature_set': str(tuple(features_subset)),  # String para poder agrupar luego
                        'fold': fold_idx,
                        'accuracy': acc,
                        'gps': gps_score
                    })
                    fold_idx += 1

            if pbar: pbar.update(1)

    if pbar: pbar.close()

    return pd.DataFrame(results)






############################ VERSION PARALELIZADA ##########################################3

def evaluate_single_combination(features_subset, X_values, y, models_dict, skf):
    """
    Función que será ejecutada por cada núcleo para una combinación específica.
    """
    local_results = []
    # Bucle por Modelo
    for model_name, model_instance in models_dict.items():
        fold_idx = 1
        # Bucle por Folds
        for train_index, test_index in skf.split(X_values, y):
            X_train, X_test = X_values[train_index], X_values[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = clone(model_instance)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            local_results.append({
                'model': model_name,
                'n_features': len(features_subset),
                'feature_set': str(tuple(features_subset)),
                'fold': fold_idx,
                'accuracy': accuracy_score(y_test, y_pred),
                'gps': compute_gps(y_test, y_pred)
            })
            fold_idx += 1
    return local_results

def brute_force_evaluation_parallel(X, y, models_dict, k_folds=5, verbose=True):
    features = X.columns.tolist()
    n_features = len(features)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # --- Configuración de núcleos ---
    total_cores = os.cpu_count()
    use_cores = max(1, (total_cores -1))
    if verbose:
        print(f"Servidor detectado: {total_cores} núcleos. Usando: {use_cores}")

    # Generar todas las combinaciones primero
    all_combos = []
    for r in range(1, n_features + 1):
        for combo in combinations(features, r):
            all_combos.append(list(combo))

    # Ejecución paralela
    # delayed envuelve la función para que no se ejecute inmediatamente
    results_nested = Parallel(n_jobs=use_cores)(
        delayed(evaluate_single_combination)(
            combo, X[combo].values, y, models_dict, skf
        ) for combo in tqdm(all_combos, disable=not verbose, desc="Evaluando combinaciones")
    )

    # Aplanar la lista de listas de resultados
    flattened_results = [item for sublist in results_nested for item in sublist]
    return pd.DataFrame(flattened_results)


# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    list_datasets = [#'Australian.csv','bands.csv','credit-g.csv',
                     'plasma_retinol.csv',
                     'pollution.csv','vehicle2.csv','diabetic_retinopathy.csv',
                     'sylvine.csv','ring.csv','pyrim.csv'] # 'parkinsons.csv', # 'wdbc.csv','ionosphere.csv','spambase.csv','sonar.csv'
    models = {
        "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
        "KNN": KNeighborsClassifier()
    }

    path2 = "datasets"
    # os.makedirs('Results_FS_bruto', exist_ok=True)

    for file in list_datasets:
        read_csv = f"{path2}/{file}"

        df = pd.read_csv(read_csv)
        y = format_labels(df['y'])
        X_raw = df.drop('y', axis=1)
        cols = X_raw.columns

        # Escalado
        X_scaled = StandardScaler().fit_transform(X_raw)
        X = pd.DataFrame(X_scaled, columns=cols)

        dataset_name = file.split(".")[0]
        print(f"\nProcesando: {dataset_name}")

        df_resultados_raw = brute_force_evaluation_parallel(X, y, models, k_folds=5)

        name_csv = f'Results_FS_bruto/Results_FS_bruto_{dataset_name}.csv'
        df_resultados_raw.to_csv(name_csv, index=False)
        print(f"Guardado en: {name_csv}")


###################### ESTO ES TOTALMENTE SECUENCIAL #######################################

# list_datasets = ['parkinsons2.csv']#,'ionosphere.csv', 'sonar.csv', 'spambase.csv'
#                 # 'spambase.csv']
#                 # 'wdbc.csv',
#                 #  'musk2.csv','parkinsons.csv',
#                 #  'ozone.csv','sonar.csv','spambase.csv',
#                 #  'Colon.csv','arcene_train.csv','gisette_train.csv']
#
#
# # Modelos a probar
# models = {  # "LogReg": LogisticRegression(max_iter=1000, random_state=0),
#     # "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
#     "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
#     # "RandomForest": RandomForestClassifier(random_state=0),
#     "KNN": KNeighborsClassifier()
#     # "NaiveBayes": GaussianNB(),
#     # "DecisionTree": DecisionTreeClassifier(random_state=0),
#     # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
# }
#
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
#     feature_names = X.columns
#     df_resultados_raw = brute_force_evaluation(X, y, models, k_folds=5)
#     name_csv = 'Results_FS_bruto/Results_FS_bruto_'+str(dataset_name)+'.csv'
#     df_resultados_raw.to_csv(name_csv,index=False)
#
#




#
# # ---------------------------------------------------------
# # 3. EJEMPLO DE USO Y ANÁLISIS
# # ---------------------------------------------------------
#
# if __name__ == "__main__":
#     # A. Generar datos dummy (Clasificación Binaria)
#     from sklearn.datasets import make_classification
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.tree import DecisionTreeClassifier
#
#     # Creamos un dataset pequeño para el ejemplo (6 features = 63 combinaciones)
#     # NOTA: Si usas muchas features (>15), esto tardará mucho tiempo.
#     X_data, y_data = make_classification(n_samples=200, n_features=6, n_informative=3,
#                                          n_redundant=1, random_state=42)
#
#     # Convertimos a Pandas para tener nombres de columnas
#     feature_names = [f'Var_{i}' for i in range(X_data.shape[1])]
#     df_X = pd.DataFrame(X_data, columns=feature_names)
#     df_y = pd.Series(y_data)
#
#     # Modelos a probar
#     models = {  # "LogReg": LogisticRegression(max_iter=1000, random_state=0),
#         # "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
#         "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
#         # "RandomForest": RandomForestClassifier(random_state=0),
#         "KNN": KNeighborsClassifier()
#         # "NaiveBayes": GaussianNB(),
#         # "DecisionTree": DecisionTreeClassifier(random_state=0),
#         # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
#     }
#
#     # C. Ejecutar la búsqueda
#     # Guardamos el resultado crudo (raw)
#     df_resultados_raw = brute_force_evaluation(df_X, df_y, models, k_folds=5)
#
#     print("\n--- Procesamiento de Resultados ---")
#
#     # --- ANÁLISIS 1: Mejor combinación TOTAL (Promedio de los folds) ---
#     # Agrupamos por Modelo y Set de variables, promediando métricas
#     resumen_total = df_resultados_raw.groupby(['model', 'feature_set', 'n_features'])[
#         ['accuracy', 'gps']].mean().reset_index()
#
#     # Ordenamos por GPS descendente
#     mejores_globales = resumen_total.sort_values(by='gps', ascending=False)
#
#     print("\n>>> Top 3 Combinaciones Globales (Promedio CV):")
#     print(mejores_globales.head(3).to_string(index=False))
#
#     # --- ANÁLISIS 2: Mejor combinación POR FOLD ---
#     # Queremos ver qué combinación ganó en cada fold específico
#     # Agrupamos por Modelo y Fold, y sacamos el índice del máximo GPS
#     idx_mejores_fold = df_resultados_raw.groupby(['model', 'fold'])['gps'].idxmax()
#     mejores_por_fold = df_resultados_raw.loc[idx_mejores_fold]
#
#     print("\n>>> Ganadores Individuales por Fold (ejemplo primeros 5):")
#     cols_mostrar = ['model', 'fold', 'n_features', 'feature_set', 'accuracy', 'gps']
#     print(mejores_por_fold[cols_mostrar].head(10).to_string(index=False))



