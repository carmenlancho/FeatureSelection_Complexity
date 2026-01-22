## 22/01/2026
## Con los datasets artificiales ya hemos visto que las medidas de complejidad detectan cuáles son las variables
## realmente informativas. Ahora vamos a hacer una pequeña evaluación con datos reales.
## Para ello, hacemos FS por fuerza bruta, es decir, calculamos la performance con todos los subocnjuntos posibles de variables
# Esto lo hacemos en el script FS_fuerzaBruta
# En este script lo que vamos a hacer es calcular la complejidad de todos los subconjuntos posibles de variables
# para discernir si también con datos reales las medidas de complejidad pillan cuál es el mejor subconjunto de vars


import numpy as np
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from All_measures import *


def evaluate_complexity_single_combination(features_subset, X, y):
    """
    Extrae la fila 'dataset' de las medidas de complejidad para un subconjunto.
    """
    # DataFrame temporal con las columnas seleccionadas
    X_subset = X[list(features_subset)].copy()
    datos_temp = X_subset
    datos_temp['y'] = y  # Añadimos la columna objetivo

    # Calculamos complejidad
    _, df_medidas, _ = all_measures(datos_temp, save_csv=False, path_to_save=None, name_data=None)


    # Extraer solo la fila 'dataset' (complejidad a nivel dataset)
    # Usamos .loc['dataset'] para obtener la serie de esa fila
    fila_dataset = df_medidas.loc['dataset'].to_dict()

    # Formato del resultado final
    res = {
        'n_features': len(features_subset),
        'feature_set': str(tuple(features_subset)),
        **fila_dataset
    }
    return res


def complexity_brute_force_parallel(X, y, verbose=True):
    features = X.columns.tolist()
    n_features = len(features)

    # Configuración de núcleos para el servidor (50%)
    total_cores = os.cpu_count()
    use_cores = max(1, 2*total_cores // 3)

    # Generar todas las combinaciones posibles
    all_combos = []
    for r in range(1, n_features + 1):
        for combo in combinations(features, r):
            all_combos.append(combo)

    if verbose:
        print(f"--- Iniciando Cálculo de Complejidad Paralelo ---")
        print(f"Combinaciones totales: {len(all_combos)}")
        print(f"Utilizando {use_cores} núcleos del servidor.")

    # Ejecución en paralelo
    results = Parallel(n_jobs=use_cores)(
        delayed(evaluate_complexity_single_combination)(
            combo, X, y
        ) for combo in tqdm(all_combos, disable=not verbose, desc="Procesando")
    )

    return pd.DataFrame(results)


if __name__ == "__main__":
    list_datasets = ['parkinsons2.csv']
    path2 = "datasets"
    output_folder = 'Results_Complexity_Bruto'
    os.makedirs(output_folder, exist_ok=True)

    for file in list_datasets:
        read_csv = f"{path2}/{file}"

        df = pd.read_csv(read_csv)
        y = df['y'].values
        X_raw = df.drop('y', axis=1)
        cols = X_raw.columns

        # Escalado estándar (recomendado para medidas como L1, F3, etc.)
        X_scaled = StandardScaler().fit_transform(X_raw)
        X = pd.DataFrame(X_scaled, columns=cols)

        dataset_name = file.split(".")[0]
        print(f"\nAnalizando dataset: {dataset_name}")

        # Ejecutar proceso
        df_final = complexity_brute_force_parallel(X, y)

        # Guardar resultados
        name_csv = f'{output_folder}/ComplexityBruto_{dataset_name}.csv'
        df_final.to_csv(name_csv, index=False)
        print(f"Resultados guardados en: {name_csv}")



