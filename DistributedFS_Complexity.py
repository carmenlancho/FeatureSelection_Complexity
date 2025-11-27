# 21/10/2025

###################################################################################################
#####                        FS COMPLEXITY BASED, DISTRIBUTED VERSION                         #####
###################################################################################################

# En este script vamos a programar una manera de seleccionar variables un poco similar al RF
# que es de forma distribuida. Básicamente lo que vamos a hacer es un muestreo de variables
# Según el SOTA lo mejor es hacer un muestreo con reemplazamiento porque si no hay variables
# que nunca se estudian conjuntamente y se pierde esa interrelación

# La idea es la siguiente:
# Boostrap de variables: tomamos n réplicas de m random variables with replacement
# Evaluar cada réplica de forma multivariante (aquí podemos jugar con correlación, quizás merece la pena quitar primero las correladas)
# En base a la info de las n réplicas, construir un gráfico de importancia (tipo RF) que me diga
# cuánto disminuye la complejidad cada variable. Para ello, como necesito un punto de partida,
# cojo las m variables de cada réplica, calculo la complejidad y ya tengo el punto de partida.
# Ahora comienzo a quitar variables tipo backward selection y voy apuntando  lo que se va disminuyendo de complejidad
# Esto lo puedo hacer de forma aleatoria (opción 1) o guiándonos por la complejidad univariante de cada variable (opción 2)
# porque se supone que cuanto menor sea la complejidad con una variable, mejor es dicha variable.
# Con esto obtengo un gráfico de importancia de las variables
# La idea es ir metiendo variables tipo forward en base a esas variables y para cuando ya no disminuyan la complejidad


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



#
# ### EL COMPLEXITY GUIDED (COMO ESTÁ GUIADO POR LA COMPLEJIDAD UNIVARIANTE) ES CLARAMENTE   #########
# ### PEOR QUE EL RANDOM CHOICE, ASÍ QUE LO DESCARTAMOS                                      #########
#
#
# # # La complejidad con cada feature de forma univariante está en ArtificialDatasetXX_featuresComplexityRanking.csv
# # csv_file = "Results_UnivariateRanking_CM/ArtificialDataset1_featuresComplexityRanking.csv"
# # univariate_complexity = pd.read_csv(csv_file)
#
# def distributed_variable_selection_complexity_guided(X, y, dataset_name, n_replicas, m_vars,
#                                    univariate_complexity=None, # complejidad univariante
#                                    measures=["Hostility",'kDN','N1'],
#                                    filter_corr=True, corr_th=0.9, corr_method="pearson",
#                                    random_state=0, save_csv=False, path='Results_FS_Distributed'):
#
#     np.random.seed(random_state)
#     random.seed(random_state)
#
#     # Filtro previo opcional de eliminación de variables con correelación > corr_th
#     if filter_corr:
#         corr = X.corr(method=corr_method).abs()
#         upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#         to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
#         X = X.drop(columns=to_drop)
#
#     variables = X.columns.tolist()
#     results_all = pd.DataFrame()
#
#     for cm in measures:
#
#         importances = pd.Series(0.0, index=variables) # diccionario para cada complexity measure
#         # importances_norm = pd.Series(0.0, index=variables)  # diccionario para cada complexity measure
#         count_vars = pd.Series(0.0, index=variables) # cuántas veces aparece cada var para normalización
#
#         # rep = 0
#         for rep in range(n_replicas):
#             # print(rep)
#             m_vars = int(m_vars)
#             subset_vars = random.choices(variables, k=m_vars)  # sampling WITH replacement
#             Xsub = X[subset_vars]
#
#             # Multivariate complexity
#             datos = pd.DataFrame(Xsub)
#             datos['y'] = y
#             _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
#             base_complexity = df_classes.loc['dataset',cm]
#
#             current_vars = subset_vars.copy()
#             while len(current_vars) > 1: # hasta que nos quedemos sin vars
#                 # Conteo de variables participantes
#                 count_vars[current_vars] += 1
#
#                 # elegir variable a quitar
#                 df_uni_comp = univariate_complexity[univariate_complexity["level"] == "dataset"].copy()
#                 df_uni_comp = df_uni_comp.set_index("feature")[cm]
#                 guided_scores = df_uni_comp.loc[current_vars]
#                 to_remove = guided_scores.idxmax()  # quita la de mayor complejidad univariante
#
#                 current_vars.remove(to_remove)
#                 Xtemp = X[current_vars]
#
#                 # Complexity removing one variable from previous subset
#                 datos_temp = pd.DataFrame(Xtemp)
#                 datos_temp['y'] = y
#                 _, df_classes_temp, _ = all_measures_FS(datos_temp, save_csv=False, path_to_save=None, name_data=None)
#                 new_complexity = df_classes_temp.loc['dataset',cm]
#
#                 # Cambio de complejidad
#                 delta = new_complexity - base_complexity # mee interesan diferencias positivas, es decir,
#                 # quitar la variable aumenta la complejidad, luego la variable es útil
#                 # Las diferencias negativas me dicen que esa variable aumentaba la complejidad, luego no la quiero
#                 importances[to_remove] += delta
#
#                 # Actualizamos base_complexity para próximo paso
#                 base_complexity = new_complexity
#                 # Conteo de variables participantes
#                 count_vars[current_vars] += 1
#
#         # Normalizamos importancias
#         count_vars = count_vars.replace(0, np.nan) # para no tener problemas con los 0 en la división
#         importances_norm = importances / count_vars # por probabilidad, con n_replicas grande, deben ser similares estos números
#         importances_norm.sort_values(ascending=False, inplace=True)
#
#
#         results_norm = pd.DataFrame.from_dict(importances_norm)
#         results_norm.columns = [cm + '_importances_norm']
#         results = pd.DataFrame.from_dict(importances)
#         results.columns = [cm + '_importances']
#         results_count = pd.DataFrame.from_dict(count_vars)
#         results_count.columns = ['count_vars']
#         results_complete = pd.concat([results, results_norm, results_count], axis=1)
#
#         results_all = pd.concat([results_all, results_complete], axis=1)
#
#     if filter_corr: # aclaramos las que se hayan quitado por correlación
#         results_all = results_all.reindex(results_all.index.union(to_drop))
#         results_all.loc[to_drop,:] = np.nan
#
#
#     if save_csv:
#         name_csv = f"{path}/{dataset_name}_ComplexityGuidedDistributed.csv"
#         results_all.to_csv(name_csv, index=True)
#
#     return results_all


#
# def plot_importances(importances_norm):
#     for m, imp in importances_norm.items():
#         imp.plot(kind="barh", figsize=(8,6), title=f"Variable importance ({m})")
#         plt.xlabel("Avg complexity in backward")
#         plt.gca().invert_yaxis()
#         plt.show()






# n_replicas = 5
# m_vars = 5
def distributed_variable_selection_complexity_random(X, y, dataset_name, n_replicas, m_vars,
                                   measures=["Hostility", "N1", "kDN"],
                                   filter_corr=True, corr_th=0.9, corr_method="pearson",
                                   random_state=0, save_csv=False, path='Results_FS_Distributed'):

    np.random.seed(random_state)
    random.seed(random_state)

    # Filtro previo opcional de eliminación de variables con correelación > corr_th
    if filter_corr:
        corr = X.corr(method=corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        if len(to_drop) > 0:
            X = X.drop(columns=to_drop)


    variables = X.columns.tolist()
    importances = {m: pd.Series(0.0, index=variables) for m in measures} # diccionario para cada complexity measure
    importances_norm = {m: pd.Series(0.0, index=variables) for m in measures}  # diccionario para cada complexity measure
    count_vars = pd.Series(0.0, index=variables) # cuántas veces aparece cada var para normalización


    # rep = 0
    for rep in range(n_replicas):
        # print(rep)
        m_vars = int(m_vars)
        subset_vars = random.choices(variables, k=m_vars)  # sampling WITH replacement
        Xsub = X[subset_vars]

        # Multivariate complexity
        datos = pd.DataFrame(Xsub)
        datos['y'] = y
        _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
        base_complexity = df_classes.loc['dataset',measures]

        current_vars = subset_vars.copy()
        while len(current_vars) > 1: # hasta que nos quedemos sin vars
            # Conteo de variables participantes
            count_vars[current_vars] += 1

            # elegir variable a quitar randomly
            to_remove = random.choice(current_vars)

            current_vars.remove(to_remove)
            Xtemp = X[current_vars]

            # Complexity removing one variable from previous subset
            datos_temp = pd.DataFrame(Xtemp)
            datos_temp['y'] = y
            _, df_classes_temp, _ = all_measures_FS(datos_temp, save_csv=False, path_to_save=None, name_data=None)
            new_complexity = df_classes_temp.loc['dataset',measures]

            # Cambio de complejidad
            delta = new_complexity - base_complexity # mee interesan diferencias positivas, es decir,
            # quitar la variable aumenta la complejidad, luego la variable es útil
            # Las diferencias negativas me dicen que esa variable aumentaba la complejidad, luego no la quiero
            for m in measures:
                importances[m][to_remove] += delta[m]

            # Actualizamos base_complexity para próximo paso
            base_complexity = new_complexity
            # Conteo de variables participantes
            count_vars[current_vars] += 1

    # Normalizamos importancias
    count_vars = count_vars.replace(0, np.nan) # para no tener problemas con los 0 en la división
    for m in measures:
        # importances[m] = importances[m] / n_replicas # esta opción no la veo justa
        importances_norm[m] = importances[m] / count_vars # por probabilidad, con n_replicas grande, deben ser similares estos números
        importances_norm[m].sort_values(ascending=False, inplace=True)

    results_norm = pd.DataFrame.from_dict(importances_norm)
    results_norm = results_norm.add_suffix('_importances_norm')
    results = pd.DataFrame.from_dict(importances)
    results = results.add_suffix('_importances')
    results_count = pd.DataFrame.from_dict(count_vars)
    results_count.columns = ['count_vars']
    results_complete = pd.concat([results, results_norm, results_count], axis=1)

    if filter_corr: # aclaramos las que se hayan quitado por correlación
        results_complete = results_complete.reindex(results_complete.index.union(to_drop))
        results_complete.loc[to_drop,:] = np.nan

    if save_csv:
        name_csv = f"{path}/{dataset_name}_ComplexityRandomDistributed.csv"
        results_complete.to_csv(name_csv, index=True)


    return importances_norm, importances, count_vars, results_complete

# n_replicas = 200
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
# # La complejidad con cada feature de forma univariante está en ArtificialDatasetXX_featuresComplexityRanking.csv
#
# csv_file = "Results_UnivariateRanking_CM/ArtificialDataset2_featuresComplexityRanking.csv"
# univariate_complexity = pd.read_csv(csv_file)
#
# ### Random
# p = X.shape[1]
# m_vars= np.floor(np.sqrt(p)) # como en el RF
# distributed_variable_selection_complexity_random(X, y, dataset_name, n_replicas, m_vars,
#                                    measures=["Hostility", "N1", "kDN"],
#                                    filter_corr=True, corr_th=0.9, corr_method="pearson",
#                                    random_state=0, save_csv=True, path='Results_FS_Distributed')

## Una primera selección de datos es: 2,3,7,8,10,11,12,14,16,17,18,19,20,21




def distributed_complexity_random_neg_out(X, y, dataset_name, n_replicas, m_vars,
                                   measures=["Hostility", "N1", "kDN"],
                                   filter_corr=True, corr_th=0.9, corr_method="pearson",
                                   random_state=0, save_csv=False, path='Results_FS_Distributed'):
    np.random.seed(random_state)
    random.seed(random_state)

    # Filtro correlación
    if filter_corr:
        corr = X.corr(method=corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        X = X.drop(columns=to_drop)

    variables = X.columns.tolist()

    # Dicts para guardar
    importances = {m: pd.Series(0.0, index=variables) for m in measures}
    importances_norm = {m: pd.Series(0.0, index=variables) for m in measures}
    count_vars = pd.Series(0.0, index=variables)
    removed_vars = {m: pd.Series(0.0, index=variables) for m in measures} # las que quitamos por negativas

    # Cada medida tiene su propio conjunto de variables activas
    active_vars = {m: set(variables) for m in measures}
    permanently_removed = {m: set() for m in measures}

    # Main loop
    # rep=0
    for rep in range(n_replicas):
        # m_vars = int(m_vars)

        # m = 'Hostility'
        for m in measures:
            print(m)
            # # Si ya se eliminaron todas las variables, saltamos
            # if len(active_vars[m]) < 2:
            #     continue
            available_vars = list(active_vars[m] - permanently_removed[m])
            print(available_vars)
            # p = len(available_vars)
            m_vars = int(np.floor(np.sqrt(p)))
            subset_vars = random.sample(available_vars, k=m_vars) # sin remplazamiento
            Xsub = X[subset_vars]

            datos = pd.DataFrame(Xsub)
            datos['y'] = y
            _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
            base_complexity = df_classes.loc['dataset', m]

            current_vars = subset_vars.copy()
            while len(current_vars) > 1:
                # print('Current vars')
                # print(current_vars)
                count_vars[current_vars] += 1
                to_remove = random.choice(current_vars)
                current_vars.remove(to_remove)

                Xtemp = X[current_vars]
                datos_temp = pd.DataFrame(Xtemp)
                datos_temp['y'] = y
                _, df_classes_temp, _ = all_measures_FS(datos_temp, save_csv=False, path_to_save=None, name_data=None)
                new_complexity = df_classes_temp.loc['dataset', m]

                # cambio de complejidad
                delta = new_complexity - base_complexity

                importances[m][to_remove] += delta
                # print(delta)

                # Si la complejidad baja (delta < 0), eliminamos la variable de las opciones de esta medida
                if delta < -1e-3: # le ponemos un margen de tolerancia
                    active_vars[m].remove(to_remove)
                    permanently_removed[m].add(to_remove)
                    removed_vars[m][to_remove] += 1
                    # print(delta)
                    # print(to_remove)


                base_complexity = new_complexity
                count_vars[current_vars] += 1

    # Normalizamos aparición variables (POR PENSAR SERIAMNTE)
    count_vars = count_vars.replace(0, np.nan)
    for m in measures:
        importances_norm[m] = importances[m] / count_vars
        importances_norm[m].sort_values(ascending=False, inplace=True)

    # Formato resultados
    results_norm = pd.DataFrame.from_dict(importances_norm)
    results_norm = results_norm.add_suffix('_importances_norm')

    results = pd.DataFrame.from_dict(importances)
    results = results.add_suffix('_importances')

    results_count = pd.DataFrame(count_vars, columns=['count_vars'])

    removed_df = pd.DataFrame.from_dict(removed_vars)
    removed_df = removed_df.add_suffix('_removed_count')

    results_complete = pd.concat([results, results_norm, results_count, removed_df], axis=1)

    if filter_corr: # aclaramos las que se hayan quitado por correlación
        results_complete = results_complete.reindex(results_complete.index.union(to_drop))
        results_complete.loc[to_drop, :] = np.nan

    if save_csv:
        name_csv = f"{path}/{dataset_name}_ComplexityRandomDistributed_NegOut.csv"
        results_complete.to_csv(name_csv, index=True)

    return importances_norm, importances, count_vars, removed_vars, results_complete

#
#
# n_replicas = 3
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
# # La complejidad con cada feature de forma univariante está en ArtificialDatasetXX_featuresComplexityRanking.csv
#
#
#
# ### Random
# p = X.shape[1]
# dataset_name = 'PRUEBA'
# m_vars= np.floor(np.sqrt(p)) # como en el RF
# importances_norm, importances, count_vars, removed_vars, results_complete = distributed_complexity_random_neg_out(X, y, dataset_name, n_replicas, m_vars,
#                                    measures=["Hostility", "N1", "kDN"],
#                                    filter_corr=True, corr_th=0.9, corr_method="pearson",
#                                    random_state=0, save_csv=False, path='Results_FS_Distributed')
#
#
#
#


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




def evaluate_distributed_fs_cv(X, y, k, model, dataset_name, measures=["Hostility", "N1", "kDN"],
                               cv_splits=5, random_state=0, n_replicas = 200):
    """
    Realiza CV evaluando el métod distributed:
      - selecciona top-k variables según cada medida
      - entrena y evalúa modelo en cada fold

    Devuelve:
      - importances_df: todas las importancias obtenidas (por fold y medida)
      - performance_df: métricas de train/test (por fold, medida)
    """

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    importances_records = []
    performance_records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Calcular importancias distributed en el fold de entrenamiento
        p = X.shape[1]
        m_vars = np.floor(np.sqrt(p))  # como en el RF
        importances_norm, importances, count_vars, removed_vars, imp_df = distributed_complexity_random_neg_out(X_train, y_train, dataset_name, n_replicas, m_vars,
                                                         measures=measures,
                                                         filter_corr=True, corr_th=0.9, corr_method="pearson",
                                                         random_state=0, save_csv=False)
        # Guardamos nombre de variable y fold
        imp_df = imp_df.reset_index().rename(columns={"index": "feature"})
        imp_df["fold"] = fold
        importances_records.append(imp_df)

        # Seleccionamod top-k variables por cada medida
        measures_norm = [s + '_importances_norm' for s in measures]
        # measure = 'Hostility_importances_norm'
        for measure in measures_norm:
            imp_m = imp_df[measure]
            top_feats = (imp_m.sort_values(ascending=False).index.tolist())[:k]
            feats = imp_df.loc[top_feats,'feature']

            # Training modelo
            X_train_sel = X_train[feats]
            X_test_sel = X_test[feats]

            # Complejidad con el subset de variables
            datos = pd.DataFrame(X_train_sel)
            datos['y'] = y_train
            _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
            subset_complexity = df_classes.loc["dataset", measures].to_dict()


            model.fit(X_train_sel, y_train)

            # Train
            y_pred_train = model.predict(X_train_sel)
            acc_train = accuracy_score(y_train, y_pred_train)
            gps_train = compute_gps(y_train, y_pred_train)

            # Test
            y_pred_test = model.predict(X_test_sel)
            acc_test = accuracy_score(y_test, y_pred_test)
            gps_test = compute_gps(y_test, y_pred_test)

            performance_records.append({
                "fold": fold,
                "measure": measure,
                "n_features": k,
                "top_features": list(feats), # variables seleccionadas
                "acc_train": acc_train,
                "gps_train": gps_train,
                "acc_test": acc_test,
                "gps_test": gps_test,
                **{f"complexity_{m}": subset_complexity[m] for m in measures}
            })

    importances_df = pd.concat(importances_records, ignore_index=True)
    performance_df = pd.DataFrame(performance_records)

    return importances_df, performance_df

#
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#
#
# k=10
# model = RandomForestClassifier(random_state=0)
# importances_df, performance_df = evaluate_distributed_fs_cv(X, y, k, model, dataset_name, measures=["Hostility", "N1", "kDN"],
#                                cv_splits=5, random_state=0, n_replicas=3)
#



def run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
                                       measures=["Hostility", "N1", "kDN"],
                                       cv_splits=5, n_replicas=200, random_state=0,
                                       path="Results_FS_Distributed_CV", save_csv=False):
    """
    Ejecuta evaluate_distributed_fs_cv para varios modelos y resume resultados.
    """
    k = len(dict_info_feature["informative"])  # nº de variables informativas

    all_importances = []
    all_performance = []

    for model_name, model in models_dict.items():
        print(f"\n Classifier: {model_name}")
        imp_df, perf_df = evaluate_distributed_fs_cv(
            X=X, y=y, k=k, model=model,dataset_name=dataset_name, measures=measures,
            cv_splits=cv_splits, random_state=random_state, n_replicas=n_replicas)

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
            "complexity_Hostility": ["mean", "std"],
            "complexity_N1": ["mean", "std"], # la complejidad por modelo es la misma, pero es para que cuadre el DF
            "complexity_kDN": ["mean", "std"]
        }))

    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    if save_csv:
        name_csv1 = f"{path}/{dataset_name}_DistributedCVRandom_FeatureImportance_Folds.csv"
        importances_all.to_csv(name_csv1, index=False)
        name_csv2 = f"{path}/{dataset_name}_DistributedCVRandom_Performance_Folds.csv"
        performance_all.to_csv(name_csv2, index=False)
        name_csv3 = f"{path}/{dataset_name}_DistributedCVRandom_SummaryResults.csv"
        summary.to_csv(name_csv3, index=False)

    return importances_all, performance_all, summary

# models_dict = {"LogReg": LogisticRegression(max_iter=1000, random_state=0),
#     "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
#                "KNN": KNeighborsClassifier()}
#
#
# dataset_name = 'prueba'
# imp_all, perf_all, summary = run_distributed_cv_multiple_models(
#     X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["Hostility", "N1", "kDN"],
#     cv_splits=5, n_replicas=10, random_state=0,
#     path="Results_FS_Distributed_CV", save_csv=True)
#


### Segundo filtro de variables
# 2, 7, 12, 14, 18, 20, 21

models_dict = {#"LogReg": LogisticRegression(max_iter=1000, random_state=0),
    # "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
    "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
    # "RandomForest": RandomForestClassifier(random_state=0),
    "KNN": KNeighborsClassifier()
    # "NaiveBayes": GaussianNB(),
    # "DecisionTree": DecisionTreeClassifier(random_state=0),
    # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
    }


n_replicas = 200
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)
#
#
#
#
# #### Dataset 7
# dataset_name = 'ArtificialDataset7'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=589,noise_std=0.05)
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)
#




# #### Dataset 12
# dataset_name = 'ArtificialDataset12'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=987,noise_std=0.5)
#
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)
#


# #### Dataset 14
# dataset_name = 'ArtificialDataset14'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
#                                          n_redundant_linear=30,n_redundant_nonlinear=40,
#                                         flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
#                                                      random_state=95,noise_std=0.5)
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)
#

# #### Dataset 18
# dataset_name = 'ArtificialDataset18'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)


# #### Dataset 20
# dataset_name = 'ArtificialDataset20'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
#                                          n_redundant_linear=60,n_redundant_nonlinear=60,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)
#
# #### Dataset 21
# dataset_name = 'ArtificialDataset21'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
#                                          n_redundant_linear=100,n_redundant_nonlinear=100,
#                                         flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
#                                                      random_state=996,noise_std=0.5)
#
# run_distributed_cv_multiple_models(X, y, dict_info_feature, dataset_name, models_dict,
#                                 measures=["Hostility", "N1", "kDN"],
#                                     cv_splits=5, n_replicas=n_replicas, random_state=0,
#                                         path="Results_FS_Distributed_CV", save_csv=True)


# Number of distinct clusters (144) found smaller than n_clusters (160). Possibly due to duplicate points in X.



###############################################################################################################
#####                                PLOT COMPLEXITY RESULTADOS                                           #####
###############################################################################################################
# Hacemos un plot tipo importancia de variables en RF




def plot_complexity_importances(df, dataset_name="Dataset", guided=True, save_path=None):
    """
    Dibuja:
    1) Importancias medias por variable y medida de complejidad
    2) Frecuencia de aparición (count_vars)

    guided: True si el dataset es guiado por complejidad, False si es random
    """
    sns.set(style="whitegrid", font_scale=1.1)

    title_prefix = "Guided by Complexity" if guided else "Random choice"

    # Reordenamos según la media de importancias (positivas = útiles puesto que quitarlas aumenta la complejidad)
    df_plot = df.copy()
    df_plot["mean_importance_norm"] = df_plot[["Hostility_importances_norm", "N1_importances_norm", "kDN_importances_norm"]].mean(axis=1)
    df_plot = df_plot.sort_values("mean_importance_norm", ascending=False)

    # Gráfico de importancias
    plt.figure(figsize=(10, 6))
    df_melt = df_plot.melt(value_vars=["Hostility_importances_norm", "N1_importances_norm", "kDN_importances_norm"],
                           var_name="Measure", value_name="Importance_norm", ignore_index=False).reset_index()

    sns.barplot(data=df_melt, x="index", y="Importance_norm", hue="Measure")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=45)
    plt.xlabel("Variable")
    plt.ylabel("Mean importance (ΔComplexity)")
    plt.title(f"{dataset_name} — {title_prefix}\nPositive = variable reduces complexity")
    plt.legend(title="Complexity measure")
    plt.tight_layout()

    # if save_path:
    #     plt.savefig(f"{save_path}/{dataset_name}_importances_{'guided' if guided else 'random'}.png",
    #                 dpi=300, bbox_inches="tight")
    plt.show()

    # Gráfico de frecuencias
    plt.figure(figsize=(8, 4))
    sns.barplot(x=df_plot.index, y="count_vars", data=df_plot, color="skyblue")
    plt.xticks(rotation=45)
    plt.xlabel("Variable")
    plt.ylabel("Count")
    plt.title(f"{dataset_name} — Variable occurrence count ({title_prefix})")
    plt.tight_layout()

    # if save_path:
    #     plt.savefig(f"{save_path}/{dataset_name}_count_{'guided' if guided else 'random'}.png",
    #                 dpi=300, bbox_inches="tight")
    plt.show()


#
# df_random = pd.read_csv("Results_FS_Distributed/ArtificialDataset2_ComplexityRandomDistributed.csv", index_col=0)
# df_guided = pd.read_csv("Results_FS_Distributed/ArtificialDataset2_ComplexityGuidedDistributed.csv", index_col=0)
# # df = df_random
#
# plot_complexity_importances(df_random, dataset_name="Dataset2", guided=False)
# plot_complexity_importances(df_guided, dataset_name="Dataset2", guided=True)


######### CORR SPEARMAN ENTRE RANKING COMPLEJIDAD

def compute_spearman_correlations(df, measures=["Hostility_importances_norm", "N1_importances_norm", "kDN_importances_norm"]):
    """
    Calcula la correlación de Spearman entre las medidas de complejidad
    ignorando NaN.
    Devuelve un DataFrame con la matriz de correlaciones.
    """
    # Filtramos solo las columnas relevantes y quitamos filas con NaN
    df_corr = df[measures].dropna(how="any")

    # Calculamos correlación
    corr, _ = spearmanr(df_corr)
    corr_matrix = pd.DataFrame(corr, index=measures, columns=measures)

    return corr_matrix

#
# df = pd.read_csv("Results_FS_Distributed/ArtificialDataset2_ComplexityRandomDistributed.csv", index_col=0)
#
#
# corr_matrix = compute_spearman_correlations(df)
# print(corr_matrix)
#
#





##########################################################################################################
#############                                IMPORTANCIA NEGATIVA                            #############
##########################################################################################################


# En los gráficos de importancia de complejidad (similar a RF) han salido algunas variables con valores negativos
# Vamos a estudiar la naturaleza de dichas variables (noise, redudant, etc.) para ver si pillamos bien
# cuáles son las malas


def analyze_negative_importances(csv_path, dict_info_feature):
    """
    Analiza las variables con importancia negativa en el métod distribuido.
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Selección de medidas
    measures = ["Hostility_importances_norm", "N1_importances_norm", "kDN_importances_norm"]

    # Naturaleza variables en diccionario
    feature_types = {}
    for t, feats in dict_info_feature.items():
        for f in feats:
            feature_types[f] = t

    summary_list = []

    for m in measures:
        negatives = df[df[m] < 0][m]
        # if negatives.empty:
        #     continue

        # Clasificación de cada variable negativa
        neg_types = [feature_types.get(f, "unknown") for f in negatives.index]
        neg_df = pd.DataFrame({"feature": negatives.index, "importance": negatives.values, "type": neg_types})

        # Resumen por tipo
        type_counts = neg_df["type"].value_counts(normalize=True) * 100
        type_counts = type_counts.round(2)

        # Total negativas y proporciones
        summary_entry = {
            "measure": m,
            "n_negatives": len(neg_df),
            "pct_informative": type_counts.get("informative", 0),
            "pct_noise": type_counts.get("noise", 0),
            "pct_redundant_linear": type_counts.get("formulas_linear", 0),
            "pct_redundant_nonlinear": type_counts.get("formulas_nonlinear", 0),
        }

        summary_list.append(summary_entry)

    summary_df = pd.DataFrame(summary_list)
    return summary_df


# dataset_name = 'ArtificialDataset12'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=987,noise_std=0.5)
#
# csv_path = "Results_FS_Distributed/ArtificialDataset12_ComplexityRandomDistributed.csv"
# summary_neg = analyze_negative_importances(csv_path, dict_info_feature)
#
#



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
            # print(f"[{m}] Réplica {rep + 1}/{n_replicas}")


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


dataset_name = 'ArtificialDataset12'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=987,noise_std=0.5)


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
        # top-k variables por medida
        measures_norm = [s + '_importances_norm' for s in measures]
        for measure, m_name in zip(measures_norm, measures):
            imp_m = imp_df[measure]
            top_feats = (imp_m.sort_values(ascending=False).index.tolist())[:k]
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




def run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
    measures=["Hostility", "N1", "kDN"],cv_splits=5, n_replicas=200, random_state=0,
    tau=0.01,path="Results_FS_Distributed_CV", save_csv=False):

    k = len(dict_info_feature["informative"])  # nº de variables informativas

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


n_replicas = 100
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
# #### Dataset 7
# dataset_name = 'ArtificialDataset7'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=589,noise_std=0.05)
#
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)
#
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
#
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)
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
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)
#
#
# #### Dataset 18
# dataset_name = 'ArtificialDataset18'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)
#
#
# #### Dataset 20
# dataset_name = 'ArtificialDataset20'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
#                                          n_redundant_linear=60,n_redundant_nonlinear=60,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)

# #### Dataset 21
# dataset_name = 'ArtificialDataset21'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
#                                          n_redundant_linear=100,n_redundant_nonlinear=100,
#                                         flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
#                                                      random_state=996,noise_std=0.5)
#
# run_distributed_cv_multiple_models2(X, y, dict_info_feature, dataset_name, models_dict,
#     measures=["kDN"],cv_splits=5, n_replicas=n_replicas, random_state=0,
#     tau=0.01,path="Results_FS_Distributed_CV", save_csv=True)



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

    df = df.loc[df.model == 'KNN',:]

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


#
# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset2_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# # perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset2_OutHigh_EvolutivePerformance.csv',index=False)
#
#
#
# #### Dataset 7
# dataset_name = 'ArtificialDataset7'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=589,noise_std=0.05)
#
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset7_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset7_OutHigh_EvolutivePerformance.csv',index=False)
#
#
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
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset12_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset12_OutHigh_EvolutivePerformance.csv',index=False)
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
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset14_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset14_OutHigh_EvolutivePerformance.csv',index=False)
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
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset18_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset18_OutHigh_EvolutivePerformance.csv',index=False)
#
#
# #### Dataset 20
# dataset_name = 'ArtificialDataset20'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
#                                          n_redundant_linear=60,n_redundant_nonlinear=60,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset20_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset20_OutHigh_EvolutivePerformance.csv',index=False)
#

## ESTEE FALTA
# #### Dataset 21
# dataset_name = 'ArtificialDataset21'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
#                                          n_redundant_linear=100,n_redundant_nonlinear=100,
#                                         flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
#                                         random_state=996,noise_std=0.5)
#
#
# path_csv = 'Results_FS_Distributed_CV/ArtificialDataset21_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
# importances_dict = load_importances_per_fold(path_csv)
#
# perf_final = evaluate_incremental_k(X, y, importances_dict, models, dataset_name, cv_splits=5, random_state=0)
# perf_final.to_csv('Results_FS_Distributed_CV/ArtificialDataset21_OutHigh_EvolutivePerformance.csv',index=False)


###3 Plot de performance evolutiva
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




###########################################################################################
####                   ESTUDIO CARACTERÍSTICAS DE LAS VARIABLES                        ####
###########################################################################################
# Hacemos un script que caracterice el tipo de variable para ver cómo nos va quedando el ranking


# Función para extraer dependencias entre variables
def extract_dependencies(formula):
    """
    Extrae todas las variables f### que aparecen en la fórmula.
    """
    return re.findall(r"f\d+", formula)

# Diccionario  de dependeencias
def build_dependencies(dict_info_feature):
    deps = {}

    # Fórmulas lineales
    for var, formula in dict_info_feature["formulas_linear"].items():
        deps[var] = extract_dependencies(formula)

    # Fórmulas no lineales
    for var, formula in dict_info_feature["formulas_nonlinear"].items():
        deps[var] = extract_dependencies(formula)

    return deps

### Dataset 2
dataset_name = 'ArtificialDataset2'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                    flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)

dep2 = build_dependencies(dict_info_feature)


# poner tipo de cada variable en formato diccionario
def build_type_dict(dict_info_feature):
    type_dict = {}

    for v in dict_info_feature["informative"]:
        type_dict[v] = "informative"

    for v in dict_info_feature["noise"]:
        type_dict[v] = "noise"

    for v in dict_info_feature["redundant_linear"]:
        type_dict[v] = "redundant_linear"

    for v in dict_info_feature["redundant_nonlinear"]:
        type_dict[v] = "redundant_nonlinear"

    return type_dict

type_dict2 = build_type_dict(dict_info_feature)



def caracterize_features_ranking(ranking, type_dict, dependencies):
    """
    ranking: lista ordenada de variables según importancia
    """
    selected = set()
    results = []

    for var in ranking:
        t = type_dict.get(var, "unknown")
        deps = dependencies.get(var, [])

        # Variable ruidosa
        if t == "noise":
            label = "noise"
        # lineal o no lineal
        elif t in ["redundant_linear", "redundant_nonlinear"]:
            if all(d in selected for d in deps) and len(deps) > 0:
                # Totalmente redundante
                label = t  # redundant_linear o redundant_nonlinear
            else:
                # Aún aporta algo
                label = "informative_derived"

        # Informativa original
        elif t == "informative":
            # PERO puede ser redundante
            # Si ya tenemos sus dependientes en selected
            redundant_now = False
            redundant_type = None
            for dep_var, dep_sources in dependencies.items(): # dep_var es la que depende de la informativa
                # si var participa en la formula
                if var in dep_sources and dep_var in selected: # la dependiente se ha seleccionado

                    # todas las otras fuentes necesarias para reconstruir var
                    other_sources = set(dep_sources) - {var}

                    # si TODAS ya están seleccionadas, la info de var está cubierta
                    if other_sources.issubset(selected):
                        redundant_now = True
                        redundant_type = type_dict.get(dep_var, None)
                        break

            if redundant_now:
                # convertimos informativa en redundante
                if redundant_type == "redundant_linear":
                    label = "informative_redundant_linear"
                else:
                    label = "informative_redundant_nonlinear"
            else:
                label = "informative"
        # no sabemos
        else:
            label = "unknown"

        results.append((var, label))
        selected.add(var)

    return results



path_csv = 'Results_FS_Distributed_CV/ArtificialDataset2_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
importances_dict = load_importances_per_fold(path_csv)

dfs = []
for fold, v in importances_dict.items():
    df = v["kDN_importances_norm"].copy()
    df["fold"] = fold
    dfs.append(df)

importances_all = pd.concat(dfs, ignore_index=True)


ranking = importances_all.loc[importances_all.fold==1]
dep2 = build_dependencies(dict_info_feature)
type_dict2 = build_type_dict(dict_info_feature)
results = caracterize_features_ranking(ranking['feature'], type_dict2, dep2)



def analyze_topk_with_labels(df_rankings, type_dict, dependencies, k_real, dataset_name="dataset"):
    """
    df_rankings: dataframe con columnas:
        - feature
        - kDN_importances_norm
        - fold

    type_dict: {feature: tipo_base}
    dependencies: {feature: [sources]}
    k_real: nº real de variables informativas
    """

    all_folds = sorted(df_rankings["fold"].unique())
    fold_results = []   # filas por fold
    all_labels = []     # para saber qué etiquetas existen

    for fold in all_folds:
        df_fold = df_rankings[df_rankings["fold"] == fold]
        df_sorted = df_fold.sort_values("kDN_importances_norm", ascending=False)

        ranking = df_sorted["feature"].tolist()
        labelled = caracterize_features_ranking(ranking, type_dict, dependencies)

        # guardar todas las labels para conocer el universo
        all_labels.extend([lab for _, lab in labelled])

        # LIMITAR AL TOP-k
        topk = labelled[:k_real]

        # contaje
        counts = {}
        for _, lab in topk:
            counts[lab] = counts.get(lab, 0) + 1

        # normalizar a porcentajes
        counts_pct = {f"pct_{lab}": counts.get(lab, 0) / k_real for lab in counts}

        counts_pct["fold"] = fold
        fold_results.append(counts_pct)

    # dataframe por fold
    df_folds = pd.DataFrame(fold_results).fillna(0)

    # labels globales
    unique_labels = sorted(set(all_labels))
    pct_cols = [c for c in df_folds.columns if c.startswith("pct_")]

    # resumen promedio por dataset
    summary = df_folds[pct_cols].mean().to_frame().T
    summary["dataset"] = dataset_name
    summary["k_real"] = k_real


    return summary, df_folds, unique_labels


summary, df_folds, labels = analyze_topk_with_labels(
    df_rankings=importances_all,
    type_dict=type_dict2,
    dependencies=dep2,
    k_real=10)


path_csv = 'Results_FS_Distributed_CV/ArtificialDataset2_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv'
importances_dict = load_importances_per_fold(path_csv)


ranking = importances_all.loc[importances_all.fold==1]
dep2 = build_dependencies(dict_info_feature)
type_dict2 = build_type_dict(dict_info_feature)
results = caracterize_features_ranking(ranking['feature'], type_dict2, dep2)



def position_distribution_boxviolin(rankings, labels_by_fold, max_pos=None, show_violin=True):
    """
    rankings: lista de listas → ranking por fold (list of variables)
    labels_by_fold: lista paralela donde cada elemento es:
        dict {var: label}  (o lista de (var,label) que será convertido a dict)
    max_pos: máximo de posiciones a considerar (útil para rankings grandes)

    Devuelve:
        - df_pos: DataFrame con columnas ['variable','label','pos','fold']
        - fig: figura matplotlib
    """

    records = []

    for f_idx, ranking in enumerate(rankings):
        lablist = labels_by_fold[f_idx]

        # Convertir a diccionario si viene como lista de tuplas
        if isinstance(lablist, dict):
            label_dict = lablist
        else:
            label_dict = {v: l for v, l in lablist}

        # recorrer ranking
        for pos, var in enumerate(ranking, start=1):

            if max_pos and pos > max_pos:
                break

            label = label_dict.get(var, "unknown")

            records.append({
                "variable": var,
                "label": label,
                "pos": pos,
                "fold": f_idx + 1
            })

    df_pos = pd.DataFrame(records)

    # ORDENAR  labels por mediana de su posición
    order = df_pos.groupby("label")["pos"].median().sort_values().index.tolist()

    # --------- Gráfico ----------
    fig, ax = plt.subplots(figsize=(12, 6))

    if show_violin:
        sns.violinplot(
            data=df_pos,
            x="label",
            y="pos",
            order=order,
            inner=None,
            ax=ax
        )

    sns.boxplot(
        data=df_pos,
        x="label",
        y="pos",
        order=order,
        width=0.2,
        showcaps=True,
        boxprops={'facecolor': 'none'},
        ax=ax
    )

    ax.set_ylabel("Posición en el ranking (menor = mejor)")
    ax.set_xlabel("Tipo de variable (label exacta)")
    ax.set_title("Distribución de posiciones por tipo de variable (exact labels)")
    plt.xticks(rotation=40)
    plt.tight_layout()

    return df_pos, fig



