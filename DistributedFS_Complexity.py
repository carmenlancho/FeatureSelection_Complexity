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
        X = X.drop(columns=to_drop)


    variables = X.columns.tolist()
    importances = {m: pd.Series(0.0, index=variables) for m in measures} # diccionario para cada complexity measure
    importances_norm = {m: pd.Series(0.0, index=variables) for m in measures}  # diccionario para cada complexity measure
    count_vars = pd.Series(0.0, index=variables) # cuántas veces aparece cada var para normalización


    # rep = 0
    for rep in range(n_replicas):
        print(rep)
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





# # La complejidad con cada feature de forma univariante está en ArtificialDatasetXX_featuresComplexityRanking.csv
# csv_file = "Results_UnivariateRanking_CM/ArtificialDataset1_featuresComplexityRanking.csv"
# univariate_complexity = pd.read_csv(csv_file)

def distributed_variable_selection_complexity_guided(X, y, dataset_name, n_replicas, m_vars,
                                   univariate_complexity=None, # complejidad univariante
                                   measures=["Hostility",'kDN','N1'],
                                   filter_corr=True, corr_th=0.9, corr_method="pearson",
                                   random_state=0, save_csv=False, path='Results_FS_Distributed'):

    np.random.seed(random_state)
    random.seed(random_state)

    # Filtro previo opcional de eliminación de variables con correelación > corr_th
    if filter_corr:
        corr = X.corr(method=corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        X = X.drop(columns=to_drop)

    variables = X.columns.tolist()
    results_all = pd.DataFrame()

    for cm in measures:

        importances = pd.Series(0.0, index=variables) # diccionario para cada complexity measure
        # importances_norm = pd.Series(0.0, index=variables)  # diccionario para cada complexity measure
        count_vars = pd.Series(0.0, index=variables) # cuántas veces aparece cada var para normalización

        # rep = 0
        for rep in range(n_replicas):
            print(rep)
            subset_vars = random.choices(variables, k=m_vars)  # sampling WITH replacement
            Xsub = X[subset_vars]

            # Multivariate complexity
            datos = pd.DataFrame(Xsub)
            datos['y'] = y
            _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
            base_complexity = df_classes.loc['dataset',cm]

            current_vars = subset_vars.copy()
            while len(current_vars) > 1: # hasta que nos quedemos sin vars
                # Conteo de variables participantes
                count_vars[current_vars] += 1

                # elegir variable a quitar
                df_uni_comp = univariate_complexity[univariate_complexity["level"] == "dataset"].copy()
                df_uni_comp = df_uni_comp.set_index("feature")[cm]
                guided_scores = df_uni_comp.loc[current_vars]
                to_remove = guided_scores.idxmax()  # quita la de mayor complejidad univariante

                current_vars.remove(to_remove)
                Xtemp = X[current_vars]

                # Complexity removing one variable from previous subset
                datos_temp = pd.DataFrame(Xtemp)
                datos_temp['y'] = y
                _, df_classes_temp, _ = all_measures_FS(datos_temp, save_csv=False, path_to_save=None, name_data=None)
                new_complexity = df_classes_temp.loc['dataset',cm]

                # Cambio de complejidad
                delta = new_complexity - base_complexity # mee interesan diferencias positivas, es decir,
                # quitar la variable aumenta la complejidad, luego la variable es útil
                # Las diferencias negativas me dicen que esa variable aumentaba la complejidad, luego no la quiero
                importances[to_remove] += delta

                # Actualizamos base_complexity para próximo paso
                base_complexity = new_complexity
                # Conteo de variables participantes
                count_vars[current_vars] += 1

        # Normalizamos importancias
        count_vars = count_vars.replace(0, np.nan) # para no tener problemas con los 0 en la división
        importances_norm = importances / count_vars # por probabilidad, con n_replicas grande, deben ser similares estos números
        importances_norm.sort_values(ascending=False, inplace=True)


        results_norm = pd.DataFrame.from_dict(importances_norm)
        results_norm.columns = [cm + '_importances_norm']
        results = pd.DataFrame.from_dict(importances)
        results.columns = [cm + '_importances']
        results_count = pd.DataFrame.from_dict(count_vars)
        results_count.columns = ['count_vars']
        results_complete = pd.concat([results, results_norm, results_count], axis=1)

        results_all = pd.concat([results_all, results_complete], axis=1)

    if filter_corr: # aclaramos las que se hayan quitado por correlación
        results_all = results_all.reindex(results_all.index.union(to_drop))
        results_all.loc[to_drop,:] = np.nan


    if save_csv:
        name_csv = f"{path}/{dataset_name}_ComplexityGuidedDistributed.csv"
        results_all.to_csv(name_csv, index=True)

    return results_all



def plot_importances(importances_norm):
    for m, imp in importances_norm.items():
        imp.plot(kind="barh", figsize=(8,6), title=f"Variable importance ({m})")
        plt.xlabel("Avg complexity in backward")
        plt.gca().invert_yaxis()
        plt.show()


# ME FALTA HACER LA CADENA DE EJECUCIÓN Y SELECCIONAR LOS DATASETS PORQUE TARDA MUCHO Y ALGUNOS SON
# MUY IGUALEES


### Random

### Dataset 1
dataset_name = 'ArtificialDataset1'
X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                        flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01)


n_replicas=2
m_vars=5
distributed_variable_selection_complexity_random(X, y, dataset_name, n_replicas, m_vars,
                                   measures=["Hostility", "N1", "kDN"],
                                   filter_corr=True, corr_th=0.9, corr_method="pearson",
                                   random_state=0, save_csv=True, path='Results_FS_Distributed')


## Guided
# La complejidad con cada feature de forma univariante está en ArtificialDatasetXX_featuresComplexityRanking.csv
csv_file = "Results_UnivariateRanking_CM/ArtificialDataset1_featuresComplexityRanking.csv"
univariate_complexity = pd.read_csv(csv_file)

distributed_variable_selection_complexity_guided(X, y, dataset_name, n_replicas, m_vars,
                                                            univariate_complexity=univariate_complexity,
                                                            # complejidad univariante
                                                            measures=['Hostility','kDN','N1'],
                                                            filter_corr=True, corr_th=0.9,
                                                            corr_method="pearson",
                                                            random_state=0, save_csv=True,
                                                            path='Results_FS_Distributed')







