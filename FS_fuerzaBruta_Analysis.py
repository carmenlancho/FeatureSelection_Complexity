## 11/02/2026

## En este script analizamos los resultados obtenidos con FS_fuerzaBruta.py y Complexity_fuerzaBruta.py
# La idea es estudiar, si en el caso de los datos reales, tb ocurre que el mejor subconjunto de variables
# ees el que menor complejidad tiene
# Para ello, comenzamos agregando los resultados de performance por fold y luego calculamos correlación
# de spearman entre complejidad y performance


import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr



path_csv = "Results_FS_bruto/Results_FS_bruto_Australian.csv"

# Agregamos por folds
def summarize_performance(path_csv):
    df = pd.read_csv(path_csv)

    df_summary = (df.groupby(['model', 'n_features', 'feature_set'])
        .agg(mean_accuracy=('accuracy', 'mean'),
             std_accuracy=('accuracy', 'std'),
             mean_gps=('gps', 'mean'),
             std_gps=('gps', 'std')).reset_index())

    return df_summary


# Ranking de performance por modelo y dataset
def add_performance_ranking(df_summary):
    df_summary['rank_accuracy'] = (df_summary['mean_accuracy'].rank(ascending=False, method='average'))
    df_summary['rank_gps'] = (df_summary['mean_gps'].rank(ascending=False, method='average'))

    return df_summary


path_complexity_csv = 'Results_FS_bruto/ComplexityBruto_Australian.csv'
# Leemos complejidad y hacemos el ranking por medida de complejidad
def add_complexity_ranking(path_complexity_csv):
    df_c = pd.read_csv(path_complexity_csv)

    complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
                           'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']

    for measure in complexity_measures:
        df_c[f'rank_{measure}'] = (df_c[measure].rank(ascending=True, method='average'))

    return df_c


# Unimos los dfs con los rankings de performance y de complejidad
def merge_perf_complexity(df_perf, df_complex):
    df_merged = pd.merge(df_perf,df_complex,on=['n_features', 'feature_set'],how='inner')
    return df_merged

df_merged = merge_perf_complexity(df_summary, df_c)


## Calculamos correlación de Spearmaan entre los rankings
# Queremos altas y positivas
def compute_spearman(df_merged, perf_rank_col):
    complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
                           'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']

    results = []

    for measure in complexity_measures:
        rho, pval = spearmanr(df_merged[perf_rank_col],df_merged[f'rank_{measure}'])

        results.append({'complexity_measure': measure,
            'spearman_rho': rho,
            'p_value': pval})

    return pd.DataFrame(results)


df_spear_acc = compute_spearman(df_merged, 'rank_accuracy')
df_spear_gps = compute_spearman(df_merged, 'rank_gps')


# Obtenemos tb corr de spearman dentro de cada número de features
# por si hay una reelación de cuántas menos variables más simple o algo raro

def spearman_within_nfeatures(df, perf_metric='mean_accuracy'):
    complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
                           'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']

    results = []

    for model in df['model'].unique():
        df_model = df[df['model'] == model]
        for n in sorted(df_model['n_features'].unique()):
            df_n = df_model[df_model['n_features'] == n]
            # si hay pocos subconjuntos no tiene sentido calcular correlación
            if len(df_n) < 3:
                continue
            for measure in complexity_measures:
                rho, pval = spearmanr(df_n[perf_metric],df_n[measure])

                results.append({
                    'model': model,
                    'n_features': n,
                    'complexity_measure': measure,
                    'spearman_rho': rho,
                    'p_value': pval,
                    'n_subsets': len(df_n)})

    return pd.DataFrame(results)


df_spear_acc = spearman_within_nfeatures(df_merged, perf_metric='mean_accuracy')
df_spear_gps = spearman_within_nfeatures(df_merged, perf_metric='mean_gps')

df = df_merged
model = 'KNN'
n_features = 13
complexity_measure = 'Hostility'

# Dentro del número de variables consideradas (k) estudiamos el comportamiento
# Queremos que sea lineal decreciente
def plot_perf_vs_complexity(df, model, n_features,
                            complexity_measure,
                            perf_metric='mean_accuracy'):
    df_plot = df[(df['model'] == model) &
                 (df['n_features'] == n_features)]

    plt.figure(figsize=(6, 4))

    sns.regplot(
        data=df_plot,
        x=complexity_measure,
        y=perf_metric,
        lowess=True,
        scatter_kws={'alpha': 0.6}
    )

    plt.title(f'{model} | {n_features} features')
    plt.tight_layout()
    plt.show()

# Mapa de correlaciones para poder visualizar todo
def heatmap_correlations(df_spearman, model):
    df_model = df_spearman[df_spearman['model'] == model]

    pivot = df_model.pivot_table(
        index='complexity_measure',
        columns='n_features',
        values='spearman_rho'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', center=0, annot=False)
    plt.title(f'Spearman rho - {model}')
    plt.tight_layout()
    plt.show()

heatmap_correlations(df_spear_acc, model='SVM-rbf')



# summary = (
#     df_spear_acc
#     .groupby('complexity_measure')
#     .agg(mean_rho=('spearman_rho','mean'),
#          pct_negative=('spearman_rho', lambda x: (x < 0).mean()),
#          pct_significant=('p_value', lambda x: (x < 0.05).mean()))
#     .reset_index()
# )


# Generalizamos ahora para todos los datasets

complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
                       'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']


def compute_global_spearman(path_perf, path_complex):
    # ---- Performance ----
    df_perf = pd.read_csv(path_perf)

    df_perf_summary = (
        df_perf.groupby(['model', 'n_features', 'feature_set'])
        .agg(mean_accuracy=('accuracy', 'mean'),
             mean_gps=('gps', 'mean'))
        .reset_index()
    )

    # ---- Complexity ----
    df_complex = pd.read_csv(path_complex)

    # ---- Merge ----
    df = pd.merge(
        df_perf_summary,
        df_complex,
        on=['n_features', 'feature_set'],
        how='inner'
    )

    results = []

    for model in df['model'].unique():
        df_model = df[df['model'] == model]

        for perf_metric in ['mean_accuracy', 'mean_gps']:

            for measure in complexity_measures:
                rho, pval = spearmanr(
                    df_model[perf_metric],
                    df_model[measure]
                )

                results.append({
                    'model': model,
                    'performance_metric': perf_metric,
                    'complexity_measure': measure,
                    'spearman_rho': rho,
                    'p_value': pval
                })

    return pd.DataFrame(results)

datasets = ['Australian','bands','credit-g','diabetic_retinopathy',
            'plasma_retinol','pollution','vehicle2']

all_results = []

for ds in datasets:
    path_perf = f'Results_FS_bruto/Results_FS_bruto_{ds}.csv'
    path_complex = f'Results_FS_bruto/ComplexityBruto_{ds}.csv'

    df_ds = compute_global_spearman(path_perf, path_complex)
    df_ds['dataset'] = ds

    all_results.append(df_ds)

df_all = pd.concat(all_results, ignore_index=True)


def build_table(df_all, model, perf_metric):
    df_subset = df_all[
        (df_all['model'] == model) &
        (df_all['performance_metric'] == perf_metric)
        ]

    table = df_subset.pivot(
        index='dataset',
        columns='complexity_measure',
        values='spearman_rho'
    )

    return table

table_svm_acc = build_table(df_all, 'SVM-rbf', 'mean_accuracy')
table_svm_gps = build_table(df_all, 'SVM-rbf', 'mean_gps')

table_knn_acc = build_table(df_all, 'KNN', 'mean_accuracy')
table_knn_gps = build_table(df_all, 'KNN', 'mean_gps')



## Hacemos ahora la versión de la correlación dentro de los n_features y sin ranking

complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
                       'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']


def compute_within_nfeatures_dataset(path_perf, path_complex, dataset_name):
    # ---- Performance ----
    df_perf = pd.read_csv(path_perf)

    df_perf_summary = (
        df_perf.groupby(['model', 'n_features', 'feature_set'])
        .agg(mean_accuracy=('accuracy', 'mean'),
             mean_gps=('gps', 'mean'))
        .reset_index()
    )

    # ---- Complexity ----
    df_complex = pd.read_csv(path_complex)

    # ---- Merge ----
    df = pd.merge(
        df_perf_summary,
        df_complex,
        on=['n_features', 'feature_set'],
        how='inner'
    )

    results = []

    for model in df['model'].unique():

        df_model = df[df['model'] == model]

        for perf_metric in ['mean_accuracy', 'mean_gps']:

            for n in sorted(df_model['n_features'].unique()):

                df_n = df_model[df_model['n_features'] == n]

                # Evitamos casos con pocos subconjuntos
                if len(df_n) < 3:
                    continue

                for measure in complexity_measures:
                    rho, pval = spearmanr(
                        df_n[perf_metric],
                        df_n[measure]
                    )

                    results.append({
                        'dataset': dataset_name,
                        'model': model,
                        'performance_metric': perf_metric,
                        'n_features': n,
                        'complexity_measure': measure,
                        'spearman_rho': rho,
                        'p_value': pval,
                        'n_subsets': len(df_n)
                    })

    return pd.DataFrame(results)

# Ejecutamos para todos los datasets
datasets = ['Australian','bands','credit-g','diabetic_retinopathy',
            'plasma_retinol','pollution','vehicle2']

all_results = []

for ds in datasets:
    path_perf = f'Results_FS_bruto/Results_FS_bruto_{ds}.csv'
    path_complex = f'Results_FS_bruto/ComplexityBruto_{ds}.csv'

    df_ds = compute_within_nfeatures_dataset(path_perf, path_complex, ds)

    all_results.append(df_ds)

df_within_all = pd.concat(all_results, ignore_index=True) # tiene todo el estudio

# Ahora lo resumimos
# Hacemos media de rho por dataset y medida
def build_within_summary(df, model, perf_metric):
    df_sub = df[
        (df['model'] == model) &
        (df['performance_metric'] == perf_metric)
        ]

    summary = (
        df_sub.groupby(['dataset', 'complexity_measure'])
        .agg(mean_rho=('spearman_rho', 'mean'))
        .reset_index()
    )

    table = summary.pivot(
        index='dataset',
        columns='complexity_measure',
        values='mean_rho'
    )

    return table

# Estas tablas contienen la media de las correlaciones a través de todos los n_features
table_svm_acc_within = build_within_summary(df_within_all, 'SVM-rbf', 'mean_accuracy')
table_svm_gps_within = build_within_summary(df_within_all, 'SVM-rbf', 'mean_gps')

table_knn_acc_within = build_within_summary(df_within_all, 'KNN', 'mean_accuracy')
table_knn_gps_within = build_within_summary(df_within_all, 'KNN', 'mean_gps')


def summarize_measure_behavior(df, model, perf_metric):
    df_sub = df[
        (df['model'] == model) &
        (df['performance_metric'] == perf_metric)
        ]

    summary = (
        df_sub.groupby('complexity_measure')
        .agg(mean_rho=('spearman_rho', 'mean'),
             pct_negative=('spearman_rho', lambda x: (x <= -0.7).mean()),
             pct_significant=('p_value', lambda x: (x < 0.05).mean()))
        .reset_index()
    )

    return summary


summarize_measure_behavior(df_within_all, 'KNN', 'mean_accuracy')
