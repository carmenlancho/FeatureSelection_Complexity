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



# path_csv = "Results_FS_bruto/Results_FS_bruto_bodyfat.csv"

# Agregamos por folds
def summarize_performance(path_csv):
    df = pd.read_csv(path_csv)

    df_summary = (df.groupby(['model', 'n_features', 'feature_set'])
        .agg(mean_accuracy=('accuracy', 'mean'),
             std_accuracy=('accuracy', 'std'),
             mean_gps=('gps', 'mean'),
             std_gps=('gps', 'std')).reset_index())

    return df_summary

# Elegimos mejor modelo global para cada dataset

def selec_modelo_perf(df_summary):
    # media de mean_accuracy por modelo (promediando sobre todos los feature_set)
    model_perf_acc = (df_summary.groupby('model')['mean_accuracy'].mean().reset_index())
    best_model_acc = model_perf_acc.loc[model_perf_acc['mean_accuracy'].idxmax(), 'model']
    # igual para GPS
    model_perf_gps = (df_summary.groupby('model')['mean_gps'].mean().reset_index())
    best_model_gps = model_perf_gps.loc[model_perf_gps['mean_gps'].idxmax(), 'model']

    # mejor modelo global
    model_perf_global = (df_summary.groupby('model')[['mean_accuracy', 'mean_gps']].mean().reset_index())
    model_perf_global['score_global'] = (model_perf_global['mean_accuracy'] + model_perf_global['mean_gps']) / 2
    best_model_global = model_perf_global.loc[model_perf_global['score_global'].idxmax(), 'model']

    # Seleccionamos mejor modelo (por ahora global)
    best_model = best_model_global  # best_model_acc, best_model_gps o best_model_global
    df_best_model = df_summary[df_summary['model'] == best_model].copy()

    return df_best_model


# Ranking de performance por modelo y dataset
def add_performance_ranking(df_best_model):
    df_best_model['rank_accuracy'] = (df_best_model['mean_accuracy'].rank(ascending=False, method='average'))
    df_best_model['rank_gps'] = (df_best_model['mean_gps'].rank(ascending=False, method='average'))

    return df_best_model


# path_complexity_csv = 'Results_FS_bruto/ComplexityBruto_Australian.csv'
# path_complexity_csv = 'Results_FS_bruto/ComplexityCVBruto_bodyfat.csv'

## Agregamos resultados por folds
def summarize_complexity(path_complexity_csv):
    complexity_cols = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD','N1', 'N2', 'LSC',
        'F1', 'F2', 'F3', 'F4','L1']
    df_complex = pd.read_csv(path_complexity_csv)

    group_cols = ['n_features', 'feature_set', 'k_folds']

    # estos resultados contienen directamentee la media, fallo mío que no he sacado la std
    df_complex_summary = (df_complex.groupby(group_cols)[complexity_cols].agg(['mean']).reset_index())

    # Nombres de columnas: (Hostility, mean) -> Hostility_mean
    df_complex_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in df_complex_summary.columns.values]

    # quitamos k_folds porque ya no aporta nada
    df_complex_summary.drop(['k_folds'],axis=1,inplace=True)

    return df_complex_summary



# Hacemos el ranking por medida de complejidad
def add_complexity_ranking(df_complex_summary):

    complexity_measures = ['Hostility_mean', 'kDN_mean', 'DCP_mean', 'TD_U_mean', 'CLD_mean',
                           'N1_mean', 'N2_mean', 'LSC_mean', 'F1_mean', 'F2_mean', 'F3_mean', 'F4_mean', 'L1_mean']

    for measure in complexity_measures:
        df_complex_summary[f'rank_{measure}'] = (df_complex_summary[measure].rank(ascending=True, method='average'))

    return df_complex_summary


# Unimos los dfs con los rankings de performance y de complejidad
def merge_perf_complexity(df_perf, df_complex):
    df_merged = pd.merge(df_perf,df_complex,on=['n_features', 'feature_set'],how='inner')
    return df_merged

# df_merged = merge_perf_complexity(df_best_model, df_complex_summary)


## Calculamos correlación de Spearmaan entre los rankings
# Queremos altas y positivas
def compute_spearman(df_merged, perf_rank_col):
    complexity_measures = ['Hostility_mean', 'kDN_mean', 'DCP_mean', 'TD_U_mean', 'CLD_mean',
                           'N1_mean', 'N2_mean', 'LSC_mean', 'F1_mean', 'F2_mean', 'F3_mean', 'F4_mean', 'L1_mean']

    results = []

    for measure in complexity_measures:
        rho, pval = spearmanr(df_merged[perf_rank_col],df_merged[f'rank_{measure}'])

        results.append({'complexity_measure': measure,
            'spearman_rho': rho,
            'p_value': pval})

    return pd.DataFrame(results)


# df_spear_acc = compute_spearman(df_merged, 'rank_accuracy')
# df_spear_gps = compute_spearman(df_merged, 'rank_gps')

# df = df_merged
# Obtenemos tb corr de spearman dentro de cada número de features
# por si hay una reelación de cuántas menos variables más simple o algo raro

def spearman_within_nfeatures(df, perf_metric='mean_accuracy'):
    complexity_measures = ['Hostility_mean', 'kDN_mean', 'DCP_mean', 'TD_U_mean', 'CLD_mean',
                           'N1_mean', 'N2_mean', 'LSC_mean', 'F1_mean', 'F2_mean', 'F3_mean', 'F4_mean', 'L1_mean']

    results = []
    model = df['model'][0]

    # for model in df['model'].unique():
    #     df_model = df[df['model'] == model]
    for n in sorted(df['n_features'].unique()):
        df_n = df[df['n_features'] == n]
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


# df_spear_acc = spearman_within_nfeatures(df_merged, perf_metric='mean_accuracy')
# df_spear_gps = spearman_within_nfeatures(df_merged, perf_metric='mean_gps')
# #
# df = df_merged


# Dentro del número de variables consideradas (k) estudiamos el comportamiento
# Queremos que sea lineal decreciente
def plot_perf_vs_complexity(df, n_features,
                            complexity_measure,
                            perf_metric='mean_accuracy'):
    model = df['model'][0]
    df_plot = df[(df['n_features'] == n_features)]

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

# n_features = 13
# complexity_measure = 'Hostility_mean'
#
# plot_perf_vs_complexity(df_merged, n_features,
#                             complexity_measure,
#                             perf_metric='mean_accuracy')

# Mapa de correlaciones para poder visualizar tod
def heatmap_correlations(df_spearman):
    model = df_spearman['model'][0]

    pivot = df_spearman.pivot_table(
        index='complexity_measure',
        columns='n_features',
        values='spearman_rho'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', center=0, annot=False)
    plt.title(f'Spearman rho - {model}')
    plt.tight_layout()
    plt.show()

# heatmap_correlations(df_spear_acc)



# summary = (
#     df_spear_acc
#     .groupby('complexity_measure')
#     .agg(mean_rho=('spearman_rho','mean'),
#          pct_negative=('spearman_rho', lambda x: (x < 0).mean()),
#          pct_significant=('p_value', lambda x: (x < 0.05).mean()))
#     .reset_index()
# )


# Generalizamos ahora para todos los datasets

# complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
#                        'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']
#

#

def compute_global_spearman(path_perf, path_complex):
    complexity_measures = ['Hostility_mean', 'kDN_mean', 'DCP_mean', 'TD_U_mean', 'CLD_mean',
                           'N1_mean', 'N2_mean', 'LSC_mean', 'F1_mean', 'F2_mean', 'F3_mean', 'F4_mean', 'L1_mean']
    # ---- Performance ----
    df_perf = pd.read_csv(path_perf, engine='python')

    df_perf_summary = (
        df_perf.groupby(['model', 'n_features', 'feature_set'])
        .agg(mean_accuracy=('accuracy', 'mean'),
             mean_gps=('gps', 'mean'))
        .reset_index()
    )

    # Selección del mejor modelo de forma global (mix de accuracy y gps)
    df_best_model = selec_modelo_perf(df_perf_summary)
    model = df_best_model['model'].unique()[0]
    # ranking para cada una de las medidas de rendimiento empleadas
    df_best_model['rank_accuracy'] = (df_best_model['mean_accuracy'].rank(ascending=False, method='average'))
    df_best_model['rank_gps'] = (df_best_model['mean_gps'].rank(ascending=False, method='average'))


    # ---- Complexity ----
    df_complex_summary = summarize_complexity(path_complex) # agregamos por folds (solo tenemos la media)
    df_complex_rank = add_complexity_ranking(df_complex_summary) # calculamos rankings por complexity measures

    # ---- Merge ----
    # Unimos los dfs con los rankings de performance y de complejidad
    df_merged =  merge_perf_complexity(df_best_model, df_complex_rank)

    results = []


    results = []
    for perf_rank_col, perf_name in [('rank_accuracy', 'mean_accuracy'),('rank_gps', 'mean_gps')]:
        for measure in complexity_measures:
            rho, pval = spearmanr(df_merged[perf_rank_col], df_merged[f'rank_{measure}'], nan_policy='omit')
            results.append({
                'model': model,
                'performance_metric': perf_name,
                'complexity_measure': measure,
                'spearman_rho': rho,
                'p_value': pval,
                'n_pairs': df_merged[[perf_rank_col, f'rank_{measure}']].dropna().shape[0]
            })


    return pd.DataFrame(results)


path_perf = "Results_FS_bruto/Results_FS_bruto_vehicle2.csv"
path_complex = 'Results_FS_bruto/ComplexityCVBruto_zoo.csv'
df_res_corrs = compute_global_spearman(path_perf, path_complex)



import csv

# df_perf = pd.read_csv(
#     path_perf,
#     engine='python',      # parser en Python (más flexible)
#     sep=',',              # cambia si NO es coma
#     quoting=csv.QUOTE_NONE,  # no trata las comillas como delimitadores
#     on_bad_lines='skip'      # salta las líneas mal formadas
# ) # para el tocho

datasets = ['bodyfat','boston','cleve',
            'heart-statlog','zoo','vehicle2']
# el resto no los tengo completos aun
# parkinsons y 'diabetic_retinopathy'  lo quito porque peta el ordenador

all_results = []

for ds in datasets:
    path_perf = f'Results_FS_bruto/Results_FS_bruto_{ds}.csv'
    path_complex = f'Results_FS_bruto/ComplexityCVBruto_{ds}.csv'

    df_ds = compute_global_spearman(path_perf, path_complex)
    df_ds['dataset'] = ds

    all_results.append(df_ds)

df_all = pd.concat(all_results, ignore_index=True)


def build_table(df_all, perf_metric):
    df_subset = df_all[(df_all['performance_metric'] == perf_metric)]

    table = df_subset.pivot(
        index='dataset',
        columns='complexity_measure',
        values='spearman_rho'
    )

    return table

table_acc = build_table(df_all, 'mean_accuracy')
table_gps = build_table(df_all,'mean_gps')

table_acc.to_csv("tabla_spearman_{}.csv".format('mean_accuracy'))
table_gps.to_csv("tabla_spearman_{}.csv".format('mean_gps'))



## Hacemos ahora la versión de la correlación dentro de los n_features y sin ranking

complexity_measures = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD',
                       'N1', 'N2', 'LSC', 'F1', 'F2', 'F3', 'F4', 'L1']


def compute_within_nfeatures_dataset(path_perf, path_complex, dataset_name):
    complexity_measures = ['Hostility_mean', 'kDN_mean', 'DCP_mean', 'TD_U_mean', 'CLD_mean',
                           'N1_mean', 'N2_mean', 'LSC_mean', 'F1_mean', 'F2_mean', 'F3_mean', 'F4_mean', 'L1_mean']
    # ---- Performance ----
    df_perf = pd.read_csv(path_perf, engine='python')

    df_perf_summary = (
        df_perf.groupby(['model', 'n_features', 'feature_set'])
        .agg(mean_accuracy=('accuracy', 'mean'),
             mean_gps=('gps', 'mean'))
        .reset_index()
    )

    # Selección del mejor modelo de forma global (mix de accuracy y gps)
    df_best_model = selec_modelo_perf(df_perf_summary)
    model = df_best_model['model'].unique()[0]
    # ranking para cada una de las medidas de rendimiento empleadas
    df_best_model['rank_accuracy'] = (df_best_model['mean_accuracy'].rank(ascending=False, method='average'))
    df_best_model['rank_gps'] = (df_best_model['mean_gps'].rank(ascending=False, method='average'))

    # ---- Complexity ----
    df_complex_summary = summarize_complexity(path_complex)  # agregamos por folds (solo tenemos la media)
    df_complex_rank = add_complexity_ranking(df_complex_summary)  # calculamos rankings por complexity measures

    # ---- Merge ----
    # Unimos los dfs con los rankings de performance y de complejidad
    df_merged = merge_perf_complexity(df_best_model, df_complex_rank)


    results = []

    for perf_metric in ['mean_accuracy', 'mean_gps']:
        for n in sorted(df_merged['n_features'].unique()):

            df_n = df_merged[df_merged['n_features'] == n]
            # Evitamos casos con pocos subconjuntos
            if len(df_n) < 5:
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
datasets = ['bodyfat','boston','cleve',
            'heart-statlog','zoo']

all_results = []

for ds in datasets:
    path_perf = f'Results_FS_bruto/Results_FS_bruto_{ds}.csv'
    path_complex = f'Results_FS_bruto/ComplexityCVBruto_{ds}.csv'

    df_ds = compute_within_nfeatures_dataset(path_perf, path_complex, ds)

    all_results.append(df_ds)

df_within_all = pd.concat(all_results, ignore_index=True) # tiene tod el estudio

# Ahora lo resumimos
# Hacemos media de rho por dataset y medida
def build_within_summary(df, perf_metric):
    df_sub = df[(df['performance_metric'] == perf_metric)]

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
table_acc_within = build_within_summary(df_within_all,'mean_accuracy')
table_gps_within = build_within_summary(df_within_all, 'mean_gps')

table_acc_within.to_csv("tabla_spearman_within_{}.csv".format('mean_accuracy'))
table_gps_within.to_csv("tabla_spearman_within_{}.csv".format('mean_gps'))



def summarize_measure_behavior(df, perf_metric):
    df_sub = df[
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


summarize_measure_behavior(df_within_all, 'mean_accuracy')
summarize_measure_behavior(df_within_all, 'mean_gps')



### Visualización para el congreso IDEAL 2026
# boxplot

import seaborn as sns
import matplotlib.pyplot as plt

# --- prepara data ---
d = df_all.copy()
d = d.rename(columns={"dataset":"Dataset", "spearman_rho":"rho"})

# nombres bonitos
d["Performance"] = d["performance_metric"].replace({
    "mean_accuracy": "Acc",
    "mean_gps": "GPS"
})

# acorta nombres de complejidad (quita _mean y deja label corto)
d["Complexity"] = (d["complexity_measure"]
                   .str.replace("_mean","", regex=False)
                   .replace({"Hostility":"Host", "TD_U":"TDU"}))

# orden x (el que tú quieras)
order = ["F1","L1","kDN","N1","Host","N2","LSC","DCP","TDU","CLD"]
order = [x for x in order if x in d["Complexity"].unique()]

# paletas: cajas suaves, puntos con GPS azul oscuro
palette_box = {"Acc": "#D9FDFF", "GPS": "#A8BFFF"}   # suaves
palette_pts = {"Acc": "#1f77b4", "GPS": "#031573"}   # azul / azul oscuro


fig, ax = plt.subplots(figsize=(9,5.2))

sns.boxplot(
    data=d, x="Complexity", y="rho", hue="Performance",
    order=order, palette=palette_box,
    width=0.6, fliersize=0,
    #boxprops=dict(alpha=0.25),
    #whiskerprops=dict(alpha=0.7),
    #capprops=dict(alpha=0.7),
    #medianprops=dict(color="black", linewidth=1.2),
    ax=ax
)

sns.stripplot(
    data=d, x="Complexity", y="rho", hue="Performance",
    order=order, palette=palette_pts,
    dodge=True, jitter=0.20, alpha=0.7, size=3.1,
    linewidth=0, zorder=3, ax=ax
)

# Leyenda limpia (evitar duplicados)
h, l = ax.get_legend_handles_labels()
ax.legend(h[:2], l[:2], title="", loc="lower left", bbox_to_anchor=(1.01, 0))

ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Spearman correlation")
ax.set_xlabel("")
ax.set_title(" ")
ax.set_ylim([-.9, 1])

plt.tight_layout()
plt.show()

