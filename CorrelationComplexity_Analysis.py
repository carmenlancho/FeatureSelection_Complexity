##### 04/02/2026
#### En este script vamos a analizar los resultados del script CorrelationComplexity



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
import glob

# esto porque a veces tnemos 2 filas como índicee
# y a veeces solo
def flatten_columns(cols):
    new_cols = []
    for c in cols:
        if isinstance(c, tuple):
            c0, c1 = c
            # casos tipo ('corr_type', 'Unnamed: 2_level_1')
            if "unnamed" in str(c1).lower() or c1 == "" or pd.isna(c1):
                new_cols.append(str(c0).strip().lower())
            else:
                new_cols.append(f"{c0}_{c1}".strip().lower())
        else:
            new_cols.append(str(c).strip().lower())
    return new_cols


def load_datasets(base_path="Results_CorrelationComplexity"):
    dfs = []

    path = os.path.join(base_path, "*.csv")
    files = glob.glob(path)

    for f in files:
        if 'Summary' in f:
            # leer doble header
            df = pd.read_csv(f, header=[0, 1])
            # flatten columns
            df.columns = flatten_columns(df.columns)
            df["dataset_name"] = os.path.basename(f).replace("_CorrComplexityUnivariateFilter_Folds_SummaryResults.csv", "")
            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


df = load_datasets("Results_CorrelationComplexity")

# Separamos artificial y real
df_artificial = df[df["dataset_name"].str.contains("Artificial", case=False, na=False)]
df_real = df[~df["dataset_name"].str.contains("Artificial", case=False, na=False)]

## Funciones para agregar resultados
def aggregate_performance(df):
    group_cols = ["model", "measure", "corr_type", "corr_th"]

    agg = df.groupby(group_cols).agg(
        acc_test_mean=("acc_test_mean", "mean"),
        acc_test_std=("acc_test_mean", "std"),
        gps_test_mean=("gps_test_mean", "mean"),
        gps_test_std=("gps_test_mean", "std"),

        nfeat_mean=("n_features_mean", "mean"),
        nfeat_std=("n_features_mean", "std"),

        comp_all_kdn=("complexity_all_kdn_mean", "mean"),
        comp_sel_kdn=("complexity_sel_kdn_mean", "mean"),
        comp_all_n1=("complexity_all_n1_mean", "mean"),
        comp_sel_n1=("complexity_sel_n1_mean", "mean"),
        comp_all_host=("complexity_all_hostility_mean", "mean"),
        comp_sel_host=("complexity_sel_hostility_mean", "mean"),
    ).reset_index()


    return agg

agg_art = aggregate_performance(df_artificial)
agg_real = aggregate_performance(df_real)


def summary_tables(agg):
    # Por modelo
    t_model = agg.groupby(["model"]).mean(numeric_only=True)

    # Por métrica de complejidad
    t_measure = agg.groupby(["measure"]).mean(numeric_only=True)

    # Por correlación
    t_corr = agg.groupby(["corr_type"]).mean(numeric_only=True)

    return t_model, t_measure, t_corr

t_model, t_measure, t_corr = summary_tables(agg_art)
t_model, t_measure, t_corr = summary_tables(agg_real)


## Estudio gráfico

def filter_perf(df, model, metric_col):
    """
    NONE vs {kDN,N1,Hostility} × {pearson,spearman}
    """
    # baseline (NONE)
    base = df[(df["model"] == model) & (df["measure"] == "ALL")][[metric_col]].copy()
    base["group"] = "NONE"

    # complejidad
    others = df[
        (df["model"] == model) &
        (df["measure"].isin(["kDN", "N1", "Hostility"])) &
        (df["corr_type"].isin(["pearson", "spearman"]))
    ][["measure", "corr_type", metric_col]].copy()

    others["group"] = others["measure"] + "_" + others["corr_type"]

    out = pd.concat([
        base[[metric_col, "group"]],
        others[[metric_col, "group"]]
    ], ignore_index=True)

    return out


def make_boxplot(data, value_col, title, ylabel):
    groups = data["group"].unique().tolist()
    values = [data[data["group"] == g][value_col].dropna().values for g in groups]

    plt.figure()
    plt.boxplot(values, labels=groups)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



## Knn
acc_knn = filter_perf(df_artificial, "KNN", "acc_test_mean")

make_boxplot(acc_knn,"acc_test_mean",
    "Accuracy comparison (KNN)\nNONE vs complexity measures",
    "Accuracy (test)")

gps_knn = filter_perf(df_artificial, "KNN", "gps_test_mean")

make_boxplot(
    gps_knn,
    "gps_test_mean",
    "GPS comparison (KNN)\nNONE vs complexity measures",
    "GPS (test)"
)


acc_knn = filter_perf(df_real, "KNN", "acc_test_mean")

make_boxplot(acc_knn,"acc_test_mean",
    "Accuracy comparison (KNN)\nNONE vs complexity measures",
    "Accuracy (test)")


gps_knn = filter_perf(df_real, "KNN", "gps_test_mean")

make_boxplot(
    gps_knn,
    "gps_test_mean",
    "GPS comparison (KNN)\nNONE vs complexity measures",
    "GPS (test)"
)


# svm
acc_svm = filter_perf(df_artificial, "SVM-rbf", "acc_test_mean")

make_boxplot(
    acc_svm,
    "acc_test_mean",
    "Accuracy comparison (SVM-rbf)\nNONE vs complexity measures",
    "Accuracy (test)"
)


gps_svm = filter_perf(df_artificial, "SVM-rbf", "gps_test_mean")

make_boxplot(
    gps_svm,
    "gps_test_mean",
    "GPS comparison (KNN)\nNONE vs complexity measures",
    "GPS (test)"
)


acc_svm = filter_perf(df_real, "SVM-rbf", "acc_test_mean")

make_boxplot(
    acc_svm,
    "acc_test_mean",
    "Accuracy comparison (SVM-rbf)\nNONE vs complexity measures",
    "Accuracy (test)"
)


gps_svm = filter_perf(df_real, "SVM-rbf", "gps_test_mean")

make_boxplot(
    gps_svm,
    "gps_test_mean",
    "GPS comparison (KNN)\nNONE vs complexity measures",
    "GPS (test)"
)


## Complejidad
def complexity_box(df):
    rows = []

    # -------- baselines por métrica --------
    base = df[df["measure"] == "ALL"]

    rows.append(pd.DataFrame({
        "value": base["complexity_all_kdn_mean"],
        "group": "ALL_kDN"
    }))

    rows.append(pd.DataFrame({
        "value": base["complexity_all_n1_mean"],
        "group": "ALL_N1"
    }))

    rows.append(pd.DataFrame({
        "value": base["complexity_all_hostility_mean"],
        "group": "ALL_Hostility"
    }))

    # -------- filtrados --------
    for m in ["kDN", "N1", "Hostility"]:
        for c in ["pearson", "spearman"]:
            sub = df[(df["measure"] == m) & (df["corr_type"] == c)]

            rows.append(pd.DataFrame({
                "value": sub[f"complexity_sel_{m.lower()}_mean"],
                "group": f"{m}_{c}"
            }))

    return pd.concat(rows, ignore_index=True)



comp_df = complexity_box(df_artificial)
comp_df = complexity_box(df_real)

order = [
    "ALL_kDN", "kDN_pearson", "kDN_spearman",
    "ALL_N1", "N1_pearson", "N1_spearman",
    "ALL_Hostility", "Hostility_pearson", "Hostility_spearman"
]

groups = [g for g in order if g in comp_df["group"].unique()]
values = [comp_df[comp_df["group"] == g]["value"].dropna().values for g in groups]

plt.figure()
plt.boxplot(values, labels=groups)
plt.title("Complexity: initial vs filtered\nPer-complexity-measure baselines")
plt.ylabel("Complexity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


## Número de variables

def nfeatures_box(df):
    rows = []

    # baseline
    base = df[df["measure"] == "ALL"]
    rows.append(pd.DataFrame({
        "value": base["n_features_mean"],
        "group": "ALL"
    }))

    # filtrado por complejidad
    for m in ["kDN", "N1", "Hostility"]:
        for c in ["pearson", "spearman"]:
            sub = df[(df["measure"] == m) & (df["corr_type"] == c)]
            rows.append(pd.DataFrame({
                "value": sub["n_features_mean"],
                "group": f"{m}_{c}"
            }))

    return pd.concat(rows, ignore_index=True)


nf_df = nfeatures_box(df_real)

groups = nf_df["group"].unique().tolist()
values = [nf_df[nf_df["group"] == g]["value"].dropna().values for g in groups]

plt.figure()
plt.boxplot(values, labels=groups)
plt.title("Number of features: initial vs filtered\nComplexity-space correlation filtering")
plt.ylabel("Number of features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



### Lo haceeemos ahora vrsión línea para poder analizar mejor

def filter_perf_long(df, model, metric_col):
    """
    Formato largo con dataset_name para líneas
    """
    rows = []

    # NONE
    base = df[(df["model"] == model) & (df["measure"] == "ALL")]
    for _, r in base.iterrows():
        rows.append({
            "dataset": r["dataset_name"],
            "group": "NONE",
            "value": r[metric_col]
        })

    # complejidad
    others = df[
        (df["model"] == model) &
        (df["measure"].isin(["kDN", "N1", "Hostility"])) &
        (df["corr_type"].isin(["pearson", "spearman"]))
    ]

    for _, r in others.iterrows():
        rows.append({
            "dataset": r["dataset_name"],
            "group": f'{r["measure"]}_{r["corr_type"]}',
            "value": r[metric_col]
        })

    return pd.DataFrame(rows)


def make_lineplot(df_long, title, ylabel, order):
    plt.figure()

    datasets = df_long["dataset"].unique()

    x = np.arange(len(order))

    for d in datasets:
        sub = df_long[df_long["dataset"] == d]
        y = []
        for g in order:
            val = sub[sub["group"] == g]["value"]
            y.append(val.values[0] if len(val) else np.nan)

        plt.plot(x, y, marker='o', linewidth=1)

    plt.xticks(x, order, rotation=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


perf_order = ["NONE","kDN_pearson", "kDN_spearman",
    "N1_pearson", "N1_spearman","Hostility_pearson", "Hostility_spearman"]

# knn - accuracy
acc_knn_art = filter_perf_long(df_artificial, "KNN", "acc_test_mean")
make_lineplot(
    acc_knn_art,
    "Accuracy evolution per dataset (KNN) — Artificial",
    "Accuracy (test)",
    perf_order
)

acc_knn_real = filter_perf_long(df_real, "KNN", "acc_test_mean")
make_lineplot(
    acc_knn_real,
    "Accuracy evolution per dataset (KNN) — Real",
    "Accuracy (test)",
    perf_order
)

# svm - accuracy
acc_svm_art = filter_perf_long(df_artificial, "SVM-rbf", "acc_test_mean")
make_lineplot(
    acc_svm_art,
    "Accuracy evolution per dataset (SVM-rbf) — Artificial",
    "Accuracy (test)",
    perf_order
)

acc_svm_real = filter_perf_long(df_real, "SVM-rbf", "acc_test_mean")
make_lineplot(
    acc_svm_real,
    "Accuracy evolution per dataset (SVM-rbf) — Real",
    "Accuracy (test)",
    perf_order
)

# knn - gps
gps_knn_art = filter_perf_long(df_artificial, "KNN", "gps_test_mean")
make_lineplot(
    gps_knn_art,
    "GPS evolution per dataset (KNN) — Artificial",
    "GPS (test)",
    perf_order
)

gps_knn_real = filter_perf_long(df_real, "KNN", "gps_test_mean")
make_lineplot(
    gps_knn_real,
    "GPS evolution per dataset (KNN) — Real",
    "GPS (test)",
    perf_order
)

# svm - gps

gps_svm_art = filter_perf_long(df_artificial, "SVM-rbf", "gps_test_mean")
make_lineplot(
    gps_svm_art,
    "GPS evolution per dataset (SVM-rbf) — Artificial",
    "GPS (test)",
    perf_order
)

gps_svm_real = filter_perf_long(df_real, "SVM-rbf", "gps_test_mean")
make_lineplot(
    gps_svm_real,
    "GPS evolution per dataset (SVM-rbf) — Real",
    "GPS (test)",
    perf_order
)


# Complejidad
def complexity_long(df):
    rows = []

    base = df[df["measure"] == "ALL"]

    for _, r in base.iterrows():
        rows.append({
            "dataset": r["dataset_name"],
            "group": "ALL_kDN",
            "value": r["complexity_all_kdn_mean"]
        })
        rows.append({
            "dataset": r["dataset_name"],
            "group": "ALL_N1",
            "value": r["complexity_all_n1_mean"]
        })
        rows.append({
            "dataset": r["dataset_name"],
            "group": "ALL_Hostility",
            "value": r["complexity_all_hostility_mean"]
        })

    for m in ["kDN", "N1", "Hostility"]:
        for c in ["pearson", "spearman"]:
            sub = df[(df["measure"] == m) & (df["corr_type"] == c)]
            for _, r in sub.iterrows():
                rows.append({
                    "dataset": r["dataset_name"],
                    "group": f"{m}_{c}",
                    "value": r[f"complexity_sel_{m.lower()}_mean"]
                })

    return pd.DataFrame(rows)


comp_art = complexity_long(df_artificial)
comp_real = complexity_long(df_real)

comp_order = [
    "ALL_kDN", "kDN_pearson", "kDN_spearman",
    "ALL_N1", "N1_pearson", "N1_spearman",
    "ALL_Hostility", "Hostility_pearson", "Hostility_spearman"
]

make_lineplot(
    comp_art,
    "Complexity evolution per dataset — Artificial",
    "Complexity",
    comp_order
)

make_lineplot(
    comp_real,
    "Complexity evolution per dataset — Real",
    "Complexity",
    comp_order
)


# nfeaats
def nfeatures_long(df):
    rows = []

    base = df[df["measure"] == "ALL"]
    for _, r in base.iterrows():
        rows.append({
            "dataset": r["dataset_name"],
            "group": "ALL",
            "value": r["n_features_mean"]
        })

    for m in ["kDN", "N1", "Hostility"]:
        for c in ["pearson", "spearman"]:
            sub = df[(df["measure"] == m) & (df["corr_type"] == c)]
            for _, r in sub.iterrows():
                rows.append({
                    "dataset": r["dataset_name"],
                    "group": f"{m}_{c}",
                    "value": r["n_features_mean"]
                })

    return pd.DataFrame(rows)


nf_art = nfeatures_long(df_artificial)
nf_real = nfeatures_long(df_real)

nf_order = [
    "ALL",
    "kDN_pearson", "kDN_spearman",
    "N1_pearson", "N1_spearman",
    "Hostility_pearson", "Hostility_spearman"
]

make_lineplot(
    nf_art,
    "Number of features evolution per dataset — Artificial",
    "Number of features",
    nf_order
)

make_lineplot(
    nf_real,
    "Number of features evolution per dataset — Real",
    "Number of features",
    nf_order
)



####################################################################################################
############                    CON THRESHOLDS Y PARA MÁS DATASTS                       ############
####################################################################################################


path2 = 'Results_CorrelationComplexity'
df = pd.read_csv(path2+'/Australian_CorrComplexityUnivariateFilter_Thresholds_Folds.csv')
# Agregamos por folds y por parámetros de interés

# Para que no se cargue los NONE que es el baseline de referencia
df_fixed = df.copy()

# strategy NONE
df_fixed["corr_th"] = df_fixed["corr_th"].fillna("NONE")
df_fixed["measure"] = df_fixed["measure"].fillna("NONE")
# tipo categórico
df_fixed["corr_th"] = df_fixed["corr_th"].astype(str)
df_fixed["measure"] = df_fixed["measure"].astype(str)


group_cols = ["dataset", "model", "strategy", "pre_corr", "corr_th", "measure"]

agg_cols = ["n_features","acc_train", "gps_train","acc_test", "gps_test",
            "complexity_all_kDN", "complexity_all_Hostility", "complexity_all_N1",
            "complexity_sel_kDN", "complexity_sel_Hostility", "complexity_sel_N1"]

df_agg = (df_fixed.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index())

# Aplanar MultiIndex de columnas
df_agg.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_agg.columns]
# Separamos dataset column en dataset y filter
df_agg[['dataset', 'filtro']] = df_agg['dataset'].str.split('_', n=1, expand=True)


### Función para agregar por folds los datos y darles el formato correcto
# es decir, separar lo del filtro y arreglar lo de que entienda NONE comoc ategórico

def aggregate_folds(df):

    df_fixed = df.copy()

    # Baseline (NONE)
    df_fixed["corr_th"] = df_fixed["corr_th"].fillna("NONE")
    df_fixed["measure"] = df_fixed["measure"].fillna("NONE")

    df_fixed["corr_th"] = df_fixed["corr_th"].astype(str)
    df_fixed["measure"] = df_fixed["measure"].astype(str)

    group_cols = ["dataset", "model", "strategy", "pre_corr", "corr_th", "measure"]

    agg_cols = ["n_features","acc_train", "gps_train","acc_test", "gps_test",
                "complexity_all_kDN", "complexity_all_Hostility", "complexity_all_N1",
                "complexity_sel_kDN", "complexity_sel_Hostility", "complexity_sel_N1"]

    df_agg = (df_fixed.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index())

    # Aplanar MultiIndex
    df_agg.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_agg.columns]

    # Separar dataset y filtro
    df_agg[['dataset', 'filtro']] = df_agg['dataset'].str.rsplit('_', n=1, expand=True) # por la derecha

    return df_agg

# Hacemos la carga global de todos los datasets a analizar
files = glob.glob(f"{path2}/*_Thresholds_Folds.csv")
dfs = []
for f in files:
    df = pd.read_csv(f)
    df_agg = aggregate_folds(df)
    dfs.append(df_agg)

df_global = pd.concat(dfs, ignore_index=True) # todos leídos y con formato correcto

# Identificamos el baseline (no hacer nada) para hacer las comparaciones
baseline_mask = ((df_global["strategy"] == "NONE") &
                 (df_global["measure"] == "NONE") &
                 (df_global["corr_th"] == "NONE"))

# Calculamos la diferencia entre el baseline y nuestra propuesta
def add_baseline_deltas(df):

    base = df[(df.strategy == "NONE") &(df.measure == "NONE") &(df.corr_th == "NONE")
    ][["dataset","model","filtro","acc_test_mean","gps_test_mean","n_features_mean"]]

    base = base.rename(columns={"acc_test_mean": "acc_test_mean_base",
                               "gps_test_mean": "gps_test_mean_base",
                                "n_features_mean": "n_feat_mean_base"})

    df = df.merge(base, on=["dataset","model","filtro"], how="left")

    # Deltas: incremento en performance con respecto al baseline
    df["delta_acc_test"] = df["acc_test_mean"] - df["acc_test_mean_base"]
    df["delta_gps_test"] = df["gps_test_mean"] - df["gps_test_mean_base"]
    df["delta_n_feat"] = df["n_features_mean"] - df["n_feat_mean_base"]

    return df

df_baseline = add_baseline_deltas(df_global)

# Elegimos medida para las comparaciones
def get_measure_df(df, measure, filter_name):
    df_m = df[((df.measure == measure) | ((df.measure == "NONE") & (df.strategy == "NONE"))) & (df.filtro == filter_name)].copy()
    return df_m


df_m = get_measure_df(df_baseline, 'kDN', 'PreFilter')
print(df_m[['dataset','model','strategy','measure','corr_th','filtro']])


def boxplot_performance_delta(df, measure, filter_name):
    """
    Boxplot de delta_acc_test y delta_gps_test para cada corr_th
    lado a lado, mostrando encima la delta_n_feat como texto.
    """
    df_m = df[((df['measure'] == measure) | ((df['measure'] == "NONE") & (df['strategy'] == "NONE"))) &
              (df['filtro'] == filter_name)].copy()

    # Filtramos solo los que no son baseline para los boxplots
    df_plot = df_m[df_m['strategy'] != "NONE"].copy()

    order = sorted(df_plot['corr_th'].unique(), key=str)

    # Creamos un DataFrame en formato largo para facilidad de boxplot lado a lado
    df_long = df_plot.melt(id_vars=['corr_th'], value_vars=['delta_gps_test', 'delta_acc_test'],
                           var_name='metric', value_name='delta')

    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df_long, x='corr_th', y='delta', hue='metric', order=order, palette=['skyblue', 'salmon'])
    sns.stripplot(data=df_long, x='corr_th', y='delta', hue='metric', dodge=True, alpha=0.5, size=4, order=order,
                  palette=['blue', 'red'], jitter=True)

    # delta_n_feat como texto horizontal arriba
    feat_means = df_plot.groupby('corr_th')["delta_n_feat"].mean()
    ymax = plt.ylim()[1]
    for i, v in enumerate(order):
        if v in feat_means:
            plt.text(i, ymax + 0.01, f"Δ_feat={feat_means[v]:.1f}", ha='center', va='bottom', fontsize=10, rotation=0)

    plt.title(f'Performance CorrCompl - baseline (Δ) ({measure}, {filter_name})')
    plt.ylabel('Delta GPS / Delta Accuracy')
    plt.xlabel('Correlation threshold')
    plt.legend(title='Metric')
    plt.ylim(top=ymax + 0.05)  # espacio para el texto arriba
    # Línea horizontal en 0
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.show()


def add_delta_complexity(df, measure):
    """
    Añade delta de complejidad: cada fila con un método se compara con el baseline
    correspondiente a su dataset y filter.
    """
    df = df.copy()

    # Filas baseline
    baseline = df[
        (df.strategy == "NONE") & (df.measure == "NONE")
    ][['dataset', 'filtro', f'complexity_all_{measure}_mean']]

    # Renombramos para merge
    baseline = baseline.rename(columns={f'complexity_all_{measure}_mean': f'complexity_all_{measure}_base'})

    # Merge con todas las filas
    df = df.merge(baseline, on=['dataset', 'filtro'], how='left')

    # Calculamos delta
    df[f'delta_complexity_{measure}'] = df[f'complexity_sel_{measure}_mean'] - df[f'complexity_all_{measure}_base']

    return df


def boxplot_complexity_delta(df, measure, filter_name):
    """
    Boxplot de las diferencias en complejidad entre el baseline y cada selección,
    usando la columna delta_complexity_{measure}.
    """
    df_m = df[((df['measure'] == measure) | ((df['measure'] == "NONE") & (df['strategy'] == "NONE"))) &
              (df['filtro'] == filter_name)].copy()

    # Filtramos solo los que no son baseline
    df_plot = df_m[df_m['strategy'] != "NONE"].copy()

    order = sorted(df_plot['corr_th'].unique(), key=str)

    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df_plot, x='corr_th', y=f'delta_complexity_{measure}', order=order, color='lightgreen')
    sns.stripplot(data=df_plot, x='corr_th', y=f'delta_complexity_{measure}', color='green',
                  alpha=0.5, size=4, order=order, jitter=True)


    plt.title(f'Complexity - baseline ({measure}, {filter_name})')
    plt.ylabel('Delta Complexity')
    plt.xlabel('Correlation threshold')

    # Línea horizontal en 0
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.show()



# df_m = get_measure_df(df_baseline, 'Hostility', 'PreFilter')
# df_m = add_delta_complexity(df_m, 'Hostility')
#
#
# boxplot_complexity_delta(df_m, measure, filter_name)
#


measures = ['kDN', 'N1', 'Hostility']
filters = ['NoPreFilter', 'PreFilter']

for measure in measures:
    for f in filters:
        print(f"Generating plots for {measure} - {f}")
        boxplot_performance_delta(df_baseline, measure, f)
        df_m = get_measure_df(df_baseline, measure, f)
        df_m2 = add_delta_complexity(df_m, measure)
        boxplot_complexity_delta(df_m2, measure, f)



# df= df_baseline
