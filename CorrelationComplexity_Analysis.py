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


