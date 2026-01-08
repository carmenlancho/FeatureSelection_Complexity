# 08/01/2026 script para comparar los resultados de los métodos del SOTA con los nuestros

import pandas as pd
import matplotlib.pyplot as plt
import os


# Tabla comparativa mostrando media y std

def build_comparison_table(df_compl, df_sota):

    df = df_compl[["model","measure","acc_test_mean","acc_test_std","gps_test_mean","gps_test_std"]].copy()
    df = df.rename(columns={"measure": "method"})

    df_comp = pd.concat([df, df_sota],ignore_index=True)

    return df_comp


# df_compl = pd.read_csv("Results_FS_Distributed_CV/spambase_DistributedCVRandom_OutHigh_SummaryResults.csv")
# df_sota = pd.read_csv("Results_FS_SOTA_CV/spambase_SOTA_CV_SummaryResults.csv")
#
# df_comparison = build_comparison_table(df_compl, df_sota)


#######################################################################
#####                         RANKINGS                            #####
#######################################################################

# # Comparamos rankings
# df_compl_fi = pd.read_csv("Results_FS_Distributed_CV/wdbc_DistributedCVRandom_OutHigh_FeatureImportance_Folds.csv")
# df_sota_fi = pd.read_csv("Results_FS_SOTA_CV/wdbc_SOTA_CV_FeatureImportance_Folds.csv")

def ranks_comparison(df_sota_fi,df_compl_fi):

    # Calculamos los rankings
    df = df_sota_fi.copy()
    df["rank"] = df.groupby(["fold", "model", "method"])["score"].rank(ascending=False, method="average")
    df_compl_fi["rank"] = df_compl_fi.groupby(["fold", "model"])["kDN_importances_norm"].rank(ascending=False,
                                                                                              method="average")
    df_compl_fi.dropna(subset=['rank'], inplace=True)  # quitamos los NAs del ranking
    df_compl_fi['method'] = "kDN_importances"
    # Juntamos los rankings
    df_compl_fi_ranked = df_compl_fi[["fold", "model", "method", "feature", "rank"]].copy()
    df_rank = pd.concat([df, df_compl_fi_ranked], ignore_index=True)

    # Hacemos la media de los rankings por fold
    df_avg = df.groupby(["model", "method", "feature"]).agg(
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        mean_score=("score", "mean")).reset_index().sort_values(["model", "method", "mean_rank"])
    df_compl_fi_avg = df_compl_fi.groupby(["model", "method", "feature"]).agg(
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        mean_score=("kDN_importances_norm", "mean")).reset_index().sort_values(["model", "method", "mean_rank"])
    # Dataset conjunto de rankings
    df_all = pd.concat([df_avg, df_compl_fi_avg], ignore_index=True)

    return df_rank, df_all




# df_rank, df_all = ranks_comparison(df_sota_fi,df_compl_fi)




def build_ranking_table(df_all, model):
    # Ordenamos las variables en función de nuestro ranking
    feature_order = df_all[(df_all["model"] == model) & (df_all["method"] == 'kDN_importances')].sort_values("mean_rank")[
        "feature"].tolist()
    df_model = df_all[df_all["model"] == model]

    table = (df_model.pivot(
            index="feature",
            columns="method",
            values=["mean_rank", "std_rank"]).loc[feature_order])
    return table


# ranking_table = build_ranking_table(df_all,model="SVM-rbf")



# ¿Hasta qué punto los métodos inducen rankings similares, de media, a través de los folds?


def spearman_per_fold(df, model):
    """
    Compute Spearman correlation between methods for each fold,
    then average across folds.
    """
    df_model = df[df["model"] == model]

    # methods = df_model["method"].unique()
    folds = df_model["fold"].unique()

    corr_matrices = []

    for fold in folds:
        df_fold = df_model[df_model["fold"] == fold]

        # feature x method
        mat = df_fold.pivot(index="feature", columns="method", values="rank")

        corr = mat.corr(method="spearman")
        corr_matrices.append(corr)

    mean_corr = sum(corr_matrices) / len(corr_matrices)
    return mean_corr


def plot_corrplot(corr_df, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_df, interpolation="nearest")
    plt.colorbar(label="Spearman ρ")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title(title)

    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            plt.text(
                j,
                i,
                f"{corr_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


# model_name = "SVM-rbf"
# corr_mean = spearman_per_fold(df_rank, model_name)
#
# plot_corrplot(corr_mean,title=f"Mean Spearman correlation across folds ({model_name})")


