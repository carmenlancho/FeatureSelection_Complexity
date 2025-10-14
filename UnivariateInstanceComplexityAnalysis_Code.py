###################################################################################################################
######                 COMPLEJIDAD INSTANCIAS ARROJADA POR CADA VARIABLE DE FORMA UNIVARIANTE                ######
###################################################################################################################

# En este script vamos a poner el código que nos permite analizar la relación entre las complejidades
# a nivel instancia que arrojan las distintas variables de forma univariante
# La idea es estudiar si, variables que están relacionadas entre sí, otorgan complejidades similares
# de modo que se podría hacer un filtro previo de dichas variables redundantes
# Es un poco seguir la idea de eliminar variables con alta correlación pero en base a las complejidades
# En este script vamos a poner las funciones necesarias para analizar los resultados
# y en el notebook UnivariateInstanceComplexity_Analysis.ipnyb ponemos los resultados ya más visuales



import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
import matplotlib.pyplot as plt
import glob


import os
root_path = os.getcwd()





def analyze_variable_relationships(dataset_files, measures=["Hostility", "N1", "kDN"],show_plots=False):
    """
    Analiza la relación entre variables (complejidad univariante)
    para cada dataset y medida de complejidad (formato largo).

    Reestructura el dataset (pivot) y calcula:
        - Correlación Pearson
        - Correlación Spearman
        - Distancia Euclídea
        - Dendrograma jerárquico (linkage=ward)
    """

    # dataset_files = glob.glob("Results_UnivariateRanking_CM/ArtificialDataset*_featuresComplexityInstances.csv")
    results_summary = []

    for file in dataset_files:
        idx1 = file.find('/')
        idx2 = file.find('_', idx1 + len('/'))
        dataset_name = file[idx1 + len('/'):idx2]
        # print(dataset_name)

        df = pd.read_csv(file)

        for measure in measures:
            # print(measure)

            # ToWide: instancias por filas y features por columnas, para cada measure
            df_wide_m = df.pivot(index="instance_id", columns="feature", values=measure)

            # Correlaciones
            corr_pearson = df_wide_m.corr(method="pearson")
            corr_spearman = df_wide_m.corr(method="spearman")

            # Distancias euclídea
            dist_euclid = pd.DataFrame(squareform(pdist(df_wide_m.T, metric="euclidean")),
                index=df_wide_m.columns,columns=df_wide_m.columns)

            # Quitamos valores de la diagonal
            mask_offdiag = ~np.eye(corr_pearson.shape[0], dtype=bool)

            pearson_vals = corr_pearson.values[mask_offdiag]
            spearman_vals = corr_spearman.values[mask_offdiag]
            euclid_vals = dist_euclid.values[mask_offdiag]

            # Resúmenes
            # --- Calcular estadísticas descriptivas ---
            summary_stats = {
                "dataset": dataset_name,
                "measure": measure,
                "pearson_max": np.max(pearson_vals),
                "pearson_min": np.min(pearson_vals),
                "pearson_mean": np.mean(pearson_vals),
                "pearson_median": np.median(pearson_vals),
                "pearson_std": np.std(pearson_vals),

                "spearman_max": np.max(spearman_vals),
                "spearman_min": np.min(spearman_vals),
                "spearman_mean": np.mean(spearman_vals),
                "spearman_median": np.median(spearman_vals),
                "spearman_std": np.std(spearman_vals),

                "euclid_max": np.max(euclid_vals),
                "euclid_min": np.min(euclid_vals),
                "euclid_mean": np.mean(euclid_vals),
                "euclid_median": np.median(euclid_vals),
                "euclid_std": np.std(euclid_vals)
            }

            results_summary.append(summary_stats)

            if show_plots:
                # máscara triangular inferior dejando diagonal
                mask_triu = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
                # Heatmap y dendrograma
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                sns.heatmap(corr_pearson, ax=axes[0,0], cmap="coolwarm", center=0, vmin=-1, vmax=1,
                                               mask=mask_triu, square=True)
                axes[0,0].set_title("Pearson Corr")

                sns.heatmap(corr_spearman, ax=axes[0,1], cmap="coolwarm", center=0, vmin=-1, vmax=1,
                                                           mask=mask_triu, square=True)
                axes[0,1].set_title("Spearman Corr")

                sns.heatmap(dist_euclid, ax=axes[1,0], cmap="viridis",mask=mask_triu, square=True)
                axes[1,0].set_title("Euclidean Dist")

                # Dendrograma (con dist euclídea)
                Z = linkage(squareform(dist_euclid), method="ward")
                dendrogram(Z, labels=df_wide_m.columns, leaf_rotation=90, ax=axes[1,1])
                axes[1,1].set_title("Hierarchical Clustering (Ward)")

                plt.suptitle(f"{dataset_name} - {measure}", fontsize=20)
                plt.tight_layout()

                # if save_figures:
                #     plt.savefig(path_datasets / f"{dataset_name}_{measure}_relations_dendrogram.png", dpi=200)
                plt.show()
                # plt.close()
    summary_df = pd.DataFrame(results_summary)

    return summary_df

# d1_uni = pd.read_csv("Results_UnivariateRanking_CM/ArtificialDataset1_featuresComplexityInstances.csv", index_col=None)
# d1_uni
#
# dataset_files = glob.glob("Results_UnivariateRanking_CM/ArtificialDataset*_featuresComplexityInstances.csv")
#


# dataset_files = ['Results_UnivariateRanking_CM/ArtificialDataset1_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset2_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset3_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset4_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset5_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset6_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset7_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset8_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset9_featuresComplexityInstances.csv',
#                 'Results_UnivariateRanking_CM/ArtificialDataset10_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset11_featuresComplexityInstances.csv',
#                   'Results_UnivariateRanking_CM/ArtificialDataset12_featuresComplexityInstances.csv',
#                 'Results_UnivariateRanking_CM/ArtificialDataset13_featuresComplexityInstances.csv',
#                 'Results_UnivariateRanking_CM/ArtificialDataset14_featuresComplexityInstances.csv',
#                     'Results_UnivariateRanking_CM/ArtificialDataset15_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset16_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset17_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset18_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset19_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset20_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset21_featuresComplexityInstances.csv',
#                  'Results_UnivariateRanking_CM/ArtificialDataset22_featuresComplexityInstances.csv']
#                  #'Results_UnivariateRanking_CM/ArtificialDataset23_featuresComplexityInstances.csv'] # se queda muy pillado

# summary_df = analyze_variable_relationships(dataset_files,show_plots=False)
# en el d23 se queda pillado



