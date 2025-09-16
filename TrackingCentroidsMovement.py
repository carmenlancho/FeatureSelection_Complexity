####################################################################################################
#########                          TRACKING CENTROIDS MOVEMENT                             #########
####################################################################################################


# root_path = '/home/carmen/PycharmProjects/Doctorado'




import copy
import numpy as np
import pandas as pd
# from pandas.conftest import all_reductions
from sklearn.cluster import KMeans
# from scipy.stats import multivariate_normal
from Normal_dataset_generator import *
import matplotlib.pyplot as plt
import matplotlib as mpl



# Hostility measure algorithm for multiclass classification problems
def hostility_measure_multiclass_Centroids(sigma, X, y, k_min, seed=0):
    """
    :param sigma: proportion of grouped points per cluster. This parameter automatically determines the number of clusters k in every layer.
    :param X: instances
    :param y: labels
    :param k_min: the minimum number of clusters allowed (stopping condition)
    :param seed: for the k-means algorithm
    :return: host_instance_by_layer - df with hostility instance values per layer (cols are number of clusters per layer, rows are points)
             data_clusters - original data and the cluster to which every original point belongs to at any layer
             results - dataframe (rows are number of clusters per layer) with hostility per class, per dataset and overlap per class
             results_per_class - Pairwise hostility per classes. Rows: who is receiving hostility, columns: who is causing the hostility (proportion of points (row) receiving hostility from the class in the column)
             probs_per_layer - dominance probability of each class in the neighborhood of each point for each layer
             k_auto - automatic recommended value of clusters k for selecting the best layer to stop
    """
    # host_instance_by_layer_df: hostility of each instance in each layer
    # data_clusters: original points and the cluster where they belong to at each layer
    # results:
    # results_per_class:
    # probs_per_layer:
    # k_auto:

    np.random.seed(seed)

    n = len(X)
    n_classes = len(np.unique(y))
    X_aux = copy.deepcopy(X)

    host_instance_by_layer = []
    centroids_dict = {} # to save movement of centroids

    # first k:
    k = int(n / sigma)
    # The minimum k is the number of classes
    minimo_k = max(n_classes, k_min)
    if k < minimo_k:
        raise ValueError("sigma too low, choose a higher value")
    else:  # total list of k values
        k_list = [k]
        while (int(k / sigma) > minimo_k):
            k = int(k / sigma)
            k_list.append(k)

        # list of classes
        list_classes = list(np.unique(y))  # to save results with the name of the class
        list_classes_total = list(np.unique(y))  # for later saving results
        list_classes_total.append('Total') # for later saving results
        name3 = 'Host_'
        col3 = []
        for t in range(n_classes):
            col3.append(name3 + str(list_classes[t]))

        columns_v =list(col3) + list(['Dataset_Host'])

        # Results is a dataset to save hostility per class, hostility of the dataset and overlap per class in every layer
        index = k_list
        results = pd.DataFrame(0, columns=columns_v, index=index)
        results_per_class = {}
        probs_per_layer = {}

        data_clusters = pd.DataFrame(X)  # to save to which cluster every original point belongs to at any layer
        # prob_bomb = np.zeros(len(X))  # to save the probability, for every original point, of its class in its cluster
        df_bomb = pd.DataFrame(0,columns=list_classes, index=data_clusters.index)

        h = 1  # to identify the layer
        # k = 1200
        for k in k_list:

            kmeds = KMeans(n_clusters=k, n_init=15, random_state=seed).fit(X_aux)
            labels_bomb1 = kmeds.labels_
            centroids_bomb = kmeds.cluster_centers_
            # We save centroids
            centroids_dict[h] = centroids_bomb

            col_now = 'cluster_' + str(h) # for the data_clusters dataframe

            if len(y) == len(labels_bomb1):  # only first k-means
                data_clusters[col_now] = labels_bomb1
                # Probability of being correctly identified derived from first k-means
                table_percen = pd.crosstab(y, labels_bomb1, normalize='columns')
                table_percen_df = pd.DataFrame(table_percen) # ESTO ES LO QUE QUIERO, TENGO QUE ENLAZARLO

                prob_bomb1 = np.zeros(len(X))
                df_bomb1 = pd.DataFrame(columns = list_classes, index = data_clusters.index)
                for i in np.unique(labels_bomb1):
                    for t in list_classes:
                        prob_bomb1[((y == t) & (labels_bomb1 == i))] = table_percen_df.loc[t, i]
                        df_bomb1[(labels_bomb1 == i)] = table_percen_df.loc[:, i]

            else:  # all except first k-means (which points are in new clusters)
                data2 = pd.DataFrame(X_aux)
                data2[col_now] = labels_bomb1
                data_clusters[col_now] = np.zeros(n)

                for j in range(k):
                    values_together = data2.index[data2[col_now] == j].tolist()
                    data_clusters.loc[data_clusters[col_old].isin(values_together), col_now] = j

                # Proportion of each class in each cluster of the current partition
                table_percen = pd.crosstab(y, data_clusters[col_now], normalize='columns')
                table_percen_df = pd.DataFrame(table_percen)
                prob_bomb1 = np.zeros(len(X))
                df_bomb1 = pd.DataFrame(columns=list_classes, index=data_clusters.index)
                for i in np.unique(labels_bomb1):
                    for t in list_classes:
                        prob_bomb1[((y == t) & (data_clusters[col_now] == i))] = table_percen_df.loc[t, i]
                        df_bomb1[(data_clusters[col_now] == i)] = table_percen_df.loc[:, i]

            # For all cases
            df_bomb += df_bomb1
            # Mean of the probabilities
            df_prob_bomb_mean = df_bomb / h
            prob_self_perspective = np.zeros(len(X))
            for t in list_classes:
                prob_self_perspective[y == t] = df_prob_bomb_mean.loc[y == t, t]

            # We save the dominance probability of every point in every layer
            probs_per_layer[k] = df_prob_bomb_mean


            h += 1  # to count current layer
            col_old = col_now

            #### Data preparation for next iterations
            # New points: medoids of previous partition
            X_aux = kmeds.cluster_centers_

            ## Hostility instance values in current layer
            host_instance = 1 - prob_self_perspective


            # We binarize putting 1 to the maximum value
            df_binary = pd.DataFrame((df_prob_bomb_mean.T.values == np.amax(df_prob_bomb_mean.values, 1)).T * 1,
                         columns=df_prob_bomb_mean.columns)
            # If there is a tie: no one wins
            df_binary.loc[df_binary.sum(axis=1)>1] = 0
            df_hostility = pd.DataFrame(-1,columns=list_classes, index=data_clusters.index)

            df_classes = pd.DataFrame(columns=list_classes_total, index=list_classes)
            host_vector_binary = np.zeros(n)
            for t in list_classes:
                # If you are the dominant class in your environment
                dominant_condition = (df_binary.loc[:, t] == 1)
                df_hostility.loc[(y==t) & (dominant_condition), t] = 0 # you do not receive hostility from your neighborhood
                # else, you receive hostility
                df_hostility.loc[(y==t) & (~dominant_condition), t] = 1
                host_vector_binary[(y==t)] = df_hostility.loc[(y==t), t]
                # Who is giving hostility? Those classes with more (or equal) presence than you in your environment
                comparison_higher_presence = (df_prob_bomb_mean.loc[(y == t), (df_prob_bomb_mean.columns != t)].values >= df_prob_bomb_mean.loc[
                    (y == t), (df_prob_bomb_mean.columns == t)].values) * 1
                df_hostility.loc[(y==t), (df_hostility.columns != t)] = comparison_higher_presence

                total_hostility_class_t = df_hostility.loc[y == t, t].mean(axis=0)
                hostility_received_per_class = np.array(df_hostility.loc[y == t, (df_hostility.columns != t)].mean(axis=0))

                df_classes.loc[df_classes.index == t,df_classes.columns == t] = 0
                df_classes.loc[df_classes.index == t, df_classes.columns == 'Total'] = total_hostility_class_t
                df_classes.loc[df_classes.index == t,
                (df_classes.columns != t) & (df_classes.columns != 'Total')] = hostility_received_per_class
            # We save detail of pairwise hostility relation in each layer
            results_per_class[k] = df_classes
            host_dataset = np.mean(host_vector_binary) # hostility of the dataset


            # Save results from all layers
            host_instance_by_layer.append(host_instance)
            results.loc[k] = df_classes['Total'].tolist() + [host_dataset]


        ## Automatic selection of layer
        results_aux = results.loc[:, results.columns.str.startswith('Host')]
        change_max = results_aux.iloc[0, :] * 1.25
        change_min = results_aux.iloc[0, :] * 0.75
        matching = results_aux[(results_aux <= change_max) & (results_aux >= change_min)]
        matching.dropna(inplace=True)  # values not matching appear with NaN, they are eliminated
        k_auto = matching.index[-1] # k value from last layer matching the condition of variability

    host_instance_by_layer = np.vstack(host_instance_by_layer)
    host_instance_by_layer_df = pd.DataFrame(host_instance_by_layer.T, columns=results.index)

    return host_instance_by_layer_df, data_clusters, centroids_dict, results, results_per_class, probs_per_layer, k_auto

################################################################################################
###############                        EJEMPLO MULTICLASE                        ###############
################################################################################################
# 05/09/2025 Comenzamos estudiando el caso binario, de modo que este ejemplo simplemente
# saca los centroides en cada capa

# seed0 = 1
# seed1 = 2
# seed2 = 3
# n0 = 1000
# n1 = 1000
# n2 = 1000
#
#
# ## Dataset multiclass 1
# mu0 = [0, 0]
# sigma0 = [[1, 0], [0, 1]]
# mu1 = [3, 3]
# sigma1 = [[1, 0], [0, 1]]
# mu2 = [2, -1]
# sigma2 = [[3, 1], [1, 1]]
#
# X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)
#
#
# sigma = 5
# delta = 0.5
# seed = 0
# k_min = 0
# (host_instance_by_layer_df, data_clusters, centroids_dict, results, results_per_class,
#     probs_per_layer, k_auto) = hostility_measure_multiclass_Centroids(sigma, X, y, k_min, seed=0)




# Hostility measure algorithm
def hostility_measure_Centroids(sigma, X, y, delta, k_min, seed=0):
    """
    :param sigma: proportion of grouped points per cluster. This parameter automatically determines the number of clusters k in every layer.
    :param X: instances
    :param y: labels
    :param delta: the probability threshold to obtain hostility per class and for the dataset
    :param k_min: the minimum number of clusters allowed (stopping condition)
    :param seed: for the k-means algorithm
    :return: host_instance_by_layer - df with hostility instance values per layer (cols are number of clusters per layer, rows are points)
             data_clusters - original data and the cluster to which every original point belongs to at any layer
             results - dataframe (rows are number of clusters per layer) with hostility per class, per dataset and overlap per class
             k_auto - automatic recommended value of clusters k for selecting the best layer to stop
    """

    np.random.seed(seed)

    n = len(X)
    n_classes = len(np.unique(y))
    X_aux = copy.deepcopy(X)

    host_instance_by_layer = []
    centroids_dict = {}  # to save movement of centroids

    # first k:
    k = int(n / sigma)
    # The minimum k is the number of classes
    minimo_k = max(n_classes, k_min)
    if k < minimo_k:
        raise ValueError("sigma too low, choose a higher value")
    else:  # total list of k values
        k_list = [k]
        while (int(k / sigma) > minimo_k):
            k = int(k / sigma)
            k_list.append(k)

        # list of classes
        list_classes = list(np.unique(y))  # to save results with the name of the class
        name2 = 'Overlap_'
        name3 = 'Host_'
        col2 = []
        col3 = []
        for t in range(n_classes):
            col2.append(name2 + str(list_classes[t]))
            col3.append(name3 + str(list_classes[t]))

        columns_v = list(col2) + list(col3) + list(['Dataset_Host'])

        # Results is a dataset to save hostility per class, hostility of the dataset and overlap per class in every layer
        index = k_list
        results = pd.DataFrame(0.0, columns=columns_v, index=index)
        # ponemos 0.0 para que se inicialice como float y no como int

        data_clusters = pd.DataFrame(X)  # to save to which cluster every original point belongs to at any layer
        prob_bomb = np.zeros(len(X))  # to save the probability, for every original point, of its class in its cluster

        h = 1  # to identify the layer
        for k in k_list:

            kmeds = KMeans(n_clusters=k, n_init=15, random_state=seed).fit(X_aux)
            labels_bomb1 = kmeds.labels_
            # num_clusters_new = len(np.unique(labels_bomb1))
            centroids_bomb = kmeds.cluster_centers_
            # We save centroids
            centroids_dict[h] = centroids_bomb

            col_now = 'cluster_' + str(h) # for the data_clusters dataframe

            if len(y) == len(labels_bomb1):  # only first k-means
                data_clusters[col_now] = labels_bomb1
                # Probability of being correctly identified derived from first k-means
                table_percen = pd.crosstab(y, labels_bomb1, normalize='columns')
                table_percen_df = pd.DataFrame(table_percen)
                prob_bomb1 = np.zeros(len(X))
                for i in np.unique(labels_bomb1):
                    for t in list_classes:
                        prob_bomb1[((y == t) & (labels_bomb1 == i))] = table_percen_df.loc[t, i]

            else:  # all except first k-means (which points are in new clusters)
                data2 = pd.DataFrame(X_aux)
                data2[col_now] = labels_bomb1
                data_clusters[col_now] = np.zeros(n)

                for j in range(k):
                    values_together = data2.index[data2[col_now] == j].tolist()
                    data_clusters.loc[data_clusters[col_old].isin(values_together), col_now] = j

                # Proportion of each class in each cluster of the current partition
                table_percen = pd.crosstab(y, data_clusters[col_now], normalize='columns')
                table_percen_df = pd.DataFrame(table_percen)
                prob_bomb1 = np.zeros(len(X))
                for i in np.unique(labels_bomb1):
                    for t in list_classes:
                        prob_bomb1[((y == t) & (data_clusters[col_now] == i))] = table_percen_df.loc[t, i]

            # For all cases
            prob_bomb += prob_bomb1
            # Mean of the probabilities
            prob_bomb_mean = prob_bomb / h
            h += 1  # to count current layer
            col_old = col_now

            #### Data preparation for next iterations
            # New points: medoids of previous partition
            X_aux = kmeds.cluster_centers_

            ## Hostility instance values in current layer
            host_instance = 1 - prob_bomb_mean

            bin_host = np.where(host_instance > 0, 1, 0)  # it refers to overlap
            bin_hi_classes = np.zeros(n_classes)
            # lost points
            host_vector_delta = np.where(host_instance >= delta, 1, 0) # hostility instance values binarized with delta
            host_dataset = np.mean(host_vector_delta) # hostility of the dataset
            host_classes = np.zeros(n_classes)
            # hostility and overlap of the classes
            for l in range(n_classes):
                ly = list_classes[l]
                bin_hi_classes[l] = np.mean(bin_host[y == ly])
                host_classes[l] = np.mean(host_vector_delta[y == ly])

            # Save results from all layers
            host_instance_by_layer.append(host_instance)
            results.loc[k] = bin_hi_classes.tolist() + host_classes.tolist() + [host_dataset]

        ## Automatic selection of layer
        results_aux = results.loc[:, results.columns.str.startswith('Host')]
        change_max = results_aux.iloc[0, :] * 1.25
        change_min = results_aux.iloc[0, :] * 0.75
        matching = results_aux[(results_aux <= change_max) & (results_aux >= change_min)]
        matching.dropna(inplace=True)  # values not matching appear with NaN, they are eliminated
        k_auto = matching.index[-1] # k value from last layer matching the condition of variability

    host_instance_by_layer = np.vstack(host_instance_by_layer)
    host_instance_by_layer_df = pd.DataFrame(host_instance_by_layer.T, columns=results.index)
    return host_instance_by_layer_df, data_clusters, centroids_dict, results, k_auto




#Example

#
# # Parameters
# seed1 = 1
# seed2 = 2
# n0 = 3000
# n1 = 3000
#
# # Dataset 1
# mu0 = [0, 0]
# sigma0 = [[1, 0], [0, 1]]
# mu1 = [3, 3]
# sigma1 = [[1, 0], [0, 1]]
#
# X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)
#
# sigma = 5
# delta = 0.5
# seed = 0
# k_min = 0
# host_instance_by_layer_df, data_clusters, centroids_dict, results, k_auto = hostility_measure_Centroids(sigma, X, y, delta, k_min, seed=0)
# # host_instance_by_layer_df me devuelve la hostilidad de cada punto en cada capa
# # con esto puedo sacar (mediante binarización) la hostilidad de cada clase en cada mini cluster
# # para ir trackeando la evolución
# # con eso, debo estudiar cómo cambia la complejidad al moverse cada cluster


###############################################################################################
###########                    MAPA VECTORIAL MOVIMIENTO CENTROIDES                 ###########
###############################################################################################



def plot_hierarchical_quivers_with_complexity(df,
                                              centroids_dict,
                                              host_instance_by_layer_df,
                                              delta=0.5,
                                              cluster_prefix='cluster_',
                                              k_auto=None,
                                              annotate=False,
                                              figsize=(18,5),
                                              arrow_scale=1,
                                              arrow_width=0.005,
                                              cmap_name="coolwarm"):

    """
    Dibuja un quiver plot por cada par de capas consecutivas en df.

    Parámetros
    ----------
    df : data_clusters`
    centroids_dict : Diccionario con centroides
    cluster_prefix : Prefijo de columnas de cluster ('cluster_').
    annotate : bool
        Si True, escribe los ids de cluster.
    figsize, arrow_scale, arrow_width, cmap_name : parámetros de plotting.
    """

    cluster_cols = [c for c in df.columns if str(c).startswith(cluster_prefix)]
    # si hay k_auto, cortar en esa capa
    if k_auto is not None:
        n_clusters_per_layer = [df[c].nunique() for c in cluster_cols]
        arr = np.array(n_clusters_per_layer)
        if (arr == k_auto).any():
            layer_limit = np.where(arr == k_auto)[0][0] + 1
            cluster_cols = cluster_cols[:layer_limit]

    n_layers = len(cluster_cols)
    if n_layers < 2:
        raise ValueError("The minimum number of layers to generate a quiver plot is 2.")


    # Binarizar host_instance para el posterior cálculo de complejidad por clase
    host_bin = (host_instance_by_layer_df >= delta).astype(int)

    fig, axes = plt.subplots(1, n_layers-1, figsize=figsize, squeeze=False)
    axes = axes[0]

    for i, ax in enumerate(axes):
        l = i + 1
        col_l = f"{cluster_prefix}{l}"
        col_next = f"{cluster_prefix}{l+1}"

        # complejidad media por cluster en capa l y l+1
        comp_old = (host_bin.iloc[:, l - 1].groupby(df[col_l]).mean().to_dict())
        comp_new = (host_bin.iloc[:, l].groupby(df[col_next]).mean().to_dict())

        # vectores
        X, Y, U, V, colors = [], [], [], [], []
        labels_old, labels_new = [], []

        unique_old = np.unique(df[col_l].dropna().values)

        for old_label in unique_old: # recorremos todos los clusters
            mask_old = df[col_l] == old_label # nos quedamos con la posición de dichas observaciones

            # nuevo label: buscamos su nueva etiqueta en el cluster siguiente
            # elegimos la primera observación (por ejemlo), son todas iguales
            new_label = df.loc[mask_old, col_next].iloc[0]

            cent_arr_old = centroids_dict.get(l) # centroides del cluster previo
            old_cent = cent_arr_old[int(old_label)] # centroide

            # obtener centroid nuevo
            cent_arr_new = centroids_dict.get(l+1)
            new_cent = cent_arr_new[int(new_label)]

            X.append(float(old_cent[0])); Y.append(float(old_cent[1]))
            U.append(float(new_cent[0] - old_cent[0])); V.append(float(new_cent[1] - old_cent[1]))
            labels_old.append(old_label); labels_new.append(new_label)

            # cambio de complejidad
            c_old = comp_old.get(old_label, 0)
            c_new = comp_new.get(new_label, 0)
            colors.append(c_new - c_old) # degradado con cmap
            # color binario
            # if c_new > c_old:
            #     colors.append("darkred")
            # else :
            #     colors.append("green")

        X, Y, U, V, colors = map(np.array, [X, Y, U, V, colors])
        # mapa de colores
        norm = mpl.colors.TwoSlopeNorm(vmin=colors.min(), vcenter=0, vmax=colors.max())
        cmap = plt.get_cmap(cmap_name)

        # Dibujar
        ax.quiver(X, Y, U, V,colors, angles='xy', scale_units='xy', scale=arrow_scale,
                  width=arrow_width, cmap=cmap, norm=norm, alpha=0.9)

        # Por si se quieren pintar los centroides
        # ax.scatter(X, Y, s=10, alpha=0.6, label='old centroids')
        # ax.scatter(np.array(X)+np.array(U), np.array(Y)+np.array(V),
        #            s=10, marker='x', alpha=0.6, label='new centroids')

        ax.set_title(f"Layer {l} → {l+1}")
        ax.axis('equal')
        # ax.legend(fontsize='small')

        if annotate:
            for xo, yo, dx, dy, oldl, newl in zip(X, Y, U, V, labels_old, labels_new):
                ax.text(xo, yo, str(oldl), fontsize=6, va='bottom', ha='right')
                ax.text(xo+dx, yo+dy, str(newl), fontsize=6, va='bottom', ha='left')

        # añadir barra de color
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="∆ complejidad")

    plt.tight_layout()
    plt.show()


# plot_hierarchical_quivers_with_complexity(
#     data_clusters,
#     centroids_dict,
#     host_instance_by_layer_df,
#     delta=0.5,
#     k_auto=k_auto)

###############################################################################################
###########                  RANKING EN FUNCIÓN DEL MOV. CENTROIDES                 ###########
###############################################################################################


def normalize(vec):
    denom = np.abs(vec).sum()
    norm_vec = vec / denom if denom > 0 else vec
    return norm_vec

def normalize_by_movement(vec, mov):
    norm_mov_vec = vec / mov if np.any(mov > 0) else vec
    return norm_mov_vec

# Salida con formato adecuado
def to_dict(vec,feature_cols):
    dict_format = {f: float(v) for f, v in zip(feature_cols, vec)}
    return dict_format



def rank_features_by_centroid_complexity(
    df,
    centroids_dict,
    host_instance_by_layer_df,
    delta=0.5,
    cluster_prefix='cluster_',
    k_auto=None
):
    """
    Calcula un score por feature basado en el movimiento de centroides y cambio
    de complejidad de clusters.

    Devuelve:
        results: dict con:
            - 'scores_raw': {feature: valor}
            - 'scores_normalized': {feature: valor}
            - 'scores_robust': {feature: valor}
            - 'ranking_raw': [features ordenadas mejor->peor]
            - 'ranking_normalized': idem
            - 'ranking_robust': idem
            - 'details': DataFrame con evolución paso a paso
    """

    # Columnas de clusters y de features
    cluster_cols = [c for c in df.columns if str(c).startswith(cluster_prefix)]
    feature_cols = [c for c in df.columns if not str(c).startswith(cluster_prefix)]
    n_features_df = len(feature_cols)

    if k_auto is not None:
        n_clusters_per_layer = [df[c].nunique() for c in cluster_cols]
        arr = np.array(n_clusters_per_layer)
        layer_limit = np.where(arr == k_auto)[0][0] + 1
        cluster_cols = cluster_cols[:layer_limit] # columnas de capas válidas (determinadas por k_auto)
    n_layers = len(cluster_cols)

    if n_layers < 2:
        raise ValueError("At least 2 layers are required for tracking computation.")

    # Binarizamos hostilidad
    host_bin = (host_instance_by_layer_df >= delta).astype(int)

    # Complejidad media por cluster
    comp_by_cluster = {}
    for l_idx, col in enumerate(cluster_cols, start=1):
        comp_by_cluster[l_idx] = host_bin.iloc[:, l_idx - 1].groupby(df[col]).mean().to_dict()

    # Scores para guardar resultados
    scores_raw = np.zeros(n_features_df)
    scores_robust = np.zeros(n_features_df)
    movement_raw = np.zeros(n_features_df)
    movement_robust = np.zeros(n_features_df)

    # Guardar detalle paso a paso
    records = []
    list_scores_raw = []
    list_scores_robust = []

    # Recorremos transiciones entre capas
    for l in range(1, n_layers):
        col_l = f"{cluster_prefix}{l}"
        col_next = f"{cluster_prefix}{l + 1}"

        unique_old = np.unique(df[col_l].dropna())
        for old_label in unique_old:
            mask_old = df[col_l] == old_label  # los viejos de ese cluster
            n_points = mask_old.sum() # para luego normalizar
            new_label = df.loc[mask_old, col_next].iloc[0]  # en el nuevo cluster

            # centroides
            old_cent = centroids_dict[l][int(old_label)]
            new_cent = centroids_dict[l + 1][int(new_label)]
            # old_cent = np.asarray(centroids_dict[l][int(old_label)])
            # new_cent = np.asarray(centroids_dict[l + 1][int(new_label)])

            # Cambio en los centroides (enfocado en features)
            change_cent = np.abs(new_cent - old_cent)

            # cambio de complejidad
            c_old = comp_by_cluster[l].get(old_label, 0.0)
            c_new = comp_by_cluster[l + 1].get(new_label, 0.0)
            change_comp = float(c_new - c_old)  # >0 --> aumenta complejidad en la nueva capa

            # Acumulamos score por feature
            scores_raw += change_cent * change_comp
            scores_robust += change_cent * change_comp * n_points
            list_scores_raw.append(change_cent * change_comp)
            list_scores_robust.append(change_cent * change_comp * n_points)
            # Acumulamos magnitudes de movimiento
            movement_raw += np.abs(change_cent)
            movement_robust += np.abs(change_cent) * n_points

            # guardar detalle
            rec = {"layer_from": l,
                "cluster_from": old_label,
                "cluster_to": int(new_label),
                "n_points": int(n_points),
                "comp_old": c_old,
                "comp_new": c_new,
                "change_comp": change_comp}
            for f, dval in zip(feature_cols, change_cent):
                rec[f"change_{f}"] = dval
            records.append(rec)

    # Normalización
    scores_normalized = normalize(scores_raw)
    scores_norm_by_movement = normalize_by_movement(scores_raw, movement_raw)
    scores_robust_norm_by_movement = normalize_by_movement(scores_robust, movement_robust)

    # Outputs legibles
    scores_dicts = {
        "raw": to_dict(scores_raw,feature_cols),
        "normalized_simple": to_dict(scores_normalized,feature_cols),
        "robust": to_dict(scores_robust,feature_cols),
        "norm_by_movement": to_dict(scores_norm_by_movement,feature_cols),
        "robust_norm_by_movement": to_dict(scores_robust_norm_by_movement,feature_cols),
    }

    rankings = {
        name: sorted(feature_cols, key=lambda f: scores_dicts[name][f])
        for name in scores_dicts}

    details_df = pd.DataFrame(records)

    all_results = {
        "scores": scores_dicts,
        "rankings": rankings,
        'list_scores_raw': list_scores_raw,
        'list_scores_robust': list_scores_robust,
        "details": details_df,
    }

    return all_results

# resultados = rank_features_by_centroid_complexity(
#     data_clusters,
#     centroids_dict,
#     host_instance_by_layer_df,
#     delta=0.5,
#     cluster_prefix='cluster_',
#     k_auto=k_auto)
# aa = resultados['details']
#



# ## Realmente este gráfico no aporta nada porque lo que nos interesaría
# ## es cambio en complejidad por variable, pero eso no lo podemos lograr
# ## El ranking que hacemos es lo más parecido
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def plot_boxplots(details_df, feature_cols):
#     """
#     Muestra:
#       1) Boxplot de los cambios de complejidad (delta_comp).
#       2) Boxplot de la magnitud de movimiento por feature.
#     """
#
#     # --- Boxplot 1: cambios de complejidad
#     plt.figure(figsize=(6,4))
#     sns.boxplot(y=details_df["change_comp"])
#     plt.axhline(0, color="red", linestyle="--", alpha=0.7)
#     plt.title("Distribución de cambios de complejidad (Δcomp)")
#     plt.ylabel("Δcomp")
#     plt.show()
#
#     # --- Boxplot 2: magnitudes por feature
#     df_magnitudes = details_df.copy()
#     for f in feature_cols:
#         df_magnitudes[f"abs_{f}"] = df_magnitudes[f"change_{f}"].abs()
#
#     melted = df_magnitudes.melt(
#         value_vars=[f"abs_{f}" for f in feature_cols],
#         var_name="feature",
#         value_name="|Δcent|"
#     )
#     melted["feature"] = melted["feature"].str.replace("abs_", "")
#
#     plt.figure(figsize=(8,5))
#     sns.boxplot(x="feature", y="|Δcent|", data=melted)
#     plt.title("Distribución de magnitudes de movimiento por feature")
#     plt.ylabel("|Δcent|")
#     plt.xlabel("Feature")
#     plt.show()
