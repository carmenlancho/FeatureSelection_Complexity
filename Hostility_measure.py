import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Hostility measure algorithm
def hostility_measure(sigma, X, y, delta, k_min, seed=0):
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

        data_clusters = pd.DataFrame(X)  # to save to which cluster every original point belongs to at any layer
        prob_bomb = np.zeros(len(X))  # to save the probability, for every original point, of its class in its cluster

        h = 1  # to identify the layer
        for k in k_list:

            kmeds = KMeans(n_clusters=k, n_init=15, random_state=seed).fit(X_aux)
            labels_bomb1 = kmeds.labels_
            # num_clusters_new = len(np.unique(labels_bomb1))

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
    return host_instance_by_layer_df, data_clusters, results, k_auto




# #Example
from Normal_dataset_generator import normal_generator2
#
# Parameters
seed1 = 1
seed2 = 3
n0 = 1000
n1 = 1000
#
# # Dataset 1
# mu0 = [0, 0]
# sigma0 = [[1, 0], [0, 1]]
# mu1 = [3, 3]
# sigma1 = [[1, 0], [0, 1]]

## Dataset multiclass 3 (only class 0 and 2)
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [1, -3]
sigma1 = [[0.3, 0], [0, 5]]


#
X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)

sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance, data_clusters, results, k_auto = hostility_measure(sigma, X, y, delta, k_min, seed=0)

