######## Script to obtain all measures and return a csv with all of them per dataset


import os
import pandas as pd
import numpy as np
# from Hostility_measure_algorithm import hostility_measure
from Hostility_multiclass_algorithm import hostility_measure_multiclass, format_labels
from L1 import L1_HD
from measures import ClassificationMeasures
from sklearn import preprocessing

# from old.ComplexityAnalysis import cols_name


# root_path = os.getcwd()


def all_measures(data,save_csv,path_to_save, name_data):

    # Hostility measure
    y = data['y'].to_numpy()
    X = data.iloc[:, 0:-1].to_numpy()
    sigma = 5
    # delta = 0.5
    # seed = 0
    k_min = 0
    host_instance, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(
        sigma, X, y, k_min, seed=0)
    host_instances = np.array(host_instance[k_auto])
    class_data_host = results.loc[k_auto]['Host_0':'Dataset_Host']
    df_class_data_host = pd.DataFrame(class_data_host)
    df_class_data_host.columns = [name_data]
    extra_results_host = {'results_per_class':results_per_class, 'probs_per_layer':probs_per_layer, 'k_auto':k_auto.item()}

    L1 = L1_HD(X, y)

    y_format = format_labels(data['y'])
    data['y'] = y_format
    p = ClassificationMeasures(data)
    kdn = p.k_disagreeing_neighbors()

    DS = p.disjunct_size()
    DCP = p.disjunct_class_percentage()
    TD_U = p.tree_depth_unpruned()
    TD_P = p.tree_depth_pruned()
    MV = p.minority_value()
    CB = p.class_balance()
    CLD = p.class_likeliood_diff()
    N1 = p.borderline_points()  # N1
    N2 = p.intra_extra_ratio()  # N2
    LSC = p.local_set_cardinality()
    LSradius = p.ls_radius()
    H = p.harmfulness()
    U = p.usefulness()
    F1 = p.f1()
    F2 = p.f2()
    F3 = p.f3()
    F4 = p.f4()


    dict_measures = {'Hostility': host_instances, 'kDN': kdn, 'DS': DS, 'DCP': DCP,
                     'TD_U': TD_U, 'TD_P': TD_P, 'MV': MV, 'CB': CB, 'CLD': CLD, 'N1': N1, 'N2': N2,
                     'LSC': LSC, 'LSradius': LSradius, 'H': H, 'U': U, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4,
                     'L1':L1, 'y':y}

    df_measures = pd.DataFrame(dict_measures)

    # Values per class and dataset
    df_classes_dataset = pd.DataFrame(df_measures.groupby('y').mean())
    df_classes_dataset.loc["dataset"] = df_measures.mean()[:-1]
    df_classes_dataset['Hostility'] = np.array(class_data_host)

    if (save_csv == True):
        # To save the results
        os.chdir(path_to_save)
        nombre_csv = 'ComplexityMeasures_InstanceLevel_' + name_data + '.csv'
        df_measures.to_csv(nombre_csv, encoding='utf_8_sig')

        nombre_csv2 = 'ComplexityMeasures_ClassDatasetLevel_' + name_data + '.csv'
        df_class_data_host.to_csv(nombre_csv2, encoding='utf_8_sig')

    return df_measures, df_classes_dataset, extra_results_host



# # ## Ejemplo individual
# root_path = os.getcwd()
# path_csv = os.chdir(root_path+'/datasets')
#
# name_data = 'bodyfat'
# data_aux = pd.read_csv('bodyfat.csv')
# X = data_aux.iloc[:,:-1].to_numpy() # all variables except y
# X = preprocessing.scale(X)
# y = data_aux[['y']].to_numpy()
# data = pd.DataFrame(X)
# data['y']  = y
# data.columns = data_aux.columns
# save_csv = False
# path_to_save = []
# df_measures, df_classes_dataset = all_measures(data, save_csv, path_to_save, name_data)


# ## Ejemplo lista de csv
# path_csv = os.chdir(root_path+'/datasets')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv'):
#         total_name_list.append(filename)
# total_name_list.sort()
# # len(total_name_list)
#
# # total_name_list = ['Data13.csv']
#
# complex_info = pd.DataFrame()
#
# for data_file in total_name_list:
#     os.chdir(root_path + '/datasets')
#     print(data_file)
#     file = data_file
#     name_data = data_file[0:-4]
#     data_aux = pd.read_csv(file)
#     X = data_aux.iloc[:,:-1].to_numpy() # all variables except y
#     X = preprocessing.scale(X)
#     y = data_aux[['y']].to_numpy()
#     data = pd.DataFrame(X)
#     data['y']  = y
#     data.columns = data_aux.columns
#     # data = pd.read_csv('Dataset9_6000_estandarizado.csv')
#     # path_to_save = root_path+'/Results_Complexity_InstanceLevel'
#     # path_to_save = root_path+'/Results_GB'
#     # save_csv = True
#     save_csv = False
#     # df_measures, df_classes_dataset = all_measures(data, save_csv, path_to_save, name_data)
#     _, df_class_data_host = all_measures(data,save_csv,path_to_save, name_data)
#     # df_class_data_host['level'] = df_class_data_host.index
#     df_class_data_host.reset_index(inplace=True)
#     df_class_data_host['dataset'] = name_data
#     complex_info = pd.concat([complex_info,df_class_data_host])



# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
# predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)
#
# # data (as pandas dataframes)
# X = predict_students_dropout_and_academic_success.data.features
# y = predict_students_dropout_and_academic_success.data.targets
#
#
# df_final = pd.concat([X, y], axis=1)
# df_final = df_final.rename(columns={"Target": "y"})
# X = preprocessing.scale(X)
# data = pd.DataFrame(X)
# data['y']  = y
# data.columns = df_final.columns
# save_csv = False
# path_to_save = 'ccc'
# name_data = 'drop'
# df_measures, df_classes_dataset, extra_results_host = all_measures(data, save_csv, path_to_save, name_data)



