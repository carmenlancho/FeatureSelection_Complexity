## 07/01/2026

# Copiamos el código del script SOTA_methods_CV y lo adaptamos para datos reales

############################################################################################
#########          SOTA METHODS WITH CV AND PERFORMANCE AND COMPLEXITY             #########
############################################################################################



from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
import xgboost as xgb
import pandas as pd
import numpy as np
from All_measures import *
from sklearn.datasets import make_classification
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
        random_state=random_state)

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





## Función para ejecutar diversos métodos de FS tipo filtro del SOTA
def select_features_by_filters_and_corr(X, y, feature_names,k=None,methods=None,random_state=0,
                                        filter_corr=False,corr_method="pearson",corr_th=0.9):
    """
    Aplica varios métodos de filtro

    - X: np.ndarray or DataFrame
    - y: array-like
    - feature_names: list of names (length = X.shape[1])
    - k: número de features a seleccionar (si None -> k = n_informative_guess ~ sqrt(n_features) fallback)
    - methods: lista de strings entre {"mutual_info","f_classif","rf","relief",'xgboost'}
    """
    if methods is None:
        methods = ["mutual_info", "f_classif", "rf", "relief","xgboost"]

    X_df = X.copy()# if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    # Filtro previo opcional de eliminación de variables con correelación > corr_th
    if filter_corr:
        corr = X_df.corr(method=corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        if len(to_drop) > 0:
            X_df = X_df.drop(columns=to_drop)
            feature_names = [f for f in feature_names if f not in to_drop]


    # Xarr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    Xarr = X_df.values
    n_features = Xarr.shape[1]
    if k is None:
        k = max(1, int(np.sqrt(n_features)))  # heuristic value

    results = {}


    # mutual information
    if "mutual_info" in methods:
        mi = mutual_info_classif(Xarr, y, random_state=random_state)
        s = pd.Series(mi, index=feature_names).sort_values(ascending=False)
        results["mutual_info"] = {"scores": s, "selected": list(s.index[:k])}

    # ANOVA F (f_classif)
    if "f_classif" in methods:
        F, p = f_classif(Xarr, y)
        s = pd.Series(F, index=feature_names).sort_values(ascending=False)
        results["f_classif"] = {"scores": s, "selected": list(s.index[:k])}

    # Random Forest importance
    if "rf" in methods:
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        rf.fit(Xarr, y)
        imp = rf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["rf"] = {"scores": s, "selected": list(s.index[:k])}

    # ReliefF
    if "relief" in methods:
        rf_sel = ReliefF(n_features_to_select=Xarr.shape[1]) # n_neighbors usamos el valor por defecto de la librería
        rf_sel.fit(Xarr, y)
        scores = rf_sel.feature_importances_
        s = pd.Series(scores, index=feature_names).sort_values(ascending=False)
        results["relief"] = {"scores": s, "selected": list(s.index[:k])}

    # XGBoost
    if "xgboost" in methods:
        xgb_clf = xgb.XGBClassifier(eval_metric="logloss",random_state=random_state)
        xgb_clf.fit(Xarr, y)
        imp = xgb_clf.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        results["xgb"] = {"scores": s, "selected": list(s.index[:k])}


    return results


def compute_gps(y_true, y_pred):
    """
    Calcula GPS para un problema binario.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    TN, FP, FN, TP = cm.ravel()

    # métricas base
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    # F1+ y F1-
    F1_pos = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else 0
    F1_neg = 2 * (NPV * TNR) / (NPV + TNR) if (NPV + TNR) > 0 else 0

    # GPS
    GPS = 2 * (F1_pos * F1_neg) / (F1_pos + F1_neg) if (F1_pos + F1_neg) > 0 else 0
    return GPS


# k = 3
def evaluate_sota_fs_cv(X, y, k, model,
                        methods=["mutual_info", "f_classif", "rf", "relief", "xgboost"],
                        cv_splits=5, random_state=0,
                        filter_corr = True,
                        corr_th=0.9, corr_method="pearson"):
    """
    Realiza CV evaluando métodos de FS del estado del arte (filter-based):
      - aplica selección de features en cada fold de entrenamiento
      - evalúa rendimiento (train/test) y complejidad en el subset seleccionado
      - repite para cada métod

    Devuelve:
      - selections_df: todas las selecciones e importancias (por fold, métod, filtro)
      - performance_df: rendimiento (train/test) por fold, métod y filtro
      - complexity_df: medidas de complejidad del subset seleccionado
    """

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    selections_records = []
    performance_records = []
    complexity_records = []
    model_name = model.__class__.__name__

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        feature_names = X.columns.tolist()

        fs_results = select_features_by_filters_and_corr(X_train, y_train, feature_names, k=k,
                                                        methods=methods,random_state=random_state,
                                                filter_corr=filter_corr,corr_method=corr_method,corr_th=corr_th)

        for method_name, info in fs_results.items():
            selected_feats = info["selected"]
            scores = info["scores"]

            # Guardamos selección
            for f, sc in scores.items():
                selections_records.append({
                        "fold": fold,
                        "method": method_name,
                        "filter_corr": filter_corr,
                        "feature": f,
                        "score": sc,
                        "selected": f in selected_feats})

            # Evaluamos modelo con selected variables
            X_train_sel = X_train[selected_feats]
            X_test_sel = X_test[selected_feats]

            model.fit(X_train_sel, y_train)

            # Predicciones
            y_pred_train = model.predict(X_train_sel)
            y_pred_test = model.predict(X_test_sel)

            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            gps_train = compute_gps(y_train, y_pred_train)
            gps_test = compute_gps(y_test, y_pred_test)



            performance_records.append({
                    "fold": fold,
                    "model": model_name,
                    "method": method_name,
                    "filter_corr": filter_corr,
                    "n_features": len(selected_feats),
                    "acc_train": acc_train,
                    "gps_train": gps_train,
                    "acc_test": acc_test,
                    "gps_test": gps_test})

            # Complejidad del subset
            datos = pd.DataFrame(X_train_sel.copy())
            datos["y"] = y_train
            _, df_classes, _ = all_measures_FS(datos, save_csv=False, path_to_save=None, name_data=None)
            df_total = df_classes.loc["dataset"]

            complexity_record = {
                    "fold": fold,
                    "method": method_name,
                    "filter_corr": filter_corr,
                    **df_total.to_dict(),
                    "n_features": len(selected_feats)}
            complexity_records.append(complexity_record)

    # formato a dataframe
    selections_df = pd.DataFrame(selections_records)
    performance_df = pd.DataFrame(performance_records)
    complexity_df = pd.DataFrame(complexity_records)

    # Summary por fold
    summary_perf = performance_df.groupby(["method", "filter_corr"]).agg(
        acc_test_mean=("acc_test", "mean"),
        acc_test_std=("acc_test", "std"),
        acc_test_max=("acc_test", "max"),
        gps_test_mean=("gps_test", "mean"),
        gps_test_std=("gps_test", "std"),
        gps_test_max=("gps_test", "max"),
    ).reset_index()


    return selections_df, performance_df, complexity_df, summary_perf


def run_evaluate_sota_fs_multiple_models_real(X, y, models_dict,dataset_name,
                                         methods=["mutual_info", "f_classif", "rf", "relief", "xgboost"],
                                         cv_splits=5, random_state=0,
                                         filter_corr=True, corr_th=0.9, corr_method="pearson",
                                         path='Results_FS_SOTA_CV', save_csv=False):

    p = X.shape[1]
    k = np.ceil(p/3)
    if (k>=1000):
        k = np.ceil(np.sqrt(p))
    k = int(k)


    all_selections = []
    all_performance = []
    all_complexity = []
    all_summary = []

    for model_name, model in models_dict.items():
        print(f"\n Classifier: {model_name}")

        selections_df, performance_df, complexity_df, summary_perf = evaluate_sota_fs_cv(
            X=X, y=y, k=k, model=model,
            methods=methods, cv_splits=cv_splits, random_state=random_state,
            filter_corr=filter_corr, corr_th=corr_th, corr_method=corr_method)

        # etiqueta del modelo
        selections_df["model"] = model_name
        performance_df["model"] = model_name
        complexity_df["model"] = model_name
        summary_perf["model"] = model_name

        # Guardamos
        all_selections.append(selections_df)
        all_performance.append(performance_df)
        all_complexity.append(complexity_df)
        all_summary.append(summary_perf)

    # Concatenamos resultados
    selections_all = pd.concat(all_selections, ignore_index=True)
    performance_all = pd.concat(all_performance, ignore_index=True)
    complexity_all = pd.concat(all_complexity, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)

    # Unimos performance y complexity
    perf_comp =  pd.merge(left=performance_all,right=complexity_all, how='left',
        left_on=['fold', 'model', 'method', 'filter_corr', 'n_features'],
        right_on=['fold', 'model', 'method', 'filter_corr', 'n_features'])



    # Resumen global por modelo y métod
    summary_global = summary_all.groupby(["model", "method"]).agg(
        acc_test_mean=("acc_test_mean", "mean"),
        acc_test_std=("acc_test_std", "mean"),
        gps_test_mean=("gps_test_mean", "mean"),
        gps_test_std=("gps_test_std", "mean")).reset_index()

    results = {
        "selections_all": selections_all,
        "performance_all": performance_all,
        "complexity_all": complexity_all,
        "summary_all": summary_all,
        "summary_global": summary_global}

    if save_csv:
        name_csv1 = f"{path}/{dataset_name}_SOTA_CV_FeatureImportance_Folds.csv"
        selections_all.to_csv(name_csv1, index=False)
        name_csv2 = f"{path}/{dataset_name}_SOTA_CV_Performance_Folds.csv"
        perf_comp.to_csv(name_csv2, index=False)
        name_csv3 = f"{path}/{dataset_name}_SOTA_CV_SummaryResults.csv"
        summary_global.to_csv(name_csv3, index=False)

    return results





# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#


# model = RandomForestClassifier(random_state=0)
# k=3
# selections_df, performance_df, complexity_df, summary_perf = evaluate_sota_fs_cv(
#     X, y, k=len(dict_info_feature["informative"]),
#     model=model,cv_splits=5, random_state=0)


# models_dict = {
#     "RF": RandomForestClassifier(random_state=0),
#     "SVM": SVC(kernel="rbf", probability=True, random_state=0)}
#
# k = len(dict_info_feature["informative"])  # nº de variables informativas
# run_evaluate_sota_fs_multiple_models_real(X, y, k=k, models_dict=models_dict,cv_splits=5, random_state=0,save_csv=True)

# models_dict = {#"LogReg": LogisticRegression(max_iter=1000, random_state=0),
#     # "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
#     "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
#     # "RandomForest": RandomForestClassifier(random_state=0),
#     "KNN": KNeighborsClassifier()
#     # "NaiveBayes": GaussianNB(),
#     # "DecisionTree": DecisionTreeClassifier(random_state=0),
#     # "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
#     }


models_dict = {"LogReg": LogisticRegression(max_iter=1000, random_state=0),
    "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
    "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
    "RandomForest": RandomForestClassifier(random_state=0),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
    }


# os.chdir("datasets")
# for file in glob.glob("*.csv"):
#     print(file)
# file = 'ionosphere.csv'
list_datasets = [ #'bodyfat.csv',
    'boston.csv','cleve.csv',
            'heart-statlog.csv','zoo.csv','vehicle2.csv']#['spambase.csv','ionosphere.csv', 'sonar.csv','wdbc.csv']
                # 'parkinsons.csv']
                # 'wdbc.csv',
                #  'musk2.csv','parkinsons.csv',
                #  'ozone.csv','sonar.csv','spambase.csv',
                #  'Colon.csv','arcene_train.csv','gisette_train.csv']

def format_labels(y):
    """
    Ensure labels y are integers starting at 0: 0,1,...,n_classes-1.
    Works with categorical (str) or numeric labels.
    """
    y = pd.Series(y)  # ensure consistent type

    # Case 1: non-numeric labels (strings, categories, mixed)
    if not np.issubdtype(y.dtype, np.number):
        categories = sorted(y.unique())  # alphabetic order
        mapping = {cat: i for i, cat in enumerate(categories)}
        y = y.map(mapping)

    # Case 2: numeric but not starting at 0
    else:
        unique_vals = sorted(y.unique())
        if unique_vals[0] != 0 or unique_vals != list(range(len(unique_vals))):
            mapping = {val: i for i, val in enumerate(unique_vals)}
            y = y.map(mapping)

    return y.to_numpy()

path2 = "datasets"
for file in list_datasets:
    # os.makedirs(path2, exist_ok=True)
    read_csv = f"{path2}/{file}"
    df = pd.read_csv(read_csv)
    # print(df)
    print(read_csv)
    y = format_labels(df['y'])
    cols = df.drop('y', axis=1).columns
    X = df.drop('y', axis=1)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    # df[cols] = StandardScaler(with_mean=True, with_std=True).fit_transform(df)
    X = pd.DataFrame(X)
    X.columns = cols
    dataset_name = file.split(".")[0]
    print(dataset_name)

    run_evaluate_sota_fs_multiple_models_real(X, y, models_dict=models_dict,dataset_name=dataset_name,
                                              cv_splits=5, random_state=0,save_csv=True)






## 27/12/2025

# Evaluamos los datasets añadiendo una a una las variablees siguiendo el ranking de los métodos del SOTA
# para saber su performance en cada uno de los casos y comparar con nuestro métoddo


def load_all_filter_rankings_from_csv(csv_path,methods=None,filter_corr=True):
    """
    Devuelve:
      rankings[method][fold] = lista ordenada de features
    """
    df = pd.read_csv(csv_path)

    if methods is not None:
        df = df[df["method"].isin(methods)]

    df = df[df["filter_corr"] == filter_corr]
    df = df.loc[df.model == 'KNN', :] # aquí los modelos no interactuan, realmente esa columna sobra en los csvs

    rankings = {}

    for method, df_m in df.groupby("method"):
        rankings[method] = {}
        for fold, df_f in df_m.groupby("fold"):
            df_sorted = df_f.sort_values("score", ascending=False)
            rankings[method][fold] = df_sorted["feature"].tolist()

    return rankings



def evaluate_incremental_k_all_methods(X, y,rankings_all_methods,models,dataset_name,
                                        cv_splits=5,random_state=0):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        for method, rankings_by_fold in rankings_all_methods.items():

            # if fold_id not in rankings_by_fold:
            #     continue

            ranking = rankings_by_fold[fold_id]
            K = len(ranking)

            for k in range(1, K + 1):

                selected = ranking[:k]
                Xt = X_train[selected]
                Xs = X_test[selected]

                for model_name, model in models.items():

                    clf = model
                    clf.fit(Xt, y_train)
                    pred = clf.predict(Xs)

                    acc = accuracy_score(y_test, pred)
                    gps = compute_gps(y_test, pred)

                    rows.append({
                        "dataset": dataset_name,
                        "fold": fold_id,
                        "method": method,
                        "k": k,
                        "n_available_features": len(ranking),
                        "model": model_name,
                        "acc_test": acc,
                        "gps_test": gps})

    perf_final = pd.DataFrame(rows)

    return perf_final


methods = ["mutual_info", "f_classif", "rf", "relief", "xgboost"]

# models = {
#     "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
#     "KNN": KNeighborsClassifier()
# }


models = {"LogReg": LogisticRegression(max_iter=1000, random_state=0),
    "SVM-linear": SVC(kernel="linear", probability=True, random_state=0),
    "SVM-rbf": SVC(kernel="rbf", probability=True, random_state=0),
    "RandomForest": RandomForestClassifier(random_state=0),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=0)
    }




path2 = "datasets"
list_datasets =['bodyfat.csv']
for file in list_datasets:
    # os.makedirs(path2, exist_ok=True)
    read_csv = f"{path2}/{file}"
    df = pd.read_csv(read_csv)
    # print(df)
    print(read_csv)
    y = format_labels(df['y'])
    cols = df.drop('y', axis=1).columns
    X = df.drop('y', axis=1)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    # df[cols] = StandardScaler(with_mean=True, with_std=True).fit_transform(df)
    X = pd.DataFrame(X)
    X.columns = cols
    dataset_name = file.split(".")[0]
    print(dataset_name)

# rankings = load_all_filter_rankings_from_csv(
#     csv_path="Results_FS_SOTA_CV/ionosphere_SOTA_CV_FeatureImportance_Folds.csv",
#     methods=methods,filter_corr=True)
#
# performance_filters = evaluate_incremental_k_all_methods(X=X,y=y,rankings_all_methods=rankings,
#                                                         models=models,dataset_name="ionosphere",cv_splits=5)
# performance_filters.to_csv('Results_FS_SOTA_CV/ionosphere_SOTA_EvolutivePerformance.csv',index=False)




