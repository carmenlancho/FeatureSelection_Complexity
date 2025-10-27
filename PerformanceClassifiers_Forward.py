### 27/10/2025
# En este script vamos a hacer un código para evaluar cómo va cambiando la performance de los modelos
# al ir añadiendo de 1 en 1 las variables
#  Primero metemos las informativas y luego las otros modo random
# Lo hacemos así porque también quiero estudiar  si la performance se estanca o sigue aumentando
# dado que en algunos casos del distributed hemos visto que aunque ya están todas las variables
# informativas, ha seguido aumentando con variables irrelevantes


from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




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
        random_state=random_state
    )

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


# save_csv = True
def evaluate_incremental_models(X, y, feature_info, random_state=42, cv_splits=10,save_csv=False,
                                path = 'Results_PerformanceClassifiers_Forward'):
    rng = np.random.default_rng(random_state)

    # Orden de variables
    informative = feature_info.query("feature_type == 'informative'")["feature_name"].tolist()
    other_vars = feature_info.query("feature_type != 'informative'")["feature_name"].tolist()
    rng.shuffle(other_vars)
    ordered_features = informative + other_vars

    # Modelos
    models = {
        # "LogReg": LogisticRegression(max_iter=1000),
        # "SVM-linear": SVC(kernel="linear"),
        "SVM-rbf": SVC(kernel="rbf"),
        # "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        # "NaiveBayes": GaussianNB(),
        # "DecisionTree": DecisionTreeClassifier(random_state=random_state)
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    # classes = np.unique(y)

    # Resultados
    results_records = []   # fold-by-fold
    summary_records = []   # resumen mean/std
    detailed_results = {}  # por subset_k

    # Evaluación incremental
    for k in range(1, len(ordered_features) + 1):
        subset = ordered_features[:k]
        Xsub = X[subset].values
        subset_name = f"Top{k}"

        detailed_results[subset_name] = {}

        for model_name, model in models.items():
            fold_acc = []
            fold_gps = []
            # fold_acc_class = {cls: [] for cls in classes}

            for fold, (train_idx, test_idx) in enumerate(skf.split(Xsub, y), 1):
                X_train, X_test = Xsub[train_idx], Xsub[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                gps = compute_gps(y_test, y_pred)

                # acc_per_class = {}
                # for c in classes:
                #     idx = (y_test == c)
                #     acc_per_class[int(c)] = accuracy_score(y_test[idx], y_pred[idx])

                # Registro fold a fold
                record = {
                    "subset_k": k,
                    "variables_incluidas": ",".join(subset),
                    "model": model_name,
                    "fold": fold,
                    "acc": acc,
                    "gps": gps
                }
                # for cls, val in acc_per_class.items():
                #     record[f"acc_class_{cls}"] = val
                #     fold_acc_class[cls].append(val)

                results_records.append(record)
                fold_acc.append(acc)
                fold_gps.append(gps)

            # Medias y std
            mean_acc = np.mean(fold_acc)
            std_acc = np.std(fold_acc)
            mean_gps = np.mean(fold_gps)
            std_gps = np.std(fold_gps)

            # mean_acc_class = {f"mean_acc_class_{int(c)}": np.mean(vals)
            #                   for c, vals in fold_acc_class.items()}
            # std_acc_class = {f"std_acc_class_{int(c)}": np.std(vals)
            #                  for c, vals in fold_acc_class.items()}

            summary_values = {
                "subset_k": k,
                "variables_incluidas": ",".join(subset),
                "model": model_name,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_gps": mean_gps,
                "std_gps": std_gps
            }
            # summary_values.update(mean_acc_class)
            # summary_values.update(std_acc_class)
            summary_records.append(summary_values)

            detailed_results[subset_name][model_name] = {
                "fold_acc": fold_acc,
                "fold_gps": fold_gps,
                # "fold_acc_class": fold_acc_class,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_gps": mean_gps,
                "std_gps": std_gps
                # "mean_acc_class": mean_acc_class,
                # "std_acc_class": std_acc_class
            }

        # DataFrames finales
    results_df = pd.DataFrame(results_records).set_index(["subset_k", "model", "fold"])
    summary_df = pd.DataFrame(summary_records).set_index(["subset_k", "model"])

    # Guardar
    if save_csv and dataset_name:
        results_df.to_csv(f"{path}/{dataset_name}_informative_order_folds.csv")
        summary_df.to_csv(f"{path}/{dataset_name}_informative_order_summary.csv")

    return results_df, summary_df, detailed_results


# ### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset2_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
#
#
#
# ### Dataset 3
# dataset_name = 'ArtificialDataset3'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=25,n_noise=5,
#                                          n_redundant_linear=7,n_redundant_nonlinear=8,
#                                          flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=0,noise_std=0.05)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset3_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
# #### Dataset 7
# dataset_name = 'ArtificialDataset7'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=20,n_noise=10,
#                                          n_redundant_linear=10,n_redundant_nonlinear=10,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=589,noise_std=0.05)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset7_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
# #### Dataset 8
# dataset_name = 'ArtificialDataset8'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=40,n_noise=15,
#                                          n_redundant_linear=15,n_redundant_nonlinear=15,
#                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
#                                                      random_state=86785,noise_std=0.1)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset8_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
# #### Dataset 10
# dataset_name = 'ArtificialDataset10'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=2000,n_informative=6,n_noise=20,
#                                          n_redundant_linear=20,n_redundant_nonlinear=15,
#                                         flip_y=0, class_sep=0.8, n_clusters_per_class=2, weights=[0.3],
#                                                      random_state=959,noise_std=0.3)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset10_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
#
# #### Dataset 11
# dataset_name = 'ArtificialDataset11'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=20,n_noise=20,
#                                          n_redundant_linear=20,n_redundant_nonlinear=15,
#                                         flip_y=0, class_sep=0.6, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=959,noise_std=0.1)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset11_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
# #### Dataset 12
# dataset_name = 'ArtificialDataset12'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=25,n_noise=30,
#                                          n_redundant_linear=30,n_redundant_nonlinear=30,
#                                         flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
#                                                      random_state=987,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset12_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
#
# #### Dataset 14
# dataset_name = 'ArtificialDataset14'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=3000,n_informative=30,n_noise=40,
#                                          n_redundant_linear=30,n_redundant_nonlinear=40,
#                                         flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
#                                                      random_state=95,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset14_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
#
# #### Dataset 16
# dataset_name = 'ArtificialDataset16'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=50,n_noise=40,
#                                          n_redundant_linear=30,n_redundant_nonlinear=40,
#                                         flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.2],
#                                                      random_state=756,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset16_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
#
# #### Dataset 17
# dataset_name = 'ArtificialDataset17'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=5000,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.3, class_sep=0.6, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=756,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset17_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
#
# #### Dataset 18
# dataset_name = 'ArtificialDataset18'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=70,n_noise=40,
#                                          n_redundant_linear=40,n_redundant_nonlinear=40,
#                                         flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
#                                                      random_state=9462,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset18_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
# #### Dataset 19
# dataset_name = 'ArtificialDataset19'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=150,n_noise=50,
#                                          n_redundant_linear=50,n_redundant_nonlinear=50,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=655,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset19_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
# #### Dataset 20
# dataset_name = 'ArtificialDataset20'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=500,n_informative=300,n_noise=60,
#                                          n_redundant_linear=60,n_redundant_nonlinear=60,
#                                         flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
#                                                      random_state=4556,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset20_features.csv")
#
# evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
#
# #### Dataset 21
# dataset_name = 'ArtificialDataset21'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=300,n_noise=100,
#                                          n_redundant_linear=100,n_redundant_nonlinear=100,
#                                         flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
#                                                      random_state=996,noise_std=0.5)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset21_features.csv")
#
# results_df, summary_df, detailed_results = evaluate_incremental_models(X, y, feature_info,save_csv=True)
#


##### Plot





def plot_models_two_panel_performance(summary_df,dataset_name="Dataset",n_informative=None,
    figsize=(14,5),show_std=True):

    summary_df = summary_df.reset_index() # porque tiene index multilevel

    # copia
    df = summary_df.copy()
    # df["subset_k"] = pd.to_numeric(df["subset_k"], errors="coerce")

    sns.set(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # palette
    model_list = list(df["model"].unique())
    palette = sns.color_palette("tab10", len(model_list))

    # ACCURACY
    ax = axes[0]
    for i, m in enumerate(model_list):
        d = df[df["model"] == m].sort_values("subset_k")
        ax.plot(d["subset_k"], d["mean_acc"], label=m, color=palette[i], linewidth=2)
        if show_std and "std_acc" in d.columns:
            ax.fill_between(d["subset_k"],
                            d["mean_acc"] - d["std_acc"],
                            d["mean_acc"] + d["std_acc"],
                            color=palette[i], alpha=0.15)
    # Num var informativas
    ax.axvline(n_informative, color="black", linestyle="--", linewidth=1.5, label="Informative vars")
    ax.set_xlabel("Nº de features")
    ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy vs Nº de features")
    ax.grid(alpha=0.4, linestyle="--")

    # GPS
    ax = axes[1]
    for i, m in enumerate(model_list):
        d = df[df["model"] == m].sort_values("subset_k")
        ax.plot(d["subset_k"], d["mean_gps"], label=m, color=palette[i], linewidth=2)
        if show_std and "std_gps" in d.columns:
            ax.fill_between(d["subset_k"],
                            d["mean_gps"] - d["std_gps"],
                            d["mean_gps"] + d["std_gps"],
                            color=palette[i], alpha=0.15)
    # Num var informativas
    ax.axvline(n_informative, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Nº de features")
    ax.set_ylabel("GPS")
    # ax.set_title("GPS vs Nº de features")
    ax.grid(alpha=0.4, linestyle="--")

    # legend a main title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_list) + 1, frameon=False)
    fig.suptitle(f"{dataset_name} — Model Performance vs Nº of features", fontsize=14, y=0.93)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.show()


#### Dataset 2
# dataset_name = 'ArtificialDataset2'
# X, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
#                                          n_redundant_linear=4,n_redundant_nonlinear=2,
#                                     flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
#                                                      random_state=0,noise_std=0.01)
#
# feature_info = pd.read_csv("Synthetic_Metadata/ArtificialDataset2_features.csv")
#
# results_df, summary_df, detailed_results = evaluate_incremental_models(X, y, feature_info,save_csv=True)
#
# n_informative = len(feature_info[feature_info.feature_type == 'informative'])
#
# plot_models_two_panel_performance(summary_df,dataset_name=dataset_name,n_informative=n_informative,
#     figsize=(14,5),show_std=True)








