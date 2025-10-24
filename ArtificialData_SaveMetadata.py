## 24/10/2025
### En este script creamos una versión de la función para generar los datos sintéticos
# que nos permite tb guardar metadatos y así ir evaluando cuándo se cogen las variables informativas





from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from All_measures import *
import os



def generate_synthetic_dataset_save_metadata(n_samples,n_informative, n_noise, n_redundant_linear, n_redundant_nonlinear,
    flip_y, class_sep, n_clusters_per_class, weights, random_state=42, noise_std=0.05,
    dataset_id=None, save_metadata=False, metadata_dir="Synthetic_Metadata"): # esta fila es para guardar metadatos
    rng = np.random.RandomState(random_state)

    # Datos base: solo informativas + ruido
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        flip_y=flip_y,
        class_sep=class_sep,
        n_clusters_per_class=n_clusters_per_class,
        weights=weights,
        shuffle=False,
        random_state=random_state)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    formulas = {}
    formulas_nonlinear = {}

    # Redundantes lineales
    for j in range(n_redundant_linear):
        idx1, idx2 = rng.choice(n_informative, size=2, replace=False)
        coef1, coef2 = rng.uniform(-2, 2, size=2)
        new_name = f"f{df.shape[1]}"
        new_feature = coef1 * df[f"f{idx1}"] + coef2 * df[f"f{idx2}"]
        if noise_std > 0:
            new_feature += rng.normal(0, noise_std, size=n_samples)
        df[new_name] = new_feature
        formulas[new_name] = f"{coef1:.2f}*f{idx1} + {coef2:.2f}*f{idx2}" + ("" if noise_std == 0 else " + ruido")

    # Redundantes no lineales
    for j in range(n_redundant_nonlinear):
        idx = rng.choice(n_informative, size=2, replace=False)
        func = rng.choice([np.sin, np.cos, np.square, np.exp])
        new_name = f"f{df.shape[1]}"
        new_feature = func(df[f"f{idx[0]}"]) + df[f"f{idx[1]}"]
        if noise_std > 0:
            new_feature += rng.normal(0, noise_std, size=n_samples)
        df[new_name] = new_feature
        formulas_nonlinear[new_name] = f"{func.__name__}(f{idx[0]}) + f{idx[1]}" + ("" if noise_std == 0 else " + ruido")

    dict_info_feature = {
        "informative": [f"f{i}" for i in range(n_informative)],
        "noise": [f"f{i}" for i in range(n_informative, n_informative + n_noise)],
        "redundant_linear": list(formulas.keys()),
        "redundant_nonlinear": list(formulas_nonlinear.keys()),
        "formulas_linear": formulas,
        "formulas_nonlinear": formulas_nonlinear
    }

    # Estandarización
    df[df.columns] = StandardScaler(with_mean=True, with_std=True).fit_transform(df)

    # Guardar metadatos
    if save_metadata:
        os.makedirs(metadata_dir, exist_ok=True)
        meta_list = []

        for f in dict_info_feature["informative"]:
            meta_list.append({"feature_name": f, "feature_type": "informative", "formula": ""})
        for f in dict_info_feature["noise"]:
            meta_list.append({"feature_name": f, "feature_type": "noise", "formula": ""})
        for f, formula in formulas.items():
            meta_list.append({"feature_name": f, "feature_type": "redundant_linear", "formula": formula})
        for f, formula in formulas_nonlinear.items():
            meta_list.append({"feature_name": f, "feature_type": "redundant_nonlinear", "formula": formula})

        df_meta = pd.DataFrame(meta_list)
        df_meta.to_csv(os.path.join(metadata_dir, f"ArtificialDataset{dataset_id}_features.csv"), index=False)

    return df, y, dict_info_feature
