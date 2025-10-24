## 24/10/2025
### En este script creamos una versión de la función para generar los datos sintéticos
# que nos permite tb guardar metadatos y así ir evaluando cuándo se cogen las variables informativas





from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from All_measures import *
import os



def generate_synthetic_dataset_save_metadata(n_samples,n_informative, n_noise, n_redundant_linear, n_redundant_nonlinear,
    flip_y, class_sep, n_clusters_per_class, weights, random_state=42, noise_std=0.05,
    dataset_id=None, save_metadata=True, metadata_dir="Synthetic_Metadata"): # esta fila es para guardar metadatos
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








### Dataset 1
dataset_name = 'ArtificialDataset1'
id = 1
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                        flip_y=0, class_sep = 1, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01,dataset_id=id)




### Dataset 2
dataset_name = 'ArtificialDataset2'
id = 2
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,n_redundant_nonlinear=2,
                                    flip_y=0, class_sep = 0.6, n_clusters_per_class=1 , weights=[0.5],
                                                     random_state=0,noise_std=0.01,dataset_id=id)
### Dataset 3
dataset_name = 'ArtificialDataset3'
id = 3
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=25,n_noise=5,
                                         n_redundant_linear=7,n_redundant_nonlinear=8,
                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=0,noise_std=0.05,dataset_id=id)

### Dataset 4
dataset_name = 'ArtificialDataset4'
id = 4
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=5000,n_informative=15,n_noise=15,
                                         n_redundant_linear=4,n_redundant_nonlinear=5,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=10,noise_std=0.01,dataset_id=id)

### Dataset5
dataset_name = 'ArtificialDataset5'
id = 5
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=5000,n_informative=25,n_noise=15,
                                         n_redundant_linear=8,n_redundant_nonlinear=7,
                                     flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=10,noise_std=0.05,dataset_id=id)

#### Dataseet 6
dataset_name = 'ArtificialDataset6'
id = 6
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=10000,n_informative=8,n_noise=15,
                                         n_redundant_linear=4,n_redundant_nonlinear=5,
                                         flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.01,dataset_id=id)


#### Dataset 7
dataset_name = 'ArtificialDataset7'
id = 7
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=20,n_noise=10,
                                         n_redundant_linear=10,n_redundant_nonlinear=10,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=589,noise_std=0.05,dataset_id=id)

#### Dataset 8
dataset_name = 'ArtificialDataset8'
id = 8
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=40,n_noise=15,
                                         n_redundant_linear=15,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=1, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=86785,noise_std=0.1,dataset_id=id)

#### Dataset 9
dataset_name = 'ArtificialDataset9'
id = 9
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=3000,n_informative=10,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=20,
                                        flip_y=0, class_sep=0.7, n_clusters_per_class=1, weights=[0.5],
                                                     random_state=959,noise_std=0.3,dataset_id=id)

#### Dataset 10
dataset_name = 'ArtificialDataset10'
id = 10
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=2000,n_informative=6,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=0.8, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=959,noise_std=0.3,dataset_id=id)

#### Dataset 11
dataset_name = 'ArtificialDataset11'
id = 11
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=3000,n_informative=20,n_noise=20,
                                         n_redundant_linear=20,n_redundant_nonlinear=15,
                                        flip_y=0, class_sep=0.6, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=959,noise_std=0.1,dataset_id=id)

#### Dataset 12
dataset_name = 'ArtificialDataset12'
id = 12
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.9, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=987,noise_std=0.5,dataset_id=id)



#### Dataset 13
dataset_name = 'ArtificialDataset13'
id = 13
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=3000,n_informative=25,n_noise=30,
                                         n_redundant_linear=30,n_redundant_nonlinear=30,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.4],
                                                     random_state=95,noise_std=0.5,dataset_id=id)


#### Dataset 14
dataset_name = 'ArtificialDataset14'
id = 14
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=3000,n_informative=30,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.2, class_sep=0.6, n_clusters_per_class=2, weights=[0.3],
                                                     random_state=95,noise_std=0.5,dataset_id=id)


#### Dataset 15
dataset_name = 'ArtificialDataset15'
id = 15
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=5000,n_informative=40,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=78,noise_std=0.1,dataset_id=id)


#### Dataset 16
dataset_name = 'ArtificialDataset16'
id = 16
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=50,n_noise=40,
                                         n_redundant_linear=30,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.4, n_clusters_per_class=1, weights=[0.2],
                                                     random_state=756,noise_std=0.5,dataset_id=id)



#### Dataset 17
dataset_name = 'ArtificialDataset17'
id = 17
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=5000,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.3, class_sep=0.6, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=756,noise_std=0.5,dataset_id=id)



#### Dataset 18
dataset_name = 'ArtificialDataset18'
id = 18
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=500,n_informative=70,n_noise=40,
                                         n_redundant_linear=40,n_redundant_nonlinear=40,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=2, weights=[0.2],
                                                     random_state=9462,noise_std=0.5,dataset_id=id)




#### Dataset 19
dataset_name = 'ArtificialDataset19'
id = 19
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=500,n_informative=150,n_noise=50,
                                         n_redundant_linear=50,n_redundant_nonlinear=50,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=655,noise_std=0.5,dataset_id=id)


#### Dataset 20
dataset_name = 'ArtificialDataset20'
id = 20
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=500,n_informative=300,n_noise=60,
                                         n_redundant_linear=60,n_redundant_nonlinear=60,
                                        flip_y=0.1, class_sep=0.6, n_clusters_per_class=1, weights=[0.3],
                                                     random_state=4556,noise_std=0.5,dataset_id=id)



#### Dataset 21
dataset_name = 'ArtificialDataset21'
id = 21
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=300,n_noise=100,
                                         n_redundant_linear=100,n_redundant_nonlinear=100,
                                        flip_y=0.1, class_sep=0.7, n_clusters_per_class=2, weights=[0.4],
                                                     random_state=996,noise_std=0.5,dataset_id=id)




#### Dataset 22
dataset_name = 'ArtificialDataset22'
id = 22
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=500,n_noise=150,
                                         n_redundant_linear=150,n_redundant_nonlinear=150,
                                        flip_y=0.2, class_sep=0.7, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=996,noise_std=0.5,dataset_id=id)



#### Dataset 23
dataset_name = 'ArtificialDataset23'
id = 23
X, y, dict_info_feature = generate_synthetic_dataset_save_metadata(n_samples=1000,n_informative=5000,n_noise=1500,
                                         n_redundant_linear=1500,n_redundant_nonlinear=1500,
                                        flip_y=0.4, class_sep=0.8, n_clusters_per_class=1, weights=[0.4],
                                                     random_state=996,noise_std=0.5,dataset_id=id)


