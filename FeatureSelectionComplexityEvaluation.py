##### 24/09/2025
#### En este script vamos a calcular (hacemos las funciones para) la complejidad de distintos conjuntos de datos
#### en diferentes versiones: todas las variables, solo las informativas, solo las redundantes o ruidosa,
### un mix de ellas, las que seleccionan algunos métodos de filtro de FS del estado del artee
### La idea es ver cómo cambia el comportamiento de las medidas de complejidad en esas circunstancias
### para evaluar, un poco a priori, si las podemos utilizar para crear un method de FS basado en
#### medidas de complejidad. en el notebook TrackingCentroides_Hostility hemos visto que las variables que son
### literalmente la copia de otras, son fáciles de pillar porque muestran el mismo comportamiento.
### Sin embargo, las que son redundantes por ser combinación lineal de otras, ya no se ve claramente cómo pillarlas
### Leyendo el SOTA veo que ese es uno de los principales problemas de los métodos de filtro.


import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from skrebate import ReliefF
# Para pymrmr hace falta instalar antes pip install numpy Cython
import pymrmr


# Función para generar datos sintéticos a nuestro gusto
def generate_synthetic_dataset(n_samples=200, n_informative=5, n_noise=5,
                        n_redundant_linear=2, random_state=42, noise_std=0.05):
    rng = np.random.RandomState(random_state)

    # Generamos solo informativas + ruido
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,   # no usamos sklearn redundantes
        n_repeated=0,
        shuffle=False,
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    formulas = {}

    # Añadimos redundantes lineales a mano para controlar cómo se han creado
    for j in range(n_redundant_linear):
        # elijo al azar 2 features informativas para combinarlas
        idx1, idx2 = rng.choice(n_informative, size=2, replace=False)
        coef1, coef2 = rng.uniform(-2, 2, size=2)

        if (noise_std==0):
            new_feature = (coef1 * df[f"f{idx1}"] +
                           coef2 * df[f"f{idx2}"])
            new_name = f"f{df.shape[1]}"
            df[new_name] = new_feature

            formulas[new_name] = f"{coef1:.2f}*f{idx1} + {coef2:.2f}*f{idx2}"
        else:
            new_feature = (coef1 * df[f"f{idx1}"] +
                           coef2 * df[f"f{idx2}"] +
                           rng.normal(0, noise_std, size=n_samples))
            new_name = f"f{df.shape[1]}"
            df[new_name] = new_feature

            formulas[new_name] = f"{coef1:.2f}*f{idx1} + {coef2:.2f}*f{idx2} + ruido"

    dict_info_feature = {
        "informative": [f"f{i}" for i in range(n_informative)],
        "noise": [f"f{i}" for i in range(n_informative, n_informative + n_noise)],
        "redundant_linear": list(formulas.keys()),
        "formulas": formulas
    }

    return df, y, dict_info_feature



df, y, dict_info_feature = generate_synthetic_dataset(n_samples=1000,n_informative=10,n_noise=2,
                                         n_redundant_linear=4,random_state=0,noise_std=0.01)

print("Informativas:", dict_info_feature["informative"])
print("Ruidosas:", dict_info_feature["noise"])
print("Redundantes lineales:", dict_info_feature["redundant_linear"])
print("Fórmulas:")
for k, v in dict_info_feature["formulas"].items():
    print(f"  {k} = {v}")
