import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def prepare_classes(X, y):
    """
    Asumimos que las X realmente son phi(X)
    Separa los embeddings por clase.

    Parameters
    ----------
    X : array-like, shape (N, d)
        Embeddings phi(x) de las N observaciones.

    y : array-like, shape (N,)
        Etiqueta de clase de cada observación.

    Returns
    -------
    classes : np.ndarray
        Clases distintas.

    X_by_class : dict
        Diccionario:
            clase -> matriz de embeddings de esa clase
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Comprobaciones de dimensiones
    if X.ndim != 2:
        raise ValueError("X debe tener forma (N, d).")

    if len(X) != len(y):
        raise ValueError("X e y deben tener el mismo número de observaciones.")

    classes = np.unique(y)

    # Guardamos los datos de cada clase por separado
    X_by_class = {c: X[y == c] for c in classes}

    return classes, X_by_class



def knn_density(query_points,reference_points,k=3,same_class=False,eps=1e-12):
    """
    Estima p(z | C_j) mediante vecinos cercanos.
    Implementamos: p_hat(z | C_j) = k / (N_j * V)
    donde: V = (2 * r_k)^d
    y r_k es la distancia al k-ésimo vecino más cercano
    de z dentro de la clase C_j.

    Parameters
    ----------
    query_points : array, shape (M, d)
        Puntos z en los que queremos evaluar la densidad.

    reference_points : array, shape (N_j, d)
        Puntos pertenecientes a la clase C_j.

    k : int
        Número de vecinos.

    same_class : bool
        True cuando query_points proceden de la misma clase
        que reference_points.

        En ese caso debemos ignorar el propio punto como
        vecino de sí mismo.

    eps : float
        Pequeño valor para evitar divisiones por cero.

    Returns
    -------
    densities : array, shape (M,)
        Estimación de p(z | C_j) para cada query point.
    """

    query_points = np.asarray(query_points, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)

    N_j, d = reference_points.shape

    # Si estamos comparando una clase consigo misma,
    # pedimos k+1 vecinos porque el vecino más cercano
    # normalmente será el propio punto, con distancia 0
    n_neighbors = k + 1 if same_class else k

    if N_j < n_neighbors:
        raise ValueError(
            f"No hay suficientes muestras en la clase: "
            f"se necesitan al menos {n_neighbors}, pero hay {N_j}."
        )

    # Construimos la estructura de vecinos cercanos.
    nn = NearestNeighbors(n_neighbors=n_neighbors,metric="euclidean")
    nn.fit(reference_points)

    # distances tiene forma: (n_query_points, n_neighbors)
    # Las distancias aparecen ordenadas de menor a mayor.
    distances, indices = nn.kneighbors(query_points)

    if same_class:
        # Quitamos el vecino de distancia 0 correspondiente
        # al propio punto.
        distances = distances[:, 1:]

    # Distancia al k-ésimo vecino.
    r_k = distances[:, -1]

    # Si existen puntos duplicados, la distancia al k-ésimo vecino podría ser 0
    # Esto produciría V = 0 y una división por cero.
    # Por ello imponemos una distancia mínima muy pequeña.
    r_k = np.maximum(r_k, eps)

    # Volumen del hipercubo de lado 2*r_k:  V = (2*r_k)^d
    V = (2.0 * r_k) ** d

    # Estimador k-NN de densidad: p_hat(z | C_j) = k / (N_j * V)
    densities = k / (N_j * V)

    return densities



def compute_S(X, y, M=100, k=3, random_state=0):
    """
    Calcula la matriz de similitud S entre clases.
    Cada entrada se estima como: S_ij ≈ (1/M) * sum_m p_hat(z_m | C_j)
    donde los z_m son M muestras de la clase C_i.

    Parameters
    ----------
    X : array, shape (N, d) - Embeddings phi(x).
    y : array, shape (N,) - Etiquetas.
    M : int - Número de muestras Monte Carlo tomadas de cada clase.
    k : int - Número de vecinos para estimar la densidad.
    random_state : int - Semilla para reproducibilidad.

    Returns
    -------
    S_df : pd.DataFrame - Matriz K x K de similitud entre clases.
    """

    classes, X_by_class = prepare_classes(X, y)
    K = len(classes)
    rng = np.random.default_rng(random_state)
    S = np.zeros((K, K), dtype=float)

    # ------------------------------------------------------
    # Elegimos M puntos Monte Carlo para cada clase.
    # ------------------------------------------------------
    mc_samples = {}

    for c in classes:
        Xc = X_by_class[c]
        N_c = len(Xc)
        # Si una clase tiene menos de M puntos, muestreamos con reemplazo.
        replace = N_c < M
        idx = rng.choice(N_c,size=M,replace=replace)
        mc_samples[c] = Xc[idx]

    # ------------------------------------------------------
    # Calculamos S_ij para todos los pares de clases.
    # ------------------------------------------------------
    for i, c_i in enumerate(classes):
        # M puntos z_m ~ C_i
        query_points = mc_samples[c_i]
        for j, c_j in enumerate(classes):
            # Todos los puntos disponibles de C_j
            reference_points = X_by_class[c_j]
            densities = knn_density(query_points=query_points,reference_points=reference_points,
                k=k,same_class=(c_i == c_j))
            # Aproximación Monte Carlo:  S_ij ≈ mean_m p_hat(z_m | C_j)
            S[i, j] = densities.mean()

    S_df = pd.DataFrame(S,index=classes,columns=classes)

    return S_df


def compute_W(S):
    """
    Construye la matriz de adyacencia W a partir de S.
    Usa la ecuación (5):
                    sum_k |S_ik - S_jk|
        w_ij = 1 - ---------------------
                    sum_k |S_ik + S_jk|
    Parameters
    ----------
    S : pd.DataFrame o np.ndarray - Matriz de similitud entre clases.

    Returns
    -------
    W_df : pd.DataFrame - Matriz de adyacencia simétrica.
    """

    # Conservamos las etiquetas si S es un DataFrame.
    if isinstance(S, pd.DataFrame):
        labels = S.index
        S_array = S.to_numpy(dtype=float)
    else:
        S_array = np.asarray(S, dtype=float)
        labels = np.arange(S_array.shape[0])

    K = S_array.shape[0]
    W = np.zeros((K, K), dtype=float)

    for i in range(K):
        # Firma completa de la clase i
        signature_i = S_array[i, :]
        for j in range(K):
            # Firma completa de la clase j
            signature_j = S_array[j, :]
            numerator = np.sum(np.abs(signature_i - signature_j))
            denominator = np.sum(np.abs(signature_i + signature_j))
            # Si ambas firmas fueran completamente cero,
            # evitamos una división 0/0.
            if denominator == 0:
                W[i, j] = 1.0
            else:
                W[i, j] = 1.0 - numerator / denominator

    W_df = pd.DataFrame(W,index=labels,columns=labels)

    return W_df


def compute_laplacian(W):
    """
    Calcula: D_ii = sum_j W_ij
    y después: L = D - W

    Parameters
    ----------
    W : pd.DataFrame o np.ndarray

    Returns
    -------
    D_df : pd.DataFrame - Matriz diagonal de grados.
    L_df : pd.DataFrame - Laplaciano del grafo.
    """

    if isinstance(W, pd.DataFrame):
        labels = W.index
        W_array = W.to_numpy(dtype=float)
    else:
        W_array = np.asarray(W, dtype=float)
        labels = np.arange(W_array.shape[0])

    # Grado de cada nodo/clase
    degrees = W_array.sum(axis=1)
    # Matriz diagonal D
    D = np.diag(degrees)
    # Laplaciano
    L = D - W_array
    D_df = pd.DataFrame(D,index=labels,columns=labels)
    L_df = pd.DataFrame(L,index=labels,columns=labels)

    return D_df, L_df



def compute_spectrum(L):
    """
    Calcula los autovalores ordenados del Laplaciano.

    Returns
    -------
    eigenvalues : np.ndarray - lambda_1 <= lambda_2 <= ... <= lambda_K
    """

    if isinstance(L, pd.DataFrame):
        L = L.to_numpy(dtype=float)

    # eigvalsh devuelve autovalores de una matriz simétrica
    eigenvalues = np.linalg.eigvalsh(L)
    # Por errores numéricos puede aparecer algo como -1e-15.
    # Teóricamente los autovalores del Laplaciano son >= 0.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    return eigenvalues


def compute_csg_from_eigenvalues(eigenvalues):
    """
    Calcula las ecuaciones (6) y (7).

    Eq. (6): delta_i = (lambda_{i+1} - lambda_i) / (K - i)

    Eq. (7):  CSG = sum(cummax(delta))

    Parameters
    ----------
    eigenvalues : array-like - Autovalores ordenados del Laplaciano.

    Returns
    -------
    csg : float - Valor final de CSG.

    details : pd.DataFrame
        Tabla con eigengaps, eigengaps normalizados y máximo acumulado.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    K = len(eigenvalues)
    if K < 2:
        raise ValueError("Se necesitan al menos dos clases para calcular CSG.")

    # Eigengaps: lambda_{i+1} - lambda_i
    gaps = np.diff(eigenvalues)

    # Índices: i = 0, 1, 2, ..., K-1, siguiendo el paper
    i = np.arange(K - 1)

    # Eq. (6): normalized_gap_i =   gap_i / (K - i)
    normalized_gaps = gaps / (K - i)

    # Máximo acumulado
    cumulative_max = np.maximum.accumulate(normalized_gaps)

    # Eq. (7)
    csg = cumulative_max.sum()

    details = pd.DataFrame({
        "i": i,
        "lambda_i": eigenvalues[:-1],
        "lambda_i+1": eigenvalues[1:],
        "eigengap": gaps,
        "K-i": K - i,
        "normalized_eigengap": normalized_gaps,
        "cummax": cumulative_max
    })

    return csg, details


def compute_csg(X, y, M=100, k=3, random_state=0):
    """
    Calcula toda la medida CSG siguiendo el pipeline:

        X
        -> S
        -> W
        -> D
        -> L
        -> eigenvalues
        -> normalized eigengaps
        -> cummax
        -> CSG

    Parameters
    ----------
    X : array-like, shape (N, d) - Embeddings phi(x).
    y : array-like, shape (N,) - Etiquetas.
    M : int, default=100 - Número de muestras Monte Carlo por clase.
    k : int, default=3 - Número de vecinos para la estimación de densidad.
    random_state : int - Semilla aleatoria.

    Returns
    -------
    result : dict
        Diccionario que contiene todos los resultados intermedios y el CSG final.
    """

    # Matriz de similitud Monte Carlo S
    S = compute_S(X=X,y=y,M=M,k=k,random_state=random_state)
    # Matriz de adyacencia simétrica W
    W = compute_W(S)
    # Laplaciano L = D - W
    D, L = compute_laplacian(W)
    # Espectro del Laplaciano
    eigenvalues = compute_spectrum(L)
    # CSG
    csg, spectral_details = compute_csg_from_eigenvalues(eigenvalues)

    # Devolvemos todas las cosas
    result = {"CSG": csg,"S": S,"W": W,"D": D,"L": L,
        "eigenvalues": eigenvalues,"spectral_details": spectral_details}

    return result



# Semilla para obtener siempre los mismos datos
rng = np.random.default_rng(42)

# ---------------------------------------------------------
# Parámetros
# ---------------------------------------------------------
# Número de puntos por clase
n = 200
# Dimensionalidad del embedding
d = 2

# Centros de las tres clases
centro_0 = np.array([0.0, 0.0])
centro_1 = np.array([3.0, 0.5])
centro_2 = np.array([1.5, 3.0])

# Desviación típica.
# Cuanto mayor sea, mayor será el solapamiento entre clases.
sigma = 1.0

# ---------------------------------------------------------
# Generamos las tres clases
# ---------------------------------------------------------
X0 = rng.normal(loc=centro_0,scale=sigma,size=(n, d))
X1 = rng.normal(loc=centro_1,scale=sigma,size=(n, d))
X2 = rng.normal(loc=centro_2,scale=sigma,size=(n, d))

# ---------------------------------------------------------
# Unimos todos los puntos en una única matriz X_emb
# ---------------------------------------------------------
X_emb = np.vstack([X0,X1,X2])

# ---------------------------------------------------------
# Creamos las etiquetas
# ---------------------------------------------------------
y = np.concatenate([np.zeros(n, dtype=int),np.ones(n, dtype=int),np.full(n, 2, dtype=int)])

print("Forma de X_emb:", X_emb.shape)
print("Forma de y:", y.shape)
print("Clases:", np.unique(y))


result = compute_csg(X=X_emb,y=y,M=100,k=3,random_state=42)

print("CSG =", result["CSG"])