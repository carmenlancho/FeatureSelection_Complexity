###########################################################################
#########          FUNCION PARA CREAR NORMALES BIVARIANTES        #########
###########################################################################
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X_neg = mn0.rvs(size=n0, random_state=seed1)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X_pos = mn1.rvs(size=n1, random_state=seed2)

    X = np.vstack((X_neg, X_pos))
    y = np.array([0] * len(X_neg) + [1] * len(X_pos))

    data = pd.DataFrame(X,columns=['x','y'])

    # Plot
    # For labels
    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=70, c='#040082', marker=".", label='negative')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=70, c='C1', marker="+", label='positive')
    #for i, txt in enumerate(labels):
    #    plt.annotate(txt, (data.iloc[i, 0], data.iloc[i, 1]))
    # plt.axis('off') # no sale ni el recuadro
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.ylabel('y')
    # plt.xlabel('x')
    plt.show()

    return X, y



def normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X0 = mn0.rvs(size=n0, random_state=seed0)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X1 = mn1.rvs(size=n1, random_state=seed1)

    mn2 = multivariate_normal(mean=mu2, cov=sigma2)
    X2 = mn2.rvs(size=n2, random_state=seed2)

    X = np.vstack((X0, X1, X2))
    y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2))

    data = pd.DataFrame(X, columns=['x', 'y'])

    # Plot

    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    idx_2 = np.where(y == 2)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=30, c='#040082', marker=".", label='0')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=30, c='C1', marker="+", label='1')
    plt.scatter(data.iloc[idx_2].x, data.iloc[idx_2].y, s=30, c='#96DBF2', marker="*", label='2')

    plt.xticks([])
    plt.yticks([])
    plt.show()

    return X, y





def normal_generator4(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, mu3, sigma3, n3, seed0, seed1, seed2, seed3):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X0 = mn0.rvs(size=n0, random_state=seed0)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X1 = mn1.rvs(size=n1, random_state=seed1)

    mn2 = multivariate_normal(mean=mu2, cov=sigma2)
    X2 = mn2.rvs(size=n2, random_state=seed2)

    mn3 = multivariate_normal(mean=mu3, cov=sigma3)
    X3 = mn3.rvs(size=n3, random_state=seed3)

    X = np.vstack((X0, X1, X2, X3))
    y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2) + [3] * len(X3))

    data = pd.DataFrame(X, columns=['x', 'y'])

    # Plot

    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    idx_2 = np.where(y == 2)
    idx_3 = np.where(y == 3)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=10, c='k', marker=".", label='0')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=10, c='c', marker="+", label='1')
    plt.scatter(data.iloc[idx_2].x, data.iloc[idx_2].y, s=10, c='purple', marker="*", label='2')
    plt.scatter(data.iloc[idx_3].x, data.iloc[idx_3].y, s=10, c='orange', marker="h", label='3')

    plt.xticks([])
    plt.yticks([])
    plt.show()

    return X, y





def normal_generator5(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, mu3, sigma3, n3, mu4, sigma4, n4, seed0, seed1, seed2, seed3, seed4):
    mn0 = multivariate_normal(mean=mu0, cov=sigma0)
    X0 = mn0.rvs(size=n0, random_state=seed0)

    mn1 = multivariate_normal(mean=mu1, cov=sigma1)
    X1 = mn1.rvs(size=n1, random_state=seed1)

    mn2 = multivariate_normal(mean=mu2, cov=sigma2)
    X2 = mn2.rvs(size=n2, random_state=seed2)

    mn3 = multivariate_normal(mean=mu3, cov=sigma3)
    X3 = mn3.rvs(size=n3, random_state=seed3)

    mn4 = multivariate_normal(mean=mu4, cov=sigma4)
    X4 = mn4.rvs(size=n4, random_state=seed4)

    X = np.vstack((X0, X1, X2, X3, X4))
    y = np.array([0] * len(X0) + [1] * len(X1) + [2] * len(X2) + [3] * len(X3) + [4] * len(X4))

    data = pd.DataFrame(X, columns=['x', 'y'])

    # Plot

    labels = list(data.index)
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    idx_2 = np.where(y == 2)
    idx_3 = np.where(y == 3)
    idx_4 = np.where(y == 4)
    plt.scatter(data.iloc[idx_0].x, data.iloc[idx_0].y, s=10, c='k', marker=".", label='0')
    plt.scatter(data.iloc[idx_1].x, data.iloc[idx_1].y, s=10, c='c', marker="+", label='1')
    plt.scatter(data.iloc[idx_2].x, data.iloc[idx_2].y, s=10, c='purple', marker="*", label='2')
    plt.scatter(data.iloc[idx_3].x, data.iloc[idx_3].y, s=10, c='orange', marker="h", label='3')
    plt.scatter(data.iloc[idx_4].x, data.iloc[idx_4].y, s=10, c='crimson', marker="d", label='3')

    plt.xticks([])
    plt.yticks([])
    plt.show()

    return X, y



