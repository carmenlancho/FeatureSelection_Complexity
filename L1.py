from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
def L1_HD(X, y):
    # only for binary classification problems
    y_pos_neg = np.where(y == 0, -1, 1)
    y_pos_neg = y_pos_neg.ravel()
    n = len(X)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    # clf = GridSearchCV(SVC(kernel='linear'), param_grid, scoring='accuracy')
    clf = GridSearchCV(LinearSVC(random_state=0,max_iter=100000), param_grid, scoring='accuracy')
    clf = clf.fit(X, y_pos_neg)
    svc = clf.best_estimator_.fit(X, y_pos_neg)  # SVM lineal
    dec_bound = svc.decision_function(X)  # decision boundary
    w_norm = np.linalg.norm(svc.coef_)
    dist = dec_bound / w_norm  # distancia al decision boundary
    # Nos quedamos con las distancias de los puntos misclassified
    pred = svc.predict(X)
    # indices = [i for i in range(len(y_pos_neg)) if y_pos_neg[i] != pred[i]] # los misclassified
    L1_HD = dist * y_pos_neg  # negativo en los casos misclassified y positivo en el resto

    return L1_HD
