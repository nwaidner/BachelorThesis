from sklearn.cluster import KMeans
import numpy as np


def k_means(X_train, X_test, y_train, y_test, task_name, random_state):

    mask = ~y_train
    X_train_cleaned = X_train[mask]
    y_train_cleaned = y_train[mask]

    kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    kmeans.fit(X_train_cleaned)

    distances = np.min(kmeans.transform(X_test), axis=1)

    threshold = np.percentile(distances, 60)

    y_pred_anomaly = distances > threshold

    name = k_means.__name__

    return y_test, y_pred_anomaly, name, task_name, random_state



