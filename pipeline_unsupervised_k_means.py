from sklearn.cluster import KMeans


def k_means_classifier(X_train, X_test, y_train, y_test, task_name, random_state):

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_train)

    distances = kmeans.predict(X_test)

    y_pred = distances > 1.5

    name = k_means_classifier.__name__

    return y_test, y_pred, name, task_name, random_state

