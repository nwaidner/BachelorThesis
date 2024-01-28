from sklearn.neighbors import KNeighborsClassifier


def knn(X_train, X_test, y_train, y_test, task_name, random_state):

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    name = knn.__name__

    return y_test, y_pred, name, task_name, random_state
