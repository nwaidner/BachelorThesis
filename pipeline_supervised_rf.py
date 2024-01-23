from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, X_test, y_train, y_test, task_name, random_state):

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    name = random_forest.__name__


    return y_test, y_pred, name, task_name, random_state