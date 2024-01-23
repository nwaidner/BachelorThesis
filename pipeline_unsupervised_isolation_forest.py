from sklearn.ensemble import IsolationForest


def isolation_forest(X_train, X_test, y_train, y_test, task_name, random_state):

    clf = IsolationForest(contamination=0.04820708799148221, random_state=42)
    clf.fit(X_train)

    test_anomaly_scores = clf.decision_function(X_test)

    threshold = 0.07

    y_pred = test_anomaly_scores < threshold

    name = isolation_forest.__name__

    return y_test, y_pred, name, task_name, random_state
