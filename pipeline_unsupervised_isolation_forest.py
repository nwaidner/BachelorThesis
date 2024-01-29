from sklearn.ensemble import IsolationForest
import numpy as np


def isolation_forest(X_train, X_test, y_train, y_test, task_name, random_state):

    contamination = np.sum(y_train == True) / len(y_train)

    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X_train)

    test_anomaly_scores = clf.decision_function(X_test)

    threshold = 0.07

    y_pred = test_anomaly_scores < threshold

    name = isolation_forest.__name__

    return y_test, y_pred, name, task_name, random_state
