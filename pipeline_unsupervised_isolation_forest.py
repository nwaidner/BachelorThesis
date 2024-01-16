from sklearn.ensemble import IsolationForest


def isolation_forest_classifier(X_train, X_test, y_train, y_test, task_name, random_state):

    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_train)

    # Anomaliescores für die Testdaten berechnen
    test_anomaly_scores = clf.decision_function(X_test)

    # Schwellwert
    threshold = 0.0572

    y_pred = test_anomaly_scores < threshold

    name = isolation_forest_classifier.__name__

    # Auswerten
    return y_test, y_pred, name, task_name, random_state




'''for threshold in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    predictions = test_anomaly_scores < threshold

    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    print(f'Threshold: {threshold}, Balanced Accuracy: {balanced_accuracy:.4f}')
'''