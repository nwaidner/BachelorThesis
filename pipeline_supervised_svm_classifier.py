from sklearn.svm import SVC


def svm_classifier(X_train, X_test, y_train, y_test, task_name, random_state):

    # Train/ Test
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    name = svm_classifier.__name__

    return y_test, y_pred, name, task_name, random_state






