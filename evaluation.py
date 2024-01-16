from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score


def print_nan_percentages_df_columns(df):
    nan_percentage = (df.isna().mean() * 100).round(2)

    for column, percentage in nan_percentage.items():
        print(f"Percentage of NaN values in '{column}': {percentage}%")


def eval_model(y_test, y_pred):
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Balanced Accuracy: {bal_acc}")


def feature_imp(clf, X):
    feature_importances = clf.feature_importances_
    feature_names = X.columns
    feature_importance_list = list(zip(feature_names, feature_importances))
    feature_importance_list.sort(key=lambda x: x[1], reverse=True)
    print("Feature Importances:")
    for feature, importance in feature_importance_list:
        print(f"Feature: {feature}, Importance: {importance}")