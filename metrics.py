from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import csv


def print_metrics(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_metrics_csv(y_test, y_pred, file_path, classifier_name, task_name, random_state):
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([task_name, random_state, classifier_name, accuracy, balanced_accuracy,f1])

    print(f"Metrics appended to {file_path}")


def write_in_csv(file_path, x):

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(x)
        writer.writerow([])





