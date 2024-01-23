from pipeline_supervised_knn import knn
from pipeline_supervised_rf import random_forest
from pipeline_supervised_svm import svm
from pipeline_unsupervised_autoencoder import autoencoder
from pipeline_unsupervised_isolation_forest import isolation_forest
from pipeline_unsupervised_k_means import k_means
from data_import import get_merged_df_from_file
from date_preperation import prep
from Metric_s import save_metrics_csv
import concurrent.futures


def test_each_pipeline(X_train_base, X_test_base, y_train_base, y_test_base, task_name, random_state_base, file_path):
    classifiers = [
        #knn,
        #svm,
        autoencoder,
        #k_means,
        isolation_forest,
        #random_forest
    ]

    def run_classifier(classifier):
        y_test, y_pred, name, _, random_state = classifier(X_train_base, X_test_base, y_train_base, y_test_base,
                                                           task_name, random_state_base)
        save_metrics_csv(y_test, y_pred, file_path, name, task_name, random_state)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_classifier, classifiers)


def test_each_scenario(df, random_state, file_path):
    scenarios = [
        ("RANDOM FILES", ['-', '-', '-', '-', '-', '-', '-', '-']),
        ("OHNE SENSORDEFECT 1 IN TRAININGSDATEN", ['sensordefect 1', '-', '-', '-', '-', '-', '-', '-']),
        ("OHNE SENSORDEFECT 2 IN TRAININGSDATEN", ['sensordefect 2', '-', '-', '-', '-', '-', '-', '-']),
        ("OHNE SENSORDEFECT 3 IN TRAININGSDATEN", ['sensordefect 3', '-', '-', '-', '-', '-', '-', '-']),
        ("OHNE SENSORDEFECT 4 IN TRAININGSDATEN", ['sensordefect 4', '-', '-', '-', '-', '-', '-', '-']),
        ("OHNE SENSORDEFECT 1 IN TRAININGSDATEN x TEST NUR SENSORDEFECT 1",
         ['sensordefect 1', '-', '-', '-', 'sensordefect 2', 'sensordefect 3', 'sensordefect 4', 'kein defekt']),
    ]

    def execute_task(task):
        task_name, task_params = task
        X_train, X_test, y_train, y_test = prep(input_df=df, random_state=random_state,
                                                remove_from_train_1=task_params[0],
                                                remove_from_train_2=task_params[1],
                                                remove_from_train_3=task_params[2],
                                                remove_from_train_4=task_params[3],
                                                remove_from_test_1=task_params[4],
                                                remove_from_test_2=task_params[5],
                                                remove_from_test_3=task_params[6],
                                                remove_from_test_4=task_params[7])
        test_each_pipeline(X_train, X_test, y_train, y_test, task_name=task_name, random_state_base=random_state, file_path=file_path)

    with  concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(execute_task, scenarios)


df = get_merged_df_from_file()
test_each_scenario(df=df, random_state=42, file_path='final_metrics_relu.csv')



