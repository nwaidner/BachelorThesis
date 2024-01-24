from pipeline_supervised_knn import knn
from pipeline_supervised_rf import random_forest
from pipeline_supervised_svm import svm
from pipeline_unsupervised_autoencoder import autoencoder_model
from pipeline_unsupervised_isolation_forest import isolation_forest
from pipeline_unsupervised_k_means import k_means
from data_import import get_merged_df_from_file
import data_preperation as prep
from Metric_s import save_metrics_csv
import concurrent.futures
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Standard df
df = get_merged_df_from_file()

total_samples_df = len(df)
class_distribution_df = df['Defect'].value_counts()

# Split By unit
unit_1, unit_2 = prep.split_by_unit(df)

total_samples_unit1_mit_WL = len(unit_1)
class_distribution_unit1_mit_WL = unit_1['Defect'].value_counts()

total_samples_unit2_mit_WL = len(unit_2)
class_distribution_unit2_mit_WL = unit_2['Defect'].value_counts()

#Cut wirkleistung unit 1
unit_1_ohne_WL = prep.drop_wirkleistung_umrichter(unit_1)

total_samples_unit1_ohne_WL = len(unit_1_ohne_WL)
class_distribution_unit1_ohne_WL = unit_1_ohne_WL['Defect'].value_counts()

#Cut wirkleistung unit 2
unit_2_ohne_WL = prep.drop_wirkleistung_umrichter(unit_2)

total_samples_unit2_ohne_WL = len(unit_2_ohne_WL)
class_distribution_unit2_ohne_WL = unit_2_ohne_WL['Defect'].value_counts()

#Split by TT Unit 1, mit WL
X_u1_mitWL, y_u1_mitWL = prep.split_by_x_y(unit_1)
X_train_u1_mitWL, X_test_u1_mitWL, y_train_u1_mitWL, y_test_u1_mitWL = (train_test_split(X_u1_mitWL, y_u1_mitWL, test_size=0.2, random_state=42))

total_samples_unit1_mit_WL_Train = len(y_train_u1_mitWL)
class_distribution_unit1_mit_WL_Train = y_train_u1_mitWL.value_counts()

total_samples_unit1_mit_WL_Test = len(y_test_u1_mitWL)
class_distribution_unit1_mit_WL_Test = y_test_u1_mitWL.value_counts()

#Split by TT Unit 1, ohne WL
X_u1_ohneWL, y_u1_ohneWL = prep.split_by_x_y(unit_1_ohne_WL)
X_train_u1_ohneWL, X_test_u1_ohneWL, y_train_u1_ohneWL, y_test_u1_ohneWL = (train_test_split(X_u1_ohneWL, y_u1_ohneWL, test_size=0.2, random_state=42))

total_samples_unit1_ohne_WL_Train = len(y_train_u1_ohneWL)
class_distribution_unit1_ohne_WL_Train = y_train_u1_ohneWL.value_counts()

total_samples_unit1_ohne_WL_Test = len(y_test_u1_ohneWL)
class_distribution_unit1_ohne_WL_Test = y_test_u1_ohneWL.value_counts()

#Split by TT Unit 2, mit WL
X_u2_mitWL, y_u2_mitWL = prep.split_by_x_y(unit_2)
X_train_u2_mitWL, X_test_u2_mitWL, y_train_u2_mitWL, y_test_u2_mitWL = (train_test_split(X_u2_mitWL, y_u2_mitWL, test_size=0.2, random_state=42))

total_samples_unit2_mit_WL_Train = len(y_train_u2_mitWL)
class_distribution_unit2_mit_WL_Train = y_train_u2_mitWL.value_counts()

total_samples_unit2_mit_WL_Test = len(y_test_u2_mitWL)
class_distribution_unit2_mit_WL_Test = y_test_u2_mitWL.value_counts()

#Split by TT Unit 2, ohne WL
X_u2_ohneWL, y_u2_ohneWL = prep.split_by_x_y(unit_2_ohne_WL)
X_train_u2_ohneWL, X_test_u2_ohneWL, y_train_u2_ohneWL, y_test_u2_ohneWL = (train_test_split(X_u2_ohneWL, y_u2_ohneWL, test_size=0.2, random_state=42))

total_samples_unit2_ohne_WL_Train = len(y_train_u2_ohneWL)
class_distribution_unit2_ohne_WL_Train = y_train_u2_ohneWL.value_counts()

total_samples_unit2_ohne_WL_Test = len(y_test_u2_ohneWL)
class_distribution_unit2_ohne_WL_Test = y_test_u2_ohneWL.value_counts()


# Ausgabe
print('unit2 ohne WL')

print('train')
print(total_samples_unit2_ohne_WL_Train)
print(class_distribution_unit2_ohne_WL_Train)

print('test')
print(total_samples_unit2_ohne_WL_Test)
print(class_distribution_unit2_ohne_WL_Test)

