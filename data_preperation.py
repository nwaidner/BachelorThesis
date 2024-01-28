import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_df_by_min_max_sta_mit(df):
    minimum = df.loc[:, df.columns.str.contains('Min, FC00') | df.columns.str.contains('Unit') | df.columns.str.contains('Zeitstempel')]
    maximum = df.loc[:, df.columns.str.contains('Max, FD00') | df.columns.str.contains('Unit') | df.columns.str.contains('Zeitstempel')]
    standardabweichung = df.loc[:, df.columns.str.contains('StAw, FE00') | df.columns.str.contains('Unit') | df.columns.str.contains('Zeitstempel')]
    mittelwert = df.loc[:, df.columns.str.contains('Mittelw, ') | df.columns.str.contains('Unit') | df.columns.str.contains('Zeitstempel')]

    return minimum, maximum, standardabweichung, mittelwert


def split_by_unit(input_df):
    grouped = input_df.groupby('Unit')
    unit_1 = grouped.get_group('unit 1').copy()
    unit_2 = grouped.get_group('unit 2').copy()

    return unit_1, unit_2


def split_by_x_y(input_df):
    y = input_df[['Defect']]
    X = input_df.drop(['Unit', 'Zeitstempel', 'Defect'], axis=1, inplace=False)

    return X, y


def y_to_bool(y_train, y_test):
    y_train['Defect'] = y_train['Defect'].notna()
    y_test['Defect'] = y_test['Defect'].notna()

    return y_train, y_test


def drop_column_with_high_nan(df, threshold):
    missing_percentage = (df.isnull().mean() * 100)
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)

    return df_cleaned


def y_to_1d_array(y_train, y_test):
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    return y_train, y_test


def standardise(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    return X_train, X_test


def lineare_interpolation_impute(X_train, X_test):
    x_train = drop_column_with_high_nan(X_train, 20)
    x_test = drop_column_with_high_nan(X_test, 20)

    x_train_imputed = pd.DataFrame(x_train).interpolate(method='linear', axis=0).values

    x_test_imputed = pd.DataFrame(x_test).interpolate(method='linear', axis=0).values

    return x_train_imputed, x_test_imputed


def remove_label(X, y, label_to_remove):
    if label_to_remove == '-':
        return X, y
    y_forbidden_indices = y.index[y['Defect'] == label_to_remove].tolist()
    X.drop(y_forbidden_indices, inplace=True)
    y.drop(y_forbidden_indices, inplace=True)

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    return X, y


def drop_wirkleistung_umrichter(df):
    df_copy = df.copy()  # Create a copy of the DataFrame

    df_copy = df_copy[df_copy['Wirkleistung Umrichter (Min, FC001)'] >= 20]
    df_copy.drop('Wirkleistung Umrichter (Min, FC001)', axis=1, inplace=True)

    return df_copy


def prep(input_df, random_state, remove_from_train_1, remove_from_train_2, remove_from_train_3, remove_from_train_4, remove_from_test_1, remove_from_test_2, remove_from_test_3, remove_from_test_4):

    unit_1, unit_2 = split_by_unit(input_df)

    df = unit_1

    df = drop_wirkleistung_umrichter(df)

    X, y = split_by_x_y(df)

    X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.2, random_state=random_state))

    X_train, y_train = remove_label(X_train, y_train, remove_from_train_1)
    X_train, y_train = remove_label(X_train, y_train, remove_from_train_2)
    X_train, y_train = remove_label(X_train, y_train, remove_from_train_3)
    X_train, y_train = remove_label(X_train, y_train, remove_from_train_4)

    X_test, y_test = remove_label(X_test, y_test, remove_from_test_1)
    X_test, y_test = remove_label(X_test, y_test, remove_from_test_2)
    X_test, y_test = remove_label(X_test, y_test, remove_from_test_3)
    X_test, y_test = remove_label(X_test, y_test, remove_from_test_4)

    y_train, y_test = y_to_bool(y_train, y_test)

    y_train, y_test = y_to_1d_array(y_train, y_test)

    X_train, X_test = standardise(X_train, X_test)

    X_train, X_test = lineare_interpolation_impute(X_train, X_test)

    return X_train, X_test, y_train, y_test
