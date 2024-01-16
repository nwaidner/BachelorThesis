import pandas as pd

FILE_PATH_CASES = "/Users/nickwaidner/Desktop/Thesis_NewData/Data/BA_nick_cases.parquet"
FILE_PATH_RAW_DF = "/Users/nickwaidner/Desktop/Thesis_NewData/Data/BA_nick_df.parquet"
FILE_PATH_MERGED_DF = "/Users/nickwaidner/Desktop/Thesis/Projekt/Data/final_merged.parquet"


def get_raw_df_from_file():
    return pd.read_parquet(FILE_PATH_RAW_DF, engine='pyarrow')


def get_cases_df_from_file():
    return pd.read_parquet(FILE_PATH_CASES, engine='pyarrow')


def get_merged_df_from_file():
    return pd.read_parquet(FILE_PATH_MERGED_DF, engine='pyarrow')
