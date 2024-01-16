import pandas as pd
from datetime import datetime


def merge_df_and_cases_save_as_file(df, cases, filepath):
    class TimeFrame:
        def __init__(self, startzeit, endzeit):
            self.startzeit = startzeit
            self.endzeit = endzeit

        def in_Zeitraum(self, timestamp):
            if self.startzeit <= timestamp <= self.endzeit:
                return True
            else:
                return False

    def _periods_to_timestamp_list(timestamps_all, periods):

        timeframes = []

        for period in periods:
            start = datetime.strptime(period['start'], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(period['end'], "%Y-%m-%d %H:%M:%S")
            timeframe = TimeFrame(start, end)
            timeframes.append(timeframe)

        timestamps_new = []
        for timeframe in timeframes:
            for timestamp in timestamps_all:
                if timeframe.in_Zeitraum(timestamp):
                    timestamps_new.append(timestamp)

        return timestamps_new

    timestamps = df['Zeitstempel']

    i = 0
    for index, row in cases.iterrows():
        current_time = datetime.now()
        i = i + 1
        print(f'started case {i} at {current_time}')
        ts_list = _periods_to_timestamp_list(timestamps_all=timestamps, periods=row['periods'])
        for timestamp in ts_list:
            neue_zeile = {'Unit': row['unit'], 'Defect': row['defect_type'], 'Zeitstempel': timestamp}
            if 'expanded_cases' not in locals():
                expanded_cases = pd.DataFrame(neue_zeile, index=[0])
            else:
                expanded_cases = pd.concat([expanded_cases, pd.DataFrame(neue_zeile, index=[0])], ignore_index=True)

        current_time = datetime.now()
        print(f'finished case {i} at {current_time}')

    df_merged = df.merge(expanded_cases, on=['Zeitstempel', 'Unit'], how='left')

    df_merged.sort_values(by=['Unit', 'Zeitstempel'], ascending=True, inplace=True)

    df_merged.to_parquet(filepath, engine='pyarrow')

