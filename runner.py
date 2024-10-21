import math
import os

import pandas as pd

from aggregators.maneuver_data_aggregator_v3 import DbV3DriverManeuverDataAggregator
from pipeline_runner.visualization import visualize


def read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def aggregate(maneuver_data_aggregator, original_df, path) -> pd.DataFrame:
    aggregated_df = maneuver_data_aggregator._transform(original_df)
    aggregated_df = clear_data(aggregated_df)
    aggregated_df['ac'] = aggregated_df['ac_x'] * aggregated_df['ac_x'] + aggregated_df['ac_y'] * aggregated_df['ac_y']
    file_path = f"resources/pipeline/transformed/{path}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    aggregated_df.to_csv(file_path, index_label='index')
    return aggregated_df


def clear_data_by_5_quantile(df, column_name):
    # Вычисляем 5-й и 95-й квантиль
    lower_quantile = df[column_name].quantile(0.05)
    upper_quantile = df[column_name].quantile(0.95)

    # Фильтруем данные, оставляя только те значения, которые находятся между квантилями
    return df[(df[column_name] >= lower_quantile) & (df[column_name] <= upper_quantile)]


def clear_data(df):
    columns = ['speed', 'gr_z', 'ac_x']
    for column in columns:
        df = clear_data_by_5_quantile(df, column)
    df = DbV3DriverManeuverDataAggregator.interpolate_df(df, df.columns.tolist())
    df['timestamp'] = df['datetime'] // 1000
    return df


def run():
    maneuver_data_aggregator = DbV3DriverManeuverDataAggregator()  #
    # simple_aggregator = DbDriverManeuverDataAggregator(id=2000)  # тут просто разворачиваем массив данных гироскопа в 3 колонки
    # paths = ['/data/maneuver_template/raw/5257-no_maneuver_1659018130487.csv']
    paths = ['pure_stat_event_new_202410212247.csv']
    for path in paths:
        original_df = read(f"/Users/vvshu/PycharmProjects/gps2gyro/{path}")
        original_df.sort_values(by='datetime', inplace=True)
        original_df.reset_index(inplace=True)

        aggregated_df = aggregate(maneuver_data_aggregator, original_df, path)

        visualize(original_df, aggregated_df, path)


if __name__ == '__main__':
    run()
