import os
import csv

import pandas as pd
from similaritymeasures import similaritymeasures
from tqdm import tqdm

from aggregator import ManeuverDataAggregator
from comparasion import calculate_initial_vector, calculate_coordinates

aggregator = ManeuverDataAggregator()


def calculate(df) -> list:
    df = aggregator.transform(df)
    initial_vector = calculate_initial_vector(df)
    origin_df = df[['x', 'y']]
    df = df[['speed', 'gr_x', 'gr_y', 'gr_z']]
    df = calculate_coordinates(df, initial_vector)

    df_coord = df['coordinate'].apply(lambda r: (r[0], r[1])).values.tolist()
    origin_coord = list(zip(origin_df.x, origin_df.y))

    exp_data = df_coord
    num_data = origin_coord

    df = similaritymeasures.frechet_dist(exp_data, num_data)
    return [df]


def walk_data_and_calculate(driver_id: str, calculation_func, writer):
    raw_data_folder = f"./drivers/{driver_id}/maneuver_data/raw_data"
    for root, dirs, file_names in os.walk(raw_data_folder):
        for file_name in tqdm(file_names):
            try:
                event_df = pd.read_csv(f"{raw_data_folder}/{file_name}", index_col="index")
                event_df = event_df[event_df["speed"] > 10]
                if event_df.shape[0] < 10:
                    continue
                result = calculation_func(event_df)
                full_result_row = [driver_id, file_name.replace('.csv', '')] + result
                writer.writerow(full_result_row)
            except Exception as ex:
                print(f"Error with file {file_name}")
                print(f"{ex}")


if __name__ == '__main__':
    ids = []
    with open(f"./results/result_{','.join(ids)}.csv", "w+") as file:
        writer = csv.writer(file, delimiter=',')
        for driver_id in ids:
            walk_data_and_calculate(driver_id, calculate, writer)
