import math
import os

import pandas as pd
import numpy as np
from numpy import cos, sin, arcsin, sqrt

from aggregators.angles.angles import get_angles
from aggregators.interpolation import interpolate_data


class DbV3DriverManeuverDataAggregator:
    @staticmethod
    def interpolate_df(df, columns=None):
        if columns is None:
            columns = ['x', 'y', 'z', 'speed', 'latitude', 'longitude']
        start_dt = df.sort_values(by='datetime', ascending=True)['datetime'].iloc[0]
        end_dt = df.sort_values(by='datetime', ascending=True)['datetime'].iloc[-1]
        dts = df['datetime'].values.tolist()
        new_dts = list(range(start_dt // 100 * 100 + 100, end_dt // 100 * 100 + 100, 100))
        new_df = pd.DataFrame()
        for column in columns:
            data = df[column].values.tolist()
            new_data = interpolate_data(dts, data, new_dts)
            new_df[column] = new_data
        new_df['datetime'] = new_dts
        return new_df

    @staticmethod
    def dist(start_lat, start_lon, curr_lat, curr_lon):
        start_lat, start_lon, curr_lat, curr_lon = map(np.deg2rad, [start_lat, start_lon, curr_lat, curr_lon])
        # todo work only above the equator
        sign = 1
        if start_lat == curr_lat:
            sign = 1 if curr_lon > start_lon else -1
        if start_lon == curr_lon:
            sign = 1 if curr_lat > start_lat else -1

        dlon = curr_lon - start_lon
        dlat = curr_lat - start_lat
        a = sin(dlat / 2) ** 2 + cos(start_lat) * cos(curr_lat) * sin(dlon / 2) ** 2
        c = 2 * arcsin(sqrt(a))
        dist = 6367000 * c
        return sign * dist

    def _transform(self, df):
        df = df[['latitude', 'longitude', 'speed', 'datetime']].sort_values(by='datetime')

        df = df.drop_duplicates(subset=['latitude', 'longitude'], keep='first')

        start_lat, start_lon = df[['latitude', 'longitude']].iloc[0, :]
        coordinates = df[['latitude', 'longitude']].apply(
            lambda r: (self.dist(start_lat, start_lon, r['latitude'], start_lon),
                       self.dist(start_lat, start_lon, start_lat, r['longitude'])),
            result_type='expand',
            axis='columns'
        )
        coordinates.columns = ['y', 'x']
        # coordinates['z'] = [random.uniform(0.05, 0.2) for x in range(coordinates.shape[0])] # todo return
        coordinates['z'] = 0
        df = pd.concat([df, coordinates], axis=1, join="inner")
        # df.to_csv(f"{self.data_folder}/maneuver_data/log/log_coordinates.csv", index_label='index')
        file_path = "resources/pipeline/pretransformed/data/maneuver_template/raw/5257-turn_left_1659018130487_extended.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index_label='index')

        df = self.interpolate_df(df)

        # df.to_csv(f"{self.data_folder}/maneuver_data/log/log_coordinates_interpolated.csv", index_label='index')

        coordinates_delta = df.join(
            df.shift(-1), how='left', rsuffix='-next'
        ).apply(
            lambda x: (x['x-next'] - x['x'], x['y-next'] - x['y'], x['z-next'] - x['z']),
            result_type='expand',
            axis='columns'
        )
        coordinates_delta.columns = ['dx', 'dy', 'dz']
        # log_df = pd.concat([df, coordinates_delta], axis=1, join="inner")  # todo delete
        # log_df.to_csv(f"{self.data_folder}/maneuver_data/in_transform_0.csv", index_label='index')  # todo delete

        df = pd.concat([df, coordinates_delta], axis=1, join="inner")

        # df.to_csv(f"{self.data_folder}/maneuver_data/log/log_coordinates_delta.csv", index_label='index')

        # import matplotlib.pyplot as plt
        # plt.plot(df['dx'].values.tolist(), df['dy'].values.tolist())
        # plt.savefig(f"{self.data_folder}/maneuver_data/in_transform_0.png", bbox_inches='tight')

        df["dt"] = df["datetime"].to_frame().join(df["datetime"].shift(-1), how='left', rsuffix='-next').apply(
            lambda r: (r['datetime-next'] - r['datetime']) / 1000,
            result_type='expand',
            axis='columns'
        )
        # df.to_csv(f"{self.data_folder}/maneuver_data/with_coordinates/1651248590700.csv", index_label='index')

        speeds = df[['dx', 'dy', 'dz', "dt"]].apply(
            lambda r: (r["dx"] / r["dt"], r["dy"] / r["dt"], r["dz"] / r["dt"]),
            result_type='expand',
            axis='columns'
        )

        speeds.columns = ['Vx', 'Vy', 'Vz']
        df = pd.concat([df, speeds], axis=1, join="inner")
        speeds_delta = speeds.join(
            speeds.shift(-1), how='left', rsuffix='-next'
        ).apply(
            lambda r: (r['Vx-next'] - r['Vx'], r['Vy-next'] - r['Vy'], r['Vz-next'] - r['Vz']),
            result_type='expand',
            axis='columns'
        )
        speeds_delta.columns = ['dVx', 'dVy', 'dVz']
        df = pd.concat([df, speeds_delta], axis=1, join="inner")
        # df.to_csv(f"{self.data_folder}/maneuver_data/in_transform_10.csv", index_label='index')  # todo delete

        df['ac_x'] = df.apply(lambda r: r["dVx"] / r["dt"], axis='columns')
        df['ac_y'] = df.apply(lambda r: r["dVy"] / r["dt"], axis='columns')
        df['ac_z'] = df.apply(lambda r: r["dVz"] / r["dt"] - 9.81, axis='columns')

        angles_df = pd.DataFrame(get_angles(df[["x", "y", "z"]].values.tolist()), columns=["dux", "duy", "duz"])
        # angles_df.to_csv(f"{self.data_folder}/maneuver_data/angles_df_in_transform.csv",
        #                  index_label='index')  # todo delete
        df[["dux", "duy", "duz"]] = angles_df
        # df.to_csv(f"{self.data_folder}/maneuver_data/in_transform_11.csv", index_label='index')  # todo delete
        df['gr_x'] = df.apply(lambda r: r["dux"] / r["dt"], axis='columns')
        df['gr_y'] = df.apply(lambda r: r["duy"] / r["dt"], axis='columns')
        df['gr_z'] = df.apply(lambda r: r["duz"] / r["dt"], axis='columns')
        df['timestamp'] = df['datetime'].apply(lambda x: int(x) // 1000)

        df["speed in ms"] = df.apply(lambda r: r["speed"] * 1000 / 3600, axis='columns')
        df["sqrt(square of speeds components)"] = df.apply(lambda r: math.sqrt(r["dVx"] * r["dVx"] + r["dVy"] * r["dVy"] + r["dVz"] * r["dVz"]), axis='columns')
        # df.to_csv(f"{self.data_folder}/maneuver_data/log/log_final.csv", index_label='index')

        df = df[['timestamp', 'datetime', 'ac_x', 'ac_y', 'ac_z', 'gr_x', 'gr_y', 'gr_z', 'speed', 'x', 'y', 'latitude', 'longitude']]
        df.dropna(inplace=True)
        return df
