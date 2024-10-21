import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import plotly.express as px


def _calculate_initial_vector(df):
    first = df[['x', 'y']].iloc[0].to_numpy()
    second = df[['x', 'y']].iloc[1].to_numpy()

    return [second[0] - first[0], second[1] - first[1], 0]


def _rotate(vector: np.array, angels: np.array) -> np.array:
    r = R.from_euler('xyz', angels, degrees=True)
    v = r.apply(vector)
    return v


def _calculate_coordinates(df):
    initial_vector_val = _calculate_initial_vector(df)
    df['vector'] = np.empty(len(df), dtype=object)

    with_speed = 'speed' in df.columns

    previous_val = initial_vector_val
    df.at[df.index[0], 'vector'] = previous_val  # Initial value is not rotated

    df['coordinate'] = np.empty(len(df), dtype=object)
    previous_coord = [0, 0, 0]
    df.at[df.index[0], 'coordinate'] = previous_coord

    for i in range(1, len(df)):
        current_val = _rotate(np.array(previous_val), df[['gr_x', 'gr_y', 'gr_z']].iloc[
            i].to_numpy() * 0.1)  # todo вынести 0.1 в глобальную константу, это шаг по времени

        norm = np.linalg.norm(current_val)
        norm = norm if norm != 0.0 else 1
        current_coord = previous_coord + current_val * (
            df['speed'].iloc[i] * 0.1 / 3.6 / norm if with_speed else 1)  # speed in km/h
        df.at[df.index[i], 'vector'] = current_val.tolist()  # Ensure it's set as a list
        df.at[df.index[i], 'coordinate'] = current_coord.tolist()
        previous_val = current_val
        previous_coord = current_coord

    df['calculated_x'] = df['coordinate'].apply(lambda p: p[0])
    df['calculated_y'] = df['coordinate'].apply(lambda p: p[1])
    return df


def _plot(df, origin_df, file_path: str):
    x = df['coordinate'].apply(lambda p: p[0])
    y = df['coordinate'].apply(lambda p: p[1])
    z = df['coordinate'].apply(lambda p: p[2])

    # fig = plt.figure('Coordinate curve')
    fig, axis = plt.subplots(1, 2)
    # ax = fig.add_subplot(111, projection='3d')
    axis[0].plot(x, y, '-r', linewidth=3)
    axis[0].set_title("Calculated trajectory")

    # o_x = origin_df['coordinate'].apply(lambda p: p[0])
    # o_y = origin_df['coordinate'].apply(lambda p: p[1])
    print(origin_df)
    axis[1].plot(origin_df['x'], origin_df['y'], linewidth=3)
    axis[1].set_title("Real trajectory")
    plt.savefig(file_path)
    plt.close()

    # Plotting the coordinates on map
    color_scale = [(0, 'red'), (1, 'green')]
    fig = px.scatter_mapbox(df,
                            lat="latitude",
                            lon="longitude",
                            zoom=8,
                            height=600,
                            width=900)

    fig.update_layout(mapbox_style="mapbox://styles/mapbox/streets-v11"
                      # , mapbox_accesstoken="pk.eyJ1IjoidnZzaHUiLCJhIjoiY2x6cHNmZjYxMDg3dTJqcXpldDFvNXR1YiJ9.uVxq0SBdZ8bgK8LmEdcZPQ"

                      # mapbox_accesstoken="pk.eyJ1IjoidnZzaHUiLCJhIjoiY2x6cHYxM2d0MG41OTJtczN3ZjE3cTY2MiJ9.de6tBpkfQneN0u6rQALZtg",
                      )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    # import folium
    #
    # center_lat = df['latitude'].mean()
    # center_lon = df['longitude'].mean()
    #
    # # Создаем карту с центром в определенной точке и начальным уровнем масштабирования
    # m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    #
    # # Добавляем маркеры или линии траектории
    # for lat, lon in zip(df['latitude'], df['longitude']):
    #     folium.Marker([lat, lon], popup="Point").add_to(m)
    #
    # # Сохраняем карту в HTML файл или отображаем в Jupyter Notebook
    # m.save("map.html")
    #
    # # Отображение карты в Jupyter Notebook (если используется)
    # m


def euclidean_distance(trajectory1, trajectory2):
    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)

    return np.linalg.norm(trajectory1 - trajectory2, axis=1).sum()


def visualize(original_df, aggregated_df, path):
    # TODO : visualize data (way by gps coordinates, way by calculated coordinates, way by speed and gyro data)

    fig, axis = plt.subplots(1, 3)
    # way by gps coordinates

    original_gps_coordinate_df = original_df[['longitude', 'latitude']].drop_duplicates()
    axis[0].set_title("GPS trajectory")
    axis[0].xaxis.set_major_locator(plt.MultipleLocator(1))
    axis[0].plot(original_gps_coordinate_df['longitude'], original_gps_coordinate_df['latitude'], '-r')

    # way by calculated coordinates
    axis[1].set_title("Calculated trajectory")
    axis[1].plot(aggregated_df['x'], aggregated_df['y'], '-r', linewidth=3)

    # way by coordinates from speed and gyro data
    calculated_df = _calculate_coordinates(aggregated_df)
    axis[2].set_title("Speed and gyro trajectory")
    axis[2].plot(aggregated_df['x'], aggregated_df['y'], '-r', linewidth=3)
    calculated_file_path = f"resources/pipeline/visualization/data/{path}"
    os.makedirs(os.path.dirname(calculated_file_path), exist_ok=True)
    calculated_df.to_csv(calculated_file_path, index_label='index')

    # save
    file_path = f"resources/pipeline/visualization/{path}"
    file_path = file_path.replace('.csv', '.jpeg')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.close()
