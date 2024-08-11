import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from main import rotate, plot
from maneuver_data_aggregator import ManeuverDataAggregator

from aggregator import ManeuverDataAggregator


def plot(df, origin_df, file_path: str):
    x = df['coordinate'].apply(lambda p: p[0])
    y = df['coordinate'].apply(lambda p: p[1])

    fig, axis = plt.subplots(1, 2)
    axis[0].plot(x, y, '-r', linewidth=3)
    axis[0].set_title("Calculated trajectory")

    axis[1].plot(origin_df['x'], origin_df['y'], linewidth=3)
    axis[1].set_title("Real trajectory")
    plt.savefig(file_path)
    plt.close()

    fig = px.scatter_mapbox(df,
                            lat="latitude",
                            lon="longitude",
                            zoom=8,
                            height=600,
                            width=900)

    fig.update_layout(mapbox_style="mapbox://styles/mapbox/streets-v11")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def calculate_coordinates(df, initial_vector_val):
    df['vector'] = np.empty(len(df), dtype=object)

    with_speed = 'speed' in df.columns

    previous_val = initial_vector_val
    df.at[df.index[0], 'vector'] = previous_val  # Initial value is not rotated

    df['coordinate'] = np.empty(len(df), dtype=object)
    previous_coord = [0, 0, 0]
    df.at[df.index[0], 'coordinate'] = previous_coord

    for i in range(1, len(df)):
        current_val = rotate(np.array(previous_val), df[['gr_x', 'gr_y', 'gr_z']].iloc[i].to_numpy() * 0.1)

        current_coord = previous_coord + current_val * (
            df['speed'].iloc[i] * 0.1 / 3.6 / np.linalg.norm(current_val) if with_speed else 1)  # speed in km/h
        df.at[df.index[i], 'vector'] = current_val.tolist()  # Ensure it's set as a list
        df.at[df.index[i], 'coordinate'] = current_coord.tolist()
        previous_val = current_val
        previous_coord = current_coord

    return df


def calculate_initial_vector(df):
    first = df[['x', 'y']].iloc[0].to_numpy()
    second = df[['x', 'y']].iloc[1].to_numpy()

    return [second[0] - first[0], second[1] - first[1], 0]


def draw_df(df, ts, path=None):
    initial_vector = calculate_initial_vector(df)
    print(initial_vector)
    origin_df = df[['x', 'y']]
    df = df[['speed', 'gr_x', 'gr_y', 'gr_z', 'latitude', 'longitude']]
    df = calculate_coordinates(df, initial_vector)
    df.to_csv(f"./{ts}/calculated.csv", index_label='index')

    if path:
        plot(df, origin_df, path)
    else:
        plot(df, origin_df, f'./{ts}/coordinate_curve_df.png')


def draw_with_origin(df, origin, ts, path=None):
    initial_vector = calculate_initial_vector(df)
    df = df[['speed', 'gr_x', 'gr_y', 'gr_z']]
    df = calculate_coordinates(df, initial_vector)
    origin_df = calculate_coordinates(origin, initial_vector)

    df.to_csv(f"./{ts}/calculated.csv", index_label='index')
    if path:
        plot(df, origin_df, path)
    else:
        plot(df, origin_df, f'./{ts}/coordinate_curve_df.png')


def euclidean_distance(trajectory1, trajectory2):
    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)

    return np.linalg.norm(trajectory1 - trajectory2, axis=1).sum()


def main():
    tss = []
    for ts in tss:
        print('----------------------------')
        print(f"TS: {ts}")
        file_path = f'{ts}.csv'
        source = pd.read_csv(file_path, index_col="index")

        fake_aggregator = ManeuverDataAggregator()
        fake = fake_aggregator.transform(source)

        draw_df(fake, ts=ts)
        print('----------------------------')


if __name__ == '__main__':
    main()
