import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter


def interpolate_data(x, y, new_x):
    f = interpolate.interp1d(x, y, kind='quadratic')
    # data = savgol_filter(f(new_x), 3, 1)
    return f(new_x)


if __name__ == '__main__':
    data = [1, 4, 3, 2, 5]
    print(savgol_filter(data, 3, 1))
