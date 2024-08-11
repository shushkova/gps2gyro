import numpy as np

from constants import round_num


class Vector:
    def __init__(self, x, y, z):
        self.x = round(x, round_num)
        self.y = round(y, round_num)
        self.z = round(z, round_num)
        # self.normalize()

    @staticmethod
    def create(array: list):
        return Vector(array[0], array[1], array[2])

    def as_np_array(self):
        return np.array([self.x, self.y, self.z])

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def norm(self):
        v = self.as_np_array()
        return np.linalg.norm(v)

    def normalize(self):
        v = self.as_np_array()
        norm = np.linalg.norm(v)
        if np.isnan(norm) or norm == 0:
            return
        v = v / norm
        self.x = v[0]
        self.y = v[1]
        self.z = v[2]

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"


def cross_product(v1: Vector, v2: Vector) -> Vector:
    v = np.cross(v1.as_np_array(), v2.as_np_array())
    return Vector.create(v.tolist())


def dot_product(v1: Vector, v2: Vector) -> float:
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def scalar(v: Vector, c: float) -> Vector:
    return Vector(c * v.x, c * v.y, c * v.z)


def vectors_angle(v1: Vector, v2: Vector) -> float:
    v1.normalize()
    v2.normalize()
    product = dot_product(v1, v2)
    if product > 1.0:
        product = 1.0
    elif product < -1.0:
        product = -1.0

    angle = np.arccos(product)
    return angle
