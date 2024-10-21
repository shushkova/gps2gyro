import math

import numpy as np

from aggregators.angles.constants import round_num
from aggregators.angles.vector import Vector


class Quaternion:

    def __init__(self, w, x, y, z):
        self.w = round(w, round_num)
        self.x = round(x, round_num)
        self.y = round(y, round_num)
        self.z = round(z, round_num)
        self.v = Vector(self.x, self.y, self.z)
        # self.normalize()

    @staticmethod
    def create(angle: float, v: Vector):
        # w = np.cos(np.deg2rad(angle / 2))
        w = np.cos(angle / 2)
        v.normalize()
        # x = v.x * np.sin(np.deg2rad(angle / 2))
        # y = v.y * np.sin(np.deg2rad(angle / 2))
        # z = v.z * np.sin(np.deg2rad(angle / 2))
        x = v.x * np.sin(angle / 2)
        y = v.y * np.sin(angle / 2)
        z = v.z * np.sin(angle / 2)
        return Quaternion(w, x, y, z)

    def __str__(self):
        return f"w: {self.w}, x: {self.x}, y: {self.y}, z: {self.z}"

    def norm(self):
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return math.sqrt(self.norm())

    # def scale(self, val):
    #     return Quaternion(self.w * val, self.x * val, self.y * val, self.z * val)

    def normalize(self):
        l = self.length()
        if l == 0.0:
            return
        self.w = self.w * 1.0 / l
        self.x = self.x * 1.0 / l
        self.y = self.y * 1.0 / l
        self.z = self.z * 1.0 / l

    def invert(self):
        def invert_var(x):
            # return x if x == 0.0 else -x
            return -x

        q = Quaternion(self.w, invert_var(self.x), invert_var(self.y), invert_var(self.z))
        q.normalize()
        # return Quaternion(self.w, invert_var(self.x), invert_var(self.y), invert_var(self.z))
        return q
