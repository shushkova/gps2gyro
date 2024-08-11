import math

import numpy as np

from constants import round_num
from vector import Vector


class Quaternion:

    def __init__(self, w, x, y, z):
        self.w = round(w, round_num)
        self.x = round(x, round_num)
        self.y = round(y, round_num)
        self.z = round(z, round_num)
        self.v = Vector(self.x, self.y, self.z)
        self.normalize()

    @staticmethod
    def create(angle: float, v: Vector):
        w = np.cos(angle / 2)
        v.normalize()
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

    def normalize(self):
        l = self.length()
        if l == 0.0:
            return
        self.w = self.w * 1.0 / l
        self.x = self.x * 1.0 / l
        self.y = self.y * 1.0 / l
        self.z = self.z * 1.0 / l

    def invert(self):
        q = Quaternion(self.w, -self.x, -self.y, -self.z)
        return q
