import math

import numpy as np
from scipy.spatial.transform import Rotation

from aggregators.angles.quaternion import Quaternion
from aggregators.angles.vector import Vector, vectors_angle, cross_product, dot_product, scalar


def get_angles(data):
    transformed_data = []
    for i in range(1, len(data)):
        transformed_data.append((data[i][0] - data[i - 1][0], data[i][1] - data[i - 1][1], data[i][2] - data[i - 1][2]))
    data = transformed_data

    data = list(map(lambda x: Vector.create(x), data))

    quat = make_quat(data[0], data[1])
    angles = [angle_components(quat)]
    # inv = quat.invert() # todo invert before rotate and keep base quat
    mul = quat

    for i in range(1, len(data) - 1):
        inv = quat.invert()
        v1 = rotate(data[i], inv)
        v2 = rotate(data[i + 1], inv)
        q = make_quat(v1, v2)
        angles.append(angle_components(q))
        mul = product(mul, q)
    return angles


def make_quat(v1: Vector, v2: Vector):
    return Quaternion.create(vectors_angle(v1, v2), cross_product(v1, v2))


def multiply(q1: Quaternion, q2: Quaternion):
    w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    return Quaternion(w, x, y, z)


def product(q1: Quaternion, q2: Quaternion):
    w = q1.w * q2.w - dot_product(q1.v, q2.v)
    v = cross_product(q1.v, q2.v) + scalar(q1.v, q2.w) + scalar(q2.v, q1.w)
    return Quaternion(w, v.x, v.y, v.z)


# def multiply_vec(q: Quaternion, v: Vector):
#     w = -q.x * v.x - q.y * v.y - q.z * v.z
#     x = q.w * v.x + q.y * v.z - q.z * v.y
#     y = q.w * v.y - q.x * v.z + q.z * v.x
#     z = q.w * v.z + q.x * v.y - q.y * v.x
#     return Quaternion(w, x, y, z)


def rotate(v: Vector, q: Quaternion):
    vq = Quaternion(0, v.x, v.y, v.z)
    t = product(q, vq)
    r = product(t, q.invert())
    return Vector(r.x, r.y, r.z)


def sign(value: int) -> int:
    if (value < 0):
        return -1
    return 1


def board(value: int) -> int:
    if (abs(value) > 90):
        return (180 - abs(value)) * (-1) * sign(value)
    return value


def angle_components(quat: Quaternion):
    sqx = quat.x * quat.x
    sqy = quat.y * quat.y
    sqz = quat.z * quat.z
    # bank = np.rad2deg(np.arctan2(1 - 2 * (sqx + sqy), 2 * (quat.w * quat.x + quat.y * quat.z)))
    # altitude = np.rad2deg(np.arcsin(2 * (quat.w * quat.y - quat.x * quat.z)))
    # heading = np.rad2deg(np.arctan2(1 - 2 * (sqy + sqz), 2 * (quat.w * quat.z + quat.x * quat.y)))
    # bank = np.rad2deg(np.arctan2(2 * (quat.w * quat.x - quat.z * quat.x), (1 - 2 * (sqy + sqz))))
    # altitude = np.rad2deg(np.arcsin(2 * (quat.w * quat.z + quat.y * quat.x)))
    # heading = np.rad2deg(np.arctan2(2 * (quat.w * quat.x - quat.z * quat.y), (1 - 2 * (sqx + sqz))))
    bank = np.rad2deg(np.arctan2(2 * (quat.w * quat.x + quat.y * quat.z), (1 - 2 * (sqy + sqx))))
    altitude = np.rad2deg(np.arcsin(2 * (quat.w * quat.y - quat.z * quat.x)))
    heading = np.rad2deg(np.arctan2(2 * (quat.w * quat.z + quat.x * quat.y), (1 - 2 * (sqy + sqz))))
    return bank, altitude, heading


#
# if __name__ == '__main__':
#     # v1 = Vector.create([1, 0, 0])
#     # v2 = Vector.create([0, 1, 0])
#     # v3 = Vector.create([0, 0, -1])
#     v1 = Vector.create([0.2778127847846701, 0.5081340831929937, 0])
#     v2 = Vector.create([0.5185838672876475, 1.0363260869216888, 0])
#     v3 = Vector.create([0.9445634654403623, 1.8185851340331236, 0])
#
#     print(np.rad2deg(vectors_angle(v1, v2)))
#     q = make_quat(v1, v2)
#
#     print(q)
#     print(angle_components(q))
#     q_invert = q.invert()
#     print(q_invert)
#     r_v2 = rotate(v2, q_invert)
#     r_v3 = rotate(v3, q_invert)
#
#     print(r_v2)
#     print(r_v3)
#
#     print(np.rad2deg(vectors_angle(r_v2, r_v3)))
#     print(cross_product(r_v2, r_v3))
#     q2 = make_quat(r_v2, r_v3)
#
#     print(q2)
#     print(angle_components(q2))
#     rotation = Rotation.from_quat([q2.w, q2.x, q2.y, q2.z])
#     # print(rotation.as_euler("xyz", degrees=True))
#
#     # print(angle_components(Quaternion(0.7071, 0, 0, -0.7071)))
#     #
#     # print(q)
#     # print(q.invert())
#     # print(multiply_vec(q, Vector.crete(v2)))
#
#     # print(rotate(Vector.create([0, 0, 1]), Quaternion.create(np.pi, Vector.create([1, 0, 1]))))
#     # print(get_angles([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))
#     # angle = vectors_angle(v1, v2)
#     # print(angle)
#     # print(np.cos(np.deg2rad(angle)))


if __name__ == '__main__':
    v1 = Vector.create([0, 1, 0])
    v2 = Vector.create([1, 0.5, 0])
    v3 = Vector.create([0.5, 1, 0])

    print(np.rad2deg(vectors_angle(v1, v2)))
    q = make_quat(v1, v2)
    print(q)
    print(angle_components(q))
    print(q.invert())
    v2 = rotate(v2, q.invert())
    v3 = rotate(v3, q.invert())

    q2 = make_quat(v2, v3)
    print(q)
    print(multiply(q, q2))
    print(product(q, q2))
