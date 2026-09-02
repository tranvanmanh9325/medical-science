import taichi as ti
import numpy as np
import torch


class point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


class bbox2d:

    def __init__(self) -> None:
        self.lower = point(999999, 99999)
        self.upper = point(0, 0)

    def print(self):
        print(self.lower.x, self.lower.y)
        print(self.upper.x, self.upper.y)


@ti.data_oriented
class bbox3d:

    def __init__(self, padding, x0, x1, y0, y1, z0, z1):
        self.padding = padding
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1


def get_op_map(op_field, threshold=10.0):
    dim = op_field.shape[0]
    op_map = np.zeros((dim, dim))
    for i in reversed(range(dim)):
        slice = op_field[:, :, i]
        mask = slice > threshold
        op_map[mask] = i
    return op_map


def padding_from_opmap(op_map, bottom=0.9):  # please check again on the morning
    dim = op_map.shape[0]
    boundary = int(dim * bottom)
    box = bbox2d()
    for i in range(op_map.shape[0]):
        for j in range(op_map.shape[1]):
            if op_map[i][j] == 0.0:
                continue
            if i < box.lower.x:
                box.lower.x = i
            if i > box.upper.x:
                box.upper.x = i
            if j < box.lower.y:
                box.lower.y = j
            if j > box.upper.y:
                box.upper.y = j
    cnt = 0
    pad_points = []
    for i in range(box.lower.x, box.upper.x + 1):
        for j in range(box.lower.y, box.upper.y + 1):
            for k in range(boundary):
                if k > op_map[i, j] and op_map[i, j] > 0:
                    cnt += 1
                    pad_points.append(np.array([i, j, k], dtype=float) / dim)

    return torch.tensor(np.array(pad_points), dtype=torch.float32, device="cuda")


@ti.func
def value_clamp(x, threshold):
    if x < -threshold:
        x = -threshold
    if x > threshold:
        x = threshold
    return x


@ti.func
def velocity_clamp(v, threshold):
    x = value_clamp(v[0], threshold)
    y = value_clamp(v[1], threshold)
    z = value_clamp(v[2], threshold)
    return ti.Vector([x, y, z])


# @ti.func
# def pp(x):
#     """
#     Print function used to debug.
#     ATTENTION: ti.init(ti.cpu)
#     """
#     print("YZYZYZYZYZYZYZYZYZYZY")
#     print(x)


# @ti.func
# def ppa(x):
#     print("Angular Velocity:")
#     print(x)


# @ti.func
# def ppd(x):
#     print("Distance:")
#     print(x)


# @ti.func
# def ppg(x):
#     print("Grid Position:")
#     print(x)


# @ti.func
# def field_copy(dest, src):
#     assert dest.s


# @ti.kernel
# def avg_external_forces(num: float):
#     scale = 1.0 / num
#     for i in ti.grouped(external_forces):
#         external_forces[i] *= scale
