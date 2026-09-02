import taichi as ti
import pybullet as p
from MPM.config import n_grid, grid_shape

MAX_VERTEX_NUM = 10  # 25000
FACE_NUM = 20  # 50000
MAX_FACE_NUM = 3 * FACE_NUM
PI = 3.1415926
vertices = ti.Vector.field(3, dtype=float, shape=MAX_VERTEX_NUM)
indices = ti.field(dtype=int, shape=MAX_FACE_NUM)
num_vertices = ti.field(dtype=int, shape=())
rotation_matrix = ti.Matrix.field(3, 3, dtype=float, shape=())
reverse_rotation_matrix = ti.Matrix.field(3, 3, dtype=float, shape=())
reverse_rotation_matrix[None] = ti.Matrix(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
reverse_offset_vector = ti.Vector.field(3, dtype=float, shape=())
position = ti.Vector.field(3, dtype=float, shape=grid_shape)
transformed_pos=ti.Vector.field(3, dtype=float, shape=grid_shape)
sdf_grid_size = 256 # high precision SDF
tmp_sdf = ti.field(dtype=float, shape=grid_shape)
co_mask = ti.field(dtype=int, shape=grid_shape)
static_sdf = ti.field(dtype=float, shape=(256,) * 3)


def init_static_sdf(np_array):
    static_sdf.from_numpy(np_array)

def init_switch_base(field):
    position.copy_from(field)

@ti.func
def clamp(x, t):
    flag = False
    if x < -t:
        x = -t
        flag = True
    if x >= t:
        x = t - 1
        flag = True
    return x, flag


@ti.func
def in_interval(x, l, r):
    res = 1
    if x >= l and x <= r:
        res = 0
    elif x < l:
        res = -1
    return res


@ti.func
def distance(flag, pos, l, r):
    res = 0.0
    if flag == 1:
        res = pos - r
    elif flag == 0:
        res = ti.min(r - pos, pos - l)
    else:
        res = ti.abs(l - pos)
    return res


@ti.func
def cube_distance(pos, a, b, c):
    """Compute the nearest distance between the given point and a cuboid."""
    x = in_interval(pos[0], -a / 2, a / 2)
    y = in_interval(pos[1], -b / 2, b / 2)
    z = in_interval(pos[2], -c / 2, c / 2)

    dx = distance(x, pos[0], -a / 2, a / 2)
    dy = distance(y, pos[1], -b / 2, b / 2)
    dz = distance(z, pos[2], -c / 2, c / 2)

    tag = ti.abs(x) + ti.abs(y) + ti.abs(z)
    res = 99999.9
    if tag == 0:
        res = ti.min(dx, dy, dz)
    elif tag == 1:
        # face
        res = ti.abs(x) * dx + ti.abs(y) * dy + ti.abs(z) * dz
    elif tag == 2:
        # line
        res = ti.sqrt(ti.abs(x) * dx**2 + ti.abs(y) * dy**2 + ti.abs(z) * dz**2)
    else:
        # point
        res = ti.sqrt(dx**2 + dy**2 + dz**2)
    return res


@ti.func
def pp(x):
    """
    Print function used to debug.
    ATTENTION: ti.init(ti.cpu)
    """
    print("YZYZYZYZYZYZYZYZYZYZY")
    print(x)


@ti.kernel
def apply_reverse_transform(tag: int, a: float, b: float, c: float, mode:int, scale:float):
    # switch reference frame
    t = sdf_grid_size // 2
    for i in ti.grouped(position):
        v = 10000.0
        transformed_pos[i] = (
            reverse_rotation_matrix[None] @ position[i]*scale + reverse_offset_vector[None]
        )

        if mode == 0:  # needle's sdf
            x, y, z = transformed_pos[i] * t
            x, flag_x = clamp(x, t)
            y, flag_y = clamp(y, t)
            z, flag_z = clamp(z, t)
            if flag_x or flag_y or flag_z:
                continue
            idx = ti.Vector([x, y, z], dt=int) + t
            v = static_sdf[idx[0], idx[1], idx[2]]
        elif mode == 1:
            v = cube_distance(transformed_pos[i], a, b, c)

        if v < tmp_sdf[i]:
            tmp_sdf[i] = v
            if v < 0.05:
                co_mask[i] = tag

import time
def switch_reference_frame_and_update_sdf(i_rot_list, i_pos_list, co_obj, scale):
    co_mask.fill(-1)
    tmp_sdf.fill(9999.0)
    tag = 0
    
    for i_rot, i_pos in zip(i_rot_list, i_pos_list):
        reverse_rotation_matrix.from_numpy(i_rot)
        reverse_offset_vector.from_numpy(i_pos)

        a, b, c = 0.0, 0.0, 0.0
        mode=0
        if co_obj[tag][1] != -1:
            a, b, c = p.getCollisionShapeData(co_obj[tag][0], co_obj[tag][1])[0][3]
            mode=1
        
        apply_reverse_transform(tag, a, b, c,mode, scale)
        tag += 1
        
        # print(f"inner time{(t1-t0)*1000}")
        
        
    # return tmp_sdf.to_numpy(), co_mask.to_numpy()  #whether necessary?
    # return tmp_sdf, co_mask


def main():
    result_dir = "../video"
    video_manager = ti.tools.VideoManager(
        output_dir=result_dir, framerate=60, automatic_build=True
    )
    window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    while window.running:
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)

        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(vertices, indices, two_sided=True)
        angle = 0.05
        # rotate(0, 0, angle)
        canvas.scene(scene)
        # video_manager.write_frame(window.get_image_buffer_as_numpy())
        window.show()


if __name__ == "__main__":
    # filename = './model/cow.obj'
    # start_time = time.time()
    # load_mesh_fast(filename)
    # use_time = time.time() - start_time
    # print(f"Load Successfully.\nTime Consumption:{use_time}s")
    # main()
    apply_reverse_transform()
