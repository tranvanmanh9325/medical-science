EXPORT = False
if EXPORT:
    export_file = "./frames/mpm3d.ply"
else:
    export_file = ""

import time
import numpy as np
import taichi as ti
import pybullet as p
import skimage.measure
from MPM.config import n_grid, grid_shape

arch=ti.cpu
if ti._lib.core.with_metal():
    arch=ti.metal
elif ti._lib.core.with_vulkan():
    arch=ti.vulkan
elif ti._lib.core.with_cuda():
    arch=ti.cuda

ti.init(arch=arch)

import MPM.sdf as sdf

MAX_COLLISION_OBJECTS = 3
dim = 3
steps, dt = 100, 5e-4
steps, dt = 25, 5e-4  # -->sweet pot
# steps, dt = 500, 1e-5

timestep = steps * dt

n_particles = 24000

dx = 1 / n_grid
inv_dx = n_grid
p_rho = 1000

p_vol = (dx * 0.5) ** dim
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3  # important for stable simulation

E = ti.field(float, shape=())
nu = ti.field(float, shape=())
mu = ti.field(float, shape=())
la = ti.field(float, shape=())

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(dim, dim, float, n_particles)
FJ = ti.field(float, n_particles)
# F_visual = ti.Vector.field(dim, float, n_particles)

# rearrange euler grids to reduce computation cost
F_grid_v = ti.Vector.field(dim, float, shape=grid_shape)
F_grid_m = ti.field(float, grid_shape)
F_grid_deformation = ti.field(float, grid_shape)
pos = ti.Vector.field(dim, dtype=float, shape=grid_shape)

# CDF = ti.Vector.field(dim, dtype=float, shape=n_grid**dim)

grid_pos = ti.Vector.field(dim, dtype=float, shape=n_grid**dim)
ref_point = ti.Vector([0.3, 0.0, 0.0])
normal = ti.Vector([1.0, 1.0, 1.0]).normalized()

fe = 1.0  # friction coefficient

neighbour = (3,) * dim
ORIANGE = ti.Vector((0.9294117647058824, 0.3333333333333333, 0.23137254901960785))
PI = 3.1415926


# co_obj = ti.Vector.field(2, int, shape=MAX_COLLISION_OBJECTS)  # (obj_id,link_id)]
co_obj=np.zeros((MAX_COLLISION_OBJECTS,2),dtype=int)
co_v = ti.Vector.field(
    dim, dtype=float, shape=MAX_COLLISION_OBJECTS
)  # translational velocity
co_w = ti.Vector.field(
    dim, dtype=float, shape=MAX_COLLISION_OBJECTS
)  # angular velocity
centroid = ti.Vector.field(dim, dtype=float, shape=MAX_COLLISION_OBJECTS)

soft_body_base_position = ti.Vector.field(3, dtype=float, shape=())


# SDF = ti.field(float, grid_shape)
# collision_mask = ti.field(dtype=int, shape=grid_shape)
SDF =None
collision_mask = None

external_forces = ti.Vector.field(3, dtype=float, shape=grid_shape)
MAX_COLLISION_NUM = 50

# fine_scale = 2.0
# density_field = ti.field(dtype=float, shape=(n_grid * int(fine_scale), ) * dim)


def set_parameters(s_E=8000, s_nu=0.2):
    print(f"MPM Parameters: Young's Modulus: {s_E}, Possion Ratio: {s_nu}")
    E[None] = s_E
    nu[None] = s_nu
    mu[None] = E[None] / (2 * (1 + nu[None]))
    la[None] = E[None] * nu[None] / ((1 + nu[None]) * (1 - 2 * nu[None]))


def set_base_position(base_position):
    soft_body_base_position[None] = ti.Vector(base_position)


def init_collision_field():
    # collision_mask.fill(-1)
    external_forces.fill([0.0, 0.0, 0.0])


def apply_external_forces(scale, mask, forces, debug=False):
    idx = mask > 0
    t = idx.nonzero()
    t = np.moveaxis(t, 0, -1)
    num_collision = idx.sum()
    if num_collision > MAX_COLLISION_NUM:
        sample_idx = np.random.choice(
            list(range(num_collision)), size=MAX_COLLISION_NUM, replace=True
        )
        t = t[sample_idx]

    t0 = time.time()
    for i in t:
        tag = mask[i[0], i[1], i[2]]
        p.applyExternalForce(
            co_obj[tag][0],
            co_obj[tag][1],
            forceObj=forces[tuple(i)],
            posObj=i * dx * scale + soft_body_base_position[None],
            flags=p.WORLD_FRAME,
        )
        if debug:
            print(forces[tuple(i)])
    t1 = time.time()
    if debug:
        print("total collision grid node:%d\nuse time:%fs" % (idx.sum(), t1 - t0))


@ti.func
def P2G():
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        # mc_base = int(F_x[p] * fine_scale / dx - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # stress = -dt * 4 * E * p_vol * (F_J[p] - 1) / dx**2
        # affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]
        I = ti.Matrix.identity(float, dim)
        F[p] = (I + dt * F_C[p]) @ F[p]
        J = 1.0
        _, sig, _ = ti.svd(F[p])
        for i in ti.static(range(dim)):
            J *= sig[i, i]
            # F_visual[p][i] = sig[i, i]

        # Neo-Hookean
        stress = mu[None] * (F[p] @ F[p].transpose() - I) + I * la[None] * ti.log(J)

        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

        # PIC
        # affine = stress
        # APIC
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
            external_forces[base + offset] += 0.1 * weight * (stress @ dpos) / dt
            # density_field[mc_base + offset] += weight * p_mass


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


@ti.func
def field_copy(dest, src):
    assert dest.s


@ti.func
def Boundary(scale, threshold=0.05):
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
            # taichi(Y) --> pyBullet(Z)
            # F_grid_v[I][1] -= dt * gravity  #taichi
            F_grid_v[I][2] -= dt * gravity  # pybullet
        # SDF collision solver
        inv_scale = 1.0 / scale
        # v1 = ti.Vector([.0, .0, .0])
        if SDF[I] < threshold and F_grid_m[I] > 0:
            i, j, k = I[0], I[1], I[2]
            # TODO!: out of boundary enquiry(maybe)

            gdx = (SDF[i + 1, j, k] - SDF[i - 1, j, k]) * 0.5 * inv_dx * inv_scale
            gdy = (SDF[i, j + 1, k] - SDF[i, j - 1, k]) * 0.5 * inv_dx * inv_scale
            gdz = (SDF[i, j, k + 1] - SDF[i, j, k - 1]) * 0.5 * inv_dx * inv_scale
            grad = ti.Vector([gdx, gdy, gdz])
            n = grad.normalized()

            # v1 = F_grid_v[I] + 0.0  #copy problem

            # angular velocity
            grid_node_pos = I * dx * scale + soft_body_base_position[None]
            tag = collision_mask[I]
            angular_velocity = co_w[tag].cross(grid_node_pos - centroid[tag])
            collision_object_velocity = co_v[tag] + angular_velocity

            # 0.05 is the mass of needle
            # decay_scale = 2 * 0.05 / (0.05 + F_grid_m[I])
            # collision_object_velocity *= decay_scale

            # v_ref = F_grid_v[I] - collision_object_velocity
            # vn = v_ref.dot(n)
            # if vn < 0.0:
            #     vt = v_ref - vn * n
            #     v_ref = vt + fe * vn * vt.normalized()
            #     F_grid_v[I] = v_ref + collision_object_velocity

            # sticky trick
            F_grid_v[I] = collision_object_velocity
        else:
            collision_mask[I] = -1

        cond = (I < bound) & (F_grid_v[I] < 0) | (I > ti.Vector(grid_shape) - bound) & (
            F_grid_v[I] > 0
        )
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])

        # add anchor points(constraints)
        anchor = I[2] == 30 and (
            (I[1] >= 10 and I[1] <= 15) or (I[1] >= 30 and I[1] <= 35)
        )

        anchor = 0

        if anchor:
            F_grid_v[I] = ti.Vector([0.0, 0.0, 0.0])
        #     F_grid_v[I][0]=.0
        #     F_grid_v[I][1]=.0
        # vt = F_grid_v[I] - F_grid_v[I][2] * ti.Vector([.0, .0, 1.0])
        # F_grid_v[I] = 0.8 * vt + F_grid_v[I][2] * ti.Vector([.0, .0, 1.0])

        # v2 = F_grid_v[I]
        # external_forces[I] += F_grid_m[I] * (v1 - v2) / dt
        if I[2] <= 3:
            F_grid_v[I][0] *= 0.1
            F_grid_v[I][1] *= 0.1


@ti.func
def G2P():
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = offset - fx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        F_v[p] = new_v

        #!:code follows may lead to velocity discontinuity(material crash)
        # stick suturing pad on floor
        # sticky_parameter = 3.0  #this parameter is used to fixed pad on table
        # if F_x[p][2] < sticky_parameter * 1.0 / n_grid:
        #     F_v[p] = ti.Vector([0.0, 0.0, 0.0])

        F_x[p] += dt * F_v[p]
        F_C[p] = new_C


@ti.kernel
def substep(scale: float, threshold: float):
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    # density_field.fill(0)
    # ti.loop_config(block_dim=n_grid, parallelize=n_grid)
    P2G()
    # ti.loop_config(block_dim=n_grid, parallelize=n_grid)
    Boundary(scale, threshold)
    # ti.loop_config(block_dim=n_grid, parallelize=n_grid)
    G2P()


@ti.kernel
def pos_generation():
    for i in ti.grouped(pos):
        pos[i] = i * dx

def init_pos():
    pos_generation()
    sdf.init_switch_base(pos)

def init_sdf(np_sdf):
    sdf.init_static_sdf(np_sdf)
    print("************Signed Distance Field has been initialized.************")


def interpolate_deformation(base, dx, dy, dz):
    deformation = 0.0
    base = tuple(base)
    x, y, z = base
    # #(0,0,0)
    # deformation += F_grid_deformation[base] * (1 - dx) * (1 - dy) * (1 - dz)
    # #(1,1,1)
    # idx = (x + 1, y + 1, z + 1)
    # deformation += F_grid_deformation[idx] * dx * dy * dz
    # #(0,0,1)
    # idx = (x, y, z + 1)
    # deformation += F_grid_deformation[idx] * (1 - dx) * (1 - dy) * dz
    for v in range(8):
        i, j, k = v & 1, v >> 1 & 1, v >> 2 & 1
        idx = (x + i, y + j, z + k)
        w = 1.0
        w *= (1 - i) + (2 * i - 1) * dx
        w *= (1 - j) + (2 * j - 1) * dy
        w *= (1 - k) + (2 * k - 1) * dz
        if F_grid_m[idx] > 0.0:
            deformation += (F_grid_deformation[idx] / F_grid_m[idx]) * w
    return deformation


# TODO: generalize to list
@ti.kernel
def init_cube():
    """
    obj_id: obj's id which need to be detected for collision

    filename: load 3D model to control the shape of soft body, if filename='', this function will generate a cube.
    """
    for i in range(n_particles):
        F[i] = ti.Matrix.identity(float, dim)
        # F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.3
        # F_x[i] = ti.Vector([ti.random() * 0.2, ti.random() * 0.2, ti.random() * 0.2])
        F_x[i] = ti.Vector([ti.random() * 0.3, ti.random() * 0.3, ti.random() * 0.05])

        F_x[i][0] += 0.1
        F_x[i][1] += 0.2
        F_x[i][2] += 0.05


@ti.kernel
def init_deformation_gradient():
    F.fill(ti.Matrix.identity(float, dim))


# @ti.kernel
# def avg_external_forces(num: float):
#     scale = 1.0 / num
#     for i in ti.grouped(external_forces):
#         external_forces[i] *= scale


def init_model(filename):
    model_array = np.load(filename)
    model_array[:, 0] += 0.1
    model_array[:, 2] += 0.01
    idx = list(range(len(model_array)))
    sampled_idx = np.random.choice(idx, size=n_particles, replace=False)
    F_x.from_numpy(model_array[sampled_idx])
    init_deformation_gradient()


def init(obj_id_list, filename,default_young=8000,default_poisson=0.2):
    # co_obj[None] = obj_id
    set_parameters(s_E=default_young,s_nu=default_poisson)
    for i in range(MAX_COLLISION_OBJECTS):
        co_obj[i] = obj_id_list[i]

    if filename == None:
        init_cube()
    else:
        init_model(filename)


def reset(filename: str):
    if filename == None:
        init_cube()
    else:
        init_model(filename=filename)
    F_v.fill([0, 0, 0])
    F_C.fill([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    F_grid_m.fill(0)
    F_grid_v.fill([0, 0, 0])


@ti.kernel
def update_J():
    F_grid_deformation.fill(0.0)
    # TODO: scatter FJ on grid nodes
    for p in FJ:
        FJ[p] = F[p].determinant()
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)

        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            F_grid_deformation[base + offset] += weight * FJ[p]


co_d_list = []
co_st_list = []


def step(
    scale,
    inverse_rot_list,
    inverse_pos_list,
    apply_force=False,
    threshold=0.05,
    visualize_deformation=False,
    debug=True
):
    """
    simulate one frame
    """
    co_begain = time.time()
    init_collision_field()

    for i in range(MAX_COLLISION_OBJECTS):
        if co_obj[i][1] == -1:
            tmp = p.getBaseVelocity(co_obj[i][0])
            co_v[i] = tmp[0]
            co_w[i] = tmp[1]
            centroid[i] = p.getBasePositionAndOrientation(co_obj[i][0])[0]
        else:
            centroid[i], _, _, _, _, _, co_v[i], co_w[i] = p.getLinkState(
                co_obj[i][0], co_obj[i][1], computeLinkVelocity=1
            )

    # print("[DEBUG] Needle's Velocity:")
    # print(co_v[None])
    # print("[DEBUG] Needle's Angular Velocity:")
    # print(co_w[None])
    # print("[DEBUG] Needle's Centroid:")
    # print(centroid[None])
    t0=time.time()
    # new_sdf, co_mask = 
    sdf.switch_reference_frame_and_update_sdf(
        i_rot_list=inverse_rot_list,
        i_pos_list=inverse_pos_list,
        co_obj=co_obj
    )
    t1=time.time()
    # print(f'sdf:{(t1-t0)*1000}')
    # for visualizationxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # np.save('./buffer/sdf', new_sdf)
    # np.save('./buffer/mask', co_mask)
    global SDF,collision_mask
    SDF=sdf.tmp_sdf
    collision_mask=sdf.co_mask
    # np.save('./sdf.npy',SDF.to_numpy())
    # np.save('./mask.npy',collision_mask.to_numpy())
    # print('SDF and mask saved')
    # exit(0)
    
    # SDF.copy_from(sdf.tmp_sdf)
    # collision_mask.copy_from(sdf.co_mask)
    # SDF.from_numpy(new_sdf)
    # collision_mask.from_numpy(co_mask)

    co_end = time.time()
    if debug:
        print(f"collision detection using {(co_end-co_begain)*1000}ms")
        co_d_list.append((co_end - co_begain) * 1000)

    s_b = time.time()
    for _ in range(steps):
        substep(scale, threshold)
    s_e = time.time()
    if debug:
        print(f"Soft body simulation used {(s_e-s_b)*1000.0}ms")
        co_st_list.append((s_e - s_b) * 1000.0)

    # if apply_force:
    #     # avg_external_forces(steps)
    #     apply_external_forces(
    #         scale,
    #         mask=collision_mask.to_numpy(),
    #         forces=external_forces.to_numpy(),
    #         debug=True,
    # )
    if visualize_deformation:
        update_J()#visualization,damage the efficiency

    # np.save('./grid_deformation', F_grid_deformation.to_numpy())
    # exit(0)
    # for debug

    # return pos.to_numpy().reshape((-1, 3)), FJ.to_numpy()



g2c = []
mc = []


def get_mesh(smooth_scale=0.2, debug=True):
    """
    Use Marching-Cubes algorithm to construct the surface from density(mass) field.
    """
    t0 = time.time()
    t = F_grid_m.to_numpy()
    t1 = time.time()
    print(f"GPU/CPU Communication time: {(t1-t0)*1000}ms")
    if debug:
        g2c.append((t1 - t0) * 1000)
    # t = density_field.to_numpy()
    mmax, mmin = t.max(), t.min()
    level = mmin + (mmax - mmin) * smooth_scale
    # default gradient direction is 'descent' which may lead to rendering problem

    t0 = time.time()
    vtx, faces, normals, _ = skimage.measure.marching_cubes(
        t, level, gradient_direction="ascent"
    )
    faces = faces.reshape(-1)
    t1 = time.time()
    print(f"Marching Cubes time: {(t1-t0)*1000}ms")
    if debug:
        mc.append((t1 - t0) * 1000)

    # print("[DEBUG]Vertices:%d, Faces:%d" % (len(vtx), len(faces)))
    # exit(0)
    
    # print(vtx.shape)
    return vtx, faces, normals


def main(show=False):
    # used to record video
    result_dir = "../video"
    video_manager = ti.tools.VideoManager(
        output_dir=result_dir, framerate=24, automatic_build=True
    )
    init()
    # SignedDistanceField = sdf.load_mesh_fast('./model/cube.obj',
    #                                          n_grid,
    #                                          scale_ratio=1.0)
    # print(SignedDistanceField.shape)

    # init_sdf(SignedDistanceField)

    window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    i = 0

    # sdf.transform(0.2, 0.0, 0.2)

    while window.running:
        i = i + 1
        angle = float(i) * PI / 180.0

        # camera.position(5.0 * ti.cos(angle), 0.0, 5.0 * ti.sin(angle))
        # camera.position(0.2 + ti.cos(angle), 0.8, 2.0 + ti.sin(angle))
        camera.position(0.2, 3.0, 0.0)
        camera.lookat(0.2, 0.0, 0.2)
        # camera.track_user_inputs(window,
        #                          movement_speed=0.005,
        #                          hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        for _ in range(steps):
            substep()

        # update_co_position()

        # update sdf by switching reference frame
        # new_sdf = sdf.switch_reference_frame_and_update_sdf(pos.to_numpy())
        # SDF.from_numpy(new_sdf)

        scene.particles(F_x, radius=0.001, color=ORIANGE)

        # scene.particles(grid_pos, radius=0.005, per_vertex_color=CDF)

        # scene.mesh(sdf.vertices, sdf.indices)
        tmp = F_grid_m.to_numpy()
        print(tmp.max(), tmp.min())
        mmax, mmin = tmp.max(), tmp.min()
        level = mmin + (mmax - mmin) / 5.0
        vtx, faces, _, _ = skimage.measure.marching_cubes(tmp, level)
        faces = faces.reshape(-1)
        soft_body_vertices = ti.Vector.field(3, dtype=float, shape=vtx.shape[0])
        soft_body_vertices.from_numpy(vtx / (n_grid * 1.0))
        soft_body_indices = ti.field(dtype=int, shape=faces.shape[0])
        soft_body_indices.from_numpy(faces)
        scene.mesh(soft_body_vertices, soft_body_indices, two_sided=True, color=ORIANGE)
        canvas.scene(scene)
        # video_manager.write_frame(window.get_image_buffer_as_numpy())
        if show:
            window.show()

        # if export_file:
        #     writer = ti.tools.PLYWriter(num_vertices=n_particles)
        #     writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        #     writer.export_frame_ascii(gui.frame, export_file)

        # also can be replace by ti.ui.Scene()
        # gui.circle(ball_center[0], radius=45, color=0x068587)
        # gui.circles(T(pos), radius=1.5, color=0xED553B)
        # video_manager.write_frame(gui.get_image())
        # gui.show()


if __name__ == "__main__":
    main(True)
