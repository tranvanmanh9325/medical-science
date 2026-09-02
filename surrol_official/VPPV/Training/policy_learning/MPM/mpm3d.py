import os
import time
import numpy as np
import taichi as ti
import pybullet as p
import skimage.measure
from plyfile import PlyData
import MPM.config as mpmconf
from MPM.config import n_grid, grid_shape
if mpmconf.USE_GPU:
    ti.init(arch=ti.cuda)
else:
    ti.init(arch=ti.cpu)
import MPM.sdf as sdf
@ti.data_oriented
class MPM_Solver:

    def __init__(self):
        self.frame = 0
        self.normalized_center = None

    def initialize_from_ply(self, model_path, MAX_COLLISION_OBJECTS=3):
        from gs.scene import GaussianModel
        gs_model = GaussianModel(sh_degree=3)
        gs_model.load_ply(model_path)
        xyz = gs_model._xyz.clone().detach()
        # normalization
        bb_min = xyz.min(0).values
        bb_max = xyz.max(0).values
        scale = 0.8
        scale /= (bb_max - bb_min).max()
        xyz = (xyz - bb_min) * scale
        self.normalize_scale = scale
        self.normalize_base = bb_min
        self.normalized_center = (xyz.min(0).values + xyz.max(0).values) * 0.5
        self.normalized_center[2] = 0.0
        self.initialize_from_torch(
            xyz,
            gs_model.get_covariance(),
            gs_model.get_opacity.squeeze(),
            MAX_COLLISION_OBJECTS,
        )

    def initialize_from_torch(self, xyz, cov, opacity=None, MAX_COLLISION_OBJECTS=3):
        assert xyz.shape[0] == cov.shape[0]
        if opacity is not None:
            assert cov.shape[0] == opacity.shape[0]

        n_particles = xyz.shape[0]
        self.init_kernel(n_particles, MAX_COLLISION_OBJECTS)
        self.F_x.from_torch(xyz)
        self.Cov.from_torch(cov)
        if opacity is not None:
            self.particle_opacity.from_torch(opacity)

        if self.normalized_center is None:
            self.normalized_center = (xyz.min(0).values + xyz.max(0).values) * 0.5
            self.normalized_center[2] = 0.0

    def init_kernel(self, n_particles, MAX_COLLISION_OBJECTS=3, gravity = 0.0, p_rho =1000):
        dim = 3
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.MAX_COLLISION_OBJECTS = MAX_COLLISION_OBJECTS
        self.dim = dim

        self.steps, self.dt = 25, 5e-4
        self.timestep = self.steps * self.dt

        self.dx = 1.0 / n_grid
        self.inv_dx = 1.0/self.dx
        self.p_rho = p_rho
        self.p_vol = (self.dx * 0.5) ** dim
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = gravity
        self.bound = 3  # important for stable simulation

        self.E = ti.field(float, shape=())
        self.nu = ti.field(float, shape=())
        self.mu = ti.field(float, shape=())
        self.la = ti.field(float, shape=())
        self.F_x = ti.Vector.field(dim, float, n_particles)
        self.F_v = ti.Vector.field(dim, float, n_particles)
        self.F_C = ti.Matrix.field(dim, dim, float, n_particles)
        self.Cov = ti.Vector.field(6, float, n_particles)
        self.Cov_deformed = ti.Vector.field(6, float, n_particles)
        self.F = ti.Matrix.field(dim, dim, float, n_particles)
        self.FJ = ti.field(float, n_particles)
        self.particle_opacity = ti.field(float, n_particles)

        self.F_grid_v = ti.Vector.field(dim, float, shape=grid_shape)
        self.F_grid_m = ti.field(float, grid_shape)
        self.F_grid_deformation = ti.field(float, grid_shape)
        self.pos = ti.Vector.field(dim, dtype=float, shape=grid_shape)
        self.opacity_field = ti.field(float, grid_shape)

        self.grid_pos = ti.Vector.field(dim, dtype=float, shape=n_grid**dim)
        self.ref_point = ti.Vector([0.3, 0.0, 0.0])
        self.normal = ti.Vector([1.0, 1.0, 1.0]).normalized()

        self.fe = 1.0  # friction coefficient
        self.neighbour = (3,) * dim
        self.ORIANGE = ti.Vector(
            (0.9294117647058824, 0.3333333333333333, 0.23137254901960785)
        )
        self.PI = 3.1415926

        # co_obj = ti.Vector.field(2, int, shape=MAX_COLLISION_OBJECTS)  # (obj_id,link_id)]
        self.co_obj = np.zeros((MAX_COLLISION_OBJECTS, 2), dtype=int)
        self.co_v = ti.Vector.field(
            dim, dtype=float, shape=MAX_COLLISION_OBJECTS
        )  # translational velocity
        self.co_w = ti.Vector.field(
            dim, dtype=float, shape=MAX_COLLISION_OBJECTS
        )  # angular velocity
        self.centroid = ti.Vector.field(dim, dtype=float, shape=MAX_COLLISION_OBJECTS)

        self.soft_body_base_position = ti.Vector.field(3, dtype=float, shape=())

        self.SDF = None
        self.collision_mask = None

        self.external_forces = ti.Vector.field(3, dtype=float, shape=grid_shape)
        self.MAX_COLLISION_NUM = 50

        self.init_deformation_gradient()  # set F to identity matrix

        self.co_d_list = []
        self.co_st_list = []
        self.g2c = []
        self.mc = []

    def get_normalize_parameter(self):
        return self.normalize_base, self.normalize_scale

    def set_normalize_parameter(self, base, scale):
        self.normalize_base = base
        self.normalize_scale = scale

    def set_boundary_box(self, bbox):
        self.bbox = bbox

    def set_parameters(self, s_E=8000, s_nu=0.2):
        print(f"MPM Parameters: Young's Modulus: {s_E}, Possion Ratio: {s_nu}")
        self.E[None] = s_E
        self.nu[None] = s_nu
        self.mu[None] = self.E[None] / (2 * (1 + self.nu[None]))
        self.la[None] = (
            self.E[None]
            * self.nu[None]
            / ((1 + self.nu[None]) * (1 - 2 * self.nu[None]))
        )

    def set_base_position(self, base_position):
        self.soft_body_base_position[None] = ti.Vector(base_position)

    def init_collision_field(self):
        # collision_mask.fill(-1)
        self.external_forces.fill([0.0, 0.0, 0.0])

    def apply_external_forces(self, scale, mask, forces, debug=False):
        idx = mask > 0
        t = idx.nonzero()
        t = np.moveaxis(t, 0, -1)
        num_collision = idx.sum()
        if num_collision > self.MAX_COLLISION_NUM:
            sample_idx = np.random.choice(
                list(range(num_collision)), size=self.MAX_COLLISION_NUM, replace=True
            )
            t = t[sample_idx]

        t0 = time.time()
        for i in t:
            tag = mask[i[0], i[1], i[2]]
            p.applyExternalForce(
                self.co_obj[tag][0],
                self.co_obj[tag][1],
                forceObj=forces[tuple(i)],
                posObj=i * self.dx * scale + self.soft_body_base_position[None],
                flags=p.WORLD_FRAME,
            )
            if debug:
                print(forces[tuple(i)])
        t1 = time.time()
        if debug:
            print("total collision grid node:%d\nuse time:%fs" % (idx.sum(), t1 - t0))

    @ti.kernel
    def compute_grid_opacity(self):
        for p in self.F_x:
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            # TODO: check boundary
            init_cov = ti.Matrix([[0 for _ in range(3)] for _ in range(3)], dt=float)
            init_cov[0, 0] = self.Cov[p][0]
            init_cov[0, 1] = self.Cov[p][1]
            init_cov[0, 2] = self.Cov[p][2]
            init_cov[1, 0] = self.Cov[p][1]
            init_cov[1, 1] = self.Cov[p][3]
            init_cov[1, 2] = self.Cov[p][4]
            init_cov[2, 0] = self.Cov[p][2]
            init_cov[2, 1] = self.Cov[p][4]
            init_cov[2, 2] = self.Cov[p][5]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.exp(-0.5 * dpos @ init_cov.inverse() @ dpos)
                self.opacity_field[base + offset] += weight * self.particle_opacity[p]

    @ti.func
    def P2G(self):
        for p in self.F_x:
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # stress = -dt * 4 * E * p_vol * (F_J[p] - 1) / dx**2
            # affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]
            I = ti.Matrix.identity(float, self.dim)
            self.F[p] = (I + self.dt * self.F_C[p]) @ self.F[p]
            J = 1.0
            _, sig, _ = ti.svd(self.F[p])
            for i in ti.static(range(self.dim)):
                J *= sig[i, i]
                # F_visual[p][i] = sig[i, i]

            # Neo-Hookean
            stress = self.mu[None] * (self.F[p] @ self.F[p].transpose() - I) + I * self.la[None] * ti.log(J)

            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress

            # PIC
            # affine = stress
            # APIC
            affine = stress + self.p_mass * self.F_C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                grid_x, grid_y, grid_z = base + offset
                if (
                    grid_x >= 0
                    and grid_y >= 0
                    and grid_z >= 0
                    and grid_x < n_grid
                    and grid_y < n_grid
                    and grid_z < n_grid
                ):
                    dpos = (offset - fx) * self.dx
                    weight = 1.0
                    for i in ti.static(range(self.dim)):
                        weight *= w[offset[i]][i]

                    self.F_grid_v[base + offset] += weight * (
                        self.p_mass * self.F_v[p] + affine @ dpos
                    )
                    self.F_grid_m[base + offset] += weight * self.p_mass
                    self.external_forces[base + offset] += (
                        0.1 * weight * (stress @ dpos) / self.dt
                    )
                    # density_field[mc_base + offset] += weight * p_mass

    @ti.func
    def Boundary(self, scale, threshold=0.05):
        inv_scale = 1.0 / scale
        for I in ti.grouped(self.F_grid_m):
            if self.F_grid_m[I] > 0:
                self.F_grid_v[I] /= self.F_grid_m[I]
                # taichi(Y) --> pyBullet(Z)
                # F_grid_v[I][1] -= dt * gravity  #taichi
                self.F_grid_v[I][2] -= self.dt * self.gravity  # pybullet

            open_collision_handling = True #Control Collision detection and handling

            if open_collision_handling and SDF[I] < threshold and self.F_grid_m[I] > 0:
                tag = collision_mask[I]
                self.F_grid_v[I] = self.co_v[tag]*inv_scale
            else:
                collision_mask[I] = -1

            cond = (I < self.bound) & (self.F_grid_v[I] < 0) | (
                I > ti.Vector(grid_shape) - self.bound
            ) & (self.F_grid_v[I] > 0)
            self.F_grid_v[I] = ti.select(cond, 0, self.F_grid_v[I])

            # add anchor points(constraints)
            anchor = I[2] == 30 and (
                (I[1] >= 10 and I[1] <= 15) or (I[1] >= 30 and I[1] <= 35)
            )

            anchor = 0

            if anchor:
                self.F_grid_v[I] = ti.Vector([0.0, 0.0, 0.0])

            # bounding box constraint
            # bbox = self.bbox
            # grid_x, grid_y, grid_z = I[0], I[1], I[2]
            # if (
            #     grid_x < bbox.x0 + bbox.padding
            #     or grid_x >= bbox.x1 - bbox.padding
            #     or grid_y < bbox.y0 + bbox.padding
            #     or grid_y >= bbox.y1 - bbox.padding
            #     or grid_z < bbox.z0 + bbox.padding
            #     or grid_z >= bbox.z1 - bbox.padding
            # ):
            #     #self.F_grid_v[I] = ti.Vector([0.0, 0.0, 0.0], dt=float)
            #     pass

            #     F_grid_v[I][0]=.0
            #     F_grid_v[I][1]=.0
            # vt = F_grid_v[I] - F_grid_v[I][2] * ti.Vector([.0, .0, 1.0])
            # F_grid_v[I] = 0.8 * vt + F_grid_v[I][2] * ti.Vector([.0, .0, 1.0])

            # v2 = F_grid_v[I]
            # external_forces[I] += F_grid_m[I] * (v1 - v2) / dt
            if I[2] <= 3:
                self.F_grid_v[I][0] *= 0.1
                self.F_grid_v[I][1] *= 0.1

    @ti.func
    def G2P(self):
        for p in self.F_x:
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(self.F_v[p])
            new_C = ti.zero(self.F_C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                grid_x, grid_y, grid_z = base + offset
                if (
                    grid_x >= 0
                    and grid_y >= 0
                    and grid_z >= 0
                    and grid_x < n_grid
                    and grid_y < n_grid
                    and grid_z < n_grid
                ):
                    dpos = offset - fx
                    weight = 1.0
                    for i in ti.static(range(self.dim)):
                        weight *= w[offset[i]][i]
                    g_v = self.F_grid_v[base + offset]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.F_v[p] = new_v

            #!:code follows may lead to velocity discontinuity(material crash)
            # stick suturing pad on floor
            # sticky_parameter = 3.0  #this parameter is used to fixed pad on table
            # if F_x[p][2] < sticky_parameter * 1.0 / n_grid:
            #     F_v[p] = ti.Vector([0.0, 0.0, 0.0])

            self.F_x[p] += self.dt * self.F_v[p]
            self.F_C[p] = new_C

    @ti.kernel
    def substep(self, scale: float, threshold: float):
        # for I in ti.grouped(F_grid_m):
        #     F_grid_v[I] = ti.zero(F_grid_v[I])
        #     F_grid_m[I] = 0
        self.F_grid_v.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.F_grid_m.fill(0)
        # density_field.fill(0)
        # ti.loop_config(block_dim=n_grid, parallelize=n_grid)
        self.P2G()
        # ti.loop_config(block_dim=n_grid, parallelize=n_grid)
        self.Boundary(scale, threshold)
        # ti.loop_config(block_dim=n_grid, parallelize=n_grid)
        self.G2P()

    @ti.kernel
    def pos_generation(self):
        for i in ti.grouped(self.pos):
            self.pos[i] = i * self.dx

    def init_pos(self):
        self.pos_generation()
        sdf.init_switch_base(self.pos)

    def init_sdf(self, np_sdf):
        sdf.init_static_sdf(np_sdf)
        print("************Signed Distance Field has been initialized.************")

    def interpolate_deformation(self, base, dx, dy, dz):
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
            if self.F_grid_m[idx] > 0.0:
                deformation += (self.F_grid_deformation[idx] / self.F_grid_m[idx]) * w
        return deformation

    @ti.kernel
    def init_cube(self):
        """
        obj_id: obj's id which need to be detected for collision

        filename: load 3D model to control the shape of soft body, if filename='', this function will generate a cube.
        """
        for i in range(self.n_particles/2):
            self.F[i] = ti.Matrix.identity(float, self.dim)
            # F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.3
            # F_x[i] = ti.Vector([ti.random() * 0.2, ti.random() * 0.2, ti.random() * 0.2])
            self.F_x[i] = ti.Vector(
                [ti.random() * 0.3, ti.random() * 0.15, ti.random() * 0.05]
            )

            self.F_x[i][0] += 0.1
            self.F_x[i][1] += 0.2
            self.F_x[i][2] += 0.05
        
        for i in range(self.n_particles/2, self.n_particles):
            self.F[i] = ti.Matrix.identity(float, self.dim)
            # F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.3
            # F_x[i] = ti.Vector([ti.random() * 0.2, ti.random() * 0.2, ti.random() * 0.2])
            self.F_x[i] = ti.Vector(
                [ti.random() * 0.3, ti.random() * 0.15, ti.random() * 0.05]
            )

            self.F_x[i][0] += 0.1
            self.F_x[i][1] += 0.2
            self.F_x[i][2] += 0.15

    @ti.kernel
    def init_deformation_gradient(self):
        self.F.fill(ti.Matrix.identity(float, self.dim))

    def init_model_from_numpy(self, filename):
        model_array = np.load(filename)
        model_array[:, 0] += 0.1
        model_array[:, 2] += 0.01
        idx = list(range(len(model_array)))
        sampled_idx = np.random.choice(idx, size=self.n_particles, replace=False)
        self.F_x.from_numpy(model_array[sampled_idx])
        self.init_deformation_gradient()
    
    def init_model_from_ply(self, filename):
        ply = PlyData.read(filename)
        vtx = ply['vertex']
        pts = np.stack([vtx['x'], vtx['z'], vtx['y']], axis=-1)
        idx = list(range(len(pts)))
        np.random.seed(0)
        sampled_idx = np.random.choice(idx, size=self.n_particles, replace=True)
        np.random.seed()
        self.F_x.from_numpy(pts[sampled_idx])
        self.init_deformation_gradient()
    
    def choose_idx_given_position(self, pos, threshold):
        x, y, z = pos
        pts = self.F_x.to_numpy()
        mask_x = np.logical_and(pts[:,0]>x-threshold, pts[:,0]<x+threshold)
        mask_y = np.logical_and(pts[:,1]>y-threshold, pts[:,1]<y+threshold)
        mask_z = np.logical_and(pts[:,2]>z-threshold, pts[:,2]<z+threshold)
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        idx = np.where(mask==1)[0]
        return idx


    def init(self, obj_id_list, filename, gs_model, default_young=8000, default_poisson=0.2):
        # co_obj[None] = obj_id
        assert (
            len(obj_id_list) == self.MAX_COLLISION_OBJECTS
        ), f"Please edit MAX_COLLISION_OBJECTS in mpm3d.py to {len(obj_id_list)}."
        self.set_parameters(s_E=default_young, s_nu=default_poisson)
        for i in range(self.MAX_COLLISION_OBJECTS):
            self.co_obj[i] = obj_id_list[i]

        if filename == None:
            self.init_cube()
        elif filename.endswith(".ply"):
            if not gs_model:
                self.init_model_from_ply(filename)
        elif filename.endswith(".npy"):
            self.init_model_from_numpy(filename)
        else:
            print("Uneresolved file format!")
            exit(-1)

    # def reset(self, filename: str):
    #     if filename == None:
    #         self.init_cube()
    #     else:
    #         self.init_model(filename=filename)
    #     self.F_v.fill([0, 0, 0])
    #     self.F_C.fill([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    #     self.F_grid_m.fill(0)
    #     self.F_grid_v.fill([0, 0, 0])
            
    def reset(self):
        self.F_v.fill([0, 0, 0])
        self.F_C.fill([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.F_grid_m.fill(0)
        self.F_grid_v.fill([0, 0, 0])

    @ti.kernel
    def update_J(self):
        self.F_grid_deformation.fill(0.0)
        # TODO: scatter FJ on grid nodes
        for p in self.FJ:
            self.FJ[p] = self.F[p].determinant()
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)

            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                self.F_grid_deformation[base + offset] += weight * self.FJ[p]

    @ti.kernel
    def update_cov(self):
        for p in self.Cov:
            # Cov_deformed[p] = F[p] @ Cov[p] @ F[p].transpose()
            init_cov = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            init_cov[0, 0] = self.Cov[p][0]
            init_cov[0, 1] = self.Cov[p][1]
            init_cov[0, 2] = self.Cov[p][2]
            init_cov[1, 0] = self.Cov[p][1]
            init_cov[1, 1] = self.Cov[p][3]
            init_cov[1, 2] = self.Cov[p][4]
            init_cov[2, 0] = self.Cov[p][2]
            init_cov[2, 1] = self.Cov[p][4]
            init_cov[2, 2] = self.Cov[p][5]

            cov = self.F[p] @ init_cov @ self.F[p].transpose()

            self.Cov_deformed[p][0] = cov[0, 0]
            self.Cov_deformed[p][1] = cov[0, 1]
            self.Cov_deformed[p][2] = cov[0, 2]
            self.Cov_deformed[p][3] = cov[1, 1]
            self.Cov_deformed[p][4] = cov[1, 2]
            self.Cov_deformed[p][5] = cov[2, 2]

    def step(
        self,
        scale,
        inverse_rot_list,
        inverse_pos_list,
        apply_force=False,
        threshold=0.05,
        visualize_deformation=False,
        debug=True,
    ):
        """
        simulate one frame
        """
        self.frame += 1
        co_begain = time.time()
        self.init_collision_field()

        for i in range(self.MAX_COLLISION_OBJECTS):
            if self.co_obj[i][1] == -1:
                tmp = p.getBaseVelocity(self.co_obj[i][0])
                self.co_v[i] = tmp[0]
                self.co_w[i] = tmp[1]
                self.centroid[i] = p.getBasePositionAndOrientation(self.co_obj[i][0])[0]
            else:
                self.centroid[i], _, _, _, _, _, self.co_v[i], self.co_w[i] = (
                    p.getLinkState(
                        self.co_obj[i][0], self.co_obj[i][1], computeLinkVelocity=1
                    )
                )

        sdf.switch_reference_frame_and_update_sdf(
            i_rot_list=inverse_rot_list, i_pos_list=inverse_pos_list, co_obj=self.co_obj, scale=scale
        )

        global SDF, collision_mask
        SDF = sdf.tmp_sdf
        collision_mask = sdf.co_mask

        co_end = time.time()
        if debug:
            print(f"collision detection using {(co_end-co_begain)*1000}ms")
            self.co_d_list.append((co_end - co_begain) * 1000)

        s_b = time.time()
        for _ in range(self.steps):
            self.substep(scale, threshold)

        self.update_cov()

        s_e = time.time()
        if debug:
            print(f"Soft body simulation used {(s_e-s_b)*1000.0}ms")
            self.co_st_list.append((s_e - s_b) * 1000.0)

        # if apply_force:
        #     # avg_external_forces(steps)
        #     apply_external_forces(
        #         scale,
        #         mask=collision_mask.to_numpy(),
        #         forces=external_forces.to_numpy(),
        #         debug=True,
        # )
        if visualize_deformation:
            self.update_J()  # visualization,damage the efficiency

    def get_mesh(self, smooth_scale=0.2, debug=False):
        """
        Use Marching-Cubes algorithm to construct the surface from density(mass) field.
        """
        t0 = time.time()
        t = self.F_grid_m.to_numpy()
        t1 = time.time()
        
        if debug:
            print(f"GPU/CPU Communication time: {(t1-t0)*1000}ms")
            # self.g2c.append((t1 - t0) * 1000)

        mmax, mmin = t.max(), t.min()
        level = mmin + (mmax - mmin) * smooth_scale
        # default gradient direction is 'descent' which may lead to rendering problem

        t0 = time.time()
        vtx, faces, normals, _ = skimage.measure.marching_cubes(
            t, level, gradient_direction="ascent"
        )
        faces = faces.reshape(-1)
        t1 = time.time()
        
        if debug:
            print(f"Marching Cubes time: {(t1-t0)*1000}ms")
            # self.mc.append((t1 - t0) * 1000)

        # print("[DEBUG]Vertices:%d, Faces:%d" % (len(vtx), len(faces)))
        # exit(0)

        # print(vtx.shape, vtx[0])
        return vtx, faces, normals

    def save_ply(self, folder_path):
        series_prefix = "eample.ply"
        np_pos = self.F_x.to_numpy()

        writer = ti.tools.PLYWriter(num_vertices=self.n_particles)
        writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        writer.export_frame_ascii(self.frame, os.path.join(folder_path, series_prefix))
