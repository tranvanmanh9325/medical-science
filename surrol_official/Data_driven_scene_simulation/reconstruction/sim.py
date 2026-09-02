# import warp as wp
import time
import torch
import numpy as np
from tqdm import tqdm
from padding import padding
import matplotlib.pyplot as plt
from scene import GaussianModel
from argparse import ArgumentParser
from gaussian_renderer import render
from arguments import PipelineParams
from warpMPM import wp, save_data_at_frame, MPM_Simulator_WARP, wp_bbox


class tiny_camera:
    def __init__(
        self,
        FoVx,
        FoVy,
        image_height,
        image_width,
        world_view_transform,
        full_proj_transform,
        camera_center,
        zfar,
    ):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = image_height
        self.image_width = image_width
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.zfar = zfar


def red_print(p: str):
    print("\033[91m" + p + "\033[0m")


if __name__ == "__main__":
    wp.init()
    # wp.config.verify_cuda=True #only for debug
    dvc = "cuda:0"

    pre_solver = MPM_Simulator_WARP(10)
    # Load Gaussian
    gs_model = GaussianModel(sh_degree=3)
    task_name = "sr1"
    model_path = (
        "./output/"
        + task_name
        + "/point_cloud/iteration_3000/point_cloud.ply"
    )

    # model_path = (
    #     "./output/"
    #     + task_name
    #     + "/point_cloud.ply"
    # )

    gs_model.load_ply(model_path)
    xyz = gs_model._xyz.clone().detach()

    # normalization
    l_min = xyz.min()
    l_max = xyz.max()
    xyz = (xyz - l_min) / (l_max - l_min)
    gs_model._xyz = torch.nn.Parameter(xyz)

    # GS pruning
    bottom = 1.0  # hyperparameter
    mask = gs_model._xyz[:, 2] > bottom
    gs_model.prune_points_without_training(mask=mask)
    volume_tensor = torch.ones(gs_model.get_gs_number) * 2.5e-8
    pre_solver.load_initial_data_from_torch(
        gs_model.get_xyz,
        gs_model.get_opacity,
        volume_tensor,
        tensor_cov=gs_model.get_covariance(),
    )

    # Pre-computation: compute the opacity field
    t0 = time.time()
    pre_solver.compute_opacity_field()
    op_field = pre_solver.export_opacity_field_to_torch().cpu().numpy()
    op_map = np.zeros((100, 100))
    for i in reversed(range(100)):
        slice = op_field[:, :, i]
        mask = slice > 10.0  # hyper-parameter
        op_map[mask] = i
    _, points = padding(op_map, bottom)
    t1 = time.time()
    red_print(f"Processing Time: {t1-t0}s")
    #####################################################
    mpm_solver = MPM_Simulator_WARP(10)
    pad_number = len(points)

    total_particle_number = gs_model.get_gs_number + pad_number
    volume_tensor = torch.ones(total_particle_number) * 2.5e-8
    red_print(f"Gaussian Number:{total_particle_number}")
    points = torch.tensor(points.astype(np.float32), device="cuda:0")
    total_xyz = torch.cat([gs_model.get_xyz, points], dim=0)
    print(total_xyz.shape)
    pad_cov = torch.zeros((pad_number, 6), dtype=torch.float32, device="cuda:0")
    total_cov = torch.cat([gs_model.get_covariance(), pad_cov], dim=0)
    print(total_cov.shape)

    # load gaussians in mpm solver
    mpm_solver.load_initial_data_from_torch(
        total_xyz, None, volume_tensor, tensor_cov=total_cov
    )


    material_params = {
        "E": 6000,
        "nu": 0.2,
        "material": "jelly",
        "g": [0.0, 0.0, 0.0],
        "density": 100000.0,
    }

    dt = 0.0005
    steps = 800

    mpm_solver.set_parameters_dict(material_params)
    mpm_solver.finalize_mu_lam()  # set mu and lambda from the E and nu input
    # add constraints
    mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0, 0.0, 1.0), "sticky", 0.0)
    mpm_solver.mpm_model.bbox = wp_bbox()
    bbox = mpm_solver.mpm_model.bbox
    bbox.padding = 3
    bbox.x0 = 7
    bbox.x1 = 50 #48
    bbox.y0 = 13
    bbox.y1 = 38 #44
    bbox.z0 = 3
    bbox.z1 = 90

    directory_to_save = "./sim_results"
    save_data_at_frame(
        mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False
    )
    # Apply external force to simulate the deformation
    f = 0.00

    # Single Force
    external_force = (-0.5*f, f, 0.0)
    force_position = (0.48, 0.43, 0.74)
    force_range = (0.05, 0.05, 0.03)

    mpm_solver.add_impulse_on_particles(
        force=external_force,
        dt=dt,
        point=force_position,
        size=force_range,
        num_dt=12000,
        start_time=0.0,
    )

    ## Multi Forces
    # external_force_list=[(f,f,0),(0,f,0),(f,0,0),(-f,0,-f)]
    # force_position_list=[(0.28, 0.28, 0.61),(0.31, 0.36, 0.5),(0.39, 0.31, 0.506),(0.31, 0.36, 0.5)]
    # force_range_list=[(0.05,0.05,0.01),(0.05, 0.05, 0.01),(0.05, 0.05, 0.01),(0.03, 0.03, 0.01)]
    # st=0.0

    # for external_force, force_position, force_range in zip(external_force_list,force_position_list,force_range_list):
    #     st = mpm_solver.add_impulse_on_particles(force= external_force,
    #                                             dt= dt,
    #                                             point= force_position,
    #                                             size= force_range,
    #                                             num_dt= 100,
    #                                             start_time= st)+23*dt*steps

    # rendering configuration
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    pp.compute_cov3D_python = True  # Using covariance instead of scales and rotations
    pp.use_deformed_cov = True

    # fixed camera from endo-dataset
    cam = tiny_camera(
        FoVx=1.0239093368021417,
        FoVy=1.0239093368021417,
        image_height=512,
        image_width=640,
        world_view_transform=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        ),
        full_proj_transform=torch.tensor(
            [
                [1.7796, 0.0000, 0.0000, 0.0000],
                [0.0000, 2.2245, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0001, 1.0000],
                [0.0000, 0.0000, -0.0100, 0.0000],
            ],
            device="cuda:0",
        ),
        camera_center=torch.tensor([0.0, 0.0, 0.0], device="cuda:0"),
        zfar=10000,
    )

    # cam parameter for ham1 and ham2
    # cam = tiny_camera(FoVx=0.6503751964113784,
    #                     FoVy=0.3340161973201237,
    #                     image_height=192,
    #                     image_width=384,
    #                     world_view_transform=torch.tensor([[1., 0., 0., 0.],
    #                         [0., 1., 0., 0.],
    #                         [0., 0., 1., 0.],
    #                         [0., 0., 0., 1.]], device='cuda:0'),
    #                     full_proj_transform=torch.tensor([[ 2.9660,  0.0000,  0.0000,  0.0000],
    #                         [ 0.0000,  5.9320,  0.0000,  0.0000],
    #                         [ 0.0000,  0.0000,  1.0001,  1.0000],
    #                         [ 0.0000,  0.0000, -0.0100,  0.0000]], device='cuda:0'),
    #                     camera_center=torch.tensor([0., 0., 0.], device='cuda:0'),
    #                     zfar=10000)

    # camera for endo-gaussian
    # cam = tiny_camera(FoVx=0.356128261356895,
    #                     FoVy=0.2862919145330625,
    #                     image_height=512,
    #                     image_width=640,
    #                     world_view_transform=torch.tensor([[ 0.9795, -0.0118,  0.1736,  0.0000],
    #                         [-0.0189,  0.9873,  0.1383,  0.0000],
    #                         [-0.1711, -0.1419,  0.9679,  0.0000],
    #                         [-0.0868, -0.0383,  1.1182,  1.0000]],device='cuda:0'),
    #                     full_proj_transform=torch.tensor([[ 5.4424, -0.0816,  0.1736,  0.1736],
    #                         [-0.1051,  6.8500,  0.1383,  0.1383],
    #                         [-0.9509, -0.9842,  0.9680,  0.9679],
    #                         [-0.4822, -0.2657,  1.1083,  1.1182]],device='cuda:0'),
    #                     camera_center=torch.tensor([0., 0., 0.], device='cuda:0'),
    #                     zfar=10000)

    # simulation loop
    step_list = []
    for k in tqdm(range(15)):
        st = time.time()
        mpm_solver.step(steps=steps, dt=dt)
        se = time.time()
        step_list.append(se - st)
        # save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)

        # canonical sapce to world
        x = mpm_solver.export_particle_x_to_torch()[
            : gs_model.get_gs_number
        ]  # we don't render the padded Gaussian
        x = x * (l_max - l_min) + l_min

        # update evoluted gaussian for new simulation results
        gs_model._xyz = torch.nn.Parameter(x)

        cov_tensor = mpm_solver.export_particle_cov_to_torch()[: gs_model.get_gs_number]
        gs_model.set_deformed_covariance(cov_tensor)

        p_rot = mpm_solver.export_particle_R_to_torch()[: gs_model.get_gs_number]
        gs_model.set_SH_rotation(p_rot)

        # render
        rt = time.time()
        img = render(cam, gs_model, pp, bg_color)["render"]  # (3, H, W) tensor
        re = time.time()
        # print(f'rendering time:{re-rt}')

        # simple visualization
        tensor = img.cpu().detach()
        array = tensor.numpy()
        array = np.transpose(array, (1, 2, 0))
        plt.axis("off")
        plt.imsave(f"./render_results/{k:06d}.png", array.clip(0, 1))
        # print(f"./render_results/{k:06d}.png saved")

    FPS = 1.0 / (np.array(step_list).mean())
    FPS = int(FPS)
    red_print(f"Avg FPS:{FPS}")

    data = {"force": external_force, "position": force_position, "scale": force_range}

    # Save the force information to a text file
    with open("./render_results/output.txt", "w") as file:
        for name, lst in data.items():
            file.write(f"{name}: {lst}\n")

    print("done!")
