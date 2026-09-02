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

if __name__ == "__main__":
    gs_model = GaussianModel(sh_degree=3)
    task_name = "stomach"
    model_path = (
        "./output/"
        + task_name
        + "/point_cloud/iteration_7000/point_cloud.ply"
    )
    gs_model.load_ply(model_path)

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
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)

    img = render(cam, gs_model, pp, bg_color)["render"]  # (3, H, W) tensor
    array = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    plt.axis("off")
    plt.imsave(f"./{task_name}.png", array.clip(0, 1))