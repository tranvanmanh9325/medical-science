import os
import sys
import numpy as np
import open3d as o3d
from PIL import Image
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement


def depth_map_to_point_cloud(depth_map, rgb, mask, camera_intrinsics):
    # Get camera intrinsic parameters
    focal_length = camera_intrinsics["focal_length"]
    principal_point = camera_intrinsics["principal_point"]

    # Initialize the point cloud
    point_cloud = []
    colors = []

    # Iterate over each pixel in the depth map
    if len(depth_map.shape)==2:
        height, width = depth_map.shape
    elif len(depth_map.shape)==3:
        height, width, _ = depth_map.shape

    for y in range(height):
        for x in range(width):
            # Get the depth value for the current pixel
            if(len(depth_map.shape)==3):
                depth = depth_map[y, x, 0]
            else:
                depth = depth_map[y, x]
            if not mask[y,x]:
                continue
            # Calculate the 3D coordinates
            X = (x - principal_point[0]) * depth / focal_length
            Y = (y - principal_point[1]) * depth / focal_length
            Z = depth

            # Store the 3D point in the point cloud
            point_cloud.append([X, Y, Z])
            colors.append(rgb[y,x])

    # Convert the point cloud to a NumPy array
    point_cloud = np.array(point_cloud)
    colors = np.array(colors)

    return point_cloud, colors[...,:3]

def get_depth(dir_path:str,idx:int):
    return np.array(Image.open(os.path.join(dir_path, f"depth/frame-{idx:06d}.depth.png")))

def get_rgb(dir_path:str,idx:int):
    return np.array(Image.open(os.path.join(dir_path, f"images/frame-{idx:06d}.color.png")))
    
def merge_depth(dir_path: str, dest_img, src_img):
    
    if isinstance(dest_img, int):
        d_img = get_depth(dir_path, dest_img)
    elif isinstance(dest_img, np.ndarray):
        d_img = dest_img
    
    if isinstance(src_img, int):
        s_img = get_depth(dir_path, src_img)
    elif isinstance(src_img, np.ndarray):
        s_img = src_img
        
    mask = d_img == 0.0
    d_img[mask] = s_img[mask]
    depth_map = d_img
    return depth_map, mask

def merge_rgb(dir_path: str, dest_img, src_img, mask):
    
    if isinstance(dest_img, int):
        d_img = get_rgb(dir_path, dest_img)
    elif isinstance(dest_img, np.ndarray):
        d_img = dest_img
    
    if isinstance(src_img, int):
        s_img = get_rgb(dir_path, src_img)
    elif isinstance(src_img, np.ndarray):
        s_img = src_img
        
    mask = mask[...,np.newaxis]
    d_img = d_img*(1-mask) + s_img*mask
    rgb = d_img
    return rgb


def genPLY(dir_path, depth_map, rgb, mask):
    if len(depth_map.shape)==2:
        H, W = depth_map.shape
    elif len(depth_map.shape)==3:
        H,W,_ = depth_map.shape

    print(f"Image Shape {H}, {W}")
    cam_intr = {}
    cam_intr["focal_length"] = 569.5
    cam_intr["principal_point"] = (W / 2.0, H / 2.0)

    pcd = o3d.geometry.PointCloud()

    points, colors = depth_map_to_point_cloud(depth_map, rgb, mask, cam_intr)
    # points = np.random.rand(10000, 3)
    # colors = np.random.rand(10000, 3)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=3)
    points = np.array(down_pcd.points, dtype=np.float32)
    colors = np.array(down_pcd.colors, dtype=np.uint8)
    
    print(f"Number of points: {len(points)}")

    normals = np.zeros_like(colors, dtype=np.float32)

    # Create the PlyElement for vertices
    vertices = np.core.records.fromarrays(
        [
            points[:, 0],
            points[:, 1],
            points[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        ],
        names=["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"],
    )
    vertex_element = PlyElement.describe(vertices, "vertex")

    # Create PlyData object and save to file
    plydata = PlyData([vertex_element])
    plydata.write(os.path.join(dir_path, "d_pcd.ply"))

    print("The PLY file is generated.")


def gen_random_ply(dir_path, p_number=10000):
    pcd = o3d.geometry.PointCloud()
    points = np.random.uniform(0, 1, size=(p_number, 3))
    points[:, 2] += 3
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd

    # save pcd
    points = np.array(down_pcd.points, dtype=np.float32)
    print(f"Number of points: {len(points)}")
    colors = np.zeros_like(points, dtype=np.uint8)
    normals = colors.astype(np.float32)

    # Create the PlyElement for vertices
    vertices = np.core.records.fromarrays(
        [
            points[:, 0],
            points[:, 1],
            points[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        ],
        names=["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"],
    )
    vertex_element = PlyElement.describe(vertices, "vertex")

    # Create PlyData object and save to file
    plydata = PlyData([vertex_element])
    plydata.write(os.path.join(dir_path, "o_pcd.ply"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", type=str, default="no given data diirectory")
    args = parser.parse_args(sys.argv[1:])
    data_path = args.data_dir
    src_img = 0
    dst_img = 20
    merged_depth = merge_depth(data_path, src_img, dst_img)
    genPLY(data_path, merged_depth)
