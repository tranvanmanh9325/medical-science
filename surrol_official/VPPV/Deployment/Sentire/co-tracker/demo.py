# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio
import numpy as np
import json

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def read_images_from_path(image_root, image_file_names):
    images = []
    for filename in image_file_names:
        if '_left.jpg' in filename:
            filename = os.path.join(image_root, filename)
            image = imageio.imread(filename)

            image_array = np.array(image)
            images.append(image_array)
        
    return np.stack(images)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args()


    # load the input video frame by frame
    # video = read_video_from_path(args.video_path)
    # video = read_images_from_path()
    # video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    # segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    # segm_mask = torch.from_numpy(segm_mask)[None, None]

    if args.checkpoint is not None:
        model = CoTrackerPredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(DEFAULT_DEVICE)
    # video = video.to(DEFAULT_DEVICE)

    directory = '/research/d1/rshr/jwfu/CSR/data_jiawei_new/move'

    for root, dirs, files in os.walk(directory):
        if 'contact_points_last_manual_labeled.json' in files:
            images = [img for img in files if '_left.jpg' in img]
            images.sort(reverse=True)
            query_name = images[0][:15]
            print(f'query_name: {query_name}')
            print(f'root: {root}')

            video = read_images_from_path(root, images)
            print(f'video size: {video.shape}')
            video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
            video = video.to(DEFAULT_DEVICE)
            with open(os.path.join(root, 'contact_points_last_manual_labeled.json'), 'r') as f:
                init_data = json.load(f)
                query_normalized = init_data[query_name]
            query = torch.tensor([[0., query_normalized[0] * 740, query_normalized[1] * 512]]).unsqueeze(0).to(DEFAULT_DEVICE)

            # video = video[:, :20]
            # query = torch.tensor([[0., 510., 375.]]).unsqueeze(0).to(DEFAULT_DEVICE)
            pred_tracks, pred_visibility = model(
                video,
                queries=query,
                # grid_size=args.grid_size,
                # grid_query_frame=args.grid_query_frame,
                # backward_tracking=args.backward_tracking,
                # segm_mask=segm_mask
            )
            print("computed")
            print(f'pred_tracks size: {pred_tracks.size()}')
            print(f'pred_visibility size: {pred_visibility.size()}')
            print(f'pred_tracks: {pred_tracks.size()}')
            print('*' * 100)


            data = dict()
            pred_tracks = pred_tracks.squeeze(0)
            for i in range(pred_tracks.size(0)):
                data.update({images[i][:15]: [pred_tracks[i][0][0].item() / 740, pred_tracks[i][0][1].item() / 512]})
            pred_tracks = pred_tracks.unsqueeze(0)


            data = dict(sorted(data.items()))

            # with open(os.path.join(root, 'contact_points_full.json'), 'w') as f:
            #         json.dump(data, f)


            # save a video with predicted tracks
            seq_name = args.video_path.split("/")[-1]
            vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
            vis.visualize(
                video,
                pred_tracks,
                pred_visibility,
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
                filename=os.path.basename(root),
            )
