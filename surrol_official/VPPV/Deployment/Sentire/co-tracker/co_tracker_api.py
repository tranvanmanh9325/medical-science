import os
import torch
import argparse
import imageio
import numpy as np
import json

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

class CoTrackerVisualSurrolAPI:

    def __init__(self):
        self.model = CoTrackerPredictor(checkpoint='checkpoints/cotracker2.pth')
        self.device = 'cuda'
        self.model.to(self.device)
        self.video_st = []
        self.point = None

    def step(self, img, point=None):

        # point: [w, h]

        if point is not None:
            self.point = point
        assert self.point is not None

        # Assert img follows imageio format
        self.video_st.append(img)
        video = torch.from_numpy(self.video_st).permute(0, 3, 1, 2)[None].float()
        video = video.to(self.device)

        # NOTE: take care order of H and W
        # H, W = img.size()[-2:]
        query = torch.tensor([[0., self.point[0], self.point[1]]]).unsqueeze(0).to(self.device)

        pred_tracks, pred_visibility = self.model(
            video,
            queries=query,
            # grid_size=args.grid_size,
            # grid_query_frame=args.grid_query_frame,
            # backward_tracking=args.backward_tracking,
            # segm_mask=segm_mask
        )
        pred_tracks = pred_tracks.squeeze(0)

        new_w, new_h = pred_tracks[-1][0][0].item(), pred_tracks[-1][0][1].item()

        return [new_w, new_h]



