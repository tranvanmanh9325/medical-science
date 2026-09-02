import torch
# import einops
import sys
sys.path.append('.')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
from torchvision.ops import FeaturePyramidNetwork

# from utils.position_encodings import RotaryPositionEncoding3D
# from utils.layers import RelativeCrossAttentionModule
# from utils.utils import (
#     normalise_quat,
#     sample_ghost_points_uniform_cube,
#     sample_ghost_points_uniform_sphere,
#     compute_rotation_matrix_from_ortho6d
# )
# from utils.resnet import load_resnet50
# from utils.clip import load_clip
# from utils.resnet_2 import load_resnet50
# from utils.img2pcd import images_to_pointcloud

from resnet_3 import load_resnet50_1, load_resnet50_depthseg

from transformer import Transformer, Transformer2

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
    
class Mlp(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self._init_layer()

    def _init_layer(self):
        for layer in self.layers:
            init.zeros_(layer.bias)
            init.kaiming_normal_(layer.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AlohaTransformer(nn.Module):

    def __init__(self,
                 action_length=5,
                 hidden_dim=256,
                 robot_state_dim=10, 
                 action_size=5):
        super().__init__()
        total_num_queries = action_length
        self.action_length = action_length

        self.backbone = load_resnet50_1()
        self.query_embed = nn.Embedding(total_num_queries, hidden_dim)
        self.position_encoder = PositionEmbeddingLearned(128)

        # self.lr_img_feat_proj = nn.Conv2d(4096, 2048, kernel_size=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # We encode robot state into action head directly
        # self.robot_state_proj = nn.Linear(robot_state_dim, hidden_dim)

        self.transformer = Transformer(d_model=hidden_dim,
                                       dropout=0.1,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       num_encoder_layers=5,
                                       num_decoder_layers=5,
                                       normalize_before=True,
                                       return_intermediate_dec=True)
        
        self.action_head = Mlp(hidden_dim+robot_state_dim, 256, action_size, 3)



    def forward(self, img, state):
        """
        left_imgs / right_imgs: [B, 3, H, W]
        states: [B, 8]
        """
        bs = img.size(0)

        # TODO: if we follow the network structure of InterFuser, we can change the transformer architecture to be more similar with it

        img_feat, pos_embedding = self._compute_visual_features(img)
        # print(f'img_feat size: {img_feat.size()}')
        # print(f'pos_embedding size: {pos_embedding.size()}')
        # imgs_feat = torch.cat([left_img_feat, right_img_feat], dim=2)
        # pos_embedding = torch.cat([left_pos_embedding, right_pos_embedding], dim=2)

        hs = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)

        hs, memory = self.transformer(self.input_proj(img_feat), 
                            hs, 
                            pos_embedding,
                            mask=None)
        
        # hs with size: [num_layer, bs, 5, hidden_dim]
        num_layer, num_action = hs.size(0), hs.size(2)
        state = state.unsqueeze(0).unsqueeze(2).repeat(num_layer, 1, num_action, 1)

        # print(f'state size: {state.size()}')
        # print(f'hs size: {hs.size()}')

        action_preds = self.action_head(torch.cat([hs, state], dim=-1))

        return action_preds

    def _compute_visual_features(self, imgs):
        """Compute visual features at different scales and their positional embeddings."""

        imgs_feat = self.backbone(imgs)[3]
        # right_imgs_feat = self.backbone(right_imgs)[3]
        
        imgs_pos_embedding = self.position_encoder(imgs_feat)
        # right_imgs_pos_embedding = self.position_encoder(right_imgs_feat)


        
        return imgs_feat, imgs_pos_embedding


class AlohaDepthSegTransformer(nn.Module):

    def __init__(self,
                 action_length=5,
                 hidden_dim=256,
                 robot_state_dim=10, 
                 action_size=5):
        super().__init__()
        total_num_queries = action_length
        self.action_length = action_length

        self.backbone = load_resnet50_depthseg()
        self.query_embed = nn.Embedding(total_num_queries, hidden_dim)
        self.position_encoder = PositionEmbeddingLearned(128)

        # self.lr_img_feat_proj = nn.Conv2d(4096, 2048, kernel_size=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # We encode robot state into action head directly
        # self.robot_state_proj = nn.Linear(robot_state_dim, hidden_dim)

        self.transformer = Transformer(d_model=hidden_dim,
                                       dropout=0.1,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       num_encoder_layers=5,
                                       num_decoder_layers=5,
                                       normalize_before=True,
                                       return_intermediate_dec=True)
        
        self.action_head = Mlp(hidden_dim+robot_state_dim, 256, action_size, 3)



    def forward(self, img, state):
        """
        left_imgs / right_imgs: [B, 3, H, W]
        states: [B, 8]
        """
        bs = img.size(0)

        # TODO: if we follow the network structure of InterFuser, we can change the transformer architecture to be more similar with it

        img_feat, pos_embedding = self._compute_visual_features(img)
        # print(f'img_feat size: {img_feat.size()}')
        # print(f'pos_embedding size: {pos_embedding.size()}')
        # imgs_feat = torch.cat([left_img_feat, right_img_feat], dim=2)
        # pos_embedding = torch.cat([left_pos_embedding, right_pos_embedding], dim=2)

        hs = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)

        hs, memory = self.transformer(self.input_proj(img_feat), 
                            hs, 
                            pos_embedding,
                            mask=None)
        
        # hs with size: [num_layer, bs, 5, hidden_dim]
        num_layer, num_action = hs.size(0), hs.size(2)
        state = state.unsqueeze(0).unsqueeze(2).repeat(num_layer, 1, num_action, 1)

        # print(f'state size: {state.size()}')
        # print(f'hs size: {hs.size()}')

        action_preds = self.action_head(torch.cat([hs, state], dim=-1))

        return action_preds

    def _compute_visual_features(self, imgs):
        """Compute visual features at different scales and their positional embeddings."""

        imgs_feat = self.backbone(imgs)[3]
        # right_imgs_feat = self.backbone(right_imgs)[3]
        
        imgs_pos_embedding = self.position_encoder(imgs_feat)
        # right_imgs_pos_embedding = self.position_encoder(right_imgs_feat)


        
        return imgs_feat, imgs_pos_embedding

class AlohaTransformer2(nn.Module):

    def __init__(self,
                 action_length=5,
                 hidden_dim=256,
                 robot_state_dim=10, 
                 action_size=5):
        super().__init__()
        total_num_queries = action_length
        self.action_length = action_length

        self.backbone = load_resnet50_1()
        self.query_embed = nn.Embedding(total_num_queries, hidden_dim)
        self.position_encoder = PositionEmbeddingLearned(128)

        # self.lr_img_feat_proj = nn.Conv2d(4096, 2048, kernel_size=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # We encode robot state into action head directly
        # self.robot_state_proj = nn.Linear(robot_state_dim, hidden_dim)

        self.transformer = Transformer(d_model=hidden_dim,
                                       dropout=0.1,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       num_encoder_layers=5,
                                       num_decoder_layers=5,
                                       normalize_before=True,
                                       return_intermediate_dec=True)
        
        self.action_head = Mlp(hidden_dim, 256, action_size, 3)



    def forward(self, img, state):
        """
        left_imgs / right_imgs: [B, 3, H, W]
        states: [B, 8]
        """
        bs = img.size(0)

        # TODO: if we follow the network structure of InterFuser, we can change the transformer architecture to be more similar with it

        img_feat, pos_embedding = self._compute_visual_features(img)
        # print(f'img_feat size: {img_feat.size()}')
        # print(f'pos_embedding size: {pos_embedding.size()}')
        # imgs_feat = torch.cat([left_img_feat, right_img_feat], dim=2)
        # pos_embedding = torch.cat([left_pos_embedding, right_pos_embedding], dim=2)

        hs = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)

        hs, memory = self.transformer(self.input_proj(img_feat), 
                            hs, 
                            pos_embedding,
                            mask=None)
        
        # hs with size: [num_layer, bs, 5, hidden_dim]
        # num_layer, num_action = hs.size(0), hs.size(2)
        # state = state.unsqueeze(0).unsqueeze(2).repeat(num_layer, 1, num_action, 1)

        # print(f'state size: {state.size()}')
        # print(f'hs size: {hs.size()}')

        action_preds = self.action_head(hs)

        return action_preds

    def _compute_visual_features(self, imgs):
        """Compute visual features at different scales and their positional embeddings."""

        imgs_feat = self.backbone(imgs)[3]
        # right_imgs_feat = self.backbone(right_imgs)[3]
        
        imgs_pos_embedding = self.position_encoder(imgs_feat)
        # right_imgs_pos_embedding = self.position_encoder(right_imgs_feat)


        
        return imgs_feat, imgs_pos_embedding

class AlohaDepthSegTransformer2(nn.Module):

    def __init__(self,
                 action_length=5,
                 hidden_dim=256,
                 robot_state_dim=10, 
                 action_size=5):
        super().__init__()
        total_num_queries = action_length
        self.action_length = action_length

        self.backbone = load_resnet50_depthseg()
        self.query_embed = nn.Embedding(total_num_queries, hidden_dim)
        self.position_encoder = PositionEmbeddingLearned(128)

        # self.lr_img_feat_proj = nn.Conv2d(4096, 2048, kernel_size=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # We encode robot state into action head directly
        # self.robot_state_proj = nn.Linear(robot_state_dim, hidden_dim)

        self.transformer = Transformer(d_model=hidden_dim,
                                       dropout=0.1,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       num_encoder_layers=5,
                                       num_decoder_layers=5,
                                       normalize_before=True,
                                       return_intermediate_dec=True)
        
        self.action_head = Mlp(hidden_dim, 256, action_size, 3)



    def forward(self, img, state):
        """
        left_imgs / right_imgs: [B, 3, H, W]
        states: [B, 8]
        """
        bs = img.size(0)

        # TODO: if we follow the network structure of InterFuser, we can change the transformer architecture to be more similar with it

        img_feat, pos_embedding = self._compute_visual_features(img)
        # print(f'img_feat size: {img_feat.size()}')
        # print(f'pos_embedding size: {pos_embedding.size()}')
        # imgs_feat = torch.cat([left_img_feat, right_img_feat], dim=2)
        # pos_embedding = torch.cat([left_pos_embedding, right_pos_embedding], dim=2)

        hs = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)

        hs, memory = self.transformer(self.input_proj(img_feat), 
                            hs, 
                            pos_embedding,
                            mask=None)
        
        # hs with size: [num_layer, bs, 5, hidden_dim]
        num_layer, num_action = hs.size(0), hs.size(2)
        # state = state.unsqueeze(0).unsqueeze(2).repeat(num_layer, 1, num_action, 1)

        # print(f'state size: {state.size()}')
        # print(f'hs size: {hs.size()}')

        action_preds = self.action_head(hs)

        return action_preds

    def _compute_visual_features(self, imgs):
        """Compute visual features at different scales and their positional embeddings."""

        imgs_feat = self.backbone(imgs)[3]
        # right_imgs_feat = self.backbone(right_imgs)[3]
        
        imgs_pos_embedding = self.position_encoder(imgs_feat)
        # right_imgs_pos_embedding = self.position_encoder(right_imgs_feat)


        
        return imgs_feat, imgs_pos_embedding

class AlohaTransformer3(nn.Module):

    def __init__(self,
                 action_length=5,
                 hidden_dim=256,
                 robot_state_dim=10, 
                 action_size=5):
        super().__init__()
        total_num_queries = action_length
        self.action_length = action_length

        self.backbone = load_resnet50_1()
        self.query_embed = nn.Embedding(total_num_queries, hidden_dim)
        self.position_encoder = PositionEmbeddingLearned(128)

        # self.lr_img_feat_proj = nn.Conv2d(4096, 2048, kernel_size=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # We encode robot state into action head directly
        # self.robot_state_proj = nn.Linear(robot_state_dim, hidden_dim)

        self.transformer = Transformer2(d_model=hidden_dim,
                                       dropout=0.1,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       num_encoder_layers=5,
                                       num_decoder_layers=5,
                                       normalize_before=True,
                                       return_intermediate_dec=True)
        self.robot_state_pos_embed = nn.Embedding(1, 256)
        self.robot_state_proj = nn.Linear(robot_state_dim, 256)
        
        self.action_head = Mlp(hidden_dim, 256, action_size, 3)



    def forward(self, img, state):
        """
        left_imgs / right_imgs: [B, 3, H, W]
        states: [B, 8]
        """
        bs = img.size(0)
        robot_state_feat = self.robot_state_proj(state)
        robot_state_pos_embedding = self.robot_state_pos_embed.weight.unsqueeze(0).expand(bs, -1, -1)


        img_feat, pos_embedding = self._compute_visual_features(img)
        # imgs_feat = torch.cat([left_img_feat, right_img_feat], dim=2)
        # pos_embedding = torch.cat([left_pos_embedding, right_pos_embedding], dim=2)

        hs = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)

        feat = torch.cat([self.input_proj(img_feat).flatten(2).permute(0, 2, 1), robot_state_feat.unsqueeze(1)], dim=1).permute(1, 0, 2)
        pos_embedding = torch.cat([pos_embedding.flatten(2).permute(0, 2, 1), robot_state_pos_embedding], dim=1).permute(1, 0, 2)

        hs = self.transformer(feat, 
                            hs, 
                            pos_embedding,
                            mask=None)
        
        # hs with size: [num_layer, bs, 5, hidden_dim]
        # num_layer, num_action = hs.size(0), hs.size(2)
        # state = state.unsqueeze(0).unsqueeze(2).repeat(num_layer, 1, num_action, 1)

        # print(f'state size: {state.size()}')
        # print(f'hs size: {hs.size()}')

        action_preds = self.action_head(hs)

        return action_preds

    def _compute_visual_features(self, imgs):
        """Compute visual features at different scales and their positional embeddings."""

        imgs_feat = self.backbone(imgs)[3]
        # right_imgs_feat = self.backbone(right_imgs)[3]
        
        imgs_pos_embedding = self.position_encoder(imgs_feat)
        # right_imgs_pos_embedding = self.position_encoder(right_imgs_feat)


        
        return imgs_feat, imgs_pos_embedding


class AlohaTransformer4(nn.Module):

    def __init__(self,
                 action_length=5,
                 hidden_dim=256,
                 robot_state_dim=10, 
                 action_size=5):
        super().__init__()
        total_num_queries = action_length
        self.action_length = action_length

        self.backbone = load_resnet50_1()
        self.query_embed = nn.Embedding(total_num_queries, hidden_dim)
        self.position_encoder = PositionEmbeddingLearned(128)

        # self.lr_img_feat_proj = nn.Conv2d(4096, 2048, kernel_size=1)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # We encode robot state into action head directly
        # self.robot_state_proj = nn.Linear(robot_state_dim, hidden_dim)

        self.transformer = Transformer2(d_model=hidden_dim,
                                       dropout=0.1,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       num_encoder_layers=5,
                                       num_decoder_layers=5,
                                       normalize_before=True,
                                       return_intermediate_dec=True)
        self.action_idx_pos_embed = nn.Embedding(1, 256)
        self.action_idx_proj = nn.Linear(1, 256)
        
        self.action_head = Mlp(hidden_dim, 256, action_size, 3)



    def forward(self, img, state, action_idx):
        """
        left_imgs / right_imgs: [B, 3, H, W]
        states: [B, 8]
        """
        bs = img.size(0)
        action_idx_feat = self.action_idx_proj(action_idx)
        action_idx_pos_embedding = self.action_idx_pos_embed.weight.unsqueeze(0).expand(bs, -1, -1)


        img_feat, pos_embedding = self._compute_visual_features(img)
        # imgs_feat = torch.cat([left_img_feat, right_img_feat], dim=2)
        # pos_embedding = torch.cat([left_pos_embedding, right_pos_embedding], dim=2)

        hs = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)

        feat = torch.cat([self.input_proj(img_feat).flatten(2).permute(0, 2, 1), action_idx_feat.unsqueeze(1)], dim=1).permute(1, 0, 2)
        pos_embedding = torch.cat([pos_embedding.flatten(2).permute(0, 2, 1), action_idx_pos_embedding], dim=1).permute(1, 0, 2)

        hs = self.transformer(feat, 
                            hs, 
                            pos_embedding,
                            mask=None)
        
        # hs with size: [num_layer, bs, 5, hidden_dim]
        # num_layer, num_action = hs.size(0), hs.size(2)
        # state = state.unsqueeze(0).unsqueeze(2).repeat(num_layer, 1, num_action, 1)

        # print(f'state size: {state.size()}')
        # print(f'hs size: {hs.size()}')

        action_preds = self.action_head(hs)

        return action_preds

    def _compute_visual_features(self, imgs):
        """Compute visual features at different scales and their positional embeddings."""

        imgs_feat = self.backbone(imgs)[3]
        # right_imgs_feat = self.backbone(right_imgs)[3]
        
        imgs_pos_embedding = self.position_encoder(imgs_feat)
        # right_imgs_pos_embedding = self.position_encoder(right_imgs_feat)


        
        return imgs_feat, imgs_pos_embedding


    
if __name__ == '__main__':


    img = torch.randn((2, 3, 224, 224))
    robot_state = torch.randn((2, 10))

    aloha_transformer = AlohaTransformer3(action_length=5, hidden_dim=256, robot_state_dim=10, action_size=5)

    action_preds = aloha_transformer(img, robot_state)

    print(f'action_preds size: {action_preds.size()}')













        