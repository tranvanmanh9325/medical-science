import numpy as np
import torch
import torch.nn.functional as F

from utils.general_utils import AttrDict
from .dex import DEX


class VA(DEX):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.k = agent_cfg.k
    
    

    def get_action(self, state, noise=False):
        with torch.no_grad():
            
            
            
            
            
            
