import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/yhlong/project/VPPV_checking/SurRoL/VPPV/Training/state_regress')
# from model.transformer_with_kin import TransformerModule
#from model.transformer_aux_task_jw import TransformerModule 
#from model.transformer_with_kin_targetggoal import TransformerModule
#from model.transformer_aux_task_jw import TransformerModule
#from model.tansformer_simple import TransformerModule
#from model.transformer_aux_task import TransformerModule
from config import opts

class E2EAgent(nn.Module):
    def __init__(self, env_params=None, sampler=None, agent_cfg=None):
        super().__init__()
        opts.device='cuda:0'
        self.opts=opts
        self.net=TransformerModule(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.to(opts.device)
        self.net.eval()
        print("load v_output")
    
    def get_action(self, obs, noise=False, v_action=None, v_model=None):
        img=torch.from_numpy(obs['img']).to(self.opts.device).float().unsqueeze(0)
        #print(img.shape)
        #img=img.unsqueeze(0)

        if self.opts.use_kinematics:
            state=torch.from_numpy(obs["observation"]).to(self.opts.device).float().unsqueeze(0)
            
            with torch.no_grad():
                action_output=self.net.infer(img, state)
        else:
            with torch.no_grad():
                action_output=self.net.infer(img)
        print('action: ',action_output)
        return action_output.cpu().data.numpy().flatten()

        


    
    