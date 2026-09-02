import numpy as np
from easydict import EasyDict as edict 
import os
import PyKDL

import dvrk
import crtk

player_opts=edict()

# player_opts.ral = crtk.ral('dvrk_python_node')
# player_opts.psm1 = dvrk.psm(player_opts.ral, 'PSM1')
# player_opts.psm2 = dvrk.psm(player_opts.ral, 'PSM2')

# filepath
player_opts.base='/home/kj/lsong/vppv'
player_opts.fs_path='/home/kj/lsong/EndoscopeCalibration/endoscope_calibration_csr_0degree.yaml'
player_opts.task_base={"gauze": '/home/kj/lsong/vppv',
                       "vessel": '/home/kj/lsong/vppv',
                       "tissue_1":'/home/kj/lsong/vppv',
                       "tissue_2":'/home/kj/lsong/vppv',
                       "needle": '/home/kj/lsong/vppv',
                       "peg": '/home/kj/lsong/vppv'}
player_opts.policy_file={"gauze": 's80_DDPG_demo0_traj_best_gauze.pt',
                          "vessel":'s80_DDPG_demo0_traj_best_vessel.pt', 
                          "tissue_1":'DDPG_BluntDissect_LargeInit_ep4k.pt',
                          "tissue_2":'DDPG_tissue_p2.pt',
                          "endo":'',
                          "needle":"s80_DDPG_demo0_traj_best_needle.pt",
                          "peg":"s80_DDPG_demo0_traj_best_peg.pt"
                          }

player_opts.vmodel_file={"gauze": 'v_model.pt',
                          "vessel":'v_model.pt',
                          "tissue_1":'v_model.pt',
                          "tissue_2":'v_model.pt',
                          "endo":'',
                          "needle":"v_model.pt",
                          "peg":"v_model.pt"
                          }

player_opts.max_step={"gauze": 1,
                       "vessel":1,
                       "tissue_1":2,
                       "tissue_2":10,
                       "endo":'',
                       "needle":2,
                       "peg":1}

player_opts.save_dir=os.path.join(player_opts.base,'test_record')

# config
player_opts.init_rotate_ecm=np.array([9.70763688e-01, 2.61508163e-05, -3.74627977e-05])

player_opts.rotation = PyKDL.Rotation(0.6639591303122233, -0.7474396402652629, -0.022186875290807165, -0.7475085868864552, -0.6642185834259241, 0.006677272098193439, -0.019727792731334556, 0.012151444020882684, -0.9997315422663008)

player_opts.mani_rotation = np.array([[ 0.66395913, -0.74743964, -0.02218688],
                                      [-0.74750859, -0.66421858,  0.00667727],
                                      [-0.01972779,  0.01215144, -0.99973154]])
player_opts.Q =  np.array([[1, 0, 0, -355.45],
                          [0, 1, 0, -327.7],
                          [0, 0, 0, 693.12],
                          [0, 0, 0.23731, -0]])

player_opts.intrinsics_matrix=np.array([[693.12012738 , 0.0, 355.44816971], 
                                        [0.0,693.12012738 , 327.7015152], 
                                        [0.0, 0.0, 1.0]])
player_opts.intrinsics_matrix_right=np.array([[693.12012738 , 0.0, 355.44816971], 
                                              [0.0, 693.12012738, 327.7015152], 
                                              [0.0, 0.0, 1.0]])

player_opts.mani_intrinsics_matrix=np.array([[693.12012738   / 2.5 , 0.0, 355.44816971 / 2.5], 
                                             [0.0, 693.12012738   / 2.5 , 327.7015152 / 2.5], 
                                             [0.0, 0.0, 1.0]])


player_opts.basePSM_T_cam = np.array([[-0.9715952 , -0.08207092,  0.22196199,  0.14564693],
       [-0.1482096 ,  0.94223976, -0.30036337,  0.09838246],
       [-0.18449032, -0.3247285 , -0.92763933,  0.00676384],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
player_opts.cam_T_basePSM = np.array([[-0.9715952 , -0.1482096 , -0.18449032,  0.15733895],
       [-0.08207092,  0.94223976, -0.3247285 , -0.07855007],
       [ 0.22196199, -0.30036337, -0.92763933,  0.00349681],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
                                
# player_opts.basePSM_T_cam2 = np.array([[-0.73543307,  0.66955681, -0.10407629, -0.08368251],
#                                       [ 0.66662119,  0.68740056, -0.2882649 ,  0.1316885 ],
#                                       [-0.12146762, -0.281379  , -0.95187787, -0.0560771 ],
#                                       [ 0.        ,  0.        ,  0.        ,  1.        ]])
# player_opts.cam_T_basePSM2 = np.array([[-0.73543307,  0.66662119, -0.12146762, -0.15614078],
#                                       [ 0.66955681,  0.68740056, -0.281379  , -0.05027147],
#                                       [-0.10407629, -0.2882649 , -0.95187787, -0.02412674],
#                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])

player_opts.basePSM_T_cam2 = np.array([[-0.8128076 ,  0.57348532, -0.10226631, -0.06379491],
        [ 0.57612612,  0.76542098, -0.28672185,  0.09373904],
        [-0.08615399, -0.29196799, -0.95253986, -0.00611106],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
player_opts.cam_T_basePSM2 = np.array([[-0.8128076 ,  0.57612612, -0.08615399, -0.10638499],
        [ 0.57348532,  0.76542098, -0.29196799, -0.03694861],
        [-0.10226631, -0.28672185, -0.95253986,  0.01453193],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])

player_opts.calibration_data = {
            'baseline': 0.004214,
            'focal_length_left': 693.12012738,
            'focal_length_right':693.12012738
        }
player_opts.threshold={"gauze": 0.01,
                       "vessel":0.012,
                       "tissue_1":0.01,
                       "tissue_2":0.01,
                       "endo":0.01,
                       "needle":0.01,
                       "peg":0.01}

player_opts.goal_offset_z={"gauze": -0.02,
                       "vessel":-0.014,
                       "tissue_1":0.0,
                       "tissue_2":0.04,
                       "endo":0.0,
                       "needle":0.0,
                       "peg":0.002,
                       "peg_place":0.0,
                       "base":-0.013}


player_opts.goal_offset_x={"gauze": 0.00,
                       "vessel":0,
                       "tissue_1":0.0,
                       "tissue_2":0.0,
                       "endo":0.0,
                       "needle":0.0,
                       "peg":0.0015,
                       "peg_place":-0.0054}


player_opts.goal_offset_y={"gauze": -0.00,
                       "vessel":-0.0035,
                       "tissue_1":0.0,
                       "tissue_2":0.01,
                       "endo":0.0,
                       "needle":0.0,
                       "peg":-0.003,
                       "peg_place":-0.015}

player_opts.LND_offset = [0.0, 0.00954, 0.0] # LND 
# tool_offset = [0.0, 0.0, 0.02111] # FBF
# tool_offset = [0.0, 0.0, 0.02032] # MBF
# tool_offset = [0.0, 0.0, 0.01158] # Medium clip applier  


player_opts.tool_offset={"gauze": [0.0,0.012,0.0],
                       "vessel":[-0.006, 0.018, 0.0],
                       "tissue_1":[0.0, 0.00, 0.0],
                       "tissue_2":[0.0,0.0,0.0],
                       "endo":0.0,
                       "needle":[0,0.001,0.0],
                       "peg":[0.0,0.0,0.0]}



player_opts.kinematics_offset =  [0.0048, -0.0054, 0.0039]
player_opts.video_left='/dev/video0'
player_opts.video_right='/dev/video2'

player_opts.action_len= 1
player_opts.use_blur=False

player_opts.lift_height = 0.03

player_opts.add_seg_noise = False
player_opts.seg_noise_ratio = 0.1 # 0-0.3 ratio of pixels to be flipped

player_opts.add_seg_MarkovNoise = False
player_opts.Markov_T = 60
player_opts.Markov_theta1 = 0.5
player_opts.Markov_theta2 = 0.05
player_opts.Markov_theta3 = 0.1

player_opts.add_depth_noise = False
player_opts.depth_noise_mean = -0.004 # 0-0.003 mm gaussian noise
player_opts.depth_noise_sigma = 0.001 # 0-0.003 mm gaussian noise

player_opts.multiple_prompt = False

# if ture use the predefined propmts
player_opts.demo_mode = False
player_opts.demo_points = {
    "gauze": [[303, 101], [313, 235]],
    "vessel": [[102, 120], [80, 80]]
}