import numpy as np
from easydict import EasyDict as edict 
import os
import PyCSR

player_opts=edict()

# filepath
player_opts.base='/home/student/csr_test/animal_exp'
player_opts.fs_path='/home/student/csr_test/endoscope_calibration_1205.yaml'
player_opts.task_base={"gauze": '/home/student/csr_test',
                       "vessel": '/home/student/csr_test/vessel',
                       "tissue_1":'/home/student/yhlong/BluntDissection',
                       "tissue_2":'/home/student/yhlong/BluntDissection',
                       "needle": '/home/student/csr_test/animal_exp'}
player_opts.policy_file={"gauze": 'gauze_policy_1206.pt',
                         # "gauze": 's71_DDPG_demo0_traj_best.pt',
                       "vessel":'vessel_policy_1206.pt', # newvein_0822_multipositio_episode
                       "tissue_1":'DDPG_BluntDissect_LargeInit_ep4k.pt',
                     #   "tissue_1":  's71_DDPG_demo0_traj_best.pt',    
                       "tissue_2":'DDPG_tissue_p2.pt',
                       "endo":'',
                       "needle":"needle_policy_1206.pt"
                       #"needle":"needle_policy.pt"
                       }

player_opts.vmodel_file={"gauze": 'gauze_vision_1206.pt',
                         # "gauze": 'csr_ar_rstate_regression_best.pt',
                       "vessel":'vessel_vision_1206.pt',#'vmodel_0818.pt',
                       "tissue_1":'vision_tissue.pt',
                       # "tissue_1":'csr_ar_rstate_regression_best.pt',
                       "tissue_2":'vision_tissue_p2.pt',
                       "endo":'',
                       "needle":"needle_vision_1206.pt"
                       # "needle":"v_needle_model.pt"
                       }

player_opts.max_step={"gauze": 1,
                       "vessel":1,
                       "tissue_1":1,
                       "tissue_2":10,
                       "endo":'',
                       "needle":2}

player_opts.save_dir=os.path.join(player_opts.base,'test_record')

# config
player_opts.init_rotate_ecm=np.array([9.70763688e-01, 2.61508163e-05, -3.74627977e-05])

# player_opts.rotation=PyCSR.Rotation( -0.445465,    0.888536,    0.109839,
#      0.866506,    0.458745,   -0.196774,
#     -0.225229,  0.00752064,   -0.974277)

# player_opts.mani_rotation=np.array([[ 0.66397923, -0.74742173, -0.02218865],
#        [-0.74749038, -0.66423891,  0.0066928 ],
#        [-0.01974091,  0.01214192, -0.9997314 ]])

player_opts.rotation = PyCSR.Rotation(0.6639591303122233, -0.7474396402652629, -0.022186875290807165, -0.7475085868864552, -0.6642185834259241, 0.006677272098193439, -0.019727792731334556, 0.012151444020882684, -0.9997315422663008)
player_opts.mani_rotation = np.array([[ 0.66395913, -0.74743964, -0.02218688],
       [-0.74750859, -0.66421858,  0.00667727],
       [-0.01972779,  0.01215144, -0.99973154]])

player_opts.Q = np.array([[1, 0, 0, -4.05009071e+02],
                        [0, 1, 0, -2.88859371e+02],
                        [0, 0, 0,  7.92355233e+02],
                        [0, 0, 2.16833937e-01, -0]])


player_opts.intrinsics_matrix=np.array([[792.3552334 , 0.0, 405.00907135], [0.0, 792.3552334 , 288.85937119], [0.0, 0.0, 1.0]])
player_opts.mani_intrinsics_matrix=np.array([[792.3552334 / 2.5 , 0.0, 405.00907135 / 2.5], [0.0, 792.3552334 / 2.5 , 288.85937119 / 2.5], [0.0, 0.0, 1.0]])

'''
0.0790253,    -0.54449,   -0.835036;
    -0.386392,   -0.788904,    0.477842;
    -0.918943,    0.284889,    -0.27273
'''
'''
player_opts.basePSM_T_cam=np.load("/home/student/csr_test/animal_exp/exp_calibration/basePSM_T_cam.npy")
player_opts.cam_T_basePSM=np.load("/home/student/csr_test/animal_exp/exp_calibration/cam_T_basePSM.npy")

#print(player_opts.cam_T_basePSM)
#exit()

'''


player_opts.calibration_data = {
            'baseline': 0.004612,
            'focal_length_left': 792.3552334,
            'focal_length_right':792.3552334
        }
player_opts.threshold={"gauze": 0.01,
                       "vessel":0.012,
                       "tissue_1":0.01,
                       "tissue_2":0.01,
                       "endo":0.01,
                       "needle":0.01}

player_opts.goal_offset_z={"gauze": 0.0,
                       "vessel":0.0,
                       "tissue_1":0.0,
                       "tissue_2":0.02,
                       "endo":0.01,
                       "needle":0.0}


player_opts.goal_offset_x={"gauze": 0.0,
                       "vessel":0,
                       "tissue_1":0.0,
                       "tissue_2":0.0,
                       "endo":0.0,
                       "needle":0.0}


player_opts.goal_offset_y={"gauze": 0.0,
                       "vessel":0.0,
                       "tissue_1":0.0,
                       "tissue_2":0.01,
                       "endo":0.0,
                       "needle":0.0}

# tool_offset = [0.0, 0.0, 0.00954] # LND 
# tool_offset = [0.0, 0.0, 0.02111] # FBF
# tool_offset = [0.0, 0.0, 0.02032] # MBF
# tool_offset = [0.0, 0.0, 0.01158] # Medium clip applier  

# player_opts.tool_offset={"gauze": [0.0,0.0,0.020],
#                        "vessel":[0.0, 0.0, 0.018],
#                        "tissue_1":[0.0, 0.0, 0.012],
#                        "tissue_2":[0.0,0.0,0.0],
#                        "endo":0.0,
#                        "needle":[-0.01,-0.0087,0.036]}

player_opts.tool_offset={"gauze": [0.0,0.005,0.0],
                       "vessel":[0.0, 0.018, 0.0],
                       "tissue_1":[0.0, 0.012, 0.0],
                       "tissue_2":[0.0,0.0,0.0],
                       "endo":0.0,
                       "needle":[0,0.010,0.0]}

player_opts.kinematics_offset = [-0.0014999999999999998, -0.007499999999999999, 0.0050999999999999995]

player_opts.video_left='/dev/video0'
player_opts.video_right='/dev/video2'
player_opts.psa_num="psa3"

player_opts.action_len=1
player_opts.use_blur=False

player_opts.lift_height = 0.03

