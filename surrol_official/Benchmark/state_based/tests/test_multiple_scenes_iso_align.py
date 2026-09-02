import os, sys, time, math

from kivy.lang import Builder
import pybullet as p
from panda3d_kivy.mdapp import MDApp
from direct.gui.DirectGui import *
from panda3d.core import AmbientLight, DirectionalLight, Spotlight, PerspectiveLens
from UI_kivy import start_page,ecm_ui,needle_ui,peg_ui,menu_bar_ui

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application, ApplicationConfig
from surrol.tasks.ecm_misorient import MisOrient
from surrol.tasks.ecm_reach import ECMReach
from surrol.tasks.ecm_active_track import ActiveTrack
from surrol.tasks.ecm_static_track import StaticTrack
from surrol.tasks.gauze_retrieve import GauzeRetrieveFullDof
from surrol.tasks.needle_reach import NeedleReachFullDof
from surrol.tasks.needle_pick import NeedlePickFullDof
from surrol.tasks.peg_board import BiPegBoardFullDof
from surrol.tasks.peg_transfer import PegTransferFullDof
from surrol.tasks.peg_transfer_RL import PegTransferRL
from surrol.tasks.needle_regrasp import NeedleRegraspFullDof
from surrol.tasks.peg_transfer_bimanual import BiPegTransferFullDof
from surrol.tasks.pick_and_place import PickAndPlaceFullDof
from surrol.tasks.match_board import MatchBoardFullDof 
from surrol.tasks.needle_pick_RL import NeedlePickRL
from surrol.tasks.needle_the_rings import NeedleRingsFullDof

from surrol.tasks.ecm_env import EcmEnv, goal_distance,reset_camera
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.robots.ecm import Ecm


import numpy as np
import rospy, PyKDL
import threading
from sensor_msgs.msg import Joy
from dvrk import mtm

import torch
import torch.nn as nn
from direct.task import Task
from surrol.utils.pybullet_utils import step

app = None
hint_printed = False

def home_every_scene(arm_name='MTMR'):
    m = mtm(arm_name)
    # print with node id
    def print_id(message):
        print('%s -> %s' % (rospy.get_caller_id(), message))
    def home():
        print_id('starting enable')
        if not m.enable(2):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not m.home(2):
            sys.exit('failed to home within 10 seconds')
        # get current joints just to set size
        print_id('move to starting position')
        goal = np.copy(m.setpoint_jp())
        # go to zero position, make sure 3rd joint is past cannula
        goal.fill(0)
        m.move_jp(goal).wait()
    home()
    del m

def open_scene(id):
    # id 0: StartPage() 1: ECMPage() 2: NeedlePage() 3: PegPage() 4: PickPlacePage() 5: NeedlePick,
    #    6: NeedlePick,demo 7: PegTransfer 8: PegTransfer, demo 9: NeedleRegrasp, bimanual 10: NeedleRegrasp, demo bimanual
    #    11: BiPegTransfer,  bimanual 12: BiPegTransfer, demo bimanual 13: PickAndPlace 14: PickAndPlace, demo 15: PegBoard
    #    16: PegBoard demo17: NeedleRings,  bimanual 18: NeedleRings, demo bimanual 19: MatchBoard 20: MatchBoard, demo
    #    21: ECMReach 22: ECMReach, demo 23: MisOrient 24: MisOrient, demo 25: StaticTrack 26: StaticTrack, demo
    #    27: ActiveTrack 28: ActiveTrack, demo 29: NeedleReach 30: NeedleReach, demo 31: GauzeRetrieve 32: GauzeRetrieve, demo
    global app, hint_printed
    scene = None
    menu_dict = {0:StartPage(),1:ECMPage(),2:NeedlePage(),3:PegPage()}
    task_list =[NeedlePickFullDof,PegTransferFullDof,NeedleRegraspFullDof,\
                BiPegTransferFullDof,PickAndPlaceFullDof,BiPegBoardFullDof,\
                NeedleRingsFullDof,MatchBoardFullDof,ECMReach,MisOrient,StaticTrack,\
                ActiveTrack,NeedleReachFullDof,GauzeRetrieveFullDof]
    bimanual_list=[9,10,11,12,15,17,18]
    if id in menu_dict:
        scene = menu_dict[id]
    elif id in bimanual_list:
        home_every_scene('MTMR')
        home_every_scene('MTML')
        scene = SurgicalSimulatorBimanual(task_list[(id-5)//2], {'render_mode': 'human'},id=id) if id %2==1 else \
        SurgicalSimulatorBimanual(task_list[(id-5)//2], {'render_mode': 'human'}, id=id,demo=1)
    else:
        if id ==8:
            scene = SurgicalSimulator(PegTransferRL,{'render_mode': 'human'},id,demo=1)
        elif id ==6:
            scene = SurgicalSimulator(NeedlePickRL,{'render_mode': 'human'},id,demo=1)
        else:
            if id%2==1:
                home_every_scene('MTMR')
                scene = SurgicalSimulator(task_list[(id-5)//2],{'render_mode': 'human'},id) 
            else:
                scene = SurgicalSimulator(task_list[(id-5)//2],{'render_mode': 'human'},id,demo=1)
    if scene:
        app.play(scene)



class SelectionUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(start_page.selection_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(1))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(2))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(3))
        # self.screen.ids.btn4.bind(on_press = lambda _: open_scene(4))

        # self.screen.ids.btn11.bind(on_press = lambda _: open_scene(11))

class NeedleUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(needle_ui.needle_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(29))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(31))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(5))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(9))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(2))

class PegUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(peg_ui.Peg_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(7))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(11))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(15))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(13))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(19))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(17))
        self.screen.ids.btn7.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn8.bind(on_press = lambda _: open_scene(3))

class ECMUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(ecm_ui.ECM_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(22))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(24))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(26))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(28))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(1))

class StartPage(Scene):
    def __init__(self):
        super(StartPage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = SelectionUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class NeedlePage(Scene):
    def __init__(self):
        super(NeedlePage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = NeedleUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class PegPage(Scene):
    def __init__(self):
        super(PegPage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = PegUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)


class ECMPage(Scene):
    def __init__(self):
        super(ECMPage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = ECMUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)


class MenuBarUI(MDApp):
    def __init__(self, panda_app, display_region,id = None):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None
        self.id = id
        self.ecm_list=[i for i in range(21,29)]
        self.fund_list = [31,32,5,6,9,10,29,30]
        self.basic_list=[7,8,11,12,15,16,17,18,13,14,19,20]
    def build(self):
        self.theme_cls.theme_style = "Dark"
        if self.id % 2 == 0:
            if self.id in self.ecm_list:
                self.screen = Builder.load_string(menu_bar_ui.menu_bar_kv_haptic)
            else:
                self.screen = Builder.load_string(menu_bar_ui.menu_bar_kv_RL)
        else:
            self.screen = Builder.load_string(menu_bar_ui.menu_bar_kv_haptic)
        return self.screen

    def on_start(self):
        # scene_menu = 1
        if self.id in self.ecm_list:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(1))
        elif self.id in self.fund_list:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(2))
        elif self.id in self.basic_list:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(3))
        else:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(0))
        if self.id % 2 ==0:
            if self.id in self.ecm_list:
                self.screen.ids.btn4.bind(on_press = lambda _: (open_scene(0), open_scene(self.id)))
            else:
                self.screen.ids.btn4.bind(on_press = lambda _: (open_scene(0), open_scene(self.id-1)))
        else:
            self.screen.ids.btn4.bind(on_press = lambda _:(open_scene(0), open_scene(self.id+1)))
            
class SurgicalSimulatorBase(GymEnvScene):
    def __init__(self, env_type, env_params):
        super(SurgicalSimulatorBase, self).__init__(env_type, env_params)
        self.clutched = False
    def before_simulation_step(self):
        pass

    # foot pedal callback
    def coag_event_cb(self,data):
        # print('~~~',data)
        # if (data.buttons[0] == 1):
        #     coag_event.set()
        #     print('clutched')
        # else:
        #     print('not clutched',data.buttons[0])
        print('clutch status',data.buttons[0])
        if data.buttons[0] == 1:
            self.mr.lock_orientation_as_is()
            self.clutched = True 
            self.ml.lock_orientation_as_is()
            # self.left_clutched = True 
        else:
            self.mr.unlock_orientation()           
            self.clutched = False
            self.ml.unlock_orientation()           
            # self.left_clutched = False
        # return data.buttons[0]

    def on_env_created(self):
        """Setup extrnal lights"""
        self.ecm_view_out = self.env._view_matrix

        table_pos = np.array(self.env.POSE_TABLE[0]) * self.env.SCALING

        # ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.world3d.attachNewNode(alight)
        self.world3d.setLight(alnp)

        # directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.4, 0.4, 0.25, 1))
        # dlight.setShadowCaster(True, app.configs.shadow_resolution, app.configs.shadow_resolution)
        dlnp = self.world3d.attachNewNode(dlight)
        dlnp.setPos(*(table_pos + np.array([1.0, 0.0, 15.0])))
        dlnp.lookAt(*table_pos)
        self.world3d.setLight(dlnp)

        # spotlight
        slight = Spotlight('slight')
        slight.setColor((0.5, 0.5, 0.5, 1.0))
        lens = PerspectiveLens()
        lens.setNearFar(0.5, 5)
        slight.setLens(lens)
        slight.setShadowCaster(True, app.configs.shadow_resolution, app.configs.shadow_resolution)
        slnp = self.world3d.attachNewNode(slight)
        slnp.setPos(*(table_pos + np.array([0, 0.0, 5.0])))
        slnp.lookAt(*(table_pos + np.array([0.6, 0, 1.0])))
        self.world3d.setLight(slnp)

        self.pos=[]
        self.ml = mtm('MTML')
        print("debug: self.mtml",self.ml)
        # turn gravity compensation on/off
        self.ml.use_gravity_compensation(True)
        self.ml.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        init_pos_l = self.ml.setpoint_cp().p
        self.pos.append(np.array([init_pos_l[i] for i in range(3)])) # front(neg) back: 1; left(neg) right: 0; up down(neg): 2
        self.pos_cur_l = self.pos[0].copy()
        # self.ml = mtm('MTML')
        # print("debug: self.mtml",self.ml)
        # # turn gravity compensation on/off
        # self.ml.use_gravity_compensation(True)
        # self.ml.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        
        # psm_pose_l = self.psm1.get_current_position()
        # psm_measured_cp_l= psm_pose_l.copy()
        # goal_l = PyKDL.Frame()
        # goal_l.p = self.ml.setpoint_cp().p
        # # # goal.p[0] += 0.05
        # goal_l.M= self.ml.setpoint_cp().M
        # # psm_measured_cp = np.matmul(mapping_mat,psm_measured_cp)
        # for i in range(3):
        #     print(f"previous goal:{goal_l.M}")
        #     for j in range(3):
        #         goal_l.M[i,j]=psm_measured_cp_l[i][j]
        #         # if j==1:
        #         #     goal.M[i,j]*=-1
        #         # goal.M[i,j]=psm_pose[i][j]
        #     print(f"modified goal:{goal_l.M}")
        # print(goal_l.M.GetEulerZYX())
        # # print(rotationMatrixToEulerAngles(psm_measured_cp[:3,:3]))
        # self.ml.move_cp(goal_l).wait() #align
        
        # init_pos_l = self.ml.setpoint_cp().p
        # self.pos.append(np.array([init_pos_l[i] for i in range(3)])) # front(neg) back: 1; left(neg) right: 0; up down(neg): 2
        # self.pos_cur_l = self.pos[0].copy()



        self.mr = mtm('MTMR')

        # turn gravity compensation on/off
        self.mr.use_gravity_compensation(True)
        self.mr.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.mr_init_pose = self.mr.setpoint_cp()
        
        init_pos_r = self.mr.setpoint_cp().p
        self.pos.append(np.array([init_pos_r[i] for i in range(3)])) # front(neg) back: 1; left(neg) right: 0; up down(neg): 2
        self.pos_cur_r = self.pos[1].copy()

        init_orn_r = self.mr.setpoint_cp().M.GetEulerZYX()
        self.orn_r = np.array([init_orn_r[i] for i in range(3)]) # 
        self.orn_cur_r = self.orn_r.copy()
        print(f'mtm setpoint: {self.mr_init_pose}')

        coag_event = threading.Event()
        rospy.Subscriber('footpedals/clutch',
                        Joy, self.coag_event_cb)

        # psm_pose_r = self.psm1.get_current_position()
        # psm_measured_cp_r= psm_pose_r.copy()
        # goal_r = PyKDL.Frame()
        # goal_r.p = self.ml.setpoint_cp().p
        # # # goal.p[0] += 0.05
        # goal_r.M= self.ml.setpoint_cp().M
        # # psm_measured_cp = np.matmul(mapping_mat,psm_measured_cp)
        # for i in range(3):
        #     print(f"previous goal:{goal_r.M}")
        #     for j in range(3):
        #         goal_r.M[i,j]=psm_measured_cp_r[i][j]
        #         # if j==1:
        #         #     goal.M[i,j]*=-1
        #         # goal.M[i,j]=psm_pose[i][j]
        #     print(f"modified goal:{goal_r.M}")
        # print(goal_r.M.GetEulerZYX())
        # # print(rotationMatrixToEulerAngles(psm_measured_cp[:3,:3]))
        # self.mr.move_cp(goal_r).wait() #align

        self.first = [True,True]
        # cp_r=np.array([[cp_r[i,0],cp_r[i,1],cp_r[i,2]] for i in range(3)])
        # print(f"position is: {self.pos_r}")
        # print(f"orientation is {self.orn_r}")
        # exit()
        # print(f"cur position is: {self.pos_cur}")


    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 0.061)
        self.kivy_ui = MenuBarUI(
            self.app,
            self.ui_display_region,
            self.id
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), 
        )

    def forward(self, input):
        return self.mlp(input)


class DeterministicActor(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim=256):
        super().__init__()

        self.trunk = MLP(
            in_dim=dimo+dimg,
            out_dim=dima,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        a = self.trunk(obs)
        return torch.tanh(a)

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.zeros(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        self.total_sum += v.sum(axis=0)
        self.total_sumsq += (np.square(v)).sum(axis=0)
        self.total_count[0] += v.shape[0]

    def recompute_stats(self):
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


class SurgicalSimulator(SurgicalSimulatorBase):
    def __init__(self, env_type, env_params,id=None,demo=None):
        super(SurgicalSimulator, self).__init__(env_type, env_params)
        self.id = id
        self.full_dof_list = [5,7,13,19,29,31]
        # initTouch_right()
        # startScheduler()
        if env_type.ACTION_SIZE != 3 and env_type.ACTION_SIZE != 1:
            self.psm1_action = np.zeros(env_type.ACTION_SIZE)
            if self.id not in self.full_dof_list:
                self.psm1_action[4] = 0.5



        self.ecm_view = 1
        self.ecm_view_out = None
        self.demo = demo
        self.start_time = time.time()
        self.has_load_policy = False
        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)
        if self.id in range(21,30):
            self.toggleEcmView()

    def get_MTMR_position_action(self,psm1_action,mat=None):
        # psm_pose_world = np.eye(4)
        if self.id in self.full_dof_list:

                
            if self.first[1]:
                for i in range(3):
                    psm1_action[i] =0
                self.first[1] = False
            elif self.clutched:
                for i in range(3):
                    psm1_action[i] =0
            else:
                self.pos_cur = np.array([self.mr.setpoint_cp().p[i] for i in range(3)])
                psm1_action[0] = (self.pos_cur[1] - self.pos[1][1])*(1000)
                psm1_action[1] = (self.pos_cur[0] - self.pos[1][0])*(-1000)
                psm1_action[2] = (self.pos_cur[2] - self.pos[1][2])*(1000)
                self.pos[1] = self.pos_cur.copy()
            psm1_action[3] = self.mr.gripper.measured_jp()[0]
            goal_orn = self.mr.setpoint_cp().M
            # print("goal orn: ",goal_orn)
            for i in range(3):
                for j in range(3):
                    mat[i][j]= goal_orn[i,j]
            
        else:
            if self.first[1]:
                for i in range(3):
                    psm1_action[i] =0
                psm1_action[4] = 1
                self.first[1] = False
            else:
                self.pos_cur = np.array([self.mr.setpoint_cp().p[i] for i in range(3)])

                psm1_action[0] = (self.pos_cur[1] - self.pos[1][1])*(-500)
                psm1_action[1] = (self.pos_cur[0] - self.pos[1][0])*(750)
                psm1_action[2] = (self.pos_cur[2] - self.pos[1][2])*(750)

                self.pos[1] = self.pos_cur.copy()
            psm1_action[3]=0

    def _step_simulation_task(self, task):
        """Step simulation
        """
        if self.demo == None:
            print(f"scene id:{self.id}")
            if task.time - self.time > 1 / 240.0:
                self.before_simulation_step()

                # Step simulation
                p.stepSimulation()
                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)
                print(f"ecm view out matrix:{self.ecm_view_out}")
                self.time = task.time
        else:
            if time.time() - self.time > 1/240:
                self.before_simulation_step()

                # Step simulation
                #pb.stepSimulation()
                # self._duration = 0.1 # needle 
                self._duration = 0.1
                step(self._duration)

                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = time.time()
                # print(f"current time: {self.time}")
                # print(f"current task time: {task.time}")

                # if time.time()-self.start_time > (self.itr + 1) * time_size:
                obs = self.env._get_obs()
                obs = self.env._get_obs()['achieved_goal'] if isinstance(obs, dict) else None
                # print(f"waypoints: {self.env._waypoints}")
                success = self.env._is_success(obs,self.env._sample_goal()) if obs is not None else False
                print(f"success: {success}")
                wait_list=[12,30]
                if (self.id not in wait_list and success) or time.time()-self.start_time > 10:   
                    # if self.cnt>=6: 
                    #     self.kivy_ui.stop()
                    #     self.app.win.removeDisplayRegion(self.ui_display_region)
                    # self.before_simulation_step()
                    # self._duration = 0.2
                    # step(self._duration)
                    # self.after_simulation_step()
                    
                    open_scene(0)
                    print(f"xxxx current time:{time.time()}")
                    open_scene(self.id)
                    exempt_l = [i for i in range(21,23)]
                    if self.id not in exempt_l:
                        self.toggleEcmView()
                    # self.cnt+=1
                    return 
                    # self.start_time=time.time()
                    # self.toggleEcmView()
                    # self.itr += 1
                        
        return Task.cont

    def load_policy(self, obs, env):
        steps = str(300000)
        if self.id == 8:
            model_file = './peg_transfer_model'
        if self.id == 6:
            model_file = './needle_pick_model'

        actor_params = os.path.join(model_file, 'actor_' + steps + '.pt')
        actor_params = torch.load(actor_params)

        dim_o = obs['observation'].shape[0]
        dim_g = obs['desired_goal'].shape[0]
        dim_a = env.action_space.shape[0]
        actor = DeterministicActor(
            dim_o, dim_g, dim_a, 256
        )

        actor.load_state_dict(actor_params)

        g_norm = Normalizer(dim_g)
        g_norm_stat = np.load(os.path.join(model_file, 'g_norm_' + steps + '_stat.npy'), allow_pickle=True).item()
        g_norm.mean = g_norm_stat['mean']
        g_norm.std = g_norm_stat['std']

        o_norm = Normalizer(dim_o)
        o_norm_stat = np.load(os.path.join(model_file, 'o_norm_' + steps + '_stat.npy'), allow_pickle=True).item()
        o_norm.mean = o_norm_stat['mean']
        o_norm.std = o_norm_stat['std']

        return actor, o_norm, g_norm

    def _preproc_inputs(self, o, g, o_norm, g_norm):
        o_norm = o_norm.normalize(o)
        g_norm = g_norm.normalize(g)
 
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def before_simulation_step(self):
        if (self.id == 8 or self.id == 6) and not self.has_load_policy:
            obs = self.env._get_obs()
            self.actor, self.o_norm,self.g_norm = self.load_policy(obs,self.env)
            self.has_load_policy = True

        retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        # to do define a new getdeviceaction_right function for mtm 
        # getDeviceAction_right(retrived_action)
        if self.id in self.full_dof_list:
            retrived_action= np.array([0, 0, 0, 0], dtype = np.float32)
            mat = np.eye(4)
            self.get_MTMR_position_action(retrived_action,mat)
        else:
            self.get_MTMR_position_action(retrived_action)

        if self.demo:
            obs = self.env._get_obs()
            action = self.env.get_oracle_action(obs)
        if self.env.ACTION_SIZE != 3 and self.env.ACTION_SIZE != 1:

            self.psm1_action = retrived_action

            print(f"len of retrieved action:{len(retrived_action)}")
            if self.demo:
                if self.id ==8:
                    obs = self.env._get_obs()
                    o, g = obs['observation'], obs['desired_goal']
                    input_tensor = self._preproc_inputs(o, g, self.o_norm, self.g_norm)
                    action = self.actor(input_tensor).data.numpy().flatten()
                    print(f"retrieved action is: {self.psm1_action}")
                    self.psm1_action = action
                    self.env._set_action(self.psm1_action)
                else:
                    obs = self.env._get_obs()
                    action = self.env.get_oracle_action(obs)
                    self.psm1_action = action
                    self.env._set_action(self.psm1_action)
                    self.env._step_callback()
            else:
                
                if self.id in self.full_dof_list:
                    print("mat is",mat,'psm action is',self.psm1_action)
                    self.env._set_action(self.psm1_action,mat)
                    self.env._step_callback()
                else:
                    self.env._set_action(self.psm1_action)
                    self.env._step_callback()
        
        '''Control ECM'''
        ecm_control_idx = 3 if self.id in self.full_dof_list else 4 #for needle pick full dof test
        if retrived_action[ecm_control_idx] == 3:
            print(f'!!!retrieved action:{retrived_action}')
            self.ecm_action[0] = -retrived_action[0]*0.2
            self.ecm_action[1] = -retrived_action[1]*0.2
            self.ecm_action[2] = retrived_action[2]*0.2
        print(self.id)
        #active track
        if self.id == 27 or self.id == 28:
            self.env._step_callback()

        # self.env._set_action(self.ecm_action)
        if self.demo and (self.env.ACTION_SIZE == 3 or self.env.ACTION_SIZE == 1):
            print('ecm here 1')
            self.ecm_action = action
            self.env._set_action(self.ecm_action)
        if self.demo is None:
            print('ecm here')
            self.env._set_action_ecm(self.ecm_action)
        self.env.ecm.render_image()
        # _,_,rgb,_,_ = p.getCameraImage(
        #             width=100, height=100,
        #             viewMatrix=self.env._view_matrix,
        #             projectionMatrix=self.env._proj_matrix
        #             )
        # rgb_array = np.array(rgb, dtype=np.uint8)
        # if len(rgb_array)>0:
        #     rgb_array = np.reshape(rgb_array, (100, 100, 4))

        #     rgb_array = rgb_array[:, :, :3]

        #     imageio.imwrite('test.png',rgb_array)
        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out


    def addEcmAction(self, dim, incre):
        self.ecm_action[dim] += incre

    def setEcmAction(self, dim, val):
        self.ecm_action[dim] = val
    def resetEcmFlag(self):
        # print("reset enter")
        self.env._reset_ecm_pos()
    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

    def on_destroy(self):
        # !!! important
        # stopScheduler()
        # closeTouch_right()
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)


class SurgicalSimulatorBimanual(SurgicalSimulatorBase):
    def __init__(self, env_type, env_params, id=None, demo=None):
    # def __init__(self, env_type, env_params, jaw_states=[1.0, 1.0],id=None,demo=None):
        super(SurgicalSimulatorBimanual, self).__init__(env_type, env_params)

        # initTouch_right()
        # initTouch_left()
        # startScheduler()
        self.id=id
        self.demo = demo
        self.closed=True
        self.start_time = time.time()

        self.psm1_action = np.zeros(env_type.ACTION_SIZE // 2)

        self.psm2_action = np.zeros(env_type.ACTION_SIZE // 2)

        self.ecm_view = 0
        self.ecm_view_out = None
        exempt_l = [i for i in range(21,23)]
        if self.id not in exempt_l:
            self.toggleEcmView()
        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)


    def get_MTM_position_action(self,psm_action,who,mat=None):

        if self.first[who]:
            for i in range(3):
                psm_action[i] =0
            # psm_action[4] = 1
            self.first[who] = False
        elif self.clutched:
            for i in range(3):
                psm_action[i] =0
        else:
            self.pos_cur = np.array([self.ml.setpoint_cp().p[i] for i in range(3)]) if who == 0 else np.array([self.mr.setpoint_cp().p[i] for i in range(3)])

            psm_action[0] = (self.pos_cur[1] - self.pos[who][1])*(1000)
            psm_action[1] = (self.pos_cur[0] - self.pos[who][0])*(-1000)
            psm_action[2] = (self.pos_cur[2] - self.pos[who][2])*(1000)

            self.pos[who] = self.pos_cur.copy()
        psm_action[3]= self.mr.gripper.measured_jp()[0] if who ==1 else self.ml.gripper.measured_jp()[0]
        goal_orn = self.mr.setpoint_cp().M if who == 1 else self.ml.setpoint_cp().M 
        print("goal orn: ",goal_orn)
        for i in range(3):
            for j in range(3):
                mat[i][j]= goal_orn[i,j]
        print("bimaunal idx",who)

    def _step_simulation_task(self, task):
        """Step simulation
        """
        if self.demo == None:
            print(f"scene id:{self.id}")
            if task.time - self.time > 1 / 240.0:
                self.before_simulation_step()

                # Step simulation
                p.stepSimulation()
                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = task.time
        else:
            if time.time() - self.time > 1/240:
                self.before_simulation_step()

                # Step simulation
                #pb.stepSimulation()
                # self._duration = 0.1 # needle 
                self._duration = 0.1
                step(self._duration)

                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = time.time()
                # print(f"current time: {self.time}")
                # print(f"current task time: {task.time}")

                # if time.time()-self.start_time > (self.itr + 1) * time_size:
                obs = self.env._get_obs()
                obs = self.env._get_obs()['achieved_goal'] if isinstance(obs, dict) else None
                success = self.env._is_success(obs,self.env._sample_goal()) if obs is not None else False
                if  time.time()-self.start_time > 18:   
                    # if self.cnt>=6: 
                    #     self.kivy_ui.stop()
                    #     self.app.win.removeDisplayRegion(self.ui_display_region)

                    open_scene(0)
                    print(f"xxxx current time:{time.time()}")
                    open_scene(self.id)
                    exempt_l = [i for i in range(21,23)]
                    if self.id not in exempt_l:
                        self.toggleEcmView()
                    # self.cnt+=1
                    return 
                    # self.start_time=time.time()
                    # self.toggleEcmView()
                    # self.itr += 1
                        
        return Task.cont


    def before_simulation_step(self):

        # haptic left
        retrived_action = np.array([0, 0, 0, 0], dtype = np.float32)
        mat_r = np.eye(4)
        # getDeviceAction_left(retrived_action)
        self.get_MTM_position_action(retrived_action,1,mat_r)
        # retrived_action-> x,y,z, angle, buttonState(0,1,2)
        if self.demo:
            obs = self.env._get_obs()
            action = self.env.get_oracle_action(obs)
        self.psm1_action = retrived_action

        # # haptic right
        retrived_action_l = np.array([0, 0, 0, 0], dtype = np.float32)
        mat_l = np.eye(4)
        # getDeviceAction_right(retrived_action)
        self.get_MTM_position_action(retrived_action_l,0,mat_l)
        self.psm2_action = retrived_action_l

        if self.demo:
            self.env._set_action(action)
        else:
            # self.env._set_action(np.concatenate([self.psm2_action, self.psm1_action], axis=-1))
            print('enttter here',self.psm1_action,self.psm2_action,mat_r,mat_l)
            self.env._set_action(action = self.psm1_action,mat_r=mat_r,action_l=self.psm2_action,mat_l=mat_l)
        self.env._step_callback()
        '''Control ECM'''
        # if retrived_action[4] == 3:
        #     self.ecm_action[0] = -retrived_action[0]*0.2
        #     self.ecm_action[1] = -retrived_action[1]*0.2
        #     self.ecm_action[2] = retrived_action[2]*0.2

        # self.env._set_action(self.ecm_action)
        self.env._set_action_ecm(self.ecm_action)
        self.env.ecm.render_image()

        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out


    def addEcmAction(self, dim, incre):
        self.ecm_action[dim] += incre

    def setEcmAction(self, dim, val):
        self.ecm_action[dim] = val
    def resetEcmFlag(self):
        # print("reset enter")
        self.env._reset_ecm_pos()
    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

    def on_destroy(self):
        # !!! important
        # stopScheduler()
        # closeTouch_right()     
        # closeTouch_left()
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)


# ecm steoro size 1024x768
app_cfg = ApplicationConfig(window_width=1200, window_height=900) #for demo screen, change to 800x600, org 1200x900
app = Application(app_cfg)
open_scene(0)
app.run()