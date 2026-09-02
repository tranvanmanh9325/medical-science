import re,os
from kivy.lang import Builder
import numpy as np
import torch
import torch.nn as nn
import pybullet as p
from panda3d_kivy.mdapp import MDApp

from direct.gui.DirectGui import *
from panda3d.core import AmbientLight, DirectionalLight, Spotlight, PerspectiveLens

import math
import imageio
import time
import sys


random_shuffle = sys.argv[1] # default: fixed
user_name = sys.argv[2]
test_name = sys.argv[3]

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application, ApplicationConfig
from surrol.tasks.ecm_misorient import MisOrient
from surrol.tasks.ecm_reach import ECMReach
from surrol.tasks.ecm_active_track import ActiveTrack
from surrol.tasks.ecm_static_track import StaticTrack
from surrol.tasks.gauze_retrieve import GauzeRetrieve
from surrol.tasks.gauze_retrieve_full_dof import GauzeRetrieveFullDof   
if test_name == 'test1':
    from surrol.tasks.gauze_retrieve_full_dof_test1 import GauzeRetrieveFullDof_fixed
elif test_name == 'test2':
    from surrol.tasks.gauze_retrieve_full_dof_test2 import GauzeRetrieveFullDof_fixed
elif test_name == 'test3':
    from surrol.tasks.gauze_retrieve_full_dof_test3 import GauzeRetrieveFullDof_fixed   
from surrol.tasks.gauze_retrieve_full_dof_haptic import GauzeRetrieveFullDof_haptic
    
from surrol.tasks.needle_reach import NeedleReach
from surrol.tasks.needle_reach_full_dof import NeedleReachFullDof
if test_name == 'test1':
    from surrol.tasks.needle_reach_full_dof_test1 import NeedleReachFullDof_fixed
elif test_name == 'test2':
    from surrol.tasks.needle_reach_full_dof_test2 import NeedleReachFullDof_fixed
elif test_name == 'test3':
    from surrol.tasks.needle_reach_full_dof_test3 import NeedleReachFullDof_fixed


from surrol.tasks.needle_pick_full_dof_haptic import NeedlePickFullDof_haptic
# from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.needle_pick_full_dof import NeedlePickFullDof
from surrol.tasks.peg_board_full_dof import BiPegBoardFullDof
# from surrol.tasks.peg_transfer import PegTransfer
from surrol.tasks.peg_transfer_full_dof_haptic import PegTransferFullDof_haptic

from surrol.tasks.peg_transfer_full_dof import PegTransferFullDof
if test_name == 'test1':
    from surrol.tasks.peg_transfer_full_dof_test1 import PegTransferFullDof_fixed
elif test_name == 'test2':
    from surrol.tasks.peg_transfer_full_dof_test2 import PegTransferFullDof_fixed
elif test_name == 'test3':
    from surrol.tasks.peg_transfer_full_dof_test3 import PegTransferFullDof_fixed

from surrol.tasks.peg_transfer_RL import PegTransferRL
from surrol.tasks.needle_regrasp_bimanual import NeedleRegrasp
from surrol.tasks.needle_regrasp_full_dof import NeedleRegraspFullDof

# from surrol.tasks.peg_transfer_bimanual_org import BiPegTransfer
from surrol.tasks.peg_transfer_bimanual import BiPegTransfer
from surrol.tasks.peg_transfer_bimanual_fulldof import BiPegTransferFullDof

from surrol.tasks.ring_rail_cu import RingRailCU
from surrol.tasks.peg_board import PegBoard
from surrol.tasks.pick_and_place import PickAndPlace
from surrol.tasks.pick_and_place_full_dof import PickAndPlaceFullDof
from surrol.tasks.match_board import MatchBoard 
from surrol.tasks.match_board_full_dof import MatchBoardFullDof 
from surrol.tasks.needle_pick_RL import NeedlePickRL
from surrol.tasks.needle_the_rings import NeedleRings
from surrol.tasks.ecm_env import EcmEnv, goal_distance,reset_camera
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.robots.ecm import Ecm

from haptic_src._test import initTouch_right, closeTouch_right, getDeviceAction_right, startScheduler, stopScheduler
from haptic_src._test import initTouch_left, closeTouch_left, getDeviceAction_left

import rospy
import sys
import time
import math
# move in cartesian space
import PyKDL
import pickle
from scipy.spatial.transform import Rotation as R
import threading
from sensor_msgs.msg import Joy
from dvrk import mtm

from direct.task import Task
from surrol.utils.pybullet_utils import step
import pybullet_data
######
from dvrk_control.dvrk_control import *
from dvrk_control.test_potential_field3d import *


app = None
hint_printed = False
resetFlag = False
cnt = 0
cntt = 0

random_seed = 1024
    

def frame_to_matrix(frame):
    rotation = np.array([[frame.M[0,0], frame.M[0,1], frame.M[0,2]],
                         [frame.M[1,0], frame.M[1,1], frame.M[1,2]],
                         [frame.M[2,0], frame.M[2,1], frame.M[2,2]]])
    translation = np.array([frame.p[0], frame.p[1], frame.p[2]])

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix

def home_every_scene():
    m = mtm('MTMR')
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

home_every_scene()

def open_scene(id):
    global app, hint_printed,resetFlag
    scene = None
    menu_dict = {0:StartPage(),1:ECMPage(),2:NeedlePage(),3:PegPage(),4:PickPlacePage()}
    task_list =[NeedlePickFullDof,PegTransferFullDof,NeedleRegraspFullDof,BiPegTransferFullDof,PickAndPlaceFullDof,BiPegBoardFullDof,NeedleRings,MatchBoardFullDof,ECMReach,MisOrient,StaticTrack,ActiveTrack,NeedleReachFullDof,GauzeRetrieveFullDof]
    bimanual_list=[9,10,11,12,15,17,18]
    if id < 5:
        scene = menu_dict[id]
    elif id in bimanual_list:
        
        jaws=[1.0, 1.0] if id==9 or id==10 else [1.0,1.0]
        scene = SurgicalSimulatorBimanual(task_list[(id-5)//2], {'render_mode': 'human'}, jaw_states=jaws,id=id) if id %2==1 else \
        SurgicalSimulatorBimanual(task_list[(id-5)//2], {'render_mode': 'human'}, jaw_states=jaws,id=id,demo=1)
    else:
        if id ==8:
            scene = SurgicalSimulator(PegTransferFullDof_haptic,{'render_mode': 'human'},id,demo=1)
        elif id ==6:
            try:
                scene = SurgicalSimulator(NeedlePickFullDof_haptic,{'render_mode': 'human'},id,demo=1)
            except Exception as e:
                print(str(e))
        elif id == 32:
            scene = SurgicalSimulator(GauzeRetrieveFullDof_haptic,{'render_mode': 'human'},id,demo=1)
        else:
            print(id)
            if id%2==1:
                home_every_scene()
                if id == 5 and random_shuffle=='fixed':
                    scene = SurgicalSimulator(NeedlePickFullDof_haptic,{'render_mode': 'human'},id)
                    
                elif id == 7 and random_shuffle=='fixed':
                    scene = SurgicalSimulator(PegTransferFullDof_fixed,{'render_mode': 'human'},id) 
                elif id == 31 and random_shuffle=='fixed':
                    scene = SurgicalSimulator(GauzeRetrieveFullDof_fixed,{'render_mode': 'human'},id) 
                elif id == 29 and random_shuffle == 'fixed':
                    scene = SurgicalSimulator(NeedleReachFullDof_fixed,{'render_mode': 'human'},id) 
                else: 
                    try:
                        scene = SurgicalSimulator(task_list[(id-5)//2],{'render_mode': 'human'},id)
                    except Exception as e:
                        print(str(e))
                    
            else:
                try:
                    scene = SurgicalSimulator(task_list[(id-5)//2],{'render_mode': 'human'},id,demo=1)
                except Exception as e:
                    print(str(e))
                print(scene)
    if scene:
        app.play(scene)
    print(id)

selection_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'
    spacing: dp(10)
    padding: dp(20)

    MDLabel:
        text: "SurRol Simulator v2"
        theme_text_color: "Primary"
        font_style: "H6"
        bold: True
        size_hint: 1.0, 0.1

    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "30dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 0.9
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/ecm_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "1. Endoscope Fov Control"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Practise ECM control skills"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlepick_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "2. Fundamental Actions"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Practise fundamental actions in surgery" 
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "3. Basic Robot Skill Training Tasks"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pratise positioning and orienting objects"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

'''      

        # MDCard:
        #     orientation: "vertical"
        #     size_hint: .45, None
        #     height: box_top.height + box_bottom.height

        #     MDBoxLayout:
        #         id: box_top
        #         spacing: "20dp"
        #         adaptive_height: True

        #         FitImage:
        #             source: "images/pick&place.png"
        #             size_hint: 0.5, None
        #             height: text_box.height

        #         MDBoxLayout:
        #             id: text_box
        #             orientation: "vertical"
        #             adaptive_height: True
        #             spacing: "10dp"
        #             padding: 0, "10dp", "10dp", "10dp"

        #             MDLabel:
        #                 text: "4. Pick & Place Tasks"
        #                 theme_text_color: "Primary"
        #                 font_style: "H6"
        #                 bold: True
        #                 adaptive_height: True

        #             MDLabel:
        #                 text: "Practice positioning and orienting objects"
        #                 adaptive_height: True
        #                 theme_text_color: "Primary"

        #     MDSeparator:

        #     MDBoxLayout:
        #         id: box_bottom
        #         adaptive_height: True
        #         padding: "0dp", 0, 0, 0
                
        #         MDRaisedButton:
        #             id: btn4
        #             text: "Play"
        #             size_hint: 0.8, 1.0
        #         MDIconButton:
        #             icon: "application-settings"

Peg_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                             Basic Robot Skill Training Tasks"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Peg Transfer"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the gripper to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/bipegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Bi-Peg Transfer"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Bimanual peg transfer"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegboard.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Peg Board"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Transfer red ring from peg board to peg on floor"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pick&place.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Pick and Place"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Place colored jacks into matching containers"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/matchboard.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Match Board"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Transfer various objects into matching spaces"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn5
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlerings.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle the Rings"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pass a needle through the target ring"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn6
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"



    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn7
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn8
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn8
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn8
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0

'''

PickPlace_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                              Pick & Place Tasks"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/gauze_retrieve.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Gauze Retrieve"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: " Pick the gauze and place it at the target position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pick&place.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Pick and Place"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Place colored jacks into matching colored containers"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/matchboard.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Match Board"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pick up various objects and place them into corresponding spaces on the board"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        


    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn4
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn5
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn7
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0

'''

needle_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                        Fundamental Actions"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needle_reach.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle Reach"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the gripper to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/gauze_retrieve.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Gauze Retrieve"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pick the gauze and place it at the target position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlepick_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle Pick"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pick up the needle and move to target position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needleregrasp_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle Regrasp"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Bimanual version of needle pick"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        




    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn5
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn6
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn6
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0

'''


ECM_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                     Endoscope Fov Control"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/ecm_reach.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "ECM Reach"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the ECM to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/misorient.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "MisOrient"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Adjust ECM to minimize misorientation"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/static_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Static Track"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Make ECM track a static target cube"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/active_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Active Track"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Make ECM track a active target cube"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"




    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn5
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn6
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn6
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0


'''

class SelectionUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(selection_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(1))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(2))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(3))

class NeedleUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(needle_panel_kv)
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
        self.screen = Builder.load_string(Peg_panel_kv)
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

class PickPlaceUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(PickPlace_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(31))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(13))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(19))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(4))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(4))
        self.screen.ids.btn7.bind(on_press = lambda _: open_scene(4))

class ECMUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(ECM_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(21))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(23))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(25))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(27))
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

class PickPlacePage(Scene):
    def __init__(self):
        super(PickPlacePage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = PickPlaceUI(
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

menu_bar_kv_haptic = '''MDBoxLayout:
    md_bg_color: (1, 0, 0, 0)
    # adaptive_height: True
    padding: "0dp", 0, 0, 0
    
    MDRectangleFlatIconButton:
        icon: "exit-to-app"
        id: btn1
        text: "Exit"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "head-lightbulb-outline"
        id: btn2
        text: "AI Assistant"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "chart-histogram"
        id: btn3
        text: "Evaluation"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "help-box"
        id: btn4
        text: "Switch to RL Agent"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
'''

menu_bar_kv_RL = '''MDBoxLayout:
    md_bg_color: (1, 0, 0, 0)
    # adaptive_height: True
    padding: "0dp", 0, 0, 0
    
    MDRectangleFlatIconButton:
        icon: "exit-to-app"
        id: btn1
        text: "Exit"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "head-lightbulb-outline"
        id: btn2
        text: "AI Assistant"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "chart-histogram"
        id: btn3
        text: "Evaluation"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "help-box"
        id: btn4
        text: "Switch to Haptic Device Training"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
'''

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
            self.screen = Builder.load_string(menu_bar_kv_RL)
        else:
            self.screen = Builder.load_string(menu_bar_kv_haptic)
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
        self.mr.body_set_cf_orientation_absolute(True)
        self.mr.use_gravity_compensation(True)
        self.mr.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.mr_init_pose = self.mr.setpoint_cp()
        
        init_pos_r = self.mr.setpoint_cp().p
        self.pos.append(np.array([init_pos_r[i] for i in range(3)])) # front(neg) back: 1; left(neg) right: 0; up down(neg): 2
        self.pos_cur_r = self.pos[1].copy()

        init_orn_r = self.mr.setpoint_cp().M.GetEulerZYX()
        self.orn_r = np.array([init_orn_r[i] for i in range(3)]) # 
        self.orn_cur_r = self.orn_r.copy()
        # print('self.mr.setpoint_cp().p',self.mr.setpoint_cp().p)
        # print('self.env._get_robot_state', self.env._get_robot_state(idx=0))
        # print('pose', self.pos_cur_r)
        # print('self.orn_r', self.orn_r)
        # r = R.from_euler('zyx',self.orn_r,degrees=True)
        # calculate_orientation = r.as_matrix()
        # print('calculate_orientation', calculate_orientation)
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
        self.full_dof_list = [5,6,7,8,13,19,29,30,31,32]
        self.path = []
        self.trajectory = []
        self.record = []
        self.contact_result = False
        self.wait_flag = False
        self.overall_distance = 0
        self.first_waypoint = True
        
        # initTouch_right()
        # startScheduler()
        if env_type.ACTION_SIZE != 3 and env_type.ACTION_SIZE != 1:
            self.psm1_action = np.zeros(env_type.ACTION_SIZE)
            if self.id not in self.full_dof_list:
                self.psm1_action[4] = 0.5

            self.app.accept('w-up', self.setPsmAction, [2, 0])
            self.app.accept('w-repeat', self.addPsmAction, [2, 0.01])
            self.app.accept('s-up', self.setPsmAction, [2, 0])
            self.app.accept('s-repeat', self.addPsmAction, [2, -0.01])
            self.app.accept('d-up', self.setPsmAction, [1, 0])
            self.app.accept('d-repeat', self.addPsmAction, [1, 0.01])
            self.app.accept('a-up', self.setPsmAction, [1, 0])
            self.app.accept('a-repeat', self.addPsmAction, [1, -0.01])
            self.app.accept('q-up', self.setPsmAction, [0, 0])
            self.app.accept('q-repeat', self.addPsmAction, [0, 0.01])
            self.app.accept('e-up', self.setPsmAction, [0, 0])
            self.app.accept('e-repeat', self.addPsmAction, [0, -0.01])
            self.app.accept('space-up', self.setPsmAction, [4, 1.0])
            self.app.accept('space-repeat', self.setPsmAction, [4, -0.5])

        self.ecm_view = 1
        self.ecm_view_out = None
        self.demo = demo
        self.start_time = time.time()
        exempt_l = [i for i in range(21,23)]
        # if self.id not in exempt_l:
        #     self.toggleEcmView()
        self.has_load_policy = False

        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)
        self.app.accept('i-up', self.setEcmAction, [2, 0])
        self.app.accept('i-repeat', self.addEcmAction, [2, 0.2])
        self.app.accept('k-up', self.setEcmAction, [2, 0])
        self.app.accept('k-repeat', self.addEcmAction, [2, -0.2])
        self.app.accept('o-up', self.setEcmAction, [1, 0])
        self.app.accept('o-repeat', self.addEcmAction, [1, 0.2])
        self.app.accept('u-up', self.setEcmAction, [1, 0])
        self.app.accept('u-repeat', self.addEcmAction, [1, -0.2])
        self.app.accept('j-up', self.setEcmAction, [0, 0])
        self.app.accept('j-repeat', self.addEcmAction, [0, 0.2])
        self.app.accept('l-up', self.setEcmAction, [0, 0])
        self.app.accept('l-repeat', self.addEcmAction, [0, -0.2])
        self.app.accept('m-up', self.toggleEcmView)
        self.app.accept('r-up', self.resetEcmFlag)

    def get_MTMR_position_action(self,psm1_action,mat=None):
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

            for i in range(3):
                for j in range(3):
                    mat[i][j]= goal_orn[i,j]
            return psm1_action, mat
            
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
            return psm1_action

    def move_to_target_psm_forcebased(self, pid_x, pid_y, pid_z, target_psm_position):
        
        current_psm_position = self.env._get_robot_state(idx=0)[0:3]
        print(current_psm_position)
        diff = np.array(target_psm_position) - np.array(current_psm_position)
        distance = np.linalg.norm(diff)    
        print(distance) 
        force_scale = 2
        force = np.array([-pid_y(current_psm_position[1]), pid_x(current_psm_position[0]),pid_z(current_psm_position[2]), 0, 0, 0])* force_scale
        self.mr.body.servo_cf(force)        
        
       
        if distance < 0.05:
            self.new_action_needed = True
            self.mr.body.servo_cf(np.array([0, 0, 0, 0, 0, 0]))
            
    def MTMR2PSM(self,psm1_action,mat=None):
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
                psm1_action[0] = (self.pos_cur[1] - self.pos[1][1])*(80)
                psm1_action[1] = (self.pos_cur[0] - self.pos[1][0])*(-80)
                psm1_action[2] = (self.pos_cur[2] - self.pos[1][2])*(80)
                self.pos[1] = self.pos_cur.copy()
            psm1_action[3] = self.mr.gripper.measured_jp()[0]
            goal_orn = self.mr.setpoint_cp().M
            for i in range(3):
                for j in range(3):
                    mat[i][j]= goal_orn[i,j]
            return psm1_action, mat
        else:
            if self.first[1]:
                for i in range(3):
                    psm1_action[i] =0
                psm1_action[4] = 1
                self.first[1] = False
            else:
                self.pos_cur = np.array([self.mr.setpoint_cp().p[i] for i in range(3)])
                psm1_action[0] = (self.pos_cur[1] - self.pos[1][1])*(50)
                psm1_action[1] = (self.pos_cur[0] - self.pos[1][0])*(-50)
                psm1_action[2] = (self.pos_cur[2] - self.pos[1][2])*(50)
                self.pos[1] = self.pos_cur.copy()
            psm1_action[3]=0
            return psm1_action

    def _step_simulation_task(self, task):
        """Step simulation
        """
        if self.demo == None:
            # print(f"scene id:{self.id}")
            if task.time - self.time > 1 / 240.0:
                try:
                    self.before_simulation_step()

                    # Step simulation
                    p.stepSimulation()
                    self.after_simulation_step()
                except Exception as e:
                    print(str(e))

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)
                self.time = task.time

                obs = self.env._get_obs()
                obs = self.env._get_obs()['achieved_goal'] if isinstance(obs, dict) else None
                success = self.env._is_success(obs,self.env._sample_goal()) if obs is not None else False
                wait_list=[12]
                global cnt
                if success: 
                    cnt += 1
                if (self.id not in wait_list and success) or time.time()-self.start_time > 100:      
                    total_distance = 0
                    for idx, items in enumerate(self.record):
                        if idx == 0:
                            previous_position = items['current_pos']
                        current_position = items['current_pos']
                        total_distance += np.linalg.norm(current_position[:3] - previous_position[:3])
                        previous_position = current_position
                    print('the overall distance is: ', total_distance)

                    open_scene(0)
                    print(f"xxxx current time:{time.time()}")
                    open_scene(self.id)
                    exempt_l = [i for i in range(21,23)]
                    if self.id not in exempt_l:
                        self.toggleEcmView()
                    if self.id == 7:
                        task_name = 'peg_transfer_'
                    elif self.id == 29:
                        task_name = 'needle_reach_'
                    elif self.id ==31:
                        task_name = 'gauze_retrive_'
                    count = str(cnt)
                    directory = f'./record/{user_name}/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        print(f"Directory '{directory}' created.")
                    else:
                        print(f"Directory '{directory}' already exists.")
                    with open(directory + task_name + user_name +'_'+test_name+'_'+ count +'.pkl', 'wb') as f:
                        pickle.dump(self.record, f)      
                    with open(directory + task_name + user_name +'_'+test_name+'_'+ count +'.txt', 'a') as txt_file:
                        txt_file.write('the final distance is: ', total_distance + '\n')
                    print('success')
                    return 
        else:
            # print('999999999999999999999',self.id)
            if time.time() - self.time > 1/240:
                try:
                    self.before_simulation_step()
                    self._duration = 0.1
                    step(self._duration)

                    self.after_simulation_step()
                except Exception as e:
                    print(str(e))
   
                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = time.time()

                obs = self.env._get_obs()
                obs = self.env._get_obs()['achieved_goal'] if isinstance(obs, dict) else None
                success = self.env._is_success(obs,self.env._sample_goal()) if obs is not None else False
                wait_list=[12]
                global cntt
                if success: 
                    cntt += 1
                if (self.id not in wait_list and success) or time.time()-self.start_time > 100:   
                    self.mr.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

                    if self.id == 8:
                        task_name = 'peg_transfer_'
                    elif self.id == 30:
                        task_name = 'needle_reach_'
                    elif self.id ==32:
                        task_name = 'gauze_retrive_'
                    count = str(cntt)
                    # with open('./record/demo_record_'+ task_name + user_name +'_'+ count +'.pkl', 'wb') as f:
                    with open('./record/demo_record_'+ task_name + user_name + '.pkl', 'wb') as f:
                        pickle.dump(self.record, f)
                    total_distance = 0
                    for idx, items in enumerate(self.record):
                        if idx == 0:
                            previous_position = items['current_pos']
                        current_position = items['current_pos']
                        total_distance += np.linalg.norm(current_position[:3] - previous_position[:3])
                        previous_position = current_position
                    print('the overall distance is: ', total_distance)
                    # with open('./record/demo_record_'+ task_name + user_name + '.txt', 'wb') as txt_file:
                    #     txt.write('the final distance is: ', total_distance + '\n')
                    #     txt.write('the overall distance is: ', self.overall_distance + '\n')
                    print('success')
                    return 

        return Task.cont


    def load_policy(self, obs, env):
        steps = str(300000)
        if self.id == 8:
            model_file = '/home/kj/skjsurrol/SurRoL_skj/tests/peg_transfer_model'
        if self.id == 6:
            model_file = '/home/kj/skjsurrol/SurRoL_skj/tests/needle_pick_model'

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
    
    def load_trajectory(self):
        # file = open('/home/kj/skjsurrol/SurRoL_skj/tests/saved_peg_transfer_action_psm.pkl','rb')
        # traj = pickle.load(file)
        # file.close()
        if self.id == 8:
            traj = np.load('/home/kj/skjsurrol/SurRoL_skj/tests/absolute_peg_transfer_position.npy')
            minimum_point = np.argmin(traj[..., 2])
            print(self.contact_result)
            if self.contact_result:
                print('Now starting second part')
                traj = traj[minimum_point:]
                # exit()
            # elif self.contact_result and current_psm_position[2] < 3.7:
            #     traj = np.array([current_psm_position, [current_psm_position[0], current_psm_position[1], 3.7]])
            #     print(traj)
            else:
                print('Start first part trajectory')
                traj = traj[:minimum_point]
            return traj
        elif self.id == 6 or self.id == 32:
            traj = np.load('/home/kj/skjsurrol/SurRoL_skj/tests/absolute_cauze_retrieve_position.npy')
            # current_psm_position = self.env._get_robot_state(idx=0)
            # print(current_psm_position, traj[0])
            # initial_point = traj[0][:3]
            # diff = initial_point - current_psm_position[:3]
            # print(diff)
            # exit()
            # step = diff / 10
            # initial_traj = []
            # for i in range(11):
            #     new_position = current_psm_position + [i * step[0], i * step[1],i * step[2],0,0,0,0]
            #     initial_traj.append(np.array(new_position))
            # traj = np.stack(initial_traj, traj)
            # print(initial_traj.shape)
            # exit()
            minimum_point = np.argmin(traj[..., 2])
            print(self.contact_result)
            if self.contact_result:
                print('Now starting second part')
                traj = traj[minimum_point:]

            else:
                print('Start first part trajectory')
                traj = traj[:minimum_point]
            return traj
        elif self.id == 30:
            goal = self.env._sample_goal()
            current_psm_position = self.env._get_robot_state(idx=0)
            diff = goal - current_psm_position[:3]
            step = diff / 40
            traj = []
            for i in range(41):
                new_position = current_psm_position + [i * step[0], i * step[1],i * step[2],0,0,0,0]
                traj.append(np.array(new_position))
            print(traj)
            np.save('/home/kj/skjsurrol/SurRoL_skj/tests/absolute_needle_reach_position.npy', np.array(traj))
            return np.array(traj)
        else:
            raise ValueError('Trajectory not implemented')

    def _preproc_inputs(self, o, g, o_norm, g_norm):
        o_norm = o_norm.normalize(o)
        g_norm = g_norm.normalize(g)
 
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def before_simulation_step(self):
        if (self.id == 8 or self.id == 6 or self.id == 30 or self.id == 32) and (not self.has_load_policy or self.contact_result):
            obs = self.env._get_obs()
            print('loading policy')
            # self.actor, self.o_norm,self.g_norm = self.load_policy(obs,self.env)
            self.traj = self.load_trajectory()
            print('policy loaded')
            self.has_load_policy = True

        if self.env.ACTION_SIZE != 3 and self.env.ACTION_SIZE != 1:
            if self.demo:
                if not self.wait_flag :
                    self.stanby_time = time.time()
                    self.wait_flag = True
                dt = time.time() - self.stanby_time
                scale = min(dt, 1)
                if self.first_waypoint:
                    self.previous_psm_position = self.env._get_robot_state(idx=0)[:3]
                    self.first_waypoint = False
                current_psm_position = self.env._get_robot_state(idx=0)
                step_distance = np.linalg.norm(current_psm_position[:3] - self.previous_psm_position)
                self.overall_distance += step_distance
                print('Overall_distance now is: ', self.overall_distance)
                self.previous_psm_position = current_psm_position[:3]
                # print(self.env._get_robot_state(idx=0))
                (target_psm_position, distance), force = trajectory_forward_field_3d(self.traj, current_psm_position, k_att=10, forward_steps=3)
                
                self.record.append({'current_pos':current_psm_position, 
                                        'target_pos':target_psm_position,
                                        'force':force,
                                        'distance':distance
                                        })
                
                self.mr.body.servo_cf(force * scale)

                '''
                use relative displacement

                Update PSM using MTM position
                '''

                if self.id in self.full_dof_list:
                    retrived_action= np.array([0, 0, 0, 0], dtype = np.float32)
                    mat = np.eye(4)
                    retrived_action, mat = self.MTMR2PSM(retrived_action,mat)
                    self.psm1_action = retrived_action
                    self.env._set_action(self.psm1_action,mat)
                    self.contact_result = self.env._step_callback()
                else:
                    retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
                    retrived_action = self.MTMR2PSM(retrived_action)
                    self.psm1_action = retrived_action
                    self.env._set_action(self.psm1_action)
                    self.contact_result = self.env._step_callback()
            else: 
                if self.first_waypoint:
                    self.previous_psm_position = self.env._get_robot_state(idx=0)[:3]
                    self.first_waypoint = False
                current_psm_position = self.env._get_robot_state(idx=0)
                step_distance = np.linalg.norm(current_psm_position[:3] - self.previous_psm_position)
                self.overall_distance += step_distance
                self.previous_psm_position = current_psm_position[:3]
                print('Overall_distance now is: ', self.overall_distance)     
                if self.id == 7:
                        task_name = 'peg_transfer_'
                elif self.id == 29:
                    task_name = 'needle_reach_'
                elif self.id ==31:
                    task_name = 'gauze_retrive_'
                directory = f'./record/{user_name}/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"Directory '{directory}' created.")
                          
                with open(directory + task_name + user_name +'_'+test_name+'_'+ str(cnt) +'.txt', 'a') as txt_file:                
                    txt_file.write('the overall distance is: '+ str(self.overall_distance) + '\n')

                current_psm_position = self.env._get_robot_state(idx=0)
                self.record.append({'current_pos':current_psm_position, 
                                        })               
                if self.id in self.full_dof_list:
                    retrived_action= np.array([0, 0, 0, 0], dtype = np.float32)
                    mat = np.eye(4)
                    retrived_action, mat = self.get_MTMR_position_action(retrived_action,mat)
                    self.psm1_action = retrived_action
                    self.env._set_action(self.psm1_action,mat)
                    self.contact_result = self.env._step_callback()
                    print('contact state is:', self.contact_result)
                else:
                    retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
                    retrived_action = self.get_MTMR_position_action(retrived_action)
                    self.psm1_action = retrived_action
                    self.env._set_action(self.psm1_action)
                    self.contact_result = self.env._step_callback()

        '''Control ECM'''
        ecm_control_idx = 3 if self.id in self.full_dof_list else 4 #for needle pick full dof test
        if retrived_action[ecm_control_idx] == 3:
            self.ecm_action[0] = -retrived_action[0]*0.2
            self.ecm_action[1] = -retrived_action[1]*0.2
            self.ecm_action[2] = retrived_action[2]*0.2
        #active track
        if self.id == 27 or self.id == 28:
            self.env._step_callback()

        if self.demo and (self.env.ACTION_SIZE == 3 or self.env.ACTION_SIZE == 1):
            # print('demo ecm here')
            self.ecm_action = self.psm1_action
            self.env._set_action(self.ecm_action)
        if self.demo is None:
            # print('ecm here')
            self.env._set_action_ecm(self.ecm_action)
        self.env.ecm.render_image()
        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out

    def setPsmAction(self, dim, val):
        self.psm1_action[dim] = val
        
    def addPsmAction(self, dim, incre):
        self.psm1_action[dim] += incre

    def addEcmAction(self, dim, incre):
        self.ecm_action[dim] += incre

    def setEcmAction(self, dim, val):
        self.ecm_action[dim] = val
    def resetEcmFlag(self):
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
    def __init__(self, env_type, env_params, jaw_states=[1.0, 1.0],id=None,demo=None):
        super(SurgicalSimulatorBimanual, self).__init__(env_type, env_params)

        # initTouch_right()
        # initTouch_left()
        # startScheduler()
        self.id=id
        self.demo = demo
        self.closed=True
        self.start_time = time.time()

        self.psm1_action = np.zeros(env_type.ACTION_SIZE // 2)
        self.psm1_action[4] = jaw_states[0]

        self.psm2_action = np.zeros(env_type.ACTION_SIZE // 2)
        self.psm2_action[4] = jaw_states[1]

        self.app.accept('w-up', self.setPsmAction1, [2, 0])
        self.app.accept('w-repeat', self.addPsmAction1, [2, 0.01])
        self.app.accept('s-up', self.setPsmAction1, [2, 0])
        self.app.accept('s-repeat', self.addPsmAction1, [2, -0.01])
        self.app.accept('d-up', self.setPsmAction1, [1, 0])
        self.app.accept('d-repeat', self.addPsmAction1, [1, 0.01])
        self.app.accept('a-up', self.setPsmAction1, [1, 0])
        self.app.accept('a-repeat', self.addPsmAction1, [1, -0.01])
        self.app.accept('q-up', self.setPsmAction1, [0, 0])
        self.app.accept('q-repeat', self.addPsmAction1, [0, 0.01])
        self.app.accept('e-up', self.setPsmAction1, [0, 0])
        self.app.accept('e-repeat', self.addPsmAction1, [0, -0.01])
        self.app.accept('space-up', self.setPsmAction1, [4, 1.0])
        self.app.accept('space-repeat', self.setPsmAction1, [4, -0.5])

        self.app.accept('i-up', self.setPsmAction2, [2, 0])
        self.app.accept('i-repeat', self.addPsmAction2, [2, 0.01])
        self.app.accept('k-up', self.setPsmAction2, [2, 0])
        self.app.accept('k-repeat', self.addPsmAction2, [2, -0.01])
        self.app.accept('l-up', self.setPsmAction2, [1, 0])
        self.app.accept('l-repeat', self.addPsmAction2, [1, 0.01])
        self.app.accept('j-up', self.setPsmAction2, [1, 0])
        self.app.accept('j-repeat', self.addPsmAction2, [1, -0.01])
        self.app.accept('u-up', self.setPsmAction2, [0, 0])
        self.app.accept('u-repeat', self.addPsmAction2, [0, 0.01])
        self.app.accept('o-up', self.setPsmAction2, [0, 0])
        self.app.accept('o-repeat', self.addPsmAction2, [0, -0.01])
        self.app.accept('m-up', self.setPsmAction2, [4, 1.0])
        self.app.accept('m-repeat', self.setPsmAction2, [4, -0.5])

        self.ecm_view = 0
        self.ecm_view_out = None
        exempt_l = [i for i in range(21,23)]
        if self.id not in exempt_l:
            self.toggleEcmView()
        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)
        self.app.accept('i-up', self.setEcmAction, [2, 0])
        self.app.accept('i-repeat', self.addEcmAction, [2, 0.2])
        self.app.accept('k-up', self.setEcmAction, [2, 0])
        self.app.accept('k-repeat', self.addEcmAction, [2, -0.2])
        self.app.accept('o-up', self.setEcmAction, [1, 0])
        self.app.accept('o-repeat', self.addEcmAction, [1, 0.2])
        self.app.accept('u-up', self.setEcmAction, [1, 0])
        self.app.accept('u-repeat', self.addEcmAction, [1, -0.2])
        self.app.accept('j-up', self.setEcmAction, [0, 0])
        self.app.accept('j-repeat', self.addEcmAction, [0, 0.2])
        self.app.accept('l-up', self.setEcmAction, [0, 0])
        self.app.accept('l-repeat', self.addEcmAction, [0, -0.2])
        self.app.accept('m-up', self.toggleEcmView)
        self.app.accept('r-up', self.resetEcmFlag)

    def get_MTM_position_action(self,psm_action,who,mat=None):
        if self.first[who]:
            for i in range(3):
                psm_action[i] =0
            psm_action[4] = 1
            self.first[who] = False
        elif self.clutched:
            for i in range(3):
                psm_action[i] =0
        else:
            self.pos_cur = np.array([self.ml.setpoint_cp().p[i] for i in range(3)]) if who == 0 else np.array([self.mr.setpoint_cp().p[i] for i in range(3)])
            psm_action[0] = (self.pos_cur[1] - self.pos[who][1])*(1000)
            psm_action[1] = (self.pos_cur[0] - self.pos[who][0])*(-1000)
            psm_action[2] = (self.pos_cur[2] - self.pos[who][2])*(1000)
            # if (self.pos_cur[i] - self.pos[i])*1000>0.1:
            #     print(f"variation:{self.pos_cur[i] - self.pos[i]}")
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
        retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        mat_r = np.eye(4)
        # getDeviceAction_left(retrived_action)
        self.get_MTM_position_action(retrived_action,1,mat_r)
        # retrived_action-> x,y,z, angle, buttonState(0,1,2)
        if self.demo:
            obs = self.env._get_obs()
            action = self.env.get_oracle_action(obs)

        self.psm1_action = retrived_action


        # # haptic right
        retrived_action_l = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        mat_l = np.eye(4)
        # getDeviceAction_right(retrived_action)
        self.get_MTM_position_action(retrived_action_l,0,mat_l)

        # retrived_action-> x,y,z, angle, buttonState(0,1,2)
        # if retrived_action[4] == 2:
        #     self.psm2_action[0] = 0
        #     self.psm2_action[1] = 0
        #     self.psm2_action[2] = 0
        #     self.psm2_action[3] = 0              
        # else:
        #     self.psm2_action[0] = retrived_action[2]*0.7
        #     self.psm2_action[1] = retrived_action[0]*0.7
        #     self.psm2_action[2] = retrived_action[1]*0.7
        #     self.psm2_action[3] = -retrived_action[3]/math.pi*180*0.6
        # if retrived_action[4] == 0:
        #     self.psm2_action[4] = 1
        # if retrived_action[4] == 1:
        #     self.psm2_action[4] = -0.5
        self.psm2_action = retrived_action_l


        if self.demo:
            self.env._set_action(action)
        else:
            # self.env._set_action(np.concatenate([self.psm2_action, self.psm1_action], axis=-1))
            print('enter here',self.psm1_action,self.psm2_action,mat_r,mat_l)
            self.env._set_action(action = self.psm1_action,mat_r=mat_r,action_l=self.psm2_action,mat_l=mat_l)
        self.env._step_callback()
        '''Control ECM'''
        if retrived_action[4] == 3:
            self.ecm_action[0] = -retrived_action[0]*0.2
            self.ecm_action[1] = -retrived_action[1]*0.2
            self.ecm_action[2] = retrived_action[2]*0.2


        # self.env._set_action(self.ecm_action)
        self.env._set_action_ecm(self.ecm_action)
        self.env.ecm.render_image()

        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out

    def setPsmAction1(self, dim, val):
        self.psm1_action[dim] = val
        
    def addPsmAction1(self, dim, incre):
        self.psm1_action[dim] += incre

    def setPsmAction2(self, dim, val):
        self.psm2_action[dim] = val
        
    def addPsmAction2(self, dim, incre):
        self.psm2_action[dim] += incre

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
app_cfg = ApplicationConfig(window_width=800, window_height=600) #for demo screen, change to 800x600, org 1200x900
app = Application(app_cfg)
open_scene(0)
app.run()
