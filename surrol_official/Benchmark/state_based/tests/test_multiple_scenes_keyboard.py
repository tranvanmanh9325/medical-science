from kivy.lang import Builder
import numpy as np

from panda3d_kivy.mdapp import MDApp

from direct.gui.DirectGui import *
from panda3d.core import AmbientLight, DirectionalLight, Spotlight, PerspectiveLens

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application, ApplicationConfig
from surrol.tasks.needle_pick_org import NeedlePick
from surrol.tasks.needle_pick import NeedlePickFullDof

from surrol.tasks.peg_transfer_org import PegTransfer
from surrol.tasks.needle_regrasp_bimanual import NeedleRegrasp
from surrol.tasks.peg_transfer_bimanual import BiPegTransfer

# load and define the MTM
import dvrk
import numpy as np
import rospy
import sys
import time
import math
# move in cartesian space
import PyKDL

from dvrk import mtm
m = mtm('MTMR')
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
app = None
hint_printed = False

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R):

    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def open_scene(id):
    global app, hint_printed

    scene = None

    if id == 0:
        scene = StartPage()
    elif id == 1:
        scene = SurgicalSimulator(NeedlePickFullDof, {'render_mode': 'human'},id=1)
        home()
    elif id == 2:
        scene = SurgicalSimulator(PegTransfer, {'render_mode': 'human'},id=2)
        home()
    elif id == 3:
        scene = SurgicalSimulatorBimanual(
            BiPegTransfer, {'render_mode': 'human'}, jaw_states=[1.0, 1.0])

    if id in (1, 2):
        print('Press <W><S><A><D><Q><E><Space> to control the PSM move up/down/left/right/front/back/close jaw.')
        print('Press <T> to toggle ECM view.')
    else:
        print('Press <W><S><A><D><Q><E><Space> to control the left PSM move up/down/left/right/front/back/close jaw.')
        print('Press <I><K><J><L><M><O><Space> to control the right PSM move up/down/left/right/front/back/close jaw.')
        print('Press <T> to toggle ECM view.')
    if scene:
        app.play(scene)


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
                        text: "Pratice picking needle"
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
                    size_hint: 0.8, None
                MDIconButton:
                    icon: "application-settings"
                    size_hint: 0.2, 1.0
        

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
                        text: "Pratice picking pegs"
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
                    size_hint: 0.8, None
                MDIconButton:
                    icon: "application-settings"
                    size_hint: 0.2, 1.0

        

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
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, None
                MDIconButton:
                    icon: "application-settings"
                    size_hint: 0.2, 1.0
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
        self.screen.ids.btn1.bind(on_press=lambda _: open_scene(1))
        self.screen.ids.btn2.bind(on_press=lambda _: open_scene(2))
        self.screen.ids.btn3.bind(on_press=lambda _: open_scene(3))


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


menu_bar_kv = '''MDBoxLayout:
    md_bg_color: (1, 0, 0, 0)
    adaptive_height: True
    padding: "0dp", 0, 0, 0
    
    MDRectangleFlatIconButton:
        icon: "exit-to-app"
        id: btn1
        text: "Exit"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.3, 1.0
    MDRectangleFlatIconButton:
        icon: "head-lightbulb-outline"
        id: btn2
        text: "AI Assistant"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.3, 1.0
    MDRectangleFlatIconButton:
        icon: "chart-histogram"
        id: btn3
        text: "Evaluation"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.3, 1.0
    MDRectangleFlatIconButton:
        icon: "help-box"
        id: btn4
        text: "Help"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.3, 1.0
'''


class MenuBarUI(MDApp):
    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(menu_bar_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press=lambda _: open_scene(0))


# print with node id
def print_id(message):
    print('%s -> %s' % (rospy.get_caller_id(), message))

class SurgicalSimulatorBase(GymEnvScene):
    def __init__(self, env_type, env_params):
        super(SurgicalSimulatorBase, self).__init__(env_type, env_params)

    def before_simulation_step(self):
        pass

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
        slight.setShadowCaster(
            True, app.configs.shadow_resolution, app.configs.shadow_resolution)
        slnp = self.world3d.attachNewNode(slight)
        slnp.setPos(*(table_pos + np.array([0, 0.0, 5.0])))
        slnp.lookAt(*(table_pos + np.array([0.5, 0, 1.0])))
        self.world3d.setLight(slnp)

        self.m = mtm('MTMR')

        # turn gravity compensation on/off
        self.m.use_gravity_compensation(True)
        self.m.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.m_init_pose = self.m.setpoint_cp()
        # print(f'mtm setpoint: {self.m_init_pose}')
        init_pos = self.m.setpoint_cp().p
        self.pos = np.array([init_pos[i] for i in range(3)]) # front(neg) back: 1; left(neg) right: 0; up down(neg): 2
        self.pos_cur = self.pos.copy()

        init_orn = self.m.setpoint_cp().M.GetEulerZYX()
        self.orn = np.array([init_orn[i] for i in range(3)]) # 
        self.orn_cur = self.orn.copy()

        self.first = True
        # cp_r=np.array([[cp_r[i,0],cp_r[i,1],cp_r[i,2]] for i in range(3)])
        # print(f"position is: {self.pos}")
        # print(f"orientation is {self.orn}")
        # exit()
        # print(f"cur position is: {self.pos_cur}")

    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(
            0, 1.0, 0, 0.061)
        self.kivy_ui = MenuBarUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()

    def on_destroy(self):
        # !!! important
        # self.home()
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

    # homing example
    def home(self):
        print_id('starting enable')
        if not self.m.enable(2):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not self.m.home(2):
            sys.exit('failed to home within 10 seconds')
        # get current joints just to set size
        print_id('move to starting position')
        goal = np.copy(self.m.setpoint_jp())
        # go to zero position, make sure 3rd joint is past cannula
        goal.fill(0)
        self.m.move_jp(goal).wait()

class SurgicalSimulator(SurgicalSimulatorBase):
    def __init__(self, env_type, env_params,id=None):
        super(SurgicalSimulator, self).__init__(env_type, env_params)

        self.psm1_action = np.zeros(env_type.ACTION_SIZE)
        self.psm1_action[4] = 0.5
        
        self.ecm_view = 0
        self.ecm_view_out = None
        self.id = id

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

        self.app.accept('t-up', self.toggleEcmView)

    def before_simulation_step(self):
        if self.id==2:
            if self.first:
                for i in range(3):
                    self.psm1_action[i] =0
                self.psm1_action[4] = 1
                self.first = False
            else:
                self.pos_cur = np.array([self.m.setpoint_cp().p[i] for i in range(3)])
                # print(f"position is: {self.pos}")
                # print(f"cur position is: {self.pos_cur}")
                # for i in range(3):
                #     if i ==0:
                #         scaling = -20
                #     else:
                #         scaling = 20
                self.psm1_action[0] = (self.pos_cur[1] - self.pos[1])*(-250)
                self.psm1_action[1] = (self.pos_cur[0] - self.pos[0])*(250)
                self.psm1_action[2] = (self.pos_cur[2] - self.pos[2])*(250)
                # if (self.pos_cur[i] - self.pos[i])*1000>0.1:
                #     print(f"variation:{self.pos_cur[i] - self.pos[i]}")
                self.pos = self.pos_cur.copy()
            self.psm1_action[3]=0
        elif self.id==1:
            if self.first:
                for i in range(6):
                    self.psm1_action[i] =0
                self.psm1_action[6] = 1
                self.first = False
            else:
                cur_loc=self.m.setpoint_cp()
                self.pos_cur= np.array([cur_loc.p[i] for i in range(3)])
                self.orn_cur = np.array([cur_loc.M.GetEulerZYX()[i] for i in range(3)]) # 
                # print(f"position is: {self.pos}")
                # print(f"cur position is: {self.pos_cur}")
                # for i in range(3):
                #     if i ==0:
                #         scaling = -20
                #     else:
                #         scaling = 20
                self.psm1_action[0] = (self.pos_cur[1] - self.pos[1])*(-2500)
                self.psm1_action[1] = (self.pos_cur[0] - self.pos[0])*(2500)
                self.psm1_action[2] = (self.pos_cur[2] - self.pos[2])*(2500)

                self.psm1_action[3] = (self.orn_cur[2] - self.orn[2])*60
                self.psm1_action[4] = (self.orn_cur[1] - self.orn[1])*50
                self.psm1_action[5] = (self.orn_cur[0] - self.orn[0])*50

                # self.psm1_action[4] += (self.pos_cur[0] - self.pos[0])*(25)
                # self.psm1_action[5] += (self.pos_cur[2] - self.pos[2])*(25)

                # if (self.pos_cur[i] - self.pos[i])*1000>0.1:
                #     print(f"variation:{self.pos_cur[i] - self.pos[i]}")
                self.pos = self.pos_cur.copy()
                self.orn = self.orn_cur.copy()
            self.psm1_action[0] = 0
            self.psm1_action[1] = 0
            self.psm1_action[2] = 0
            # self.psm1_action[3] = 1 #pitch
            # self.psm1_action[4] = 0 #roll
            # self.psm1_action[5] = 0 #yaw
        self.env._set_action(self.psm1_action)
        self.env.ecm.render_image()
        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out
    def setPsmAction(self, dim, val):
        self.psm1_action[dim] = val

    def addPsmAction(self, dim, incre):
        self.psm1_action[dim] += incre

    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

class SurgicalSimulatorBimanual(SurgicalSimulatorBase):
    def __init__(self, env_type, env_params, jaw_states=[1.0, 1.0]):
        super(SurgicalSimulatorBimanual, self).__init__(env_type, env_params)

        self.psm1_action = np.zeros(env_type.ACTION_SIZE // 2)
        self.psm1_action[4] = jaw_states[0]

        self.psm2_action = np.zeros(env_type.ACTION_SIZE // 2)
        self.psm2_action[4] = jaw_states[1]

        self.ecm_view = 0
        self.ecm_view_out = None

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

        self.app.accept('t-up', self.toggleEcmView)

    def before_simulation_step(self):
        self.env._set_action(np.concatenate(
            [self.psm2_action, self.psm1_action], axis=-1))
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

    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

app_cfg = ApplicationConfig(window_width=800, window_height=600)
app = Application(app_cfg)
open_scene(0)
app.run()