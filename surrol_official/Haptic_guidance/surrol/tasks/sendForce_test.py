import numpy as np
import time

from haptic_src.touch_haptic import initTouch_right, closeTouch_right, getDeviceAction_right, startScheduler, stopScheduler
from haptic_src.touch_haptic import initTouch_left, closeTouch_left, getDeviceAction_left
from haptic_src.touch_haptic import sendForceFeedback_right, renderForce_right
from direct.task import Task
from surrol.utils.pybullet_utils import step


initTouch_right()
startScheduler()
AnchorPoint = np.array([0, 10, 10], dtype = np.float32)
PosInfo = np.array([0, 0, 0], dtype = np.float32)
ForceInfo = np.array([0, 0, 0], dtype = np.float32)
VelocityInfo = np.array([0, 0, 0], dtype = np.float32)
render_info = np.concatenate((AnchorPoint, ForceInfo, PosInfo, VelocityInfo), axis = 0)

while 1:
    # sendForceFeedback_right(AnchorPoint)
    
    print(renderForce_right(render_info))
    ForceInfo[0] = render_info[3]
    ForceInfo[1] = render_info[4]
    ForceInfo[2] = render_info[5]
    PosInfo[0] = render_info[6]
    PosInfo[1] = render_info[7]
    PosInfo[2] = render_info[8]
    VelocityInfo[0] = render_info[9]
    VelocityInfo[1] = render_info[10]
    VelocityInfo[2] = render_info[11]
    # print(PosInfo)

    # time.sleep(0.2)

