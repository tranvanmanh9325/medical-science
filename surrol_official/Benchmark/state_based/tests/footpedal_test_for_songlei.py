# %%
import dvrk
# import sys
import rospy
import numpy as np
import threading
# import argparse
from sensor_msgs.msg import Joy
# from cisst_msgs.msg import prmCartesianImpedanceGains
from dvrk import mtm

m = mtm('MTMR')
m.use_gravity_compensation(True)
m.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

# %%
coag_event = threading.Event()
# foot pedal callback
def coag_event_cb(data):
    # print('~~~',data)
    # if (data.buttons[0] == 1):
    #     coag_event.set()
    #     print('clutched')
    # else:
    #     print('not clutched',data.buttons[0])
    print(data.buttons[0])
    if data.buttons[0] == 1:
        m.lock_orientation_as_is()
    else:
        m.unlock_orientation()
    return data.buttons[0]
rospy.Subscriber('footpedals/clutch',
                 Joy, coag_event_cb)

# %%
i=0
while i<250:
    coag_event.clear()
    coag_event.wait(1/250)
    i+=1

# %%



