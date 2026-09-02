import pybullet as p
import numpy as np

p.connect(p.GUI)
psm = p.loadURDF('./psm_RL.urdf')
numjoints = p.getNumJoints(psm)  #16

# for i in range(numjoints):
#     print(i)
#     print(p.getJointInfo(psm, i))

# while 1:
#     p.stepSimulation()

# print(len(p.getLinkState(psm, 6, computeLinkVelocity=1)))
# print(p.getLinkState(psm, 7))

print(p.getCollisionShapeData(psm, 6)[0][3])
print(p.getCollisionShapeData(psm, 7))
# p.GEOM_BOX=3
