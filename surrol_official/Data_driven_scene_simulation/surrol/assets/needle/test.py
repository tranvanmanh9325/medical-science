import pybullet as p

p.connect(p.GUI)
pid = p.loadURDF('./needle_40mm_RL.urdf')
print(p.getDynamicsInfo(pid, -1))
