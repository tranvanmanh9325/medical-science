import numpy as np
import dvrk
import PyKDL
from dvrk import mtm
from simple_pid import PID

def move_to_target_forcebased(mtm,target_position):
        mtm.body_set_cf_orientation_absolute(True)
        mtm.use_gravity_compensation(True)
        current_position = mtm.setpoint_cp().p
        diff = np.array(target_position) - np.array(current_position)
        distance = np.linalg.norm(np.array([diff.x(), diff.y(), diff.z()])) 
        pid_x = PID(5, 0.01, 0.1, setpoint = target_position[0])
        pid_y = PID(5, 0.01, 0.1, setpoint = target_position[1])
        pid_z = PID(5, 0.01, 0.1, setpoint = target_position[2]) 

        while (distance > 0.01) :

            force_scale = 30.0
            force = np.array([pid_x(current_position[0]), pid_y(current_position[1]), pid_z(current_position[2]), 0, 0, 0])* force_scale
            mtm.body.servo_cf(force)        
            
            current_position = mtm.setpoint_cp().p
            diff = np.array(target_position) - np.array(current_position)
            distance = np.linalg.norm(np.array([diff.x(), diff.y(), diff.z()]))    
            print(distance)          
        mtm.body.servo_cf(np.array([0, 0, 0, 0, 0, 0]))
        print('over') 

def move_to_target_psm_forcebased(self,env,mtm,psm_action):
        mtm.body_set_cf_orientation_absolute(True)
        mtm.use_gravity_compensation(True)
        current_psm_position = env._get_robot_state(idx=0)[0:3]
        print(current_psm_position)
        target_psm_position = current_psm_position + psm_action
        print(target_psm_position)
        distance = np.linalg.norm(psm_action) 
        pid_x = PID(5, 0.01, 0.1, setpoint = target_psm_position[0])
        pid_y = PID(5, 0.01, 0.1, setpoint = target_psm_position[1])
        pid_z = PID(5, 0.01, 0.1, setpoint = target_psm_position[2]) 
        time = 0
        while (distance > 0.01) and (time < 1000):

            force_scale = 1
            force = np.array([-pid_y(current_psm_position[1]), pid_x(current_psm_position[0]),pid_z(current_psm_position[2]), 0, 0, 0])* force_scale
            mtm.body.servo_cf(force)        
            
            current_psm_position = env._get_robot_state(idx=0)[0:3]
            diff = np.array(target_psm_position) - np.array(current_psm_position)
            distance = np.linalg.norm(diff)    
            print(distance) 
            if self.id in self.full_dof_list:
                    retrived_action= np.array([0, 0, 0, 0], dtype = np.float32)
                    mat = np.eye(4)
                    retrived_action, mat = self.get_MTMR_position_action(retrived_action,mat)
                    self.psm1_action = retrived_action
                    print("mat is",mat,'psm action is',self.psm1_action)
                    self.env._set_action(self.psm1_action,mat)
                    self.env._step_callback()
            else:
                retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
                retrived_action = self.get_MTMR_position_action(retrived_action)
                self.psm1_action = retrived_action
                self.env._set_action(self.psm1_action)
                print(self.psm1_action)
                self.env._step_callback()    
            time += 1
        mtm.body.servo_cf(np.array([0, 0, 0, 0, 0, 0]))
        print('over')     

def MTM_move_to_position(mtm, target, step_num = 10):
        current_MTM_pose = mtm.setpoint_cp()
        print('current_MTM_pose is:', current_MTM_pose)
        diff = target.p - current_MTM_pose.p
        print('difference is', diff)
        distance = np.linalg.norm(np.array([diff.x(), diff.y(), diff.z()])) 
        print('distance is: ', distance)
        '''
        Using for loop
        '''
        for i in range(1, step_num+1):
            print(i)
            next_MTM_position = current_MTM_pose.p + diff*i/step_num
            next_MTM_pose = PyKDL.Frame()
            next_MTM_pose.p = PyKDL.Vector(next_MTM_position[0], next_MTM_position[1], next_MTM_position[2])
            next_MTM_pose.M = target.M
            mtm.move_cp(next_MTM_pose)

        _current_MTM_pose = mtm.setpoint_cp()
        diff = next_MTM_pose.p - _current_MTM_pose.p
        distance = np.linalg.norm(np.array([diff.x(), diff.y(), diff.z()])) 
        return distance