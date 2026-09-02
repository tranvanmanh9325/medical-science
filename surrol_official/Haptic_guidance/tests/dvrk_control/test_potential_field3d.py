import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dvrk import mtm

def forward_point_on_trajectory(p, trajectory, forward_steps=5):
    """
    Find a forward point on the trajectory from the nearest point to p.    
    :param p: numpy array, the position vector of the current point (x, y)
    :param trajectory: numpy array of shape (n, 2), the trajectory points (x, y)
    :param forward_steps: int, the number of steps to move forward along the trajectory
    :return: numpy array, the position vector of the forward point (x, y)
    """
    # Find the nearest point index on the trajectory
    distances = np.linalg.norm(trajectory[..., 0:3] - p[0:3], axis=1)
    nearest_index = np.argmin(distances)    # Calculate the forward index, ensuring it doesn't exceed the trajectory length
    forward_index = min(nearest_index + forward_steps, len(trajectory) - 1)   
    # For rotation alignment

    return np.min(distances), (trajectory[forward_index], forward_index)

def trajectory_forward_field_3d(trajectory, position, k_att, forward_steps=5):
    """
    Plot the 3D potential field with arrows pointing towards a forward point on the trajectory.    
    :param trajectory: numpy array of shape (n, 7), the trajectory points (x, y, z, roll, pich yaw)
    :param k_att: float, scaling factor for the attractive potential
    :param grid_size: int, the size of the grid for visualization
    :param forward_steps: int, the number of steps to move forward along the trajectory for potential direction

    Return:
        Force(in MTM space, which is different than in PSM): (force_x, force_y, force_z, angular_force along x, along y, along z)
    """
    p = np.array(position)
    distances, (forward_point, forward_index) = forward_point_on_trajectory(p, trajectory, forward_steps)# Gradient of the potential field directed towards the forward point
    # for position force control
    if distances > 0.05:
        grad_x = -k_att * (p[0] - forward_point[0])
        grad_y = -k_att * (p[1] - forward_point[1]) 
        grad_z = -k_att * (p[2] - forward_point[2])   
        
    else:
        grad_y, grad_x, grad_z = 0,0,0
    

    angular_x = 0
    angular_y = 0
    angular_z = 0
    distance_to_final = np.linalg.norm(trajectory[-1, 0:3] - p[0:3])
    if forward_index == len(trajectory) and distance_to_final < 0.05:
        grad_y, grad_x, grad_z = 0,0,0
    # if len(trajectory)-1-forward_index < 3 and distances < 0.3:
    #     angle_diff = np.linalg.norm(p[3:6]- trajectory[-1][3:6])
    #     print('angle_diff', angle_diff)
    #     if angle_diff > 0.5:
    #         angular_x = min(0.05 * (forward_point[3] - p[3]), 0.01)
    #         angular_y = min(0.05 * (forward_point[4] - p[4]), 0.01)
    #         angular_z = min(0.05 * (forward_point[5] - p[5]), 0.01)
   
    force = np.array([-grad_y, grad_x, grad_z, angular_x, angular_y, angular_z])
    return (forward_point, distances), force


