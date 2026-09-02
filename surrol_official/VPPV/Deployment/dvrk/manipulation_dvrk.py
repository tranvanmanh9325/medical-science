import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import time
import PyKDL
import dvrk
import crtk

from player_config import player_opts

intrinsics_matrix=player_opts.intrinsics_matrix
Q = player_opts.Q
kinematics_offset = player_opts.kinematics_offset


# Rotation matrix that aligns two vectors
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# Distance of line and point
def dist_line_from_point(x0, y0, xp1, yp1, xp2, yp2):
    return (abs(((xp2-xp1) * (yp1-y0)) - ((xp1-x0) * (yp2-yp1)))
            / math.sqrt(((xp2-xp1)**2) + ((yp2-yp1)**2)))

# Angle of two lines
def angle_between_lines(xp1, yp1, xp2, yp2, xp3, yp3, xp4, yp4):
    if xp2 != xp1:
        m1 = float((yp2-yp1)) / float((xp2-xp1))
    else:
        m1 = (yp2-yp1) * 2.0

    if xp4 != xp3:
        m2 = float((yp4-yp3)) / float((xp4-xp3))
    else:
        m2 = (yp4-yp3) * 2.0
    try:
        ret = math.degrees(math.atan((m1-m2) / (1 + (m1 * m2))))
    except ZeroDivisionError:
        ret = math.degrees(math.atan((m1-m2) / (0.0000001)))
    return ret

# Claculate line intersection
def intersection_of_lines(xp1, yp1, xp2, yp2, xp3, yp3, xp4, yp4):
    t = (((xp1-xp3)*(yp3-yp4)) - ((yp1-yp3)*(xp3-xp4))) / (((xp1-xp2)*(yp3-yp4)) - ((yp1-yp2)*(xp3-xp4)))
    Px = xp1 + (t*(xp2-xp1))
    Py = yp1 + (t*(yp2-yp1))
    return Px, Py

# Rotation matrix that aligns two vectors
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# Point distance from plane
def point_distance_from_plane(x1, y1, z1, a, b, c, d):
    """ https://mathinsight.org/distance_point_plane
    """
    return abs((a*x1) + (b*y1) + (c*z1) + d) / math.sqrt((a*a) + (b*b) + (c*c))

# Correct depth based on plane detection
def force_depth(point, plane, block_offset):
    [a,b,c,d] = plane
    dist = point_distance_from_plane(point[0], point[1], point[2],
                                     a, b, c, d)
    #print(dist)
    norm_ori = np.array([a,b,c])
    norm_unit = norm_ori / np.linalg.norm(norm_ori)
    corr_dist = block_offset - dist
    corr_point = point - (corr_dist * norm_unit)
    return corr_point

# Get orientation from detected plane
def calcu_orientation(plane_model):
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    plane_normal = np.array([-a, -b, -c])
    start_ori_vec = np.array([0.0, 0.0, -1.0])
    grasp_ori = R.from_matrix(rotation_matrix_from_vectors(start_ori_vec, plane_normal))
    # print('grasp_ori',grasp_ori.as_matrix())
    return grasp_ori.as_matrix()

# Apply object rotation to z axis
def apply_z_rotation(R, theta):

    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    R_prime = np.dot(R, R_z)
    
    return R_prime

# Get orientation from gradient of segment image
def calculate_average_gradient(image, point, window_size = 20):

    x, y = point
    half_size = window_size // 2
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gx = grad_x[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
    gy = grad_y[y-half_size:y+half_size+1, x-half_size:x+half_size+1]

    avg_gx = np.mean(gx)
    avg_gy = np.mean(gy)

    gradient_angle = np.arctan2(avg_gy, avg_gx)  
    gradient_angle = gradient_angle 
    # gradient_angle = np.pi / 4
    print('gradient_angle111111111111111',gradient_angle * 180 / np.pi)
    R_z = np.array([
        [np.cos(gradient_angle), -np.sin(gradient_angle), 0],
        [np.sin(gradient_angle),  np.cos(gradient_angle), 0],
        [0, 0, 1]
    ])
    print(R_z)
    return R_z

def convert_point_to_camera_axis(x, y, depth, intrinsics_matrix):
    ''' 
    # Example usage
    x = 100
    y = 200
    depth = 5.0
    intrinsics_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    xc, yc, zc = convert_point_to_camera_axis(x, y, depth, intrinsics_matrix)
    print(f"Camera axis coordinates: xc={xc}, yc={yc}, zc={zc}")
    '''
    # Extract camera intrinsics matrix components
    fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

    # Normalize pixel coordinates
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    # Convert to camera axis coordinates
    xc = xn * depth
    yc = yn * depth
    zc = depth

    return np.array([xc, yc, zc])

def calculate_goal(rotation_pose, psm_goal, R_z = np.array([[1.0, 0.0, 0.0], 
                                                        [0.0, 1.0, 0.0], 
                                                        [0.0, 0.0, 1.0]])):
    rotation_pose = np.dot(rotation_pose, R_z)
    PSM_rotate=PyKDL.Rotation(rotation_pose[0][0],rotation_pose[0][1],rotation_pose[0][2],
                                rotation_pose[1][0],rotation_pose[1][1],rotation_pose[1][2],
                                rotation_pose[2][0],rotation_pose[2][1],rotation_pose[2][2])
    PSM_pose = PyKDL.Vector(psm_goal[0], psm_goal[1], psm_goal[2])

    move_goal = PyKDL.Frame(PSM_rotate, PSM_pose)

    return move_goal

class Manipulator:
    def __init__(self) -> None:
        self.ral = crtk.ral('dvrk_python_node')
        self.p = dvrk.psm(self.ral, 'PSM1')
        self.psm2 = dvrk.psm(self.ral, 'PSM2')
        self.img_size=(800,600)
        self.approach_offset = 0.005
        
        self.intrinsics_matrix=player_opts.mani_intrinsics_matrix
        
        self.Q = player_opts.Q

        # self.basePSM_T_cam = player_opts.basePSM_T_cam
        # self.cam_T_basePSM = player_opts.cam_T_basePSM

    def step_manipulate_grasp(self, rotation_pose, psm_pos, step_num = 10):
        cur_pose = self.p.measured_cp()
        cur_pos = cur_pose.p

        dx = psm_pos[0] - cur_pos[0]
        dy = psm_pos[1] - cur_pos[1]
        dz = psm_pos[2] - cur_pos[2]

        self.p.move_cp(cur_pose).wait(1)
        time.sleep(1)
        print("Manipulation open jaw")
        for i in range(step_num):
            cur_pos[0] = cur_pos[0] + dx / step_num
            cur_pos[1] = cur_pos[1] + dy / step_num
            cur_pos[2] = cur_pos[2] + dz / step_num
            # print('i', i)
            # print('target goal :', cur_pos)
            move_goal = calculate_goal(rotation_pose, cur_pos)
            self.p.move_cp(move_goal).wait(1)
            time.sleep(1)
        self.p.jaw.move_jp(np.array([-0.5])).wait()
        time.sleep(1)
        print("Manipulation close jaw")

    def step_manipulate_lift(self, rotation_pose, psm_pos, step_num = 10):
        cur_pose = self.p.measured_cp()
        cur_pos = cur_pose.p

        dx = psm_pos[0] - cur_pos[0]
        dy = psm_pos[1] - cur_pos[1]
        dz = psm_pos[2] - cur_pos[2]

        for i in range(step_num):
            cur_pos[0] = cur_pos[0] + dx / step_num
            cur_pos[1] = cur_pos[1] + dy / step_num
            cur_pos[2] = cur_pos[2] + dz / step_num
            # print('i', i)
            # print('target goal :', cur_pos)
            move_goal = calculate_goal(rotation_pose, cur_pos)
            self.p.move_cp(move_goal).wait(1)
            time.sleep(1)

    def step_manipulate_place_lift(self, rotation_pose, psm_pos, step_num = 10):
        cur_pose = self.p.measured_cp()
        cur_pos = cur_pose.p

        dx = psm_pos[0] - cur_pos[0]
        dy = psm_pos[1] - cur_pos[1]
        dz = psm_pos[2] - cur_pos[2]

        for i in range(step_num):
            cur_pos[0] = cur_pos[0] + dx / step_num
            cur_pos[1] = cur_pos[1] + dy / step_num
            cur_pos[2] = cur_pos[2] + dz / step_num
            # print('i', i)
            # print('target goal :', cur_pos)
            move_goal = calculate_goal(rotation_pose, cur_pos)
            self.p.move_cp(move_goal).wait(1)
            time.sleep(1)
    def gauze_pick(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam):
        rotation_pose=player_opts.mani_rotation
        
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+ player_opts.goal_offset_z["base"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)

        self.step_manipulate_grasp(rotation_pose, psm_goal, step_num)
        print("Manipulation move to goal")

        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)

        return True
    def grasp_tissue(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam):
        rotation_pose=player_opts.mani_rotation
        
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+ player_opts.goal_offset_z["base"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)
        # R_z = calculate_average_gradient(seg_img, object_point, 25)
        R_z = np.array([[1.0,   0.0,    0.0],
                        [0.0,    1.0,   0.0],
                        [0.0,   0.0,   1.0]])
        # rotation_pose = np.dot(rotation_pose, R_z)

        self.step_manipulate_grasp(rotation_pose, psm_goal, step_num)
        print("Manipulation move to goal")

        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)

        return True
    
    def needle_pick(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam, rotation):
        rotation_pose=rotation

        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+ player_opts.goal_offset_z["base"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)
        R_z = calculate_average_gradient(seg_img, object_point, 25)
        # rotation_pose = np.dot(rotation_pose, R_z)

        self.step_manipulate_grasp(rotation_pose, psm_goal, step_num)
        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)

        return True
    
    def vessel_clip(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam):
        self.p = self.psm2
        rotation_pose=player_opts.mani_rotation
        
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0]+player_opts.goal_offset_x["vessel"], psm_goal[1]+player_opts.goal_offset_y["vessel"], psm_goal[2]+player_opts.goal_offset_z["vessel"]]

        seg_img = np.array(seg_img==True).astype(np.uint8)

        gradient_angle =  np.pi / 12
        R_z = np.array([[np.cos(gradient_angle), -np.sin(gradient_angle), 0],
                        [np.sin(gradient_angle),  np.cos(gradient_angle), 0],
                        [0, 0, 1]])
        
        # R_z = calculate_average_gradient(seg_img, object_point, 25)
        rotation_pose = np.dot(rotation_pose, R_z)
        self.step_manipulate_grasp(rotation_pose, psm_goal, step_num)

        return True

    def retract_soft_tissue(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam):
        rotation_pose=player_opts.mani_rotation
        
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+ player_opts.goal_offset_z["base"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)
        # R_z = calculate_average_gradient(seg_img, object_point, 25)
        # R_z = np.array([[1.0,   0.0,    0.0],
        #                 [0.0,    1.0,   0.0],
        #                 [0.0,   0.0,   1.0]])
        # gradient_angle =  np.pi / 6
        gradient_angle = 0.0
        R_z = np.array([
        [np.cos(gradient_angle), -np.sin(gradient_angle), 0],
        [np.sin(gradient_angle),  np.cos(gradient_angle), 0],
        [0, 0, 1]
                ])
        rotation_pose = np.dot(rotation_pose, R_z)

        self.step_manipulate_grasp(rotation_pose, psm_goal, step_num)
        print("Manipulation move to goal")

        # self.step_manipulate_lift(rotation_pose, psm_goal_lift, 5)

        return True
    
    def gauze_place(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam):
        rotation_pose=player_opts.mani_rotation
        
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]
        
        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+ player_opts.goal_offset_z["base"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)
        # R_z = calculate_average_gradient(seg_img, object_point, 25)
        R_z = np.array([[1.0,   0.0,    0.0],
                        [0.0,    1.0,   0.0],
                        [0.0,   0.0,   1.0]])
        rotation_pose = np.dot(rotation_pose, R_z)

        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)
        self.step_manipulate_lift(rotation_pose, psm_goal, step_num)
        self.step_manipulate_place_lift(rotation_pose, psm_goal_lift, step_num)

        return True

    def peg_pick(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam, rotation):
        rotation_pose=rotation

        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        # cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        psm_goal = [psm_goal[0] + player_opts.goal_offset_x["peg"], psm_goal[1] + player_opts.goal_offset_y["peg"], psm_goal[2]+ player_opts.goal_offset_z["peg"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)
        R_z = calculate_average_gradient(seg_img, object_point, 25)
        # R_z = np.array([[1.0,   0.0,    0.0],
        #                 [0.0,    1.0,   0.0],
        #                 [0.0,   0.0,   1.0]])
        rotation_pose = np.dot(rotation_pose, R_z)
        self.peg_rotation = rotation_pose
        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)
        self.p.jaw.move_jp(np.array([0.8])).wait()
        self.step_manipulate_grasp(rotation_pose, psm_goal, step_num)
        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)

        return True

    def peg_place(self, object_point, depth_img, seg_img, step_num, tool_offset, basePSM_T_cam, peg_rotation):
        rotation_pose=peg_rotation
        
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        cam_goal = [cam_goal[0] - kinematics_offset[0], cam_goal[1] - kinematics_offset[1], cam_goal[2] - kinematics_offset[2]]

        psm_goal = np.matmul(basePSM_T_cam[:3,:3],cam_goal)+basePSM_T_cam[:3,3]
        # psm_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+ player_opts.goal_offset_z["base"]]
        psm_goal = [psm_goal[0]+ player_opts.goal_offset_x["peg_place"], psm_goal[1]+ player_opts.goal_offset_y["peg_place"], psm_goal[2]+ player_opts.goal_offset_z["peg_place"]]
        psm_goal_lift = [psm_goal[0], psm_goal[1], psm_goal[2] + player_opts.lift_height]

        seg_img = np.array(seg_img==True).astype(np.uint8)


        self.step_manipulate_lift(rotation_pose, psm_goal_lift, step_num)
        self.step_manipulate_lift(rotation_pose, psm_goal, step_num)
        self.p.jaw.move_jp(np.array([0.8])).wait()
        self.step_manipulate_place_lift(rotation_pose, psm_goal_lift, step_num)

        return True

if __name__ == "__main__":
    pass
    # p= ArmProxy(node_, player_opts.psa_num)
    # while(not p.is_connected):
    #     p.measured_cp()
    # pose = p.measured_cp()
    # rotation_pose = pose.M
    # Rotation = (rotation_pose[0,0],rotation_pose[0,1],rotation_pose[0,2],
    #             rotation_pose[1,0],rotation_pose[1,1],rotation_pose[1,2],
    #             rotation_pose[2,0],rotation_pose[2,1],rotation_pose[2,2])
    # rot = np.array([[rotation_pose[0,0],rotation_pose[0,1],rotation_pose[0,2]],
    #                 [rotation_pose[1,0],rotation_pose[1,1],rotation_pose[1,2]],
    #                 [rotation_pose[2,0],rotation_pose[2,1],rotation_pose[2,2]]])
    # print("player_opts.rotation = PyKDL.Rotation{}".format(repr(Rotation)))
    # print("player_opts.mani_rotation = np.{}".format(repr(rot)))