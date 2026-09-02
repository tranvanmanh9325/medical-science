import dvrk
import crtk
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import PyKDL
from skimage.transform import  AffineTransform
import time
### camera para

intrinsics_matrix=np.array([[693.12012738, 0.0, 355.44816971], 
                            [0.000000, 693.12012738, 327.7015152], 
                            [0.000000, 0.000000, 1.000000]])

Q = np.array([[1, 0, 0, -355.45],
              [0, 1, 0, -327.7],
              [0, 0, 0, 693.12],
              [0, 0, 0.23731, -0]])


## Helper functions
# get 3d image from disparity 
def reproject_to_3d(disparity_image, Q):
    points_3d = cv2.reprojectImageTo3D(disparity_image, Q)
    return points_3d

def convert_to_open3d_point_cloud(points_3d):
    mask = points_3d[:, :, 2] > 0 
    points = points_3d[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

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

    gradient_angle = np.arctan2(avg_gy, avg_gx)  + np.pi / 2
    # gradient_angle = np.arctan2(avg_gy, avg_gx)
    print('gradient_angle',gradient_angle)
    R_z = np.array([
        [np.cos(gradient_angle), -np.sin(gradient_angle), 0],
        [np.sin(gradient_angle),  np.cos(gradient_angle), 0],
        [0, 0, 1]
    ])
    
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
        self.img_size=(800,600)
        self.approach_offset = 0.005

        self.intrinsics_matrix=np.array([[693.12012738 / 2.5, 0.0, 355.44816971 / 2.5], 
                                    [0.000000, 693.12012738 / 2.5, 327.7015152 / 2.5], 
                                    [0.000000, 0.000000, 1.000000]])

        self.Q = np.array([[1, 0, 0, -355.45],
                    [0, 1, 0, -327.7],
                    [0, 0, 0, 693.12],
                    [0, 0, 0.23731, -0]])
        
        self.basePSM_T_cam = np.array([[-0.88328225, -0.4560953 ,  0.10857969,  0.05357333],
                                    [-0.46620461,  0.87896398, -0.10037717,  0.04117293],
                                    [-0.04965608, -0.13928173, -0.98900701, -0.03160624],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
        self.cam_T_basePSM = np.array([[-0.88328225, -0.46620461, -0.04965608,  0.06494594],
                                    [-0.4560953 ,  0.87896398, -0.13928173, -0.01615715],
                                    [ 0.10857969, -0.10037717, -0.98900701, -0.03294295],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    def step_manipulate(self, rotation_pose, psm_pos, step_num = 10):
        cur_pose = self.p.measured_cp()
        cur_pos = cur_pose.p

        dx = psm_pos[0] - cur_pos[0]
        dy = psm_pos[1] - cur_pos[1]
        dz = psm_pos[2] - cur_pos[2]

        for i in range(step_num):
            cur_pos[0] = cur_pos[0] + dx / step_num
            cur_pos[1] = cur_pos[1] + dy / step_num
            cur_pos[2] = cur_pos[2] + dz / step_num
            move_goal = calculate_goal(rotation_pose, cur_pos)
            self.p.move_cp(move_goal).wait()
            # time.sleep(0.1)


    def grasp_peg(self, object_point, depth_img, seg_img, step_num = 1, offset = [0.0, 0.0, 0.0],
                  rotation_pose = np.array([[0.645682,    -0.759185,   0.0820535],
                                            [-0.763369,   -0.639067,   0.0941291],
                                            [-0.0190237,   -0.123415,   -0.992173]]) ):
        goal_x, goal_y=object_point[0], object_point[1]
        goal_depth=depth_img[goal_y][goal_x]
        print('Manipulation grasp_peg goal_depth',goal_depth)

        cam_goal =convert_point_to_camera_axis(goal_x, goal_y, goal_depth, self.intrinsics_matrix)
        psm_goal = np.matmul(self.basePSM_T_cam[:3,:3],cam_goal)+self.basePSM_T_cam[:3,3]
        print('Manipulation grasp_peg psm_goal',psm_goal)

        psm_goal = [psm_goal[0] + offset[0], psm_goal[1] + offset[1], psm_goal[2] + offset[2]]
        psm_approach_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+0.015]
        psm_lift_goal = [psm_goal[0], psm_goal[1], psm_goal[2]+0.015]

        seg_img = np.array(seg_img==True).astype(np.uint8)
        # angle = np.pi / 2
        # R_z = np.array([
        # [np.cos(angle), -np.sin(angle), 0],
        # [np.sin(angle),  np.cos(angle), 0],
        # [0, 0, 1]])
        # # R_z = calculate_average_gradient(seg_img, object_point, 50)
        # rotation_pose = np.dot(R_z, rotation_pose)

        self.step_manipulate(rotation_pose, psm_approach_goal, step_num)
        print("Manipulation grasp_peg approach peg")

        self.p.jaw.move_jp(np.array([0.85])).wait()
        print("Manipulation grasp_peg open jaw")

        self.step_manipulate(rotation_pose, psm_goal, 10)

        self.p.jaw.move_jp(np.array([-0.5])).wait()
        print("Manipulation grasp_peg grasp peg")

        self.step_manipulate(rotation_pose, psm_lift_goal, 10)
        print("Manipulation grasp_peg lift peg")

        return rotation_pose
    