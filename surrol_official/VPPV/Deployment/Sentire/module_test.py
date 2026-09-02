import numpy as np
import cv2
import yaml 
import sys
import os

from csrk.arm_proxy import ArmProxy
from csrk.node import Node
import PyCSR
from player_utils import VideoCapture
from player_utils import my_rectify

intrinsics_matrix=np.array([[792.3552334 , 0.0, 405.00907135], [0.0, 792.3552334 , 288.85937119], [0.0, 0.0, 1.0]])
intrinsics_matrix_right = np.array([[7.92355233e+02 , 0.0, 4.05009071e+02], [0.0, 7.92355233e+02 , 2.88859371e+02], [0.0, 0.0, 1.0]])
fs = cv2.FileStorage("/home/student/csr_test/endoscope_calibration_1205.yaml", cv2.FILE_STORAGE_READ)

# class ModuleTest:
#     def __init__(self, yaml_file, node_file, video_path):
#         self.yaml_file = yaml_file
#         self.node_file = node_file
#         self.video_path = video_path

#         self.intrinsics_matrix = None
#         self.cam_T_basePSM = None
#         self.basePSM_T_cam = None

#     def init_arm(self):
#         node_ = Node(self.node_file)
#         self.cap = cv2.VideoCapture(self.video_path)
#         self.psm = ArmProxy(node_, "psa3")
#         self.ecm = ArmProxy(node_, "psa2")
#         while(not self.psm.is_connected):
#             self.psm.measured_cp()
#         while(not self.ecm.is_connected):
#             self.ecm.measured_cp()

#         self.read_camera_parameters()
#     def read_camera_parameters(self):
#         try:
#             with open(self.yaml_file, 'r') as file:
#                 data = yaml.safe_load(file)
#             self.intrinsics_matrix = np.array(data['intrinsics_matrix'])
#             self.cam_T_basePSM = np.array(data['cam_T_basePSM'])
#             self.basePSM_T_cam = np.array(data['basePSM_T_cam'])
#         except Exception as e:
#             print(f"error in found {e}")
# edit for csr
node_ = Node("/home/student/csr_test/NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where you put the ndds xml file


def convert_pos(pos,matrix):
    '''
    input: ecm pose matrix 4x4
    output rcm pose matrix 4x4
    '''
    return np.matmul(matrix[:3,:3],pos)+matrix[:3,3]

def SetPoints(windowname, img):
    
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('select point: ', points)
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('quit!')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        
        print('retry')
        return SetPoints(windowname, img)

def init_ecm():
    import time
    #edit for csr
    p= ArmProxy(node_, "psa2")
    while(not p.is_connected):
        p.measured_cp()

    p.read_rtrk_arm_state()
    print("connection: ",p.is_connected)
    #edit for csr end
    robot_pose=p.measured_cp()
    time.sleep(3)

    robot_pose=p.measured_cp()
    #print(type(robot_pose))
    #print(type(robot_pose.p))
    #print(robot_pose.M)
    #print(robot_pose.M.GetEulerZYX())
    #print(robot_pose.M.GetRPY())
    robot_pos=robot_pose.p
    transform_2=robot_pose.M
    np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2], robot_pos[0]],
                            [transform_2[1,0], transform_2[1,1], transform_2[1,2], robot_pos[1]],
                            [transform_2[2,0], transform_2[2,1], transform_2[2,2],robot_pos[2]],
                            [0,0,0,1]])

    #robot_rot_euler=player.get_euler_from_matrix(np_m)
    #print('robot_pos: ',robot_pos)
    print(np_m)
    #ecm_pos=player.convert_pos(np.array([robot_pos[0],robot_pos[1],robot_pos[2]]), cam_T_basePSM)
    #print('robot_pos_ecm: ',ecm_pos)
    
    #print('robot_rot_euler: ' ,robot_rot_euler)
    
    exit()
   
def init_psm():
    import time
    #edit for csr
    p= ArmProxy(node_, "psa3")
    while(not p.is_connected):
        p.measured_cp()

    p.read_rtrk_arm_state()
    print("connection: ",p.is_connected)
    #edit for csr end
    robot_pose=p.measured_cp()
    time.sleep(3)

    robot_pose=p.measured_cp()
    #print(type(robot_pose))
    #print(type(robot_pose.p))
    #print(robot_pose.M)
    #print(robot_pose.M.GetEulerZYX())
    #print(robot_pose.M.GetRPY())
    robot_pos=robot_pose.p
    transform_2=robot_pose.M
    np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2]],
                            [transform_2[1,0], transform_2[1,1], transform_2[1,2]],
                            [transform_2[2,0], transform_2[2,1], transform_2[2,2]]])

    robot_rot_euler=player.get_euler_from_matrix(np_m)
    print('robot_pos: ',robot_pos)
    print(transform_2)
    #ecm_pos=player.convert_pos(np.array([robot_pos[0],robot_pos[1],robot_pos[2]]), cam_T_basePSM)
    #print('robot_pos_ecm: ',ecm_pos)
    
    #print('robot_rot_euler: ' ,robot_rot_euler)
    
    exit()
   

    #new_rot_euler=np.array([robot_rot_euler[0] ,robot_rot_euler[1] ,robot_rot_euler[2]])#np.array([player.rcm_init_eul[0],player.rcm_init_eul[1],robot_rot_euler[2]])
    #print('robot_rot_euler: ',new_rot_euler)
    #transform_2=player.get_matrix_from_euler(new_rot_euler)
    
    PSM2_rotate = PyCSR.Rotation(transform_2[0,0], transform_2[0,1], transform_2[0,2],
                            transform_2[1,0], transform_2[1,1], transform_2[1,2],
                            transform_2[2,0], transform_2[2,1], transform_2[2,2])
    PSM2_pose = PyCSR.Vector(robot_pos[0]+0.01, robot_pos[1], robot_pos[2])

    goal_2 = PyCSR.Frame(PSM2_rotate , PSM2_pose)
    print("goal: ",goal_2)
    p.move_cp(goal_2, acc=1, duration=1, jaw=0)
    time.sleep(1)
    
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def test_calibration_realtime():
    
    #edit for csr
    cap_0=VideoCapture("/dev/video0")
    p= ArmProxy(node_, "psa3")
    while(not p.is_connected):
        p.measured_cp()
    #edit for csr end

    for i in range(10):
        frame1=cap_0.read()
    '''
    cam_T_basePSM =np.array([[-0.74260612, -0.66947573, -0.0183957 ,  0.09987853],
       [-0.53325979,  0.60768183, -0.58852085, -0.06567567],
       [ 0.40517916, -0.4272295 , -0.80827273, -0.01241294],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    '''

    # intrinsics_matrix=np.array([[861.63411341 , 0.0, 416.9377861], [0.0, 861.63411341 , 288.38869858], [0.0, 0.0, 1.0]])
    # fs = cv2.FileStorage("/home/student/csr_test/endoscope_calibration_csr_new_machine_1204.yaml", cv2.FILE_STORAGE_READ)
    basePSM_T_cam = np.array([[ 0.84735924,  0.5220266 , -0.09731672, -0.05897265],
        [ 0.5307947 , -0.83799338,  0.12658624, -0.06112438],
        [-0.01546939, -0.15891922, -0.98717039, -0.03197937],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    cam_T_basePSM = np.array([[ 0.84735924,  0.5307947 , -0.01546939,  0.08192082],
        [ 0.5220266 , -0.83799338, -0.15891922, -0.02551867],
        [-0.09731672,  0.12658624, -0.98717039, -0.02957061],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])

    
    while True:
        robot_pose=p.measured_cp()
        frame1=cap_0.read()
        frame2=frame1.copy()

        frame1,_=my_rectify(frame1,frame2,fs)

        #joint=p.measured_jp()
        #print(player.get_bPSM_T_j6(joint).shape)
        
        #homogeneous_point_3d=cam_T_basePSM @ player.get_bPSM_T_j6(joint)
        #homogeneous_point_3d=np.array([homogeneous_point_3d[0,3],homogeneous_point_3d[1,3],homogeneous_point_3d[2,3],1])
        robot_pos=robot_pose.p
        robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        robot_pos= convert_pos(robot_pos,cam_T_basePSM)
 
    

        #homogeneous_point_3d = np.append(robot_pos, 1)  # Convert to homogeneous coordinates
        #print(homogeneous_point_3d.shape)
    
        projected_point = intrinsics_matrix @ robot_pos
        projected_point /= projected_point[2]  # Normalize by the third component
        
        u = int(projected_point[0])
        v = int(projected_point[1])
        print(u," ",v)
        #frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)

        # Read the image
        #image = cv2.imread(image_path)

        
         

            # Draw a circle at the given point (x, y)
        cv2.circle(frame1, (u, v), 10, (0, 0, 255), -1)
           # Display the image
        cv2.imshow("Image", frame1)

            # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit if 'q' is pressed
            break

        # Destroy the window and release resources
    cv2.destroyAllWindows()


        # Load your image
        #image = plt.imread("/home/kj/ar/GauzeRetrievel/cbtest1.png")

        # Create a figure and axis
        #fig, ax = plt.subplots()

        # Display the image
        #ax.imshow(image)

        # Add a point at position (u, v)
        #ax.scatter(u, v, color='red', marker='o')

        # Show the plot
        #plt.show()
    cap_0.release()

def test_calibration_realtime_change_cam():
    
    #edit for csr
    cap_0=VideoCapture("/dev/video0")
    p= ArmProxy(node_, "psa3")
    p_ecm = ArmProxy(node_, "psa2")
    #edit for csr end
    while(not p.is_connected):
        p.measured_cp()

    while(not p_ecm.is_connected):
        p_ecm.measured_cp()

    for i in range(10):
        frame1=cap_0.read()
    # intrinsics_matrix=np.array([[861.63411341 , 0.0, 416.9377861], [0.0, 861.63411341 , 288.38869858], [0.0, 0.0, 1.0]])

    # fs = cv2.FileStorage("/home/student/csr_test/endoscope_calibration_csr_new_machine_1204.yaml", cv2.FILE_STORAGE_READ)
    # PSM1
    # update here for the initial setup
    oldcam_T_basePSM = np.array([[ 0.84735924,  0.5307947 , -0.01546939,  0.08192082],
                                [ 0.5220266 , -0.83799338, -0.15891922, -0.02551867],
                                [-0.09731672,  0.12658624, -0.98717039, -0.02957061],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])

    while True:

        ecm_pose = p_ecm.measured_cp()
        ecm_pos = ecm_pose.p
        ecm_rot = ecm_pose.M

        base_T_newtip = np.array([[ecm_rot[0,0], ecm_rot[0,1], ecm_rot[0,2], ecm_pos[0]],
                                [ecm_rot[1,0], ecm_rot[1,1], ecm_rot[1,2], ecm_pos[1]],
                                [ecm_rot[2,0], ecm_rot[2,1], ecm_rot[2,2], ecm_pos[2]],
                                [0,0,0,1]])


        oldcam_T_newcam = change_camera(base_T_newtip)

        cam_T_basePSM = np.linalg.inv(oldcam_T_newcam) @ oldcam_T_basePSM 
        basePSM_T_cam= np.linalg.inv(cam_T_basePSM)
        print("cam_T_basePSM: ", cam_T_basePSM)
        print("basePSM_T_cam: ", basePSM_T_cam)

        np.save("/home/student/csr_test/animal_exp/exp_calibration/cam_T_basePSM.npy",cam_T_basePSM)

        np.save("/home/student/csr_test/animal_exp/exp_calibration/basePSM_T_cam.npy",basePSM_T_cam)

        robot_pose=p.measured_cp()
        frame1=cap_0.read()
        frame2=frame1.copy()

        
        frame1,_=my_rectify(frame1,frame2,fs)
        

        #joint=p.measured_jp()
        #print(player.get_bPSM_T_j6(joint).shape)
        
        #homogeneous_point_3d=cam_T_basePSM @ player.get_bPSM_T_j6(joint)
        #homogeneous_point_3d=np.array([homogeneous_point_3d[0,3],homogeneous_point_3d[1,3],homogeneous_point_3d[2,3],1])
        robot_pos=robot_pose.p
        robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        robot_pos=convert_pos(robot_pos,cam_T_basePSM)
 
    

        #homogeneous_point_3d = np.append(robot_pos, 1)  # Convert to homogeneous coordinates
        #print(homogeneous_point_3d.shape)
    
        projected_point = intrinsics_matrix @ robot_pos
        projected_point /= projected_point[2]  # Normalize by the third component
        
        u = int(projected_point[0])
        v = int(projected_point[1])
        print(u," ",v)
        #frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)

        # Read the image
        #image = cv2.imread(image_path)

        
         

            # Draw a circle at the given point (x, y)
        cv2.circle(frame1, (u, v), 10, (0, 0, 255), -1)
           # Display the image
        cv2.imshow("Image", frame1)

            # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit if 'q' is pressed
            break

        # Destroy the window and release resources
    cv2.destroyAllWindows()


        # Load your image
        #image = plt.imread("/home/kj/ar/GauzeRetrievel/cbtest1.png")

        # Create a figure and axis
        #fig, ax = plt.subplots()

        # Display the image
        #ax.imshow(image)

        # Add a point at position (u, v)
        #ax.scatter(u, v, color='red', marker='o')

        # Show the plot
        #plt.show()
    cap_0.release()

def test_calibration_realtime_readapi():
    
    #edit for csr
    cap_0=VideoCapture("/dev/video0")
    cap_1=VideoCapture("/dev/video2")
    p= ArmProxy(node_, "psa3")
    p_ecm = ArmProxy(node_, "psa2")
    #edit for csr end
    while(not p.is_connected):
        p.measured_cp()

    while(not p_ecm.is_connected):
        p_ecm.measured_cp()

    for i in range(10):
        frame1=cap_0.read()

    cam_T_ECMtip = np.array([[ 0.99842177, -0.04879342, -0.02780582, -0.00218275],
        [ 0.05005654,  0.99765387,  0.04670221,  0.00336121],
        [ 0.02546182, -0.04802036,  0.99852178,  0.01329402],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    ECMtip_T_cam = np.array([[ 0.99842177,  0.05005654,  0.02546182,  0.00167257],
        [-0.04879342,  0.99765387, -0.04802036, -0.00282144],
        [-0.02780582,  0.04670221,  0.99852178, -0.01349204],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    offset = [0.0, 0.0, 0.0]
    offset = [-0.0014999999999999998, -0.007499999999999999, 0.0050999999999999995]
    scale = 0.0003
    while True:

        frame1=cap_0.read()
        frame2=cap_1.read()

        frame1,frame2=my_rectify(frame1,frame2,fs)
        
        p.read_pose_twist_jaw()
        robot_pos = p.pose.p
        key = cv2.waitKey(1)
        if key == ord('w'):
            offset[1] -= scale
        elif key == ord('s'):
            offset[1] += scale
        elif key == ord('a'):
            offset[0] -= scale
        elif key == ord('d'):
            offset[0] += scale
        elif key == ord('t'):
            offset[2] += scale
        elif key == ord('g'):
            offset[2] -= scale
        robot_pos=np.array([robot_pos[0]+offset[0],robot_pos[1]+offset[1],robot_pos[2]+offset[2]])
        # robot_pos=convert_pos(robot_pos,ECMtip_T_cam)
        # robot_pos=convert_pos(robot_pos,cam_T_ECMtip)
    
        projected_point = intrinsics_matrix @ robot_pos
        projected_point /= projected_point[2]  
        
        u = int(projected_point[0])
        v = int(projected_point[1])
        print(u," ",v)

        t_matrix = np.array([ [0.999971, -0.002225, -0.007252, -4.609986e-03], 
                            [0.002233, 0.999997, 0.001088, 0.084091e-03], 
                            [0.007250, -0.001104, 0.999973, -0.099408e-03],
                            [0.000000, 0.000000, 0.000000, 1.000000]])
        
        t_matrix_inv = np.linalg.inv(t_matrix)
        robot_pos_right = convert_pos(robot_pos, t_matrix)
        projected_point_right = intrinsics_matrix_right @ robot_pos_right
        projected_point_right /= projected_point_right[2]  
        
        u_r = int(projected_point_right[0])
        v_r = int(projected_point_right[1])
        print(u_r," ",v_r)

        cv2.namedWindow("left", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("left", 800, 600)
        cv2.circle(frame1, (u, v), 10, (0, 0, 255), -1)
        cv2.imshow("left", frame1)

        cv2.namedWindow("right", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("right", 800, 600)
        cv2.circle(frame2, (u_r, v_r), 10, (0, 0, 255), -1)
        cv2.imshow("right", frame2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(offset)
            break

    cv2.destroyAllWindows()
    cap_0.release()

def change_camera(base_T_newtip):
    print(base_T_newtip)
    # exit()
    cam_T_ECMtip = np.array([[ 9.97662041e-01, -3.22074468e-02, -6.02754785e-02,
        -1.47755181e-03],
       [ 2.88424443e-02,  9.98020363e-01, -5.58879987e-02,
         4.49334853e-05],
       [ 6.19561647e-02,  5.40188427e-02,  9.96615973e-01,
        -3.16692718e-03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
    ECMtip_T_cam = np.array([[ 9.97662041e-01,  2.88424443e-02,  6.19561647e-02,
            1.66901203e-03],
        [-3.22074468e-02,  9.98020363e-01,  5.40188427e-02,
            7.86410367e-05],
        [-6.02754785e-02, -5.58879987e-02,  9.96615973e-01,
            3.06966132e-03],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]])
    
    # update here for initial setup

    base_T_oldtip=np.array([[    0.99962,    0.018577,   -0.020228,  -0.0021777],
                            [    0.02083,    -0.99285,     0.11756,    0.059764],
                            [    -0.0179,    -0.11794,    -0.99286,   -0.080003],
                            [          0,           0,           0,           1]])
    oldtip_T_newtip = np.linalg.inv(base_T_oldtip) @ base_T_newtip

    oldcam_T_newcam = cam_T_ECMtip @ oldtip_T_newtip @ ECMtip_T_cam

    return oldcam_T_newcam



if __name__=="__main__":

    # yaml_file = "/home/student/csr_test/csr_calib_hand_eye/scripts_test/camera_parameters.yaml"
    # intrinsics_matrix, cam_T_basePSM, basePSM_T_cam = read_camera_parameters(yaml_file)
    # fs = cv2.FileStorage("/home/student/csr_test/endoscope_calibration.yaml", cv2.FILE_STORAGE_READ)


    #init_ecm()

    # for obtain the psa pose
    #init_psm()

    # for check calibration
    # test_calibration_realtime()
    
    # for to get new calibration
    # test_calibration_realtime_change_cam()

    test_calibration_realtime_readapi()
    



    