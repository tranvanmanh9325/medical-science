import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import dvrk
import crtk
import PyKDL
import os
from Player import VisPlayer
from player_utils import my_rectify


def test_depth(player):
    frame1_path='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/test_image_pre/image_left4.jpg'
    frame2_path='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/test_image_pre/image_right4.jpg'
    left=np.array(Image.open(frame1_path))
    #left=left[100:-100,100:-100,:]
    print(left.shape)
    right=np.array(Image.open(frame2_path))
    #right=right[100:-100,100:-100,:]
    '''
    plt.figure(1)
    plt.imshow(left)
    plt.figure(2)
    plt.imshow(right)
    plt.show()
    '''

    player._load_sttr_model()
    depth, left_rectified=player._get_depth(left, right, filepath='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/endoscope_calibration.yml')
    
    #plt.figure(1)
    #plt.imshow(left_rectified)
    #plt.figure(2)
    #plt.imshow(cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX))
    #plt.show()
    plt.imsave( './depth_3_pre.png',depth)
    plt.imsave("./left3_pre.png", left_rectified)
    #print(min(depth))

def test_fast_sam(player):

    player._load_fastsam()
    image_path='/home/kj/ar/GauzeRetrievel/test_record/frame1_1.png'
    #input = Image.open(image_path)
    #input = input.convert("RGB")
    input=cv2.imread(image_path)
    input=cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    object_point=SetPoints("Goal Selection", input)

    seg=player._seg_with_fastsam(input, object_point[0])
    #print(seg[0].shape)
    #seg=seg[0]
    seg=np.where(seg==False,np.zeros_like(seg), np.ones_like(seg))
    print(seg.shape)
    #print(seg)
    plt.imsave('/home/kj/ar/GauzeRetrievel/test_record/fastsam/seg.png',seg)
    #cv2.imwrite('/home/kj/ar/GauzeRetrievel/test_record/fastsam/seg.png',seg*255)



def test_depth_opencv():
    frame1_path='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/test_image/image_left3.jpg'
    frame2_path='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/test_image/image_right3.jpg'

    imgL = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
    stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

def test_depth_new(player):
    frame1_path='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/test_image/image_left2.jpg'
    frame2_path='/home/student/code/arlin_dvrk/vis_demo/peg_transfer/test_image/image_right2.jpg'
    #left=np.array(Image.open(frame1_path))
    #left=left[100:-100,100:-100,:]
    #print(left.shape)
    #right=np.array(Image.open(frame2_path))
    
    limg=Image.open(frame1_path).convert('RGB')
    rimg =Image.open(frame2_path).convert('RGB')
    pred_np=player._get_depth(limg,rimg)
    pred_np_save = np.round(pred_np * 256).astype(np.uint16)        
    filename='./pred.png'
    cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    plt.imshow(pred_np)#0,1)
    plt.show()


def test_depth_igev(player):
    player._load_depth_model()
    frame1_path='/home/kj/ar/GauzeRetrievel/depth_test3l.png'
    frame2_path='/home/kj/ar/GauzeRetrievel/depth_test3r.png'
    #left=np.array(Image.open(frame1_path))
    #left=left[100:-100,100:-100,:]
    #print(left.shape)
    #right=np.array(Image.open(frame2_path))
    #image_dir='/home/kj/ar/peg_transfer/test_record/depth_realtime_debug/images'
    #out_dir='/home/kj/ar/GauzeRetrievel/test_record/depth_realtime_debug'
    #filelist=os.listdir(os.path.join(image_dir,"left"))
    #for i in range(len(filelist)):
        #frame1_path=os.path.join(image_dir)
        #frame2_path=os.path.join(image_dir)
        #limg=Image.open(frame1_path) #.convert('RGB')
        #rimg =Image.open(frame2_path) #.convert('RGB')
    limg=np.array(Image.open(frame1_path)).astype(np.uint8)
    rimg=np.array(Image.open(frame2_path)).astype(np.uint8)
    #limg=cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
    #rimg=cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
    
    limg=np.array(limg[:,:,:3])
    rimg=np.array(rimg[:,:,:3])

    depth_map=player._get_depth(limg,rimg) # /player.scaling

    #depth_map = player.convert_disparity_to_depth(disp.squeeze(), player.calibration_data['baseline'], player.calibration_data['focal_length_left'])
    #depth=cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    #print(depth[116][95])
    #print(depth[108][77])
    #print(depth[166][83])
    #plt.imshow(pred_np)#0,1)
    #plt.show()
    print(depth_map)
    print(depth_map.mean())
    print(depth_map.std())
    print(depth_map[450][294])
    plt.imsave('test_record/pred_depth_test4.png',depth_map)
    #plt.imsave( os.path.join( out_dir,"depth.jpg"),depth)
    #np.save('./igev_depth_np.npy',pred_np)
    #print(pred_np.shape)
        
def test_depth_igev_realtime(player):
    player._load_depth_model()
    cap_0=cv2.VideoCapture("/dev/video0")
    cap_1=cv2.VideoCapture("/dev/video2")
    counter=0
    while True:
        ret1, frame1 = cap_0.read()
        #cv2.imshow("Input1", frame1)
        cv2.imwrite('test_record/depth_realtime_debug/images/image_left' + str(counter) + '.jpg', frame1)

        ret2, frame2 = cap_1.read()
        cv2.imwrite('test_record/depth_realtime_debug/images/image_right' + str(counter) + '.jpg', frame1)

        #left=np.array(Image.open(frame1_path))
        #left=left[100:-100,100:-100,:]
        #print(left.shape)
        #right=np.array(Image.open(frame2_path))
        
        limg=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rimg=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        limg=np.array(limg)
        rimg=np.array(rimg)
        pred_np=player._get_depth(limg,rimg)
        #plt.imshow(pred_np)#0,1)
        #plt.show()
        plt.imsave( 'test_record/depth_realtime_debug/depth_realtime/depth' + str(counter) + '.jpg',Image.fromarray(pred_np))
        counter+=1

    #np.save('./igev_depth_np.npy',pred_np)
    #print(pred_np.shape)
def test_dam(player):
    player._load_dam()
    imagefile='/home/kj/ar/peg_transfer/test_record/frame1_1.png'
    image=cv2.imread(imagefile)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    depth=player._get_depth_with_dam(image)
    print(depth.mean())
    print(depth.std())

    plt.imsave("/home/kj/ar/peg_transfer/test_record/dam/dam_depth.png",depth)


def test_goal_selection(player):
    intrinsics_matrix=np.array([ [915.531074, -2.735185, 475.190330], [0.000000, 917.791312, 311.855098], [0.000000, 0.000000, 1.000000]])
    p = dvrk.psm('PSM2')

    player._load_depth_model()
    player._load_policy_model(filepath='./pretrained_models/state_dict.pt')

    cap_0=cv2.VideoCapture("/dev/video0")
    cap_2=cv2.VideoCapture("/dev/video2")

    # init 
    # open jaw
    p.jaw.move_jp(np.array(0.5)).wait()
    print("move done")

    # 0. define the goal
    # TODO the goal in scaled image vs. goal in simualtor?
    for i in range(10):
        ret1, frame1=cap_0.read()
        ret2, frame2=cap_2.read()
    
    #point=SetPoints("test", frame1)

    frame1=cv2.resize(frame1, (player.img_size,player.img_size))
    frame2=cv2.resize(frame2, (player.img_size,player.img_size))
    
    '''
    frame1=Image.fromarray(frame1).convert('RGB')
    frame2=Image.fromarray(frame2).convert('RGB')

    frame1=np.array(frame1)
    frame2=np.array(frame2)
    '''
    # 1. get depth from left and right image
    
    depth=player._get_depth(frame1, frame2)
    depth=cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX) #0,1

    point=SetPoints("test", frame1)

    goal_x, goal_y=point[0][0], point[0][1]
    goal_depth=depth[goal_x][goal_y]
    goal=player.convert_point_to_camera_axis(goal_x, goal_y,goal_depth, intrinsics_matrix)
    print("Selected Goal: ",goal)

def test_action(player):
    player._load_policy_model(filepath='./pretrained_models/state_dict.pt')
    depth_filepath='igev_depth_np.npy'
    image_filepath='test_image/image_left2.jpg'
    depth=np.load(depth_filepath)
    img =Image.open(image_filepath).convert('RGB')
    img=np.array(img)

    img=cv2.resize(img, (player.img_size,player.img_size))
    depth=cv2.resize(depth, (player.img_size,player.img_size))
    depth=cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX) #0,1
    seg=player._seg_with_red(img)
    robot_pos=np.zeros(3)
    robot_rot=np.zeros(3)
    jaw=np.array([0.4])
    goal=np.zeros(3)

    action=player._get_action(seg, depth, robot_pos, robot_rot, jaw, goal)

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

def init_psm():
    # create the ROS Abstraction Layer with the name of the node
    ral = crtk.ral('dvrk_python_node')

    # create a Python proxy for PSM1, name must match ROS namespace
    p = dvrk.psm(ral, 'PSM1')
    # p = dvrk.psm('PSM1')
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

    new_rot_euler=np.array([robot_rot_euler[0] ,robot_rot_euler[1] ,robot_rot_euler[2]])#np.array([player.rcm_init_eul[0],player.rcm_init_eul[1],robot_rot_euler[2]])
    #print('robot_rot_euler: ',new_rot_euler)
    transform_2=player.get_matrix_from_euler(new_rot_euler)
    
    PSM2_rotate = PyKDL.Rotation(transform_2[0,0], transform_2[0,1], transform_2[0,2],
                            transform_2[1,0], transform_2[1,1], transform_2[1,2],
                            transform_2[2,0], transform_2[2,1], transform_2[2,2])
    PSM2_pose = PyKDL.Vector(robot_pos[0]+0.01, robot_pos[1], robot_pos[2]+0.01)

    goal_2 = PyKDL.Frame(robot_pose.M, PSM2_pose)
    print("goal: ",goal_2)
    #p.move_cp(goal_2).wait()
    
def test_set_action(player):
    p = dvrk.psm('PSM2')
    p.jaw.move_jp(np.array(0.5)).wait()
    #action=np.random.rand(7)
    action=np.array([0.035    ,   0.01237744 ,-0.02809236 , 0.56227914 , 0.221693  , -0.33573622,
  0.5])
    
    PSM2_rotate = PyKDL.Rotation(-3.42020136e-01 , 9.39692624e-01 ,-1.52792067e-07 ,
                          9.39692624e-01 , 3.42020136e-01 , 7.12481127e-08,
                          1.19209290e-07 ,-1.19209290e-07 ,-1.00000000e+00)
 
    PSM2_pose = PyKDL.Vector(2.46325319e-02, 1.44550846e-03, -9.72000122e-02)

    

    goal_2 = PyKDL.Frame(PSM2_rotate, PSM2_pose)
    p.move_cp(goal_2).wait()
    
    
    #state=player._set_action(action, robot_rot, robot_pos)
    #print(state)


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

paletee=get_palette(5)

def plot_image(img,is_seg=False, is_depth=False, path='/home/student/code/SAM-rbt-sim2real/debug_result', name='img1.png'):
    if is_depth:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)
        
    #i=Image.fromarray(np.asarray(img,dtype=np.uint8))
    
    #if is_seg:
    #    np.save(os.path.join(path,'seg.npy'),img)
    #    i.putpalette(paletee)
    
    #i.save(os.path.join(path,name))

def target_goal_path(player):
    
    p = dvrk.psm('PSM2')
    
    p.jaw.move_jp(np.array(0.5)).wait()
    robot_pos=[  -0.0362473,  -0.0114265,   -0.142078]
    transform_2=player.get_matrix_from_euler(np.array([2.93450463 , 0.12247269 ,-1.19392802]))
    PSM2_rotate = PyKDL.Rotation(transform_2[0,0], transform_2[0,1], transform_2[0,2],
                            transform_2[1,0], transform_2[1,1], transform_2[1,2],
                            transform_2[2,0], transform_2[2,1], transform_2[2,2])
    PSM2_pose = PyKDL.Vector(robot_pos[0], robot_pos[1], robot_pos[2])

    goal_2 = PyKDL.Frame(PSM2_rotate, PSM2_pose)
    print("target: ",goal_2)
    p.move_cp(goal_2).wait()

    p.jaw.move_jp(np.array(-0.5)).wait()
    
    robot_pos= np.array([-0.0644053, 0.000849748,   -0.134457])
    PSM2_pose = PyKDL.Vector(robot_pos[0], robot_pos[1], robot_pos[2])
    goal_2 = PyKDL.Frame(PSM2_rotate, PSM2_pose)
    print("goal: ",goal_2)
    p.move_cp(goal_2).wait()
    p.jaw.move_jp(np.array(0.5)).wait()

def process_seg(depth):
    #x_left=154
    #y_left=147
    #x_right=174
    #y_right=184
    print(depth[116][95])
    print(depth[108][77])
    print(depth[166][83])
    new_seg=np.where((depth>0.4333)&(depth<0.5),np.ones_like(depth),np.zeros_like(depth))
    #new_seg=np.where(depth<0.4734,np.zeros_like(depth),np.ones_like(depth))
    new_seg*=255
    np.save('/home/kj/ar/peg_transfer/test_record/seg_from_depth.npy',new_seg)
    cv2.imwrite('/home/kj/ar/peg_transfer/test_record/seg_from_depth.png',new_seg)
    #return new_seg


basePSM_T_cam =np.array([[-0.94365124,  0.2832304 , -0.17118086, -0.08812556],
       [ 0.33030491,  0.77398309, -0.54023036,  0.07944245],
       [-0.02051857, -0.56633092, -0.82392249, -0.07761894],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


cam_T_basePSM =np.array([[-0.94365124,  0.33030491, -0.02051857, -0.11099265],
       [ 0.2832304 ,  0.77398309, -0.56633092, -0.08048528],
       [-0.17118086, -0.54023036, -0.82392249, -0.03612017],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


def get_waypoint_toward_object(pos_obj, orn_obj, pos_peg, goal_pos, SCALING=5):
    yaw=orn_obj[2]

    waypoints = [None, None, None, None] 
    waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * SCALING, yaw, 0.5])  # above object
    waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                    pos_obj[2] + (0.003 + 0.0102) * SCALING, yaw, 0.5])  # approach
    waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                    pos_obj[2] + (0.003 + 0.0102) * SCALING, yaw, -0.5])  # grasp
    waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * SCALING, yaw, -0.5])  # lift up

   
    return waypoints


def get_waypoint_toward_goal(pos_obj, orn_obj, pos_peg, goal_pos, SCALING=5):
    yaw=orn_obj[2]
    waypoints = [None, None, None] 

    pos_place = [goal_pos[0] + pos_obj[0] - pos_peg[0],
                     goal_pos[1] + pos_obj[1] - pos_peg[1], waypoints[0][2]]  # consider offset
   
    waypoints[4] = np.array([pos_place[0], pos_place[1],  waypoints[0][2], yaw, -0.5])  # above goal
    waypoints[5] = np.array([pos_place[0], pos_place[1], waypoints[2][2], yaw, -0.5])  # release
    
    waypoints[6] = np.array([pos_place[0], pos_place[1], waypoints[2][2], yaw, 0.5])  # release


def get_oracle_action(player, waypoint_pos, waypoint_orn, robot_pos, robot_rot):
    robot_pos_ecm=player.convert_pos(robot_pos, basePSM_T_cam)
    robot_rot_ecm=player.convert_rot(robot_rot, basePSM_T_cam)
    delta_pos=(waypoint_pos-robot_pos_ecm)
    delta_rot=(waypoint_orn-robot_rot_ecm)
    action=np.array([delta_pos[0],delta_pos[1],delta_pos[2],delta_rot[0],delta_rot[1],delta_rot[2]])
    return action


def test_perceptual_layer():
    intrinsics_matrix=np.array([ [915.531074, 0, 475.190330], [0.000000, 917.791312, 311.855098], [0.000000, 0.000000, 1.000000]])
    p = dvrk.psm('PSM2')
    player=VisPlayer()

    player._load_depth_model()
    player._load_policy_model(filepath='./pretrained_models/state_dict.pt')

    cap_0=cv2.VideoCapture("/dev/video0")
    cap_2=cv2.VideoCapture("/dev/video2")

    # init 
    # open jaw
    p.jaw.move_jp(np.array(0.5)).wait()
    print("open jaw")

    # 0. define the goal
    # TODO the goal in scaled image vs. goal in simualtor?
    for i in range(10):
        ret1, frame1=cap_0.read()
        ret2, frame2=cap_2.read()
    
    #point=SetPoints("test", frame1)

    frame1=cv2.resize(frame1, (player.img_size,player.img_size))
    frame2=cv2.resize(frame2, (player.img_size,player.img_size))
    
    point=SetPoints("Goal Selection", frame1)

    frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    depth=player._get_depth(frame1, frame2)
    #depth=cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX) #0,1

    #point=SetPoints("Goal Selection", frame1)

    goal_x, goal_y=point[0][0], point[0][1]
    goal_depth=depth[goal_x][goal_y]
    goal=player.convert_point_to_camera_axis(goal_x, goal_y,goal_depth, intrinsics_matrix)

    
    # debug goal
    #goal= np.array([-0.0577594,   0.0043639,   -0.133283])
    #goal= player.convert_pos(goal, basePSM_T_cam)
    goal=goal/player.scaling
    print("Selected Goal: ",goal)
    

    # Make the psm to the initial position
    robot_pose=p.measured_cp()
    robot_pos=robot_pose.p
    transform_2=player.get_matrix_from_euler(player.rcm_init_eul)
    
    PSM2_rotate = PyKDL.Rotation(transform_2[0,0], transform_2[0,1], transform_2[0,2],
                            transform_2[1,0], transform_2[1,1], transform_2[1,2],
                            transform_2[2,0], transform_2[2,1], transform_2[2,2])
    #PSM2_pose = PyKDL.Vector(goal[0],goal[1],goal[2])
    PSM2_pose = PyKDL.Vector(robot_pos[0], robot_pos[1], robot_pos[2])

    goal_2 = PyKDL.Frame(PSM2_rotate, PSM2_pose)
    #print("goal: ",goal_2)
    p.move_cp(goal_2).wait()
    print("move done")
    #exit()
    count=0

    #waypoint_
    
    for i in range(4):
        count+=1
        print("--------step {}----------".format(count))

        ret1, frame1=cap_0.read()
        ret2, frame2=cap_2.read()

        frame1=cv2.resize(frame1, (player.img_size,player.img_size))
        frame2=cv2.resize(frame2, (player.img_size,player.img_size))

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        
        # 1. get depth from left and right image
        depth=player._get_depth(frame1, frame2)/player.scaling
        #plt.imsave( 'test_record/pred_depth.png',depth)
        depth=cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX) #0,1
        print("finish depth")

        # 2. get seg
        seg = player._seg_with_red(frame1)
        
        print("finish seg")
    
        robot_pose=p.measured_cp()
        robot_pos=robot_pose.p
        robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        
        # can be replaced with robot_pose.M.GetRPY()
        # start
        transform_2=robot_pose.M
        np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2]],
                            [transform_2[1,0], transform_2[1,1], transform_2[1,2]],
                            [transform_2[2,0], transform_2[2,1], transform_2[2,2]]])
        robot_rot=player.get_euler_from_matrix(np_m)
        robot_pos=player.convert_pos(robot_pos, basePSM_T_cam)
        robot_rot=player.convert_rot(robot_rot, basePSM_T_cam)
        # end

        jaw=p.jaw.measured_jp()
        #action=player._get_action(seg, depth,robot_pos, robot_rot, jaw, goal)
        waypoint_pos, waypoint_orn=player._get_visual_state(seg, depth,robot_pos, robot_rot, jaw, goal)
        print("finish get visual state")

        # 4. action -> state
        action=get_oracle_action(player, waypoint_pos, waypoint_orn, robot_pos, robot_rot)
        state=player._set_action(action)

        print("finish set action")
        print("state: ",state)
        
        # 5. move 
        PSM2_rotate = PyKDL.Rotation(state[0,0], state[0,1], state[0,2],
                            state[1,0], state[1,1], state[1,2],
                            state[2,0], state[2,1], state[2,2])
        PSM2_pose = PyKDL.Vector(state[0,-1], state[1,-1], state[2,-1])

        move_goal = PyKDL.Frame(PSM2_rotate, PSM2_pose)

        if count>1:
            break
        
        p.move_cp(move_goal).wait()
        print("finish move")

        if action[6] < 0:
            # close jaw
            p.jaw.move_jp(np.array(-0.5)).wait()
        else:
            # open jaw
            p.jaw.move_jp(np.array(0.5)).wait()

        
        #if cv2.waitKey(1)==27:
        #    break
    
    cap_0.release()
    cap_2.release()

def resize_image(player):
    
    # 示例用法
    #image_path = '/home/kj/ar/peg_transfer/test_image1.png'
    player._load_depth_model()
    target_width = 256
    target_height = 256
    #limg=Image.open('/home/kj/ar/peg_transfer/test_image/image_left_forresize2.jpg')
    #rimg =Image.open('/home/kj/ar/peg_transfer/test_image/image_right_forresize2.jpg').convert('RGB')
    limg = resize_with_pad('/home/kj/ar/peg_transfer/test_image/image_left_forresize2.jpg', target_width, target_height)
    rimg = resize_with_pad('/home/kj/ar/peg_transfer/test_image/image_right_forresize2.jpg', target_width, target_height)
    #frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    #frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    cv2.imwrite('/home/kj/ar/peg_transfer/test_image1_resize.png', limg)
    #limg=np.array(limg)
    #rimg=np.array(rimg)
    depth=player._get_depth(limg,rimg)/player.scaling
    plt.imsave('/home/kj/ar/peg_transfer/test_image/pred_depth.png',depth)

    seg = player._seg_with_red(limg)
    cv2.imwrite('/home/kj/ar/peg_transfer/test_image/seg.png',seg)

    #resized_image.show()  # 显示处理后的图片
    #limg.save('/home/kj/ar/peg_transfer/test_image1_resize.png')  # 保存处理后的图片


def resize_with_pad(image_path, target_width, target_height):
    # 读取原始图片
    image = cv2.imread(image_path)

    # 计算缩放比例
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)

    # 缩放图片
    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 计算pad的大小
    pad_height = target_height - resized_image.shape[0]
    pad_width = target_width - resized_image.shape[1]

    # 加入pad
    padded_image = cv2.copyMakeBorder(resized_image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[30, 30, 30])

    return padded_image

def crop_image():

    img = cv2.imread("/home/kj/ar/peg_transfer/test22.png")
    h=300
    crop_img = img[:,100: ]
    crop_img = crop_img[:,: -100]
    print(crop_img.shape)
    cv2.resize(crop_img ,(256,256))
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)

def refine_seg():
    depth_file='/home/kj/ar/peg_transfer/test_record/depth.npy'
    depth=np.load(depth_file)
    seg_file='/home/kj/ar/peg_transfer/test_record/seg.png'
    seg=cv2.imread(seg_file)
    seg=seg[:,:,0]
    depth_thres=depth[54][78]
    print(depth[158][83])
    print(depth_thres)
    print(seg[65][70])
    new_seg=np.where( (depth < 0.88)& (depth==depth_thres),np.zeros_like(seg),seg)
    cv2.imshow( "new_seg",new_seg)
    cv2.waitKey(0)

def check_robot_pose(player):

    p = dvrk.psm('PSM1')
    robot_pose=p.measured_cp()
    robot_pos=robot_pose.p
    transform_2=robot_pose.M
    np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2]],
                            [transform_2[1,0], transform_2[1,1], transform_2[1,2]],
                            [transform_2[2,0], transform_2[2,1], transform_2[2,2]]])

    robot_rot_euler=player.get_euler_from_matrix(np_m)
    print('robot_pos: ',robot_pos)
    print('robot_rot_euler: ' ,np_m)

    joint=p.measured_jp()
    print(joint)
    pose=player.get_bPSM_T_j6(joint)
    print(pose)

def test_calibration(player):
    # basePSM_T_cam =np.array([[-0.74260612, -0.53325979,  0.40517916,  0.04417768],
    #    [-0.66947573,  0.60768183, -0.4272295 ,  0.10147299],
    #    [-0.0183957 , -0.58852085, -0.80827273, -0.04684721],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # '''
    # cam_T_basePSM =np.array([[-0.74260612, -0.66947573, -0.0183957 ,  0.09987853],
    #    [-0.53325979,  0.60768183, -0.58852085, -0.06567567],
    #    [ 0.40517916, -0.4272295 , -0.80827273, -0.01241294],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # '''

    # cam_T_basePSM =np.array([[-0.79342797, -0.606264  , -0.05400018,  0.09116146],
    #     [-0.43862051,  0.63102278, -0.63986115, -0.07202385],
    #     [ 0.42200013, -0.48399815, -0.76659095, -0.00801232],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
    basePSM_T_cam = np.array([[-0.9715952 , -0.08207092,  0.22196199,  0.14564693],
       [-0.1482096 ,  0.94223976, -0.30036337,  0.09838246],
       [-0.18449032, -0.3247285 , -0.92763933,  0.00676384],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    cam_T_basePSM = np.array([[-0.9715952 , -0.1482096 , -0.18449032,  0.15733895],
        [-0.08207092,  0.94223976, -0.3247285 , -0.07855007],
        [ 0.22196199, -0.30036337, -0.92763933,  0.00349681],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
    
    p = dvrk.psm('PSM1')
    robot_pose=p.measured_cp()
    #joint=p.measured_jp()
    robot_pos=robot_pose.p
    robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
    robot_pos=player.convert_pos(robot_pos,cam_T_basePSM)
 
    
    intrinsic_matrix=np.array([[915.531074, -2.735185, 475.190330,0],
      [0.000000, 917.791312, 311.855098,0],
      [0.000000, 0.000000, 1.000000,0],
      [0,0,0,1]])
    homogeneous_point_3d = np.append(robot_pos, 1)  # Convert to homogeneous coordinates
    
    projected_point = intrinsic_matrix @ homogeneous_point_3d
    projected_point /= projected_point[2]  # Normalize by the third component
    
    u = projected_point[0]
    v = projected_point[1]

    # Load your image
    image = plt.imread("/home/kj/ar/GauzeRetrievel/cbtest4.png")

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Add a point at position (u, v)
    ax.scatter(u, v, color='red', marker='o')

    # Show the plot
    plt.show()
    
   
def test_calibration_realtime(player):
    from player_utils import VideoCapture

    cap_0=VideoCapture("/dev/video0")
    # create the ROS Abstraction Layer with the name of the node
    ral = crtk.ral('dvrk_python_node')

    # create a Python proxy for PSM1, name must match ROS namespace
    p = dvrk.psm(ral, 'PSM1')
    # p = dvrk.psm('PSM1')

    for i in range(10):
        frame1=cap_0.read()
    '''
    cam_T_basePSM =np.array([[-0.74260612, -0.66947573, -0.0183957 ,  0.09987853],
       [-0.53325979,  0.60768183, -0.58852085, -0.06567567],
       [ 0.40517916, -0.4272295 , -0.80827273, -0.01241294],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    '''

    intrinsics_matrix=np.array([[693.12012738, 0.0, 355.44816971], [0.000000, 693.12012738, 327.7015152], [0.000000, 0.000000, 1.000000]])
    
    # basePSM_T_cam = np.array([[-0.9715952 , -0.08207092,  0.22196199,  0.14564693],
    #    [-0.1482096 ,  0.94223976, -0.30036337,  0.09838246],
    #    [-0.18449032, -0.3247285 , -0.92763933,  0.00676384],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # cam_T_basePSM = np.array([[-0.9715952 , -0.1482096 , -0.18449032,  0.15733895],
    #     [-0.08207092,  0.94223976, -0.3247285 , -0.07855007],
    #     [ 0.22196199, -0.30036337, -0.92763933,  0.00349681],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # basePSM_T_cam = np.array([[-0.8128076 ,  0.57348532, -0.10226631, -0.06379491],
    #     [ 0.57612612,  0.76542098, -0.28672185,  0.09373904],
    #     [-0.08615399, -0.29196799, -0.95253986, -0.00611106],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # cam_T_basePSM = np.array([[-0.8128076 ,  0.57612612, -0.08615399, -0.10638499],
    #     [ 0.57348532,  0.76542098, -0.29196799, -0.03694861],
    #     [-0.10226631, -0.28672185, -0.95253986,  0.01453193],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
    basePSM_T_cam = np.array([[-0.9715952 , -0.08207092,  0.22196199,  0.14564693],
                                        [-0.1482096 ,  0.94223976, -0.30036337,  0.09838246],
                                        [-0.18449032, -0.3247285 , -0.92763933,  0.00676384],
                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    cam_T_basePSM = np.array([[-0.9715952 , -0.1482096 , -0.18449032,  0.15733895],
                                        [-0.08207092,  0.94223976, -0.3247285 , -0.07855007],
                                        [ 0.22196199, -0.30036337, -0.92763933,  0.00349681],
                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    fs = cv2.FileStorage("/home/kj/ar/EndoscopeCalibration/endoscope_calibration_csr_0degree.yaml", cv2.FILE_STORAGE_READ)
    
    while True:
        frame1=cap_0.read()
        frame2=frame1.copy()

        
        frame1,_=my_rectify(frame1,frame2,fs)
        
        robot_pose=p.measured_cp()
        #joint=p.measured_jp()
        #print(player.get_bPSM_T_j6(joint).shape)
        
        #homogeneous_point_3d=cam_T_basePSM @ player.get_bPSM_T_j6(joint)
        #homogeneous_point_3d=np.array([homogeneous_point_3d[0,3],homogeneous_point_3d[1,3],homogeneous_point_3d[2,3],1])
        robot_pos=robot_pose.p
        robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        robot_pos=player.convert_pos(robot_pos,cam_T_basePSM)
 
    

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



def test_robot(player):
    p = dvrk.psm('PSM1')
    robot_pose=p.measured_cp()

    #print(type(robot_pose))
    #print(type(robot_pose.p))
    #print(robot_pose.M)
    #print(robot_pose.M.GetEulerZYX())
    #print(robot_pose.M.GetRPY())
    robot_pos=robot_pose.p
    print(robot_pos)
    transform_2=robot_pose.M
    pos=np.array([8.42418525e-02,   3.73921402e-02,   -1.63881063e-01])
    PSM2_pose = PyKDL.Vector(8.42418525e-02,   3.73921402e-02,   -1.63881063e-01)
    goal_2 = PyKDL.Frame(robot_pose.M, PSM2_pose)
    print("goal: ",goal_2)
    p.move_cp(goal_2).wait()
    print(player.convert_pos(pos,cam_T_basePSM))

def test_v_layer(player):
    import torch
    p = dvrk.psm('PSM1')
    robot_pose=p.measured_cp()
    robot_pos=robot_pose.p
    robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
    print('ori robot pos: ',robot_pos)
    '''
    depthfile='test_record/depth.npy'
    segfile='test_record/seg.npy'
    depth=np.load(depthfile)
    seg=np.load(segfile)
    
    player._load_policy_model(filepath='./pretrained_models/s56_DDPG_demo50_traj_best.pt')
    seg=torch.from_numpy(seg).to("cuda:0").float()
    depth=torch.from_numpy(depth).to("cuda:0").float()
    v_output=player.v_model.get_obs(seg.unsqueeze(0), depth.unsqueeze(0))[0]
    rel_pos=v_output[:3].cpu().data.numpy()
    #print('rel pos: ', rel_pos)
    #new_pos=robot_pos+player.convert_pos(rel_pos,basePSM_T_cam) #rel_pos
    #print('rcm rel pos: ',player.convert_pos(rel_pos,basePSM_T_cam))

    new_robot_pos=robot_pos+player.convert_pos(rel_pos,basePSM_T_cam) #player.convert_pos(new_pos,basePSM_T_cam)
    print('new_robot_pos: ',new_robot_pos)
    #new_robot_pos*=0.1
    
    PSM2_pose = PyKDL.Vector(new_robot_pos[0],new_robot_pos[1],new_robot_pos[2])
    goal_2 = PyKDL.Frame(robot_pose.M, PSM2_pose)
    p.move_cp(goal_2).wait()
    '''
    #PSM2_pose = PyKDL.Vector(robot_pos[0],robot_pos[1],robot_pos[2]+0.01)
    #goal_2 = PyKDL.Frame(robot_pose.M, PSM2_pose)
    #p.move_cp(goal_2).wait()

def test_policy(player):
    import torch
    player._load_policy_model(filepath='./pretrained_models/s80_DDPG_demo0_traj_csr_best.pt')
    print('load ok')
    '''
    p = dvrk.psm('PSM1')
    robot_pose=p.measured_cp()
    robot_pos=robot_pose.p
    ori_robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
    robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
    
    robot cam rot:  [ 9.76455759e-01 -1.98398229e-02  1.63500186e-04]
    waypoint_rot:  [-3.14159211  0.59999807  1.57079645]
    [-0.00903667 -0.00051452  0.08117111]
    [-0.00903667 -0.00051452  0.08117111]
    action:  [ 0.0232594   0.02136326 -0.01538056  0.00465123]
    
    print("pre action pos: ",robot_pos)
    # can be replaced with robot_pose.M.GetRPY()
    # start    
    robot_pos=player.convert_pos(robot_pos,cam_T_basePSM)
    robot_pos=torch.tensor(robot_pos).to(player.device)

    robot_rot=torch.tensor([9.76455759e-01 ,-1.98398229e-02 , 1.63500186e-04]).to(player.device)
    object_pos=torch.tensor([0.097165 ,   0.069824   , -0.19636]).to(player.device)
    rel_pos=object_pos-robot_pos
    waypoint_pos=torch.tensor([0.097165 ,   0.069824   , -0.19636]).to(player.device)
    waypoint_rot=torch.tensor([-3.14159211 , 0.59999807 , 1.57079645]).to(player.device)
    jaw=torch.tensor([0.6981]).to(player.device)

    goal= np.array([    0.0679368,   0.0509549,   -0.153433])
    goal=player.convert_pos(goal, cam_T_basePSM)
    goal=torch.tensor(goal).to(player.device)

    

    o_new=torch.cat([
                robot_pos, robot_rot, jaw,
                object_pos, rel_pos, waypoint_pos, waypoint_rot
            ])
    
    o_norm=player.agent.o_norm.normalize(o_new,device=player.device)
            
    g_norm=player.agent.g_norm.normalize(object_pos, device=player.device)

    input_tensor=torch.cat((o_norm, g_norm), axis=0).to(torch.float32)
    
    action = player.agent.actor(input_tensor).cpu().data.numpy().flatten()

    transform_2=robot_pose.M
    print(transform_2)
    np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2]],
                    [transform_2[1,0], transform_2[1,1], transform_2[1,2]],
                    [transform_2[2,0], transform_2[2,1], transform_2[2,2]]])
    
    state=player._set_action(action, ori_robot_pos, np_m)
    print("state: ",state)
    '''



if __name__=="__main__":
   
    player=VisPlayer()
    #test_policy(player)
    #test_depth_igev(player)
    #test_robot(player)
    #test_v_layer(player)
    #test_dam(player)
    #test_fast_sam(player)
    #check_robot_pose(player)
    # test_depth_igev_realtime(player)
    #test_depth_igev(player)
    #crop_image()
    #refine_seg()
    #resize_image(player)
    #test_perceptual_layer(player)
    # init_psm()
    #test_depth_igev(player)
    #test_set_action(player)
    #test_action(player)
    #test_goal_selection(player)
    #test_depth(player)
    #test_depth_opencv()
    #img_filepath='/home/kj/ar/peg_transfer/test_record/depth.npy'
    #img_filepath='/home/kj/ar/peg_transfer/test_record/pred_depth.png'
    #img=cv2.imread(img_filepath)
    #img=np.array(img)
    #img=np.load(img_filepath)
    
    #process_seg(img)
    #point=SetPoints("test", img)
    #print(point[0])
    test_calibration_realtime(player)
    #init_psm(player)
    #target_goal_path(player)
    #test_set_action(player)
    #test_depth_igev_realtime(player)
    '''
    seg_path='/home/kj/ar/peg_transfer/test_record/seg.npy'
    seg=np.load(seg_path)
    
    print(np.unique(seg))
    #plot_image(seg, is_seg=True, path='/home/kj/ar/peg_transfer/test_record', name='seg.png')
    #cv2.applyColorMap(seg, cv2.COLORMAP_JET)
    cv2.imwrite('/home/kj/ar/peg_transfer/test_record/seg.png',seg)
    '''

    