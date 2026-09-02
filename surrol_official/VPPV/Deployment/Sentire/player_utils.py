import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import queue, threading
import cv2 as cv

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

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    video_name='test_record/{}.mp4'.format(name.split('/')[-1])
    #self.output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (800, 600))

    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
    #t.join()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      #self.output_video.write(frame)
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def release(self):

      self.cap.release()
      #self.output_video.release()

def my_rectify(left_image, right_image, fs):

    fn_R = fs.getNode("R")
    fn_T = fs.getNode("T")
    fn_M1= fs.getNode("M1")
    fn_D1 = fs.getNode("D1")
    fn_M2= fs.getNode("M2")
    fn_D2 = fs.getNode("D2")

    stereo_R = fn_R.mat().astype(np.float64)
    stereo_T = fn_T.mat().astype(np.float64).transpose()

    stereo_M1 = fn_M1.mat()
    stereo_D1 = fn_D1.mat()
    stereo_M2 = fn_M2.mat()
    stereo_D2 = fn_D2.mat()

    # print(stereo_R)
    # print(stereo_T.shape)
    # print(stereo_M1)
    # print(stereo_D1)
    # print(stereo_M2)
    # print(stereo_D2)
    height, width, channel = left_image.shape

    # left_image_undistorted = cv.undistort(left_image, stereo_M1, stereo_D1)
    # right_image_undistorted = cv.undistort(right_image, stereo_M2, stereo_D2)

    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(stereo_M1, stereo_D1, stereo_M2, stereo_D2, (width, height), stereo_R, stereo_T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    # print(P1)
    # print(P2)
    # print(Q)
    # exit()

    leftMapX, leftMapY = cv.initUndistortRectifyMap(stereo_M1, stereo_D1, R1, P1, (width, height), cv.CV_32FC1)
    left_rectified = cv.remap(left_image, leftMapX, leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    rightMapX, rightMapY = cv.initUndistortRectifyMap(stereo_M2, stereo_D2, R2, P2, (width, height), cv.CV_32FC1)
    right_rectified = cv.remap(right_image, rightMapX, rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    # print(left_rectified.shape)
    # print(right_rectified.shape)

    return left_rectified, right_rectified

def get_F(fs):

    fn_R = fs.getNode("R")
    fn_T = fs.getNode("T")
    fn_M1= fs.getNode("M1")
    fn_D1 = fs.getNode("D1")
    fn_M2= fs.getNode("M2")
    fn_D2 = fs.getNode("D2")

    stereo_R = fn_R.mat().astype(np.float64)
    stereo_T = fn_T.mat().astype(np.float64).transpose()

    stereo_M1 = fn_M1.mat().astype(np.float64)
    stereo_D1 = fn_D1.mat()
    stereo_M2 = fn_M2.mat().astype(np.float64)
    stereo_D2 = fn_D2.mat()
    # print("M1", stereo_M1)
    # print("M2", stereo_M2)

    K1_inv_t = np.linalg.inv(stereo_M1.transpose())
    K2_inv = np.linalg.inv(stereo_M2)

    T_mat = np.array(
            [[0.0,-stereo_T[2][0],stereo_T[1][0]],
            [stereo_T[2][0],0.0,-stereo_T[0][0]],
            [-stereo_T[1][0],stereo_T[0][0],0.0]])

    F = np.dot(K1_inv_t, np.dot(np.dot(T_mat, stereo_R), K2_inv))

    return F

def add_gaussian_noise(depth_map, noise_std):
    noise = np.random.normal(0, 25, size=depth_map.shape).astype(np.uint8)
    noisy_depth_map = np.clip(depth_map +  noise, 0, 255)
    return noisy_depth_map

def gaussian_blur(image, kernel_size=(15,15), sigma=0):
   return cv2.GaussianBlur(image, kernel_size, sigma)
