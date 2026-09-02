import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
cap_0=cv2.VideoCapture("/dev/video0") # right
cap_1=cv2.VideoCapture("/dev/video2") # left
#print(cap_0.get(4))# 800 600
#print(cap_1.get(4)) #640 360
#cap_0.set(3,640)
#cap_0.set(4,360)

counter = 53
#cap_1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))

while True:
    #cap_0.set(3,640)
    #cap_0.set(4,360)
    #cap_1.set(3,800)
    #cap_1.set(4,600)
    print(cap_0.get(3))# 800 600 ->4 3
    print(cap_0.get(4)) #640 360 ->16 9
    ret1, frame1 = cap_0.read()
    #frame1=cv2.flip(frame1,0)
    
    #cv2.imwrite('test_image/fps_debug/testimage' + str(counter) + '.jpg', frame1)
    counter+=1
    cv2.imshow("Input1", frame1)
    

    ret2, frame2 = cap_1.read()
    #frame2= cv2.flip(frame2, 0)
    #frame2=cv2.resize(frame2,(800,600))
    cv2.imshow("Input2", frame2)
    #print(frame2.shape)
    #print(frame 1)
    #frame1=cv2.normalize(frame1, None, 0, 1, cv2.NORM_MINMAX) #0,1
    #frame2=cv2.normalize(frame2, None, 0, 1, cv2.NORM_MINMAX) #0,1
    #print(frame1)
    #plt.imsave( 'test_record/realtimetest/frame1_test_{}.png'.format(counter),cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    #plt.imsave( 'test_record/realtimetest/frame2_test_{}.png'.format(counter),cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

    #frame1=cv2.resize(frame1, (256,256))
    #frame2=cv2.resize(frame2, (256,256))
    #frame1=cv2.normalize(frame1, None, 0, 1, cv2.NORM_MINMAX) #0,1
    #frame2=cv2.normalize(frame2, None, 0, 1, cv2.NORM_MINMAX) #0,1
    #cv2.imshow("Input1", frame1)
    #cv2.imshow("Input2", frame2)

    #plt.imsave( 'test_record/frame1_test.png',cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    #plt.imsave( 'test_record/frame2_test.png',cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))


    if cv2.waitKey(33) == ord('a'):
        counter+=1
        cv2.imwrite('test_image/image_left_forresize' + str(counter) + '.jpg', frame1)
        cv2.imwrite('test_image/image_right_forresize' + str(counter)+ '.jpg', frame2)

    elif cv2.waitKey(1) == 27:
        break
cap_0.release()
cap_1.release()
cv2.distroyAllWindows()
