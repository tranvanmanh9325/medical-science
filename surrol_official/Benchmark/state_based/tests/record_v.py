import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
cap_0=cv2.VideoCapture("/dev/video0")
cap_1=cv2.VideoCapture("/dev/video2")
counter=0

while True:
    
    ret1, frame1 = cap_0.read()
    ret2, frame2 = cap_1.read()
    
    cv2.imshow("Input2", frame2)
    cv2.imshow("Input1", frame1)

    if cv2.waitKey(33) == ord('a'):
        counter+=1
        cv2.imwrite('xx/image_left' + str(counter) + '.jpg', frame1)
        cv2.imwrite('xx/image_right' + str(counter)+ '.jpg', frame2)

    elif cv2.waitKey(1) == 27:
        break
        
cap_0.release()
cap_1.release()
cv2.distroyAllWindows()
