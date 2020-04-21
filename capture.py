import numpy as np
import cv2
import math

# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():


    ret, frame = capture.read(-1)

    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255,0), 0)
    crop_image = frame[100:324, 100:324]
    #img_gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gesture", frame)
  #  cv2.imshow("gray crop",img_gray)
    cv2.imshow("crop image ", crop_image)
    #img=load_img(path,target_size=(224,224))

    
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("7.jpg",crop_image)
        break
    
capture.release()
del capture
cv2.destroyAllWindows()