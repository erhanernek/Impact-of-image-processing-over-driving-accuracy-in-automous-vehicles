
# importing the necessary libraries
import sys
import random
import numpy as np
import logging
import cv2
from datetime import datetime
import time
import glob
import os

today = datetime.now()
if today.hour < 10:
    h = "0"+ str(today.hour)
else:
    h = str(today.hour)
if today.minute < 10:
    m = "0"+str(today.minute)
else:
    m = str(today.minute)

root_dir = "C:/Users/TKA/Desktop/elec/numpy"

print(root_dir)

try:
    os.makedirs(root_dir)
except:
    print(" ")
try:
    inputs_file = open(root_dir + "/inputs_dept.npy","ba+") 
except:
    print("Error encountered on file opening")

# Creating a VideoCapture object to read the video


cap = cv2.VideoCapture('C:/Users/TKA/Desktop/elec/texas_depth_200x88.avi')
  

# Loop untill the end of the video
while (cap.isOpened()) :
  
    # Capture frame-by-frame
    ret, im = cap.read()
    '''
    frame = np.frombuffer(frame, dtype=np.dtype("uint8")) 
    frame = np.reshape(frame, (88, 200, 3))
    '''
    image = im[:, :, :3]/255
    np.save(inputs_file, image)
    


inputs_file.close()
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

