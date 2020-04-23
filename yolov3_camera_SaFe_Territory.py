# This code implement YOLOv3 technique in order to detect people and cars on the camera frames. 
# The classes YOLOv3 trained on you can find on the net.
# This implementation might be needed for invasion of the territory detection.


# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import time
import datetime
import yolo

path_of_file = os.path.abspath(__file__)
os.chdir(os.path.dirname(path_of_file))

h_NN = 192
w_NN = 192

h_show = 416*2
w_show = 256*2

thr_param = 0.3  # threshold when applying non-maxima suppression
conf_param = 0.5  # minimum probability to filter weak detections

warn_img = 'files/warning.png'
warn_image = cv2.imread(warn_img)
(H, W) = warn_image.shape[:2]
warn_image = cv2.resize(warn_image,(h_show//2,w_show//2))

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(['yolo-coco/weights', "yolov3.weights"])
configPath = os.path.sep.join(['yolo-coco/cfg', "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)

writer = None
(W, H) = (None, None)

yolov3 = yolo.YOLOv3(thr_param, conf_param, [0], net, 'detection')
# loop over frames from the video file stream
while True:
    time_start = time.time()
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    boxes, idxs, frame = yolov3.detect_objects(frame)

    # Display the resulting frame
    frame = cv2.resize(frame,(h_show,w_show))
    if len(idxs) > 0:
        frame[frame.shape[0]//4: frame.shape[0]//4+warn_image.shape[0], frame.shape[1]//4: frame.shape[1]//4+warn_image.shape[1],:] = np.where(warn_image>70,frame[frame.shape[0]//4: frame.shape[0]//4+warn_image.shape[0], frame.shape[1]//4: frame.shape[1]//4+warn_image.shape[1],:], warn_image)
    cv2.imshow('Frame', frame)
    time_end = time.time()
    print('FPS = ', 1/(time_end - time_start))
    if cv2.waitKey(1) == 27: # if esc button clicked, then exit loop
        break

print('Finished! you\'re awesome')    