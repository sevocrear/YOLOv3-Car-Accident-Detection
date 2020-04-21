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


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(['yolo-coco/weights', "yolov3.weights"])
configPath = os.path.sep.join(['yolo-coco/cfg', "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)

writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
    time_start = time.time()
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # frame = imutils.resize(frame, width=96)

    # # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (h_NN, w_NN),
                                swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf_param and (classID == 0):
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_param,
                            thr_param)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                    confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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