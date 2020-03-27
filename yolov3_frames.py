# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt
import glob
from vehicle_tracking import *

thr_param = 0.3
conf_param = 0.5

img_dir = "000002" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')

frame_counter = len(glob.glob(data_path))
print(data_path)
print(frame_counter)

files = []
for q in range(frame_counter):
#for q in range(50):
	path = img_dir+'/'+str(q)+'.jpg'
	files.append(path)

print(files)

cars_dict = {}
counter = 1
for f1 in files:
	image = cv2.imread(f1)
	print(type(image))
	print(str(counter)+'/'+str(frame_counter))
	counter +=1

	if type(image) is np.ndarray:	
		# load the COCO class labels our YOLO model was trained on
		labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
		LABELS = open(labelsPath).read().strip().split("\n")

		# initialize a list of colors to represent each possible class label
		np.random.seed(42)
		COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
			dtype="uint8")

		# derive the paths to the YOLO weights and model configuration
		weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
		configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])

		# load our YOLO object detector trained on COCO dataset (80 classes)
		print("[INFO] loading YOLO from disk...")
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


		# load our input image and grab its spatial dimensions
		#image = cv2.imread(img)
		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (96, 96),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > conf_param and (classID == 0 or classID == 2):
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_param,
			thr_param)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			new_boxes = []
			for i in idxs.flatten():
				new_boxes.append(boxes[i]) 
				# extract the bounding box coordinates
				
			#print(new_boxes)
			cars_dict = BuildAndUpdate(new_boxes, cars_dict)
			cars_labels = list(cars_dict)
			for boxes in new_boxes:
				(x, y) = (boxes[0], boxes[1])
				(w, h) = (boxes[2], boxes[3])
				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 1)
			
			for car_label in cars_labels:
				car_path = cars_dict[car_label][0]
				#print('p',car_path)
				if len(car_path)> 1:
					car_path = np.asarray(car_path,dtype=np.int32)
					car_path = car_path.reshape((-1,1,2))
					cv2.polylines(image,car_path,True,(0,0,255))
					

			

		# show the output image
		cv2.imshow("Image", image)
		#time.sleep(2)
		cv2.waitKey(0)
		print(classIDs)

print(cars_dict)
print(list(cars_dict))

cv2.waitKey(0)
cv2.destroyAllWindows()