# This code implements YOLOv3 technique in order to detect car accidents on the video frames. 
# If you want to use this with camera, you can easily modify it

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt
import glob
from vehicle_tracking import * # functions for tracking

thr_param = 0.3
conf_param = 0.5

data_dir = "Dataset/" 	#dataset directory
dataset_path = glob.glob(data_dir+"*/") 		#reading all sub-directories in folder
print('Sub-directories',dataset_path)

for path in dataset_path:
	split_path = path.split('/')
	folders = glob.glob(path)
	print('Processing folder',folders[0], '...')
	img_dir = split_path[1]  
	data_path = os.path.join(path,'*g')

	frame_counter = len(glob.glob(data_path))
	print('Number of frames:',frame_counter)

	files = []
	for q in range(frame_counter):
	#for q in range(10):
		path = folders[0]+'/'+str(q)+'.jpg'
		files.append(path)

	

	cars_dict = {}
	counter = 1
	for f1 in files:
		image = cv2.imread(f1)
		print('Processing frame:'+str(counter)+'/'+str(frame_counter),'in folder:'+folders[0]+'...')
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
			weightsPath = os.path.sep.join(['yolo-coco/weights', "yolov3.weights"])
			configPath = os.path.sep.join(['yolo-coco/cfg', "yolov3.cfg"])

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
				# building a list or centers we're keeping
				new_boxes = []
				for i in idxs.flatten():
					new_boxes.append(boxes[i]) 
					
					
				# building cars data
				cars_dict = BuildAndUpdate(new_boxes, cars_dict)
				cars_labels = list(cars_dict)
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, color, 1)
				
				for car_label in cars_labels:
					car_path = cars_dict[car_label][0]
					# plotting car path on image and printing car label
					if len(car_path)> 1:
						car_path = np.asarray(car_path,dtype=np.int32)
						car_path = car_path.reshape((-1,1,2))
						cv2.polylines(image,car_path,True,(0,0,255))
						label_location = car_path[len(car_path)-1][0]
						cv2.putText(image, car_label, (label_location[0], label_location[1]), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, color, 1)
						

			# show the output image
			#cv2.imshow("Image", image)
			#time.sleep(2)
			#cv2.waitKey(0)
			print('ClassIDs:',classIDs)

	#cv2.waitKey(0)
	print(cars_dict)
	print(list(cars_dict))

	#saving output image in folder output/
	cv2.imwrite('output/'+img_dir+'_final_frame.png', image)

	#building dictionary of plot data
	cars_plot_data = {}
	for label in cars_labels: 
		position  = cars_dict[label][0]
		direction = cars_dict[label][1]
		velocity = cars_dict[label][2]
		acceleration = cars_dict[label][3]
		
		x_pos = []
		y_pos = []
		angle = []
		time_frame = []
		cars_plot_data[label]={}
		for i in range(len(position)):
			x_pos.append(position[i][0])
			y_pos.append(position[i][1])
			angle.append(np.arccos(direction[i][0][0]))
			time_frame.append(i)

		cars_plot_data[label]['x'] = x_pos
		cars_plot_data[label]['y'] = y_pos
		cars_plot_data[label]['time'] = time_frame
		cars_plot_data[label]['angle'] = angle
		cars_plot_data[label]['velocity'] = velocity
		cars_plot_data[label]['acceleration'] = acceleration

	#plotting and saving cars information from each video
	#plots can be found in folder "figures/"
	plt.figure(figsize=(10,8))
	for label in cars_labels: 
		plt.plot(cars_plot_data[label]['x'],cars_plot_data[label]['y'])
	plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('position x')
	plt.ylabel('position y')
	plt.xlim((0,image.shape[0]))
	plt.ylim((0,image.shape[1]))
	plt.title('cars trajectories')
	plt.savefig('figures/'+img_dir+'_trajectory.png')



	plt.figure(figsize=(10,8))
	plt.subplots_adjust(wspace=0.5)
	plt.subplot(221)
	for label in cars_labels: 
		plt.plot(cars_plot_data[label]['time'],cars_plot_data[label]['angle'])
	plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('frame')
	plt.ylabel('angle (rad)')
	plt.title('cars angles')


	plt.subplot(222)
	for label in cars_labels: 
		plt.plot(cars_plot_data[label]['time'],cars_plot_data[label]['velocity'])
	plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('frame')
	plt.ylabel('velocity (pixel/frame)')
	plt.title('cars velocities')


	plt.subplot(223)
	for label in cars_labels: 
		plt.plot(cars_plot_data[label]['time'],cars_plot_data[label]['acceleration'])
	plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('frame')
	plt.ylabel(r'acceleration (pixel/${frame}^2$)')
	plt.title('cars accelerations')
	plt.savefig('figures/'+img_dir+'_Info.png')
	#plt.show()

	#cv2.waitKey(0)
	cv2.destroyAllWindows()
