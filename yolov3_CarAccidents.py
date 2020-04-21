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
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D


def calc_dist(coord_new_frame, coord_prev_frame):
	# The function that calculates the eucledean distance 
	# between two points (x_new, y_new) and (x_prev, y_prev)
	# It returns the float number
	coord_subtract = np.subtract(coord_new_frame, coord_prev_frame)
	dist = np.sqrt(coord_subtract[0]**2+coord_subtract[1]**2)
	return int(dist)

def draw_box_rectangle(frame, box, color, id):
	# The function just draws rectangle by given coordinates of the box
	# on the certain frame.
	# Also, it gives rectangle certain color and ID (text)

	# extract the bounding box coordinates
	# box = (x, y, w, h, centerx, centery)
	(x, y) = (box[0], box[1])
	(w, h) = (box[2], box[3])

	# draw a bounding box rectangle and label on the frame
	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
	text = "{}: {:d}".format('Car #',
		int(id))
	cv2.circle(frame, (box[4], box[5]), 5, color,2)
	cv2.putText(frame, text, (x, y - 5),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	return frame

track_dictionary = {} # dict for saving info about cars being tracked
# key: id. value: x, y, w, h, centerx, centery, color
# where:
## id - id of the human
## x - x coordinate on the frame of the left upper angle of the human rectangle
## y - y coordinate on the frame of the left upper angle of the human rectangle
## (centerx, centery) - coordinates of the rectangle's center.
## color - color of the rectangle. (r,g,b)

def track_cars(image, boxes, track_dictionary):
	# function that starts every new frame shows
	# input: boxes of detected by YOLO people and dictionary that saves info about people that have been tracked already
	# box = (x, y, w, h, centerx, centery, color)
	# output: ids of people to show in this frame and updated dictionary of tracked people if there were new people.

	idx_in_this_frame = [] #This list saves human detected on this frame in order to show them later on this frame
	if track_dictionary == {}:
		# is dictionary is free, than that's probably the first frame and there weren't any people that were tracked earlier 
		for box in boxes:
			track_dictionary[boxes.index(box)] = box
			idx_in_this_frame.append(boxes.index(box))	
			save_car(image, int(boxes.index(box)), box)
	else:
		for box in boxes:
			# try to find people that were seen in the previous frames in order to set the same id for them.

			min_dist = 5000 # ratio in order to compare people from dictionary with people in the boxes.
			for key, value in track_dictionary.copy().items():
				dist = calc_dist(box[4:6], track_dictionary[key][4:6])
				if dist <= min_dist:
					min_dist = dist
					idx = key

			if min_dist <= min(track_dictionary[idx][2:4]):
				track_dictionary[idx][0:6] = box[0:6]	 
				boxes.remove(box)
				idx_in_this_frame.append(idx)
			else:
				# new human
				k = len(track_dictionary)
				idx_in_this_frame.append(k)
				save_car(image, int(k), box)	
				track_dictionary[k] = box
				boxes.remove(box)	
	return track_dictionary, idx_in_this_frame	


def check_odd_filter(x):
	# It's function used for window and poly order calculation
	# for moving averaginf filter

	# x is the size of the window
	# y is the poly order. Should be less than x

	coeff = 1
	x = x// coeff # window size = (size of data)/coefficient
	if x <= 2: 
		x = 3
	if x % 2 == 0:
		x = x - 1
	if x <= 3:
		if x <=2:
			y = 1
		else:	
			y = 2
	else:
		y = 3		
	return (x, y)	

def save_car(frame, label, box):
  (x, y) = (box[0], box[1])
  (w, h) = (box[2], box[3])
  # draw a bounding box rectangle and label on the image
  color = (255,255,0)
  cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
  text = "{}: {:d}".format('label', int(label))
  cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
  #saving output image in folder cars/
  cv2.imwrite('cars/label_'+str(label)+'.png', frame)  
  pass

path_of_file = os.path.abspath(__file__)
os.chdir(os.path.dirname(path_of_file))

thr_param = 0.3 # threshold for YOLO detection
conf_param = 0.5 # confidence for YOLO detection
number_of_frames = 100 # number of image frames to work with in each folder (dataset)
filter_flag = 1 # use moving averaging filter or not (1-On, 0 - Off)
len_on_filter = 2 # minimum length of the data list to apply filter on it

data_dir = "Dataset/" 	#dataset directory
dataset_path = glob.glob(data_dir+"*/") 		#reading all sub-directories in folder
print('Sub-directories',dataset_path)

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

for path in dataset_path: # Loop through folders with different video frames (situations on the road)
	split_path = path.split('/')
	folders = glob.glob(path)
	print('Processing folder',folders[0], '...')
	img_dir = split_path[1]  
	data_path = os.path.join(path,'*g')

	frame_counter = len(glob.glob(data_path))
	print('Number of frames:',frame_counter)

	files = []
	#for q in range(frame_counter):
	for q in range(number_of_frames): # Loop through certain number of video frames in the folder
		path = folders[0]+'/'+str(q)+'.jpg'
		files.append(path)

	

	cars_dict = {}
	counter = 1
	for f1 in files:
		image = cv2.imread(f1)
		print('Processing frame:'+str(counter)+'/'+str(frame_counter),'in folder:'+folders[0]+'...')
		counter +=1

		if type(image) is np.ndarray:	


			# load our input image and grab its spatial dimensions
			#image = cv2.imread(img)
			(H, W) = image.shape[:2]

			# determine only the *output* layer names that we need from YOLO
			ln = net.getLayerNames()
			ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (384, 384), # (96, 96) \ (192, 192) \ (256, 256)
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# initialize our lists of detected bounding boxes, confidences,
			# and class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []
			centers = []
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
					if (confidence > conf_param) and ((classID == 2) or (classID == 3) or (classID == 5) or (classID == 7)):
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
						boxes.append([x, y, int(width), int(height), centerX, centerY])

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
				boxes_tracked = []
				for i in idxs.flatten():
					r = np.random.choice(255)
					g = np.random.choice(255)
					b = np.random.choice(255)
					color = (r,g,b)
					boxes[i].append(color)
					boxes_tracked.append(boxes[i])

				track_dictionary, idx_in_this_frame = track_cars(image, boxes_tracked, track_dictionary)

				for key in idx_in_this_frame:
					image = draw_box_rectangle(image, track_dictionary[key][0:6], track_dictionary[key][6], key)

			image = cv2.resize(image,(W,H))
			cv2.imshow("Image", image)
			if cv2.waitKey(1) == 27: # if esc button clicked, then exit loop
				break
	#cv2.waitKey(0)
	# print(cars_dict)
	# print(list(cars_dict))

	#saving output image in folder output/
	cv2.imwrite('output/'+img_dir+'_final_frame.png', image)

	# #building dictionary of plot data
	# cars_plot_data = {}
	# for label in cars_labels: 
	# 	position  = cars_dict[label][0]
	# 	direction = cars_dict[label][1]
	# 	velocity = cars_dict[label][2]
	# 	acceleration = cars_dict[label][3]
		
	# 	x_pos = []
	# 	y_pos = []
	# 	angle = []
	# 	time_frame = []
	# 	cars_plot_data[label]={}
	# 	for i in range(len(position)):
	# 		x_pos.append(position[i][0])
	# 		y_pos.append(position[i][1])
	# 		angle.append(np.arccos(direction[i][0][0]))
	# 		time_frame.append(i)

	# 	# Used condition on length of the list in order not tu use filter with very small amount of data.

	# 	if filter_flag:
	# 		if len(x_pos) > len_on_filter:
	# 			window_size, polyorder = check_odd_filter(len(x_pos))
	# 			x_pos = savgol_filter(x_pos, window_size, polyorder)
	# 		if len(y_pos) > len_on_filter:
	# 			window_size, polyorder = check_odd_filter(len(y_pos))
	# 			y_pos = savgol_filter(y_pos, window_size, polyorder)
	# 		if len(angle) > len_on_filter:
	# 			window_size, polyorder = check_odd_filter(len(angle))
	# 			angle = savgol_filter(angle, window_size, polyorder)
	# 		if len(velocity) > len_on_filter:
	# 			window_size, polyorder = check_odd_filter(len(velocity))
	# 			velocity = savgol_filter(velocity, window_size, polyorder)
	# 		if len(acceleration) > len_on_filter:
	# 			window_size, polyorder = check_odd_filter(len(acceleration))
	# 			acceleration = savgol_filter(acceleration, window_size, polyorder)

	# 	cars_plot_data[label]['x'] = x_pos
	# 	cars_plot_data[label]['y'] = y_pos
	# 	cars_plot_data[label]['time'] = time_frame
	# 	cars_plot_data[label]['angle'] = angle
	# 	cars_plot_data[label]['velocity'] = velocity
	# 	cars_plot_data[label]['acceleration'] = acceleration

	# #plotting and saving cars information from each video
	# #plots can be found in folder "figures/"

	# # plt.figure(figsize=(10,8))
	# # for label in cars_labels: 
	# # 	plt.plot(cars_plot_data[label]['x'],cars_plot_data[label]['y'])
	# # plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	# # plt.xlabel('position x')
	# # plt.ylabel('position y')
	# # plt.xlim((0,image.shape[0]))
	# # plt.ylim((0,image.shape[1]))
	# # plt.title('cars trajectories')
	# # plt.savefig('figures/'+img_dir+'_trajectory.png')

	# plt.figure(figsize=(10,8))
	# ax = plt.axes(projection='3d')
	# for label in cars_labels: 
	# 	# Data for a three-dimensional line
	# 	ax.plot3D(cars_plot_data[label]['x'],cars_plot_data[label]['y'], cars_plot_data[label]['time'])
	# ax.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('frames')
	# ax.set_title('cars trajectories')
	# plt.savefig('figures/'+img_dir+'_y_x_t.png')
	# plt.show()

	# plt.figure(figsize=(10,8))
	# plt.subplots_adjust(wspace=0.5)
	# plt.subplot(221)
	# for label in cars_labels: 
	# 	plt.plot(cars_plot_data[label]['time'],cars_plot_data[label]['angle'])
	# plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.xlabel('frame')
	# plt.ylabel('angle (rad)')
	# plt.title('cars angles')


	# plt.subplot(222)
	# for label in cars_labels: 
	# 	plt.plot(cars_plot_data[label]['time'],cars_plot_data[label]['velocity'])
	# plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.xlabel('frame')
	# plt.ylabel('velocity (pixel/frame)')
	# plt.title('cars velocities')


	# plt.subplot(223)
	# for label in cars_labels: 
	# 	plt.plot(cars_plot_data[label]['time'],cars_plot_data[label]['acceleration'])
	# plt.legend(cars_labels,loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.xlabel('frame')
	# plt.ylabel(r'acceleration (pixel/${frame}^2$)')
	# plt.title('cars accelerations')
	# plt.savefig('figures/'+img_dir+'_Info.png')
	# #plt.show()

	# #cv2.waitKey(0)
	cv2.destroyAllWindows()
