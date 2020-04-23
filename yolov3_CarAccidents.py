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
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def check_odd_filter(x):
	# It's function used for window and poly order calculation
	# for moving averaging filter

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


path_of_file = os.path.abspath(__file__)
os.chdir(os.path.dirname(path_of_file))

thr_param = 0.3 # threshold for YOLO detection
conf_param = 0.5 # confidence for YOLO detection
frame_start_with = 140
frame_end_with = 170 # number of image frames to work with in each folder (dataset)

filter_flag = 1 # use moving averaging filter or not (1-On, 0 - Off)
len_on_filter = 2 # minimum length of the data list to apply filter on it

T_var = 200 # threshold in order to show only those cars that is moving... (50 is okay)

k_overlap = 0.25 # ratio (0-1) for overlapping issue | thresh_overlap = distance_between*k_overlap
frame_overlapped_interval = 20 # the interval (- frame_overlapped_interval + frame; frame + frame_overlapped_interval) to analyze if there were accident or not

T_acc = 1.5 # theshold in order to detect acceleration anomaly



data_dir = "Dataset/" 	#dataset directory
dataset_path = glob.glob(data_dir+"*/") 		#reading all sub-directories in folder
print('Sub-directories',dataset_path)

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

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
	for q in range(frame_start_with, frame_end_with): # Loop through certain number of video frames in the folder
		path = folders[0]+'/'+str(q)+'.jpg'
		files.append(path)

	
	update_times = 1
	cars_dict = {}
	counter = 1
	images_saved = []
	for f1 in files:
		image = cv2.imread(f1)
		images_saved.append(image)
		print('Processing frame:'+str(counter)+'/'+str(frame_counter),'in folder:'+folders[0]+'...')

		if type(image) is np.ndarray:	
			time_start = time.time()

			# load our input image and grab its spatial dimensions
			(H, W) = image.shape[:2]

			# determine only the *output* layer names that we need from YOLO
			ln = net.getLayerNames()
			ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 192), # (96, 96) \ (192, 192) \ (256, 256) \ (384, 384)
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

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
					if confidence > conf_param and (classID == 2 or classID == 3 or classID == 5 or classID == 7):
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
					r = np.random.choice(255)
					g = np.random.choice(255)
					b = np.random.choice(255)
					color = (r,g,b)
					boxes[i].append(color)
					new_boxes.append(boxes[i]) 
					
					
				# building cars data
				cars_dict = BuildAndUpdate(new_boxes, cars_dict, update_times)
				cars_labels = list(cars_dict)

				for car_label in cars_labels:
					car_path = cars_dict[car_label][0]
					# plotting car path on image and printing car label
					if len(car_path)> 1:
						car_path = np.asarray(car_path,dtype=np.int32)
						car_path = car_path.reshape((-1,1,2))
						cv2.polylines(image,car_path,True, cars_dict[car_label][4],3)
						label_location = car_path[len(car_path)-1][0]
						cv2.putText(image, car_label, (label_location[0]+5, label_location[1]+5), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, cars_dict[car_label][4], 2)
						cv2.circle(image, (label_location[0], label_location[1]), 4, cars_dict[car_label][4],2)
						x = cars_dict[car_label][5][0]
						y = cars_dict[car_label][5][1]
						w = cars_dict[car_label][5][2]
						h = cars_dict[car_label][5][3]
						cv2.rectangle(image, (x, y), (x+w, y+h),cars_dict[car_label][4], 2)

			cv2.putText(image, 'frame ' + str(frame_start_with+counter), (20, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
						3, (255,255,200), 2)			

			time_end = time.time()
			print('FPS = ', 1/ (time_end - time_start))
			# show the output image
			cv2.imshow("Image", image)
			if cv2.waitKey(1) == 27:
				break
			counter +=1

	#saving output image in folder output/
	cv2.imwrite('output/'+img_dir+'_final_frame.png', image)
	cv2.destroyAllWindows()
	#building dictionary of plot data
	cars_data = {}
	for label in cars_labels: 
		position  = cars_dict[label][0]
		direction = cars_dict[label][1]
		velocity = cars_dict[label][2]
		acceleration = cars_dict[label][3]
		w,h = cars_dict[label][5][2:4]
		x_pos = []
		y_pos = []
		angle = []
		time_frame = []
		cars_data[label]={}
		for i in range(len(position)):
			x_pos.append(position[i][0])
			y_pos.append(position[i][1])
			angle.append(np.arccos(direction[i][0][0]))
			time_frame.append(i)

		# Used condition on length of the list in order not tu use filter with very small amount of data.

		if filter_flag:
			if len(x_pos) > len_on_filter:
				window_size, polyorder = check_odd_filter(len(x_pos))
				x_pos = savgol_filter(x_pos, window_size, polyorder)
			if len(y_pos) > len_on_filter:
				window_size, polyorder = check_odd_filter(len(y_pos))
				y_pos = savgol_filter(y_pos, window_size, polyorder)
			if len(angle) > len_on_filter:
				window_size, polyorder = check_odd_filter(len(angle))
				angle = savgol_filter(angle, window_size, polyorder)
			if len(velocity) > len_on_filter:
				window_size, polyorder = check_odd_filter(len(velocity))
				velocity = savgol_filter(velocity, window_size, polyorder)
			if len(acceleration) > len_on_filter:
				window_size, polyorder = check_odd_filter(len(acceleration))
				acceleration = savgol_filter(acceleration, window_size, polyorder)

		cars_data[label]['x'] = x_pos
		cars_data[label]['y'] = y_pos
		cars_data[label]['time'] = time_frame
		cars_data[label]['angle'] = angle
		cars_data[label]['velocity'] = velocity
		cars_data[label]['acceleration'] = acceleration
		cars_data[label]['car diagonal'] = np.sqrt(w**2+h**2)




	#----------------------------------------------------------------------#
	

	####						DETECTION PART 							####


	#----------------------------------------------------------------------#

	# Exlude cars that doesn't move at all (so, there wasn't any accident or smth with them)
	cars_labels_to_analyze = []
	for label in cars_labels: 
		# Data for a three-dimensional line
		print('np.var',np.var(np.sqrt(np.power(cars_data[label]['x'],2)+np.power(cars_data[label]['y'],2))))
		if np.var(np.sqrt(np.power(cars_data[label]['x'],2)+np.power(cars_data[label]['y'],2))) <T_var:
			del cars_data[label]
		else:	
			cars_labels_to_analyze.append(label)

	path = Path(cars_data)

	# Interpolate data for each car (it's needed because YOLO didn't detect car in 
	# each frame. So,we need to fill empty space)
	for label in cars_labels_to_analyze:
		interp_points = path.interpolate(label, number = frame_end_with - frame_start_with, method = 'cubic')
		cars_data[label]['x'] = interp_points[:,0]
		cars_data[label]['y'] = interp_points[:,1]
		cars_data[label]['time'] = interp_points[:,2]
		cars_data[label]['angle'] = interp_points[:,3]
		cars_data[label]['velocity'] = interp_points[:,4]
		cars_data[label]['acceleration'] = interp_points[:,5]

	checks = [0,0,0]

	#------Checking vehicle overlapps--------#
	overlapped = set()
	flag = 1
	frames = [int(i) for i in range(frame_end_with - frame_start_with)]
	for frame in frames:
		for first_car in cars_labels_to_analyze:
			for second_car in cars_labels_to_analyze:
				if (int(second_car) != int(first_car)) and (check_overlap((cars_data[first_car]['x'][frame],cars_data[first_car]['y'][frame]),(cars_data[second_car]['x'][frame],cars_data[second_car]['y'][frame]), cars_data[first_car]['car diagonal'], cars_data[second_car]['car diagonal'],k_overlap)):
					overlapped.add(first_car)
					overlapped.add(second_car)
					if flag:
						frame_overlapped = frame
						flag = 0	
	if not flag:					
		print('labels of overlapped cars:', overlapped,'. Frame of potential accident:', frame_start_with + frame_overlapped)
		checks[0] = 1
	else:
		print('There weren\'nt any accidents this time... Happily...')
		exit()
	potential_cars_labels = [label for label in overlapped]
				
	#------Checking acceleration anomaly--------#
	'''When two vehicles are overlapping, we find the acceleration of the vehicles from their speeds captured in the
	dictionary. We find the average acceleration of the vehicles
	for N frames before the overlapping condition and the
	maximum acceleration of the vehicles N frames after it.
	We find the change in accelerations of the individual vehicles
	by taking the difference of the maximum acceleration and
	average acceleration during overlapping condition'''

	frames_before = [int(i) for i in range(frame_overlapped-frame_overlapped_interval, frame_overlapped)]

	acc_average = []
	for label in potential_cars_labels:
		acc_av = 0
		t = 1
		for frame in frames_before:
			acc_av = acc_av*(t-1)/t + cars_data[label]['acceleration'][frame]/t
			t += 1
		acc_average.append(acc_av)
	acc_maximum = []
	for label in potential_cars_labels:
		acc_max = 0
		for frame in frames_before:
			if cars_data[label]['acceleration'][frame]>acc_max:
				acc_max = cars_data[label]['acceleration'][frame]
		acc_maximum.append(acc_max)

	acc_diff = np.mean(np.subtract(acc_maximum, acc_average))

	if acc_diff > T_acc:
		checks[1] = 1
	else:
		checks[0] = 0.5

	#----Angle Anomalies----#


	#----Checkings----#
	if (checks[0]+checks[1] + checks[2])>1.5:
		image = images_saved[frame_overlapped]
		print('accident happened at frame ',frame_overlapped,' between cars ', overlapped)
		for car_label in potential_cars_labels:
			cv2.circle(image, (int(cars_data[car_label]['x'][frame_overlapped]), int(cars_data[car_label]['y'][frame_overlapped])), 50,  (255,255,0), 2)
		#saving output image in folder output/
		cv2.imwrite('cars/accident.png', image)			


	#-----# Plots			
	fig = plt.figure()
	ax = fig.gca(projection='3d')	
	for label in potential_cars_labels: 	
		ax.plot(cars_data[label]['x'],cars_data[label]['y'], frames, label = label)
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	X = np.arange(0, W, W//10)
	Y = np.arange(0, W, W//10)
	X, Y = np.meshgrid(X, Y)
	Z = np.full((10,1),frame_overlapped)
	ax.plot_wireframe(X, Y, Z)
	Z = np.full((10,1),frame_overlapped - frame_overlapped_interval)
	ax.plot_wireframe(X, Y, Z)	
	Z = np.full((10,1),frame_overlapped + frame_overlapped_interval)
	ax.plot_wireframe(X, Y, Z)				   			   
	ax.set_zlabel('frames')
	ax.set_title('cars trajectories')
	plt.savefig('figures/'+img_dir+'_y_x_t.png')

	plt.show()

	plt.figure(figsize=(10,8))
	plt.subplots_adjust(wspace=0.5)
	plt.subplot(221)
	for label in potential_cars_labels: 
		plt.plot(frames,cars_data[label]['angle'], label = label)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('frame')
	plt.grid()
	plt.axvline(x=frame_overlapped - frame_overlapped_interval, color='r', linestyle='--')
	plt.axvline(x=frame_overlapped, color='k', linestyle='--')
	plt.axvline(x=frame_overlapped + frame_overlapped_interval, color='r', linestyle='--')
	plt.ylabel('angle (rad)')
	plt.title('cars angles')


	plt.subplot(222)
	for label in potential_cars_labels: 
		plt.plot(frames,cars_data[label]['velocity'], label = label)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('frame')
	plt.grid()
	plt.axvline(x=frame_overlapped - frame_overlapped_interval, color='r', linestyle='--')
	plt.axvline(x=frame_overlapped, color='k', linestyle='--')
	plt.axvline(x=frame_overlapped + frame_overlapped_interval, color='r', linestyle='--')
	plt.ylabel('velocity (pixel/frame)')
	plt.title('cars velocities')


	plt.subplot(223)
	for label in potential_cars_labels: 
		plt.plot(frames,cars_data[label]['acceleration'], label = label)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlabel('frame')
	plt.ylabel(r'acceleration (pixel/${frame}^2$)')
	plt.axvline(x=frame_overlapped - frame_overlapped_interval, color='r', linestyle='--')
	plt.axvline(x=frame_overlapped, color='k', linestyle='--')
	plt.axvline(x=frame_overlapped + frame_overlapped_interval, color='r', linestyle='--')
	plt.title('cars accelerations')
	plt.grid()
	plt.savefig('figures/'+img_dir+'_Info.png')
	plt.show()

