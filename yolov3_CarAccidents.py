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
from accident_detecting import *
import yolo
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

path_of_file = os.path.abspath(__file__)
os.chdir(os.path.dirname(path_of_file))

''' Params for yolov3 '''
thr_param = 0.3 # threshold for YOLO detection
conf_param = 0.5 # confidence for YOLO detection

W_show, H_show = 1152, 768 # Shape of images to show

''' Params for cars detection '''
frame_start_with =45 # frame to start with
frame_end_with = 80 # number of image frames to work with in each folder (dataset)

filter_flag = 1 # use moving averaging filter or not (1-On, 0 - Off)
len_on_filter = 2 # minimum length of the data list to apply filter on it

T_var = 2 # threshold in order to show only those cars that is moving... (0.5 is okay)

frame_overlapped_interval = 10 # the interval (- frame_overlapped_interval + frame; frame + frame_overlapped_interval) to analyze if there were accident or not

angle_threshold = 1 #threshold to detect crash angle
trajectory_thresold = 0.1 #threshold to detect change in path direction

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

# Initialize made by us yolov3 object:
yolov3 = yolo.YOLOv3(thr_param, conf_param, [2,3,5,6,7], net, 'tracking')

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

	# Start implementing YOLOv3 through files (frames)
	update_times = 1
	cars_dict = {}
	counter = 0
	images_saved = []
	for f1 in files:
		image = cv2.imread(f1)
		images_saved.append(image)	
		# print('Processing frame:'+str(counter+frame_start_with)+'/'+str(frame_end_with),'in folder:'+folders[0]+'...')

		if type(image) is np.ndarray:	
			time_start = time.time()
			(H, W) = image.shape[:2]
			new_boxes, new_boxes_id, image = yolov3.detect_objects(image)
					
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
					1, (255,255,200), 2)			

		time_end = time.time()
		# print('FPS = ', 1/ (time_end - time_start))


		if files.index(f1)%(2*frame_overlapped_interval) == 0 and files.index(f1) != 0:
			#building dictionary of cars data
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

				# Used condition on length of the list in order not to use filter with very small amount of data.

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
				cars_data[label]['car diagonal'] = np.sqrt(w**2+h**2)/2
			
			path = Path(cars_data)

			# Interpolate data for each car (it's needed because YOLO didn't detect car in 
			# each frame. So,we need to fill empty space)
			cars_labels_to_analyze = check_path_variance(cars_data,cars_labels, T_var)
			for label in cars_labels_to_analyze:
				interp_points = path.interpolate(label, number = 2*frame_overlapped_interval, method = 'cubic')
				cars_data[label]['x'] = interp_points[:,0]
				cars_data[label]['y'] = interp_points[:,1]
				cars_data[label]['time'] = [int(i) for i in range(2*frame_overlapped_interval)]
				cars_data[label]['angle'] = interp_points[:,3]
				cars_data[label]['velocity'] = interp_points[:,4]
				cars_data[label]['acceleration'] = interp_points[:,5]

			#------Checking vehicle overlapps--------#
			overlapped = {}
			flag = 1
			frames = [int(i) for i in range(2*frame_overlapped_interval)]
			accident_frames = set()
			for frame in frames:
				for first_car in cars_labels_to_analyze:
					for second_car in cars_labels_to_analyze:
						if (int(second_car) != int(first_car)):
							check, intersection = check_overlap((cars_data[first_car]['x'][frame],cars_data[first_car]['y'][frame]),(cars_data[second_car]['x'][frame],cars_data[second_car]['y'][frame]), cars_data[first_car]['car diagonal'], cars_data[second_car]['car diagonal'])
							if check and (overlapped.get(second_car) == None) and (overlapped.get(first_car) == None):
								overlapped[second_car] = [intersection,0,0, frame]
								overlapped[first_car] = [intersection,0,0, frame]
								flag = 0
								accident_frames.add(frame)

			if not flag:					
				# print('labels of overlapped cars:', list(overlapped),'. Frames of potential accidents:', accident_frames)
				potential_cars_labels = [label for label in list(overlapped)]
					#------Checking acceleration anomaly--------#
				'''When two vehicles are overlapping, we find the acceleration of the vehicles from their speeds captured in the
				dictionary. We find the average acceleration of the vehicles
				for N frames before the overlapping condition and the
				maximum acceleration of the vehicles N frames after it.
				We find the change in accelerations of the individual vehicles
				by taking the difference of the maximum acceleration and
				average acceleration during overlapping condition'''
				for frame_overlapped in accident_frames:
					flag = 0
					potential_cars_labels_in_frame = []
					for label in potential_cars_labels:
						if overlapped[label][3] == frame_overlapped:
							potential_cars_labels_in_frame.append(label)
							flag = 1
					if not flag:
						continue
					if frame_overlapped-frame_overlapped_interval< 0:
						minus = frame_overlapped-frame_overlapped_interval
						frames_before = [int(i) for i in range(frame_overlapped-(frame_overlapped_interval+minus), frame_overlapped)]
					else:	
						frames_before = [int(i) for i in range(frame_overlapped-frame_overlapped_interval, frame_overlapped)]

					acc_average = []
					for label in potential_cars_labels_in_frame:
						acc_av = 0
						t = 1
						for frame in frames_before:
							acc_av = acc_av*(t-1)/t + cars_data[label]['acceleration'][frame]/t
							t += 1
						acc_average.append(acc_av)
					frames_after = [int(i) for i in range(frame_overlapped, frame_overlapped+frame_overlapped_interval)]	
					acc_maximum = []
					for label in potential_cars_labels_in_frame:
						acc_max = 0
						for frame in frames_after:
							if frame<2*frame_overlapped_interval:
								if cars_data[label]['acceleration'][frame]>acc_max:
									acc_max = cars_data[label]['acceleration'][frame]
						acc_maximum.append(acc_max)

					acc_diff = np.subtract(acc_maximum, acc_average)
					for label in potential_cars_labels_in_frame:
						overlapped[label][1] = np.abs(acc_diff[potential_cars_labels_in_frame.index(label)])


					#----Angle Anomalies----#
					angle_anomalies = []
					for label in potential_cars_labels_in_frame:
						
						angle_difference = check_angle_anomaly(cars_data[label]['angle'],frame_overlapped,frame_overlapped_interval)
						overlapped[label][2] = angle_difference

						# angle_anomalies.append(angle_difference)

					# if len(angle_anomalies)>0:	
					# 	max_angle_change = max(angle_anomalies)
					# 	# print('change in angle :', max_angle_change)
					# 	if max_angle_change >= trajectory_thresold:
					# 		checks_anom = 1
					# 	else:
					# 		checks_anom = 0.5
					# else:
					# 	checks_anom = 0.5


					# #----Checkings----#


					for label in potential_cars_labels_in_frame:
						image = images_saved[frame_overlapped]
						# print('label', label, '\n')
						print(overlapped[label], 'frame = ', counter+frame_overlapped + frame_start_with- 2*frame_overlapped_interval)
						overlap = overlapped[label][0]
						acc_anomaly = overlapped[label][1]
						angle_anomaly = overlapped[label][2]
						print('sum = ', overlap*0.7 + acc_anomaly*0.27+angle_anomaly*0.9)
						# if (overlap*0.7 + acc_anomaly*0.4+check_anom)>=2:
						if (overlap*0.7 + acc_anomaly*0.25+angle_anomaly*0.9)>=2:
							print('accident happened at frame ',counter -2*frame_overlapped_interval+ frame_start_with + frame_overlapped,' with car ', label)
							cv2.circle(image, (int(cars_data[label]['x'][frame_overlapped]), int(cars_data[label]['y'][frame_overlapped])), 30,  (255,255,0), 2)
							#saving output image in folder output/
							cv2.imwrite('cars/'+img_dir+'_accident_in_frame_'+str(counter -2*frame_overlapped_interval+ frame_start_with + frame_overlapped)+'.png', image)		

							#-----# Plots			
							plot3D_graph(cars_data,frames, potential_cars_labels_in_frame, W,H, frame_overlapped, frame_overlapped_interval, img_dir, counter -2*frame_overlapped_interval+ frame_start_with + frame_overlapped)

							plot2D_graphs(cars_data, frames, potential_cars_labels_in_frame, frame_overlapped,  frame_overlapped_interval, img_dir, counter -2*frame_overlapped_interval+ frame_start_with + frame_overlapped)

				update_times = 1
				cars_dict = {}
				images_saved = []

			elif flag:
				print('There weren\'nt any overlapping cars this time... Let\'s check further...')
				update_times = 1
				cars_dict = {}
				images_saved = []

		counter +=1	

		image = cv2.resize(image,(W_show, H_show))
		cv2.imshow("Image", image)
		if cv2.waitKey(1) == 27:
			break