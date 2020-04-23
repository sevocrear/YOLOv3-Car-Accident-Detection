import numpy as np
from numpy import linalg as LA

# this function takes a list of bounding boxes and return list of centroids
def  centroid(box):   
  centroids= []
  for i in box:               
    # i = [x,y,w,h]
    dis_x = np.int32(i[2]//2)
    dis_y = np.int32(i[3]//2)
    centroid = [i[0]+dis_x, i[1]+dis_y]
    centroids.append(centroid)
  return centroids

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
  
#function takes center of car from previous frame and list of centers from current frame
# this funtion is used to helping decide which center from current frame belongs to which car
def get_closest_center(old_center, new_centers):
  centers_distance = [[],[]]
  for i in new_centers: 
    motion_vector = np.subtract(i,old_center)                       #to get position difference
    distance = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)   # distance between car center and new center
    motion_vector = np.divide(motion_vector,distance+0.000000001)    #normalizing motion ventor for later use for path angles
    centers_distance[0].append([motion_vector])                     
    centers_distance[1].append(distance)                            
  
  
  min_idx = centers_distance[1].index(min(centers_distance[1]))     #returning index of smallest distance in list

  #returning which center is closest to this car, direction (normalized) vector, distance magnitude, index
  return new_centers[min_idx], centers_distance[0][min_idx], centers_distance[1][min_idx], min_idx


# function to sort new values inside the dictionary 
def update_dict(arrays, car_bounds, new_center,vector,distance, t): 
  arrays[0].append(new_center)
  arrays[1].append(vector)
  arrays[2].append(distance)
  arrays[5] =   [car_bounds[0], car_bounds[1],  int((t-1)*arrays[5][2]/t+car_bounds[2]/t), int((t-1)*arrays[5][3]/t+car_bounds[3]/t)]
  
 
  if len(arrays[2])<3:
    arrays[3].append(0)
  else: 
    accel = arrays[2][len(arrays[2])-2]-distance  # calculate acceleration
    arrays[3].append(accel)

  return arrays


#main function for tracking algorithm 
# Used to either build or update cars data 
# if given dictionary (cars_dict) is empty, it builds a new one
# if cars_dict is not empty, it updates values
# this function calculates for each car path points(centers), direction vector, velocity, acceleration 
def BuildAndUpdate(boxes, cars_dict, update_times):
  # boxes[i] = [x, y, int(width), int(height), color]
  centers = centroid(boxes)
  if len(cars_dict)==0:             #to check if dictionary is empty 
    for i in range(len(centers)):   #takes each center and assign label as a separate car
      car_info = [[]for i in range(6)]
      label = str(i+1)                  #label is only a number 
      car_info[0].append(centers[i])    # center position 
      car_info[1].append([[0,0]]) # position vector 
      car_info[2].append(0) # velocity
      car_info[3].append(0) # acceleration
      car_info[4] = (boxes[i][4]) # color of the car
      car_info[5] = boxes[i][0:4] # x,y,w,h
      cars_dict[label]= car_info
  else:
    cars_labels = list(cars_dict)         #getting list of all labels
    for i in cars_labels:
      if len(centers)>0:
        locations = cars_dict[i][0] # (x,y) - coordinates of the centroid
        
        old_center = locations[len(locations)-1]        #taking position of car in previous frame
        
        #getting the closest point in current frame to this position
        closest_center, vector, distance, idx = get_closest_center(old_center,centers)  
        car_bounds = boxes[idx][0:4]
        if distance <= min(car_bounds[2:4]):       #applying threshold to closest distance to see if it's close enough
          cars_dict[i]= update_dict(cars_dict[i],car_bounds, closest_center, vector, distance, update_times)      #if distance less than threshold the position of car is updated
          update_times += 1
          del centers[idx]                  #delete center from list
          del boxes[idx]
    int_labels = []
    for car_label in cars_labels:
      int_labels.append(int(car_label))
   
    if len(centers)> 0:         #checking if there are still not-assigned centers
      for center in centers:
        new_label = str(max(int_labels)+1)        #creating new label for new car
        r = np.random.choice(255)
        g = np.random.choice(255)
        b = np.random.choice(255)
        color = (r,g,b)
        cars_dict[new_label] = [[center],[[[0,0]]],[0],[0], color, boxes[centers.index(center)][0:4]]
  return cars_dict



from scipy.interpolate import interp1d

class Path():
    """
    interpolation methods: ['original', 'slinear', 'quadratic', 'cubic']
    """
    def __init__(self,data):
        self.data = data

    def interpolate(self, label, number=100, method ='slinear'):
      x = np.array(self.data[label]['x'])
      y = np.array(self.data[label]['y'])
      time = np.array(self.data[label]['time'])
      angle = np.array(self.data[label]['angle'])
      acceleration = np.array(self.data[label]['velocity'])
      velocity = np.array(self.data[label]['acceleration'])
      self.points = np.stack((x, y, time, angle, velocity, acceleration), axis = 1)

      if method == 'original':
        return self.points

      # Calculate the linear length along the line:
      distance = np.cumsum(np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1)))
      distance = np.insert(distance, 0, 0)/distance[-1]
      
      # Interpolation itself:
      alpha = np.linspace(0, 1, number)
      if len(distance) <3:
        method = 'spline'
      interpolator =  interp1d(distance, self.points, kind=method, axis=0)
      interp_points = interpolator(alpha)
      return interp_points