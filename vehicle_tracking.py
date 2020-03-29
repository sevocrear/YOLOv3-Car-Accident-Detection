import numpy as np
from numpy import linalg as LA

# this function takes a list of bounding boxes and return list of centroids
def  centroid(box):   
  centroids= []
  for i in box:               
    # i = [x,y,w,h]
    dis_x = np.int32(i[2]/2)
    dis_y = np.int32(i[3]/2)
    centroid = [i[0]+dis_x, i[1]+dis_y]
    centroids.append(centroid)
  return centroids


#function takes center of car from previous frame and list of centers from current frame
# this funtion is used to helping decide which center from current frame belongs to which car
def get_closest_center(old_center, new_centers):
  centers_distance = [[],[]]
  for i in new_centers: 
    motion_vector = np.subtract(i,old_center)                       #to get position difference
    distance = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)   # distance between car center and new center
    normal = LA.norm(motion_vector)
    motion_vector = np.divide(motion_vector,normal)                 #normalizing motion ventor for later use for path angels
    centers_distance[0].append([motion_vector])                     
    centers_distance[1].append(distance)                            
  
  
  min_idx = centers_distance[1].index(min(centers_distance[1]))     #returning index of smallest distance in list

  #returning which center is closest to this car, direction (normalized) vector, distance magnitude, index
  return new_centers[min_idx], centers_distance[0][min_idx], centers_distance[1][min_idx], min_idx


# function to sort new values inside the dictionary 
def update_dict(arrays,new_center,vector,distance ): 
  arrays[0].append(new_center)
  arrays[1].append(vector)
  arrays[2].append(distance)
  
 
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
def BuildAndUpdate(boxes, cars_dict):
  centers = centroid(boxes)
  if len(cars_dict)==0:             #to check if dictionary is empty 
    for i in range(len(centers)):   #takes each center and assign label as a separate car
      car_info = [[]for i in range(4)]
      label = str(i+1)                  #label is only a number 
      car_info[0].append(centers[i])    
      car_info[1].append([0,0])
      car_info[2].append(0)
      car_info[3].append(0)
      cars_dict[label]= car_info
  else:
    cars_labels = list(cars_dict)         #getting list of all labels
    for i in cars_labels:
      if len(centers)>0:
        locations = cars_dict[i][0]
        old_center = locations[len(locations)-1]        #taking position of car in previos frame
        
        #getting the closest point in current frame to this position
        closest_center, vector, distance, idx = get_closest_center(old_center,centers)  
        car_bounds = boxes[idx][2:4]
        if distance <= min(car_bounds):       #applying threshold to closest distance to see if it's close enough
          cars_dict[i]= update_dict(cars_dict[i],closest_center, vector, distance)      #if distance less than threshold the position of car is updated
          del centers[idx]                  #delete center from list
    
    int_labels = []
    for car_label in cars_labels:
      int_labels.append(int(car_label))
   
    if len(centers)> 0:         #checking if there are still not-assigned centers
      for center in centers:
        new_label = str(max(int_labels)+1)        #creating new label for new car
        cars_dict[new_label] = [[center],[[0,0]],[0],[0]]

        

  return cars_dict