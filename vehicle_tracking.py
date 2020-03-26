import numpy as np

def  centroid(box):
  #box = [x,y,w,h]
  centroids= []
  for i in box:
    dis_x = np.int32(i[2]/2)
    dis_y = np.int32(i[3]/2)
    centroid = [i[0]+dis_x, i[1]+dis_y]
    centroids.append(centroid)
  return centroids


def get_closest_center(old_center, new_centers):
  centers_distance = [[],[]]
  for i in new_centers: 
    motion_vector = np.subtract(i,old_center)
    distance = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
    centers_distance[0].append([motion_vector])
    centers_distance[1].append(distance)
  
  
  min_idx = centers_distance[1].index(min(centers_distance[1]))

  return new_centers[min_idx], centers_distance[0][min_idx], centers_distance[1][min_idx], min_idx



def update_dict(arrays,new_center,vector,distance ): 
  arrays[0].append(new_center)
  arrays[1].append(vector)
  arrays[2].append(distance)
  
 
  if len(arrays[2])<3:
    arrays[3].append(0)
  else: 
    accel = arrays[2][len(arrays[2])-2]-distance
    arrays[3].append(accel)

  return arrays


def BuildAndUpdate(boxes, cars_dict):
  centers = centroid(boxes)
  if len(cars_dict)==0:
    for i in range(len(centers)):
      car_info = [[]for i in range(4)]
      label = str(i+1)
      car_info[0].append(centers[i])
      car_info[1].append([0,0])
      car_info[2].append(0)
      car_info[3].append(0)
      cars_dict[label]= car_info
  else:
    cars_labels = list(cars_dict) 
    for i in cars_labels:
      locations = cars_dict[i][0]
      old_center = locations[len(locations)-1]
      closest_center, vector, distance, idx = get_closest_center(old_center,centers)
      car_bounds = boxes[idx][2:4]
      if distance <= min(car_bounds)/10:
        cars_dict[i]= update_dict(cars_dict[i],closest_center, vector, distance)

  return cars_dict