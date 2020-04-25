import numpy as np
import matplotlib.pyplot as plt

def check_angle_anomaly(angle_list_1st,frame,check_frames):
  
  #iterating and calculating numerical second derivative for list of angles
  angle_change = angle_list_1st[frame-check_frames: frame+check_frames]
  if len(angle_change)>0:
    diff = max(angle_change)-min(angle_change)
  else:
    diff = 0
  
  return diff

#checks crash angle at some frame
def check_crash_angle(angle_1st_car,angle_2nd_car,threshold):
  if (angle_1st_car-angle_2nd_car) > threshold:
    check=True
  else:
    check = False
  return check
  
def check_overlap(first_car,second_car, one_diag, second_diag):
  dist_x = np.abs(first_car[0]-second_car[0])
  dist_y = np.abs(first_car[1]-second_car[1])
  diag_x = one_diag[0]+second_diag[0]
  diag_y = one_diag[1] + second_diag[1]
  if dist_x<diag_x and dist_y<diag_y:
    check = 1
  else:
    check = 0 
  return check

def check_path_variance(cars_data,cars_labels,  threshold):  
    cars_labels_to_analyze = []
    for label in cars_labels: 
        # Data for a three-dimensional line
        variance = np.abs(np.var(cars_data[label]['x'])) + np.abs(np.var(cars_data[label]['y']))
        ratio = variance/(np.sqrt(cars_data[label]['car diagonal'][0]**2+cars_data[label]['car diagonal'][1]**2))
        if ratio < threshold:
            del cars_data[label]
            cars_labels.remove(label)
        else:	
            cars_labels_to_analyze.append(label)         
    return cars_labels_to_analyze    

def plot2D_graphs(cars_data, frames, potential_cars_labels, frame_overlapped,  frame_overlapped_interval, img_dir, actual_frame, show):    
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
    plt.savefig('figures/'+img_dir+'frame_'+str(actual_frame)+'_Info.png')
    if show == 'Yes':
      plt.show()

def plot3D_graph(cars_data,frames, potential_cars_labels, W,H, frame_overlapped, frame_overlapped_interval, img_dir, actual_frame, show):
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
    plt.savefig('figures/'+img_dir+'frame_'+str(actual_frame)+'_y_x_t.png')
    if show == 'Yes':
      plt.show()  
#
