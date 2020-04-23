import numpy as np
def check_overlap(first_car,second_car, one_diag, second_diag, k):
  dist = np.sqrt((first_car[0]-second_car[0])**2 + (first_car[1]-second_car[1])**2)
  threshold = one_diag + second_diag
  if (dist < threshold*k):
    check = True
  else:
    check = False  
  return check