!/bin/bash

make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights

# copy darknet weight files, continue '-c' if partially downloaded
sudo apt-get install axel
axel https://pjreddie.com/media/files/yolov3.weights
axel https://pjreddie.com/media/files/yolov3-tiny.weights
axel https://pjreddie.com/media/files/yolov3-spp.weights
