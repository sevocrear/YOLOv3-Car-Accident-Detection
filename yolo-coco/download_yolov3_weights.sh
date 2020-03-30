# !/bin/bash
cd ..
printf "Creating needed folders ...\n"
mkdir -p Dataset figures output && cd yolo-coco
# make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights
printf "Copying weight files into yolo-coco/weights folder... It will take time ...\n"
# copy darknet weight files, continue '-c' if partially downloaded
sudo apt-get install axel
axel -a https://pjreddie.com/media/files/yolov3.weights
axel -a https://pjreddie.com/media/files/yolov3-tiny.weights
axel -a https://pjreddie.com/media/files/yolov3-spp.weights
