# !/bin/bash
cd ..
echo "Creating needed folders ...\n"
mkdir -p figures output
axel -a https://drive.google.com/drive/folders/1mg4_nvXmSbLX0uSLD7YPrN6GtnCEJVv-?usp=sharing
cd yolo-coco
# make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights
echo "Copying weight files into yolo-coco/weights folder... It will take time ...\n"
# copy darknet weight files, continue '-c' if partially downloaded
sudo apt-get install axel
axel -a https://pjreddie.com/media/files/yolov3.weights
axel -a https://pjreddie.com/media/files/yolov3-tiny.weights
axel -a https://pjreddie.com/media/files/yolov3-spp.weights
