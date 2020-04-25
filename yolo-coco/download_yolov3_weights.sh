# !/bin/bash
# This bash-script downloads weights and dataset. Also, it creates all needed files
cd ..
echo "Creating needed folders ..."
mkdir -p figures output cars
echo "Downloading needed datasets..."
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1TioFKqtfM_VSB6pESLUm5GiUgJLZBTZR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TioFKqtfM_VSB6pESLUm5GiUgJLZBTZR" -O Dataset.zip && rm -rf /tmp/cookies.txt
echo "Unzipping Dataset folder ..."
sudo apt install unzip
unzip Dataset.zip
rm Dataset.zip
cd yolo-coco
# make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights
echo "Copying weight files into yolo-coco/weights folder... It will take time ..."
# copy darknet weight files, continue '-c' if partially downloaded
sudo apt-get install axel
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=18PtBISXPhT9CLKy0i5gc8YmrH3Fl1KMj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18PtBISXPhT9CLKy0i5gc8YmrH3Fl1KMj" -O yolov3.weights && rm -rf /tmp/cookies.txt
