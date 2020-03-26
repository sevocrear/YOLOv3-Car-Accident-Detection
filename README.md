# Implementing YOLO technique in order to detect car accidents

In this repository you could find 3 scripts in order to implement YOLOv3 algorithm on videos, images and camera.

At first, clone the repository:
```bash
git clone git@github.com:Terminateit/YOLOv3-Car-Accident-Detection.git
```

then go to the folder yolo-coco in the repository:

```bash
cd YOLOv3-Car-Accident-Detection/yolo-coco
```

Run the bash-script to download weights for models:

```bash
./download_yolov3_weights.sh
```
*(May be, it the upper command will download the axel packet also (it's needed for boosting downloading from the net))*

After that, you can run any of the scripts like:

```bash
python3 yolov3_<aim>.py
```

! If you have some troubles with OpenCV, try to do:

```bash
pip install opencv-python==4.2.0.32
pip install opencv-contrib-python==4.2.0.32
```
