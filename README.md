# Implementing YOLO technique in order to detect car accidents and to Detect an invasion of private territory

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

* yolov3_camera_SaFe_Territory.py - detect people and cars on the camera frames 
* yolov3_camera.py - detect any objects on the camera frames
* yolov3_detect.py - detect objects on the camera frames for RealSense Camera
* yolov3_frames.py - detect car accidents on the video frames (Dataset is given)
* yolov3_image.py - detect objects on the image (put to input)
* yolov3_video.py - detect objects on the video frames

! If you have some troubles with OpenCV, try to do:

```bash
pip3 install opencv-python==4.2.0.32 && pip3 install opencv-contrib-python==4.2.0.32
```
