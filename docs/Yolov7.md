# How to use the Yolov7 Notebook:
## Download Yolov7 necessary files
```
!git clone https://github.com/WongKinYiu/yolov7.git
```
```
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```
## Mount to google drive
1. Create a shortcut of IntroCS_Tree folder to your own Google Drive
2. Mount your drive to Google Colab by executing two following lines of code
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```
3. Remember to change the directory to /IntroCS_Tree/YOLO/yolov7 folder and install necessary packages
```
%cd /content/gdrive/MyDrive/IntroCS_Tree/YOLO/yolov7
!pip install thop
!pip install torchprofile
!pip install -r requirements.txt
```
## Training process


