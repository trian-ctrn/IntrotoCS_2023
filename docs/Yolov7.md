# How to use the Yolov7 Notebook
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
1. Before training model, go to Environment set up to modify your directories for train_path and test_path:
```
train_path = ['/content/gdrive/MyDrive/IntroCS_Tree/ver12/train', '/content/gdrive/MyDrive/IntroCS_Tree/ver12/val']
test_path = ['/content/gdrive/MyDrive/IntroCS_Tree/ver12/test']
```
2. Change the epochs depends how you might want the model to perform
```
!python train.py --batch 16 --epochs 50 --data data/treecounting.yaml --weights 'yolov7.pt' --workers 8 --device 0
```

## Test model
1. Run Detect block
```
!python detect.py --weights runs/train/exp27/weights/best.pt --conf 0.25 --img-size 1024 --source /content/gdrive/MyDrive/IntroCS_Tree/ver12/val/images
```
2. Run Test block
```
!python3 test.py --weights runs/train/exp27/weights/best.pt --task test --data data/treecounting.yaml --img-size 1024 --conf-thres 0.40
```
