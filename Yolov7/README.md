# How to use the Yolov7 Notebook
<a href="https://github.com/trian-ctrn/IntrotoCS_2023/blob/master/Yolov7/TreeCounting_YOLO.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Mount to google drive
1. Create a IntroCS_Tree folder to your own Google Drive
2. Put the dataset into the working folder
3. Mount your drive to Google Colab by executing two following lines of code
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```
4. Remember to redirect the current working directory, download Yolov7 necessary file and install necessary packages
```
%cd /content/gdrive/MyDrive/IntroCS_Tree/Yolov7
!git clone https://github.com/WongKinYiu/yolov7.git
```
```
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```
```
%cd /content/gdrive/MyDrive/IntroCS_Tree/Yolov7/yolov7/
!pip install -r requirements.txt
!pip install thop
!pip install torchprofile
```
## Training process
1. Before training model, go to folder data and create treecounting.yaml
2. Add these lines of code to set your directories for train_path, val_path, and test_path:
```
train: /content/gdrive/MyDrive/IntroCS_Tree/ver12/train 
val: /content/gdrive/MyDrive/IntroCS_Tree/ver12/val
test: /content/gdrive/MyDrive/IntroCS_Tree/ver12/test

# number of classes
nc: 1

# class names
names: [ 'tree' ]

```
3. Change the epochs and workers depends on how you might want the model to perform
```
!python train.py --batch 16 --epochs 50 --data data/treecounting.yaml --weights 'yolov7.pt' --workers 8 --device 0
```
4. After the model finish, the output will be save to a directory, this directory is different on each time:
```
Optimizer stripped from runs/train/exp27/weights/last.pt, 74.8MB
```
## Detect model 
Remember to change the path of the weights depends on the path of the result on the train model.
```
!python detect.py --weights runs/train/exp27/weights/best.pt --conf 0.25 --img-size 1024 --source /content/gdrive/MyDrive/IntroCS_Tree/ver12/val/images
```
## Test model (Optional)
```
!python3 test.py --weights runs/train/exp27/weights/best.pt --task test --data data/treecounting.yaml --img-size 1024 --conf-thres 0.40
```

