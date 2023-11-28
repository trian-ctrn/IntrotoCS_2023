# SSD Object Detection model 

<a href="https://github.com/trian-ctrn/IntrotoCS_2023/blob/master/SSD/SSD_train.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Introduction
SSD, or Single Shot MultiBox Detector, is an efficient method for executing deep learning models on devices with limited resources. SSD models are faster and require less computational power than standard TensorFlow models, making them ideal for real-time applications.

This tutorial provides a detailed walkthrough on how to train a custom SSD object detection model, convert it into an optimized format for use with SSD,. It also includes Python code for using SSD models to perform detection tasks on images, videos.
### Using Google Colab

The easiest way to train, convert, and export a SSD object detection model is using Google Colab. Colab provides you with a free GPU-enabled virtual machine on Google's servers that comes pre-installed with the libraries and packages needed for training.

The Google Colab notebook that can be used to train custom SSD models. It goes through the process of preparing data, configuring a model for training, training the model, running it on test images, and exporting it to a downloadable TFLite format. It makes training a custom SSD model as easy as uploading an image dataset and clicking Play on a few blocks of code!

Open the Colab notebook in your browser. Work through the instructions in the notebook to start training model.

# How to use the SSD Notebook:
## Install TensorFlow Object Detection Dependencies:
1. Clone the tensorflow models repository from GitHub
```
pip uninstall Cython -y 
!git clone --depth 1 https://github.com/tensorflow/models
```
2.Install neccesary libraries
```
!pip install pyyaml==5.3
!pip install /content/models/research/
!pip install tensorflow==2.8.0
```
## Clone project and install libraries:
1. Clone project to local environment:
```
git clone https://github.com/trian-ctrn/IntrotoCS_2023.git
```
2. Install necessary libraries:
```
!pip install pycocotools --quiet
!pip install torchmetrics
!pip install git+https://github.com/albumentations-team/albumentations.git
```
## Upload Image Dataset and Prepare Training Data:
1. Upload the dataset to google drive, remember to modify your directories to the data set. The data set must contain all pictues and its labels in xml format:
```
from google.colab import drive
drive.mount('/content/gdrive')

!cp /content/gdrive/MyDrive/images.zip /content
```
2. Split images into train, validation, and test folders. You can change the ration of train, validation, test images by changing, train_percent, val_percent, test_percent:
```
import glob
from pathlib import Path
import random
import os

# Define paths to image folders
image_path = '/content/images/all'
train_path = '/content/images/train'
val_path = '/content/images/validation'
test_path = '/content/images/test'

# Get list of all images
jpg_file_list = [path for path in Path(image_path).rglob('*.jpg')]
JPG_file_list = [path for path in Path(image_path).rglob('*.JPG')]
png_file_list = [path for path in Path(image_path).rglob('*.png')]
bmp_file_list = [path for path in Path(image_path).rglob('*.bmp')]

file_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list
file_num = len(file_list)
print('Total images: %d' % file_num)

# Determine number of files to move to each folder
train_percent = 0.9  # 80% of the files go to train
val_percent = 0.1 # 10% go to validation
test_percent = 0 # 10% go to test
train_num = int(file_num*train_percent)
val_num = int(file_num*val_percent)
test_num = file_num - train_num - val_num
print('Images moving to train: %d' % train_num)
print('Images moving to validation: %d' % val_num)
print('Images moving to test: %d' % test_num)

# Select 80% of files randomly and move them to train folder
for i in range(train_num):
    move_me = random.choice(file_list)
    fn = move_me.name
    base_fn = move_me.stem
    parent_path = move_me.parent
    xml_fn = base_fn + '.xml'
    os.rename(move_me, train_path+'/'+fn)
    os.rename(os.path.join(parent_path,xml_fn),os.path.join(train_path,xml_fn))
    file_list.remove(move_me)

# Select 10% of remaining files and move them to validation folder
for i in range(val_num):
    move_me = random.choice(file_list)
    fn = move_me.name
    base_fn = move_me.stem
    parent_path = move_me.parent
    xml_fn = base_fn + '.xml'
    os.rename(move_me, val_path+'/'+fn)
    os.rename(os.path.join(parent_path,xml_fn),os.path.join(val_path,xml_fn))
    file_list.remove(move_me)

# Move remaining files to test folder
for i in range(test_num):
    move_me = random.choice(file_list)
    fn = move_me.name
    base_fn = move_me.stem
    parent_path = move_me.parent
    xml_fn = base_fn + '.xml'
    os.rename(move_me, test_path+'/'+fn)
    os.rename(os.path.join(parent_path,xml_fn),os.path.join(test_path,xml_fn))
    file_list.remove(move_me)```
3. If you have a strong processor, you can increase num_workers:
```
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
```
4. In the training block, you can adjust num_epochs and then start training:
```
num_epochs = 100

for epoch in range(num_epochs):
    print(f"Training epoch: {epoch + 1}/{num_epochs}")
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_val, device=device)

!nvidia-smi
```
5. To view results, you can uncomment these code:
```
print('EXPECTED OUTPUT\n')
plot_img_bbox(torch_to_pil(img), target)
print('MODEL OUTPUT\n')
plot_img_bbox(torch_to_pil(img), nms_prediction)
```
## Save and import model:
1. Save model:
```
files_dir = r"/content/gdrive/MyDrive/IntroCS_Tree/Faster_RCNN/weights"
version = len([ver for ver in sorted(os.listdir(files_dir)) if ver[-3:] == ".pt"])
torch.save(model.state_dict(), f"{files_dir}/test({version}).pt")
print(f"Saved to test({version}).pt")
```
2. Load model (Change your model directory):
```
num_classes = 2
model = get_object_detection_model(num_classes)
model.load_state_dict(torch.load("/content/gdrive/MyDrive/IntroCS_Tree/Faster_RCNN/weights/test(15).pt"))
model.to(device)
model.eval()
```
