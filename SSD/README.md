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
    file_list.remove(move_me)
```
3. Create Labelmap and TFRecords as your label file:
```
cat <<EOF >> /content/labelmap.txt
tree
EOF
```
##  Set Up Training Configuration:
1. Chosing model SSD:
```
chosen_model = 'ssd-mobilenet-v2-fpnlite-320'

MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
    },
    'ssd-mobilenet-v2-fpnlite-320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    },
    # The centernet model isn't working as of 9/10/22
    #'centernet-mobilenet-v2': {
    #    'model_name': 'centernet_mobilenetv2fpn_512x512_coco17_od',
    #    'base_pipeline_file': 'pipeline.config',
    #    'pretrained_checkpoint': 'centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz',
    #}
}

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
```
2. Set training parameters for the model. You can change num_steps to number that you want but recommended num steps is from 2000 to 5000:
```
num_steps = 2000

if chosen_model == 'efficientdet-d0':
  batch_size = 4
else:
  batch_size = 16
```
## Train model:
Clicking on the play button and waiting for result. It only displays logs once every 100 steps:
```
!python /content/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1
```
