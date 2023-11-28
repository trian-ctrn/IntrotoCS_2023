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
## Training phase:
1. To prepare the dataset, remember to modify your directories for train_dir and test_dir:
```
train_dir = ['/content/gdrive/MyDrive/IntroCS_Tree/ver12/train', '/content/gdrive/MyDrive/IntroCS_Tree/ver12/val']
test_dir = ['/content/gdrive/MyDrive/IntroCS_Tree/ver12/test']
```
2. In the get_object_detection_model function, you can choose to pretrain or not:
```
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```
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
