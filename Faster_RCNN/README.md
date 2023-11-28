<a href="https://github.com/trian-ctrn/IntrotoCS_2023/blob/master/Faster_RCNN/IntroCS_Faster_RCNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# How to use the Faster R-CNN Notebook:
## Link Google Drive:
1. Create a shortcut of IntroCS_Tree folder to your own Google Drive (https://drive.google.com/drive/u/1/folders/1EbMrlHBU0AREOwSJZF18Q55K0qRG5Uoa)
2. Link your Google Drive with the following 2 blocks (Make sure to modify your directories):
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```
```
%cd /content/gdrive/MyDrive/IntroCS_Tree/Faster_RCNN
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
