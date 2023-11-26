# How to use the Faster R-CNN Notebook:
## Stop Colab from disconnecting:
1. Connect your notebook to Runtime (Preferably T4 GPU)
2. Paste the following code into the web console (Ctrl + Shift + I):
```
function ConnectButton(){
  console.log("Connect pushed");
  document.querySelector("#top-toolbar > colab-connectbutton").shadowRoot.querySelector("#connect").click()
}
setInterval(ConnectButton,60000);
```
## Set up:
1. Link your Google Drive with the following 2 blocks (Make sure to modify your directories):
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```
```
%cd /content/gdrive/MyDrive/IntroCS_Tree/Faster_RCNN
```
2. Install necessary libraries:
```
!pip install pycocotools --quiet
!pip install torchmetrics
!pip install git+https://github.com/albumentations-team/albumentations.git
```
3. Import the libraries
4. To prepare the dataset, remember to modify your directories for train_dir and test_dir
5. Run the next 2 blocks to define some necessary functions
6. In the get_object_detection_model function, you can choose to pretrain or not
## Train and get result:
1. Run a few more blocks to prepare for training
2. In the training block, you can adjust num_epochs and then start training
3. Wait for the training to complete
4. Run the last few blocks to generate the detection results
## Save and import model
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
