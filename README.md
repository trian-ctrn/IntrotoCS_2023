# IntrotoCS_2023
This project is part of the Introduction to Computer Science 2023 course at the Vietnamese German University. It involves implementing various popular object detection models to identify the number of trees in an image.
## Model Zoo 
We provide some models to use in this project: 
||Name|url|size|
|---|---|--------|---|
|0|RCNN|[model](https://drive.google.com/uc?export=download&id=1-PhXe-WUdzziK2T0WM0c9aMur0bTLkH8)|217.5Mb|
|1|Yolov7|[model](https://drive.google.com/uc?export=download&id=10CUFS0mtObpQDVzm7TAxE6LvfpEC7r65)|284.6Mb|
|2|Faster RCNN|[model](https://drive.google.com/uc?export=download&id=1DtVlGqlJiOKLBB0rsJDhyCfKsQgI15R9)|158.1Mb|

## How to use the Faster R-CNN Notebook:
1. Connect your notebook to Runtime (Preferably T4 GPU)
2. Open the web console (Ctrl+Shift+I) and paste in the block of code to prevent the notebook from timing out
3. Link your Google Drive with the following 2 blocks of code (Make sure to modify your directories)
4. Run the next few blocks of code to install necessary libraries
5. To prepare the dataset, remember to modify your directories for train_dir and test_dir
6. Run the following 2 blocks of code to define some necessary functions
7. For the get_object_detection_model function, you can choose to pretrain or not
8. Run a few more blocks and in the training block, you can adjust the num_epochs
9. Run the last few blocks to generate the detection results
10. Save and import your models (Modify your directories)

## How to use the Yolov7 Notebook:
1. Connect your notebook to Runtime (Preferably T4 GPU)
2. Open the web console (Ctrl+Shift+I) and paste in the block of code to prevent the notebook from timing out
3. Link your Google Drive with the following 2 blocks of code (Make sure to modify your directories)
4. Run the next few blocks of code to install necessary libraries
