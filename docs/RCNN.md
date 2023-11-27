# Training Progress
## Dataset Preprocessing 
The dataset comprises images and annotation files divided into train, validation, and test sets. Preprocessing involves the following steps:
1. CSV Creation for Train and Validation:

Create a CSV file for each train and validation folder containing the image IDs, derived from the numbers in the image filenames. For instance, if the train file includes "IMG_1.jpg" and "IMG_5.jpg," the CSV file should contain 1 and 5, respectively.
2. Dataset Creation using 'finetune.py' and 'classifier.py':

Run 'finetune.py' and 'classifier.py' in the preprocessing folder to generate two new datasets. This process creates the 'finetune_tree' and 'classifier_tree' folders. Each folder contains CSV files for positive and negative bounding box values.

## CNN (Convolutional Neural Network)
Utilizing AlexNet for the CNN, the finetune dataset is employed for training. The steps involved are:
1. Adjusting Parameters in 'model.py':
Modify parameters in 'model.py' based on your device specifications. For instance:
```
data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), bathc_positive = 32, batch_negative =  96)
data_loader = DataLoader(data_set, batch_size=28, sampler=data_sampler, num_workers=2, drop_last=True)
```
If you have a strong GPU, consider increasing batch_size and num_workers for faster training.
2. Parameter Adjustment for Training:
- Change parameters such as optimizer, learning rate scheduler, and epochs:
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=10)
```
## SVM (Support Vector Machine)
Using features extracted from the CNN, a linear SVM is employed for classification into tree and non-tree categories:
1. Running "svm.py"
- Using features extracted from the CNN, a linear SVM is employed for classification into tree and non-tree categories:
2. Saving SVM Weights:
- Upon running the file, the weights obtained are saved in the 'models' folder. 

# Detector Implementation 
## Process 
1. Input an image
2. Calculate candidate recommendations using a selective search algorithm.
3. Process each recommendation:
    i. Compute features using the AlexNet model.
    ii. Employ the linear SVM classifier for classification.
4. Perform non-maximum suppression on candidate recommendations classified as trees

## Relevent  file 
- <mark>**RCNN/inference.py**</mark>

## Non-maximum suppression
For a detailed understanding and implementation of non-maximum suppression, refer to [this tutorial](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/)

## Implement parameters
- Non-maximum suppression threshold: 0.3
- Classifier threshold: 0.55

# Note
1. The selective search strategy can be altered:
```
config(gs, img, strategy='q')
```
Available strategies:
- 's' (Single Strategy)
- 'f' (Selective Search Fast)
- 'q' (Selective Search Quality)

# Citation
```
@misc{object-detection-algorithm,
  title = {R-CNN},
  howpublished = {[GitHub repository]},
  url = {[https://github.com/object-detection-algorithm/R-CNN]},
  year = {2020},
  author = {object-detection-algorithm}
}
```


