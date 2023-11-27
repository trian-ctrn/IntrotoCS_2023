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
From features we take from CNN, we use linear SVM to classifier which is a tree and which is not a tree. And it will return 2 numbers and put in an array which a first index is probability that the box is not contain a tree and the second index represent the probabilitu that the box contain a tree: 
1. Run the "svm.py" file and change some parameters like the upper progress base on how strong is your GPU and your computer.
2. After running the file you will get the weights and save it in the "models" folder 

# Detector Implementation 
## Process 
1. Enter an image
2. Candidate recommendations are calculated using a selective search algorithm
3. Count the candidate recommendations one by one
    i. The features were computed using the AlexNet model
    ii. The classification results were calculated by using the linear SVM classifier
4. Perform non-maximum suppression on all candidate recommendations classified as automobiles

## Python file 
- <mark>**RCNN/inference.py**</mark>

## Non-maximum suppression
Watch [here](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/)

## Implement parameters
- Non-maximum suppression threshold: 0.3
- Classifier threshold: 0.55

# Note
1. We can change the strategy of the selective 
```
config(gs, img, strategy='q')
```
Base on the config function we have 3 strategies to choose: 
```
def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy() 
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)
```

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


