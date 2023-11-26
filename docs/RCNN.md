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
- Classifier threshold: 0.6

# Note
1. We can change the strategy of the selective 
   ```
   config(gs, img, strategy='q')
   ```
Base on the code we have 3 strategies to choose: 
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

# Reference 
```
@misc{object-detection-algorithm,
  title = {R-CNN},
  howpublished = {[GitHub repository]},
  url = {[https://github.com/object-detection-algorithm/R-CNN]},
  year = {2020},
  author = {object-detection-algorithm}
}
```


