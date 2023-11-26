# Detector Implementation 
## Process 
1. Enter an image
2. Candidate recommendations are calculated using a selective search algorithm
3. Count the candidate recommendations one by one
    i. The features were computed using the AlexNet model
    ii. The classification results were calculated by using the linear SVM classifier
4. Perform non-maximal suppression on all candidate recommendations classified as automobiles

## Python file 
- <mark>**RCNN/inference.py**</mark>

