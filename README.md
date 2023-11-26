# IntrotoCS_2023
This project is part of the Introduction to Computer Science 2023 course at the Vietnamese German University. It involves implementing various popular object detection models to identify the number of trees in an image.

## Members:
- Yamashita Tri An - 10422004
- Nguyen Tran Quoc Dat - 10422017
- Nguyen Minh Giap - 10422024
- Ton That Nhat Minh - 10422050
- Nguyen Khoi Nguyen - 10422058
- Hoang Quang Nhat - 10422060
- Le Viet Tin - 10422078

## Coach:
- Dr. Nhan Le

## Dataset
Our tree dataset has 405 images of size 1024 x 1024, which have been divided into the train, validation, and test set with a ratio of 0.7:0.2:0.1. The dataset can be found [here](https://drive.google.com/drive/folders/1ylXgcWFMX43FdWxA-oHMlztELLNFdbgY?usp=sharing)

## Model Zoo 
We provide some models to use in this project: 
||Name|url|size|
|---|---|--------|---|
|0|RCNN|[model](https://drive.google.com/uc?export=download&id=1-PhXe-WUdzziK2T0WM0c9aMur0bTLkH8)|217.5Mb|
|1|Yolov7|[model](https://drive.google.com/uc?export=download&id=10CUFS0mtObpQDVzm7TAxE6LvfpEC7r65)|284.6Mb|

## How to use the Faster R-CNN Notebook:
1. Connect your notebook to Runtime (Preferably T4 GPU)
2. Open the web console (Ctrl+Shift+I) and paste in the block of code to prevent the notebook from timing out
3. Link your Google Drive with the following 2 blocks of code (Make sure to modify your directories)
4. Run the next few blocks of code to install the necessary libraries
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
4. Run the next few blocks of code to install the necessary libraries
## Acknowledgments
We would like to thank our coach Dr. Nhan Le who supported us throughout the development of this project.

## References
[1]Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C., … & Berg, A. C. (2016). Ssd: single shot multibox detector. Computer Vision – ECCV 2016, 21-37. https://doi.org/10.1007/978-3-319-46448-0_2 \
[2]C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.” arXiv, Jul. 06, 2022. doi: 10.48550/arXiv.2207.02696. \
[3]R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” \
[4]Sri, M. S., Naik, B. R., & Sankar, K. J. (2021, February 28). Object Detection Based on Faster R-CNN. International Journal of Engineering and Advanced Technology, 10(3), 72–76. https://doi.org/10.35940/ijeat.c2186.0210321

## Contacts
### Development team:
- Yamashita Tri An: 10422004@student.vgu.edu.vn
- Nguyen Tran Quoc Dat: 10422017@student.vgu.edu.vn
- Nguyen Minh Giap: 10422024@student.vgu.edu.vn
- Ton That Nhat Minh: 10422050@student.vgu.edu.vn
- Nguyen Khoi Nguyen: 10422058@student.vgu.edu.vn
- Hoang Quang Nhat: 10422060@student.vgu.edu.vn
- Le Viet Tin: 10422078@student.vgu.edu.vn
