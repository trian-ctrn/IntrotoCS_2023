import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
from selective_search import *
from util import *

def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(device=None):
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('models/best_linear_svm_alexnet_tree.pth'))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def draw_box_with_text(img, rect_list, score_list):
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def nms(rect_list, score_list):
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        iou_scores = iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = get_transform()
    model = get_model(device=device)

    gs = get_selective_search()

    test_img_path = f'ver13/images/IMG_175.jpg'
    test_xml_path = f'ver13/labels/IMG_175.xml'

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    bndboxs = take_bb_from_xml(test_xml_path)
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)


    config(gs, img, strategy='q')
    rects = get_rects(gs)
    print('Number of candidate regional: %d' % len(rects))

    # softmax = torch.softmax()

    svm_thresh = 0.55


    score_list = list()
    positive_list = list()

    # tmp_score_list = list()
    # tmp_positive_list = list()
    start = time.time()
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
    
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # tmp_score_list.append(probs[1])
            # tmp_positive_list.append(rect)

            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                print(rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end - start))

    # tmp_img2 = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img2, tmp_positive_list, tmp_score_list)
    # cv2.imshow('tmp', tmp_img2)
    # #
    # tmp_img = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img, positive_list, score_list)
    # cv2.imshow('tmp2', tmp_img)

    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)
    draw_box_with_text(dst, nms_rects, nms_scores)

    cv2.imshow('img', dst)
    cv2.waitKey(0)
