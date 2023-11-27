import os
import numpy as np
import xmltodict
import torch
import matplotlib.pyplot as plt
import re
import csv 

# Check if the directory is exist. If not create the directory
def check_directory(name_directory):
    if not os.path.exists(name_directory):
        os.mkdir(name_directory)

def get_digit(string): 
    return ''.join(c for c in string if c.isdigit())

def get_name_id(path): 
    re_pattern = re.compile('.+?(\d+)\.([a-zA-Z0-9+])')
    path_list = sorted(os.listdir(path), key=lambda x: int(re_pattern.match(x).groups()[0]))
    list_id = []
    for name in path_list:
        list_id.append(get_digit(name))
    return list_id

with open('ver13/val/tree.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    names = np.array(get_name_id("ver13/val/images"))
    for name in names:
        writer.writerow([name])

    
#Take the bounding box from the xml file
def take_bb_from_xml(xml_path):
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
    bbxs = []
    objects = xml_dict["annotation"]["object"]
    if isinstance(objects, list):
        for obj in objects:
            obj_name = obj['name']
            difficult = int(obj['difficult'])
            if 'tree'.__eq__(obj_name) and difficult != 1:
                bndbox = obj['bndbox']
                bbxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
    elif isinstance(objects, dict):
        obj_name = objects['name']
        difficult = int(objects['difficult'])
        if 'tree'.__eq__(obj_name) and difficult != 1:
            bndbox = objects['bndbox']
            bbxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
    else:
        pass

    return np.array(bbxs)

def iou(pred_box, target_box):
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]
        
    x_min = np.maximum(pred_box[0], target_box[:,0])
    y_min = np.maximum(pred_box[1], target_box[:,1])
    x_max = np.minimum(pred_box[2], target_box[:,2])
    y_max = np.minimum(pred_box[3], target_box[:,3])
    
    area_intersection = np.maximum(0.0, (x_max - x_min)) * np.maximum(0.0, (y_max - y_min))
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_target = (target_box[:,2] - target_box[:,0]) * (target_box[:,3] - target_box[:,1])
    area_union = area_pred + area_target - area_intersection
    iou_score = area_intersection / area_union
    return iou_score

def compute_ious(rects, bndboxs):
    iou_list = []
    for rect in rects:
        scores = iou(rect, bndboxs)
        iou_list.append(max(scores))
    return iou_list

def save_model(model, model_save_path):
    check_directory('./models')
    torch.save(model.state_dict(), model_save_path)

def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title('loss')
    plt.savefig('./loss.png')
