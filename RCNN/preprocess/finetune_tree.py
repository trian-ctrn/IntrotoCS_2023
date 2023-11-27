from util import *
from selective_search import *
import cv2
import time 
import shutil

def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    img = cv2.imread(jpeg_path)
    config(gs, img, strategy='q')
    rects = get_rects(gs)
    bndboxs = take_bb_from_xml(annotation_path)
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    iou_list = compute_ious(rects, bndboxs)

    positive_list = []
    negative_list = []
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)
        iou_score = iou_list[i]
        if iou_score >= 0.5:
            positive_list.append(rects[i])
        if 0 < iou_score < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list

if __name__ == "__main__":
    tree_directory = "ver13"
    finetune_directory = "finetune_ver13"
    check_directory(finetune_directory)

    gs = get_selective_search()
    for name in ['train', 'val']:
        src_directory = os.path.join(tree_directory,name)
        image_directory = os.path.join(src_directory, "images")
        annotation_directory = os.path.join(src_directory, "labels")

        dst_directory = os.path.join(finetune_directory, name)
        finetune_image_directory = os.path.join(dst_directory, 'images')
        finetune_annotation_directory = os.path.join(dst_directory, 'labels')
        check_directory(dst_directory)
        check_directory(finetune_image_directory)
        check_directory(finetune_annotation_directory)

        total_positive = 0 
        total_negative = 0 

        name_id = get_name_id(image_directory)
        src_csv_path = os.path.join(src_directory, 'tree.csv')
        dst_csv_path = os.path.join(dst_directory, 'tree.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)

        for id in name_id:
            since = time.time()
            src_annotation_path = os.path.join(annotation_directory, f"IMG_{id}.xml")
            src_jpeg_path = os.path.join(image_directory, f"IMG_{id}.jpg")
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_positive += len(positive_list)
            total_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(finetune_annotation_directory, f"IMG_{id}_1.csv")
            dst_annotation_negative_path = os.path.join(finetune_annotation_directory, f"IMG_{id}_0.csv")
            dst_jpeg_path = os.path.join(finetune_image_directory, f"IMG_{id}.jpg")
            
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(f"IMG_{id}", time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_positive))
        print('%s negative num: %d' % (name, total_negative))
    print('done')
        
        
    