import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.array([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])
    return bbox


def link_bbox_instance(instances, bbox, class_index):
    instance_ids = np.copy(instances[bbox[1]:bbox[3]+1][:, bbox[0]:bbox[2]+1])
    # remove those pixels that not belong to the object classes
    instance_ids[instance_ids < class_index * 1000] = 0
    instance_ids[instance_ids >= (class_index+1) * 1000] = 0
    # find the instance_id that appears mostly in this matrix
    unique_numbers, unique_counts = np.unique(instance_ids.flatten(), return_counts=True)
    if 0 in unique_numbers and unique_numbers.size > 1:
        instance_pixels = np.max(unique_counts[1:])
        argmax_instance_id = unique_numbers[np.argmax(unique_counts[1:]) + 1]
        # compute the bounding box that encloses the instance most tightly in order to update the bbox
        instance_ids[instance_ids != argmax_instance_id] = 0
        new_bbox = bounding_box(instance_ids)
        new_bbox = np.array([bbox[0] + new_bbox[0], bbox[1] + new_bbox[1], bbox[0] + new_bbox[2], bbox[1] + new_bbox[3]])
    else:
        instance_pixels = np.max(unique_counts)
        argmax_instance_id = unique_numbers[np.argmax(unique_counts)]
        new_bbox = bbox

    return argmax_instance_id, instance_pixels, new_bbox


gt_root_dir = '/home/wind/Data/Isaac_sim/AIUE_V01_001/demo'
sequence_names = ['000001']
for sequence_name in sequence_names:
    gt_dir = os.path.join(gt_root_dir, 'ground_truth')
    gt_label_dir = os.path.join(gt_dir, 'label_raw', sequence_name)

    # save single refined ground truth in the ground_truth_refined folder
    gt_label_refined_dir = os.path.join(gt_dir, 'label', sequence_name,)
    if not os.path.isdir(gt_label_refined_dir):
        os.makedirs(gt_label_refined_dir)

    gt_seg_dir = os.path.join(gt_dir, 'seg', sequence_name)
    gt_instance_dir = os.path.join(gt_dir, 'instance', sequence_name)

    # label_index file stores the class names and their indexes
    label_index_file = os.path.join(gt_dir, 'label_index.json')
    with open(label_index_file, 'r') as f:
        label_index = json.load(f)
    # reverse the label and index for later use
    index_label = {index: label for label, index in label_index.items()}

    # the object class index range that excludes those stuff classes
    class_index_range = np.array([label_index['bottle'], label_index['person']])

    ground_truths = {}
    for root, _, gt_names in sorted(os.walk(gt_instance_dir)):
        for gt_name in sorted(gt_names):
            print(gt_name)
            # remove file extension
            gt_name_ = gt_name[:-4]

            # label_file stores the raw ground truth for each image
            label_file = os.path.join(gt_label_dir, gt_name_+'.json')
            label_refined_file = os.path.join(gt_label_refined_dir, gt_name_ + '.json')

            # seg_file stores the semantic segmentation of the image
            seg_file = os.path.join(gt_seg_dir, gt_name_+'.png')

            # instance_file stores the instance segmentation of the image
            instance_file = os.path.join(gt_instance_dir, gt_name_+'.png')
            with open(label_file, 'r') as f:
                labels = json.load(f)

            # segs is saved in uint8 type
            segs = cv2.imread(seg_file, -1)

            # instances is stored in uint16 type so when reading it add -1 argument
            instances = cv2.imread(instance_file, -1)
            # plt.imshow(segs)
            # plt.show()

            # save the refined ground_truth in RVC format
            ground_truth = {'_metadata': {'mask_name': gt_name_ + '.png'}}
            labels_refined = []
            for label in labels:
                # if this object is completely occluded then exclude it
                if math.isnan(label['occlusion_ratio']):
                    print(label['occlusion_ratio'])
                    continue

                bbox = (np.array(label['bbox'])-1).astype(int)
                instance_class = label['class']
                bbox[0] = np.max([bbox[0], 0])
                bbox[1] = np.max([bbox[1], 0])
                bbox[2] = np.max([bbox[2], 0])
                bbox[3] = np.max([bbox[3], 0])
                # if the object only has pixels in a row or a column, remove it
                if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                    continue
                class_index = label_index[instance_class]

                # for each object bounding box, link it to one instance in the instance segmentation map by searching
                # the instance id that appears mostly in that bounding box and has the same class with the bounding box
                # at the same time, refine the original bounding box that is computed through projecting the 3D mesh
                # into the image plane by computing the tightest box enclosing the instance
                argmax_instance_id, instance_pixels, new_bbox = link_bbox_instance(instances, bbox, class_index)
                # if the found instance id is 0 (background), then exclude this object
                if argmax_instance_id == 0 or instance_class != index_label[argmax_instance_id // 1000 ]:
                    continue

                label["mask_id"] = int(argmax_instance_id)
                label["bounding_box"] = new_bbox.tolist()
                label["num_pixels"] = int(instance_pixels)
                label['class'] = instance_class
                label['raw_bbox'] = label.pop('bbox')
                labels_refined.append(label.copy())
                ground_truth['pose_rotation'] = label.pop('pose_rotation')
                ground_truth['pose_translation'] = label.pop('pose_translation')
                ground_truth['pose'] = label.pop('pose')
                ground_truth[str(argmax_instance_id)] = label
                # labels_refined.append(label)

                # seg_ids = segs[bbox[1]:bbox[3]][:, bbox[0]:bbox[2]]
                # instances = cv2.rectangle(instances, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # instances = cv2.rectangle(instances, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (255, 0, 0), 2)
                # plt.subplot(1, 2, 1)
                # plt.imshow(seg_ids)
                # plt.subplot(1, 2, 2)
                # plt.imshow(instances)
                # plt.show()

            with open(label_refined_file, 'w') as f:
                json.dump(labels_refined, f)
            ground_truths[gt_name_] = ground_truth
    gt_save_name = os.path.join(gt_label_refined_dir, 'labels.json')
    with open(gt_save_name, 'w') as f:
        json.dump(ground_truths, f)
    print('end')
