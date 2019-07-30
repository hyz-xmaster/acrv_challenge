#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from matplotlib import pyplot as plt


def compute_robot_T_camera(pose_rotation=[1, 0, 0, 0], pose_translation=[0.0, 0.0, 0.0]):
    """
    THis function computes the transformation T that transform one point in the camera coordinate
    (x - right, y - up, z - forward) to the point in the robot coordinate system (x - forward, y - left, z - up).
    (The camera is fixed on the robot base-link)
    P_r = P_c * T, P_c = [x, y, z, 1] is a row vector in the camera coordinate
    Please note that the pose_rotation quaternion is not valid because the camera coordinate is different from the robot
    coordinate in Isaac Sim. Actually, the camera coordinate is left-handed whereas the robot's is right-handed. To
    transform the camera coordinate, a axis-swap matrix is needed.

    :param pose_rotation: [quat_w, quat_x, quat_y, quat_z], a quaternion that represents a rotation
    :param pose_translation: [tr_x, tr_y, tr_z], the location of camera origin in the robot coordinate system
    :return: transformation matrix
    """
    # compute the rotation matrix
    # note that the scipy...Rotation accepts the quaternion in scalar-last format
    pose_quat = [pose_rotation[1], pose_rotation[2], pose_rotation[3], pose_rotation[0]]
    rot = Rotation.from_quat(pose_quat)
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3][:, 0:3] = rot.as_dcm().T

    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = np.array(pose_translation)

    # swap the axes of camera coordinate system to be coherent with the robot coordinate system
    swap_matrix = np.array([[0, -1, 0, 0],
                            [0,  0, 1, 0],
                            [1,  0, 0, 0],
                            [0,  0, 0, 1]], dtype=np.float)
    transformation_matrix = swap_matrix @ rotation_matrix @ translation_matrix

    return transformation_matrix


def compute_world_T_robot(pose_rotation, pose_translation):
    """
    This function computes the transformation matrix T that transforms one point in the robot coordinate system to
    the world coordinate system. This function needs the scipy package - scipy.spatial.transform.Rotation
    P_w = P_r * T, P_r = [x, y, z, 1] is a row vector in the robot coordinate (x - forward, y - left, z - up)

    :param pose_rotation: [quat_w, quat_x, quat_y, quat_z], a quaternion that represents a rotation
    :param pose_translation: [tr_x, tr_y, tr_z], the location of the robot origin in the world coordinate system
    :return: transformation matrix
    """
    # compute the rotation matrix
    # note that the scipy...Rotation accepts the quaternion in scalar-last format
    pose_quat = [pose_rotation[1], pose_rotation[2], pose_rotation[3], pose_rotation[0]]
    rot = Rotation.from_quat(pose_quat)
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3][:, 0:3] = rot.as_dcm().T

    # build the tranlation matrix
    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = np.array(pose_translation)

    # compute the tranformation matrix
    transformation_matrix = rotation_matrix @ translation_matrix

    # convert robot's right-handed coordinate system to unreal world's left-handed coordinate system
    transformation_matrix[:, 1] = -1.0 * transformation_matrix[:, 1]

    return transformation_matrix


def compute_camera_T_world(pose_rotation, pose_translation):
    """
    This function is written according to the UpdateProjectMatrix() in IssacSimBoundingBox.cpp
    It computes the transformation matrix T that transforms one point in the world coordinate into the camera coordinate
    (x - right, y - up, z - forward)
    P_c = P_w * T, P_w = [x, y, z, 1] a row vector denoting point location in the world coordinate

    :param pose_rotation: [Yaw, Pitch, Roll] (degree), represents a rotation, exported from Isaac Sim
    :param pose_translation: [tr_x, tr_y, tr_z], the location of the camera origin in the world coordinate system
    :return: transformation matrix
    """

    # build rotation matrix
    yaw_angle = pose_rotation[0] / 180.0 * np.pi
    pitch_angle = -1 * pose_rotation[1] / 180.0 * np.pi
    roll_angle = -1 * pose_rotation[2] / 180.0 * np.pi
    yaw = np.array([[np.cos(yaw_angle),  -np.sin(yaw_angle),  0],
                    [np.sin(yaw_angle),   np.cos(yaw_angle),  0],
                    [                0,                   0,  1]])
    pitch = np.array([[ np.cos(pitch_angle),  0,  np.sin(pitch_angle)],
                      [                   0,  1,                    0],
                      [-np.sin(pitch_angle),  0,  np.cos(pitch_angle)]])
    roll = np.array([[1,                   0,                    0],
                     [0,  np.cos(roll_angle),  -np.sin(roll_angle)],
                     [0,  np.sin(roll_angle),   np.cos(roll_angle)]])
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3][:, 0:3] = yaw @ pitch @ roll

    # build translation matrix
    # [1  0  0  0;
    #  0  1  0  0;
    #  0  0  1  0;
    #  dx dy dz 1]
    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = -1. * np.array(pose_translation)

    # swap the axis
    swap_matrix = np.array([[0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float)

    # compute transform_matrix
    transformation_matrix = translation_matrix @ rotation_matrix @ swap_matrix

    return transformation_matrix


def project_camera_to_image(projection_matrix, point_3d):
    """
    This function project one point in the camera coordinate system into the image plane
    image_x, image_y, image_z = point * projection_matrix.T
    :param projection_matrix: 4*4 matrix
    :param point_3d: [x1, y1, z1; x2, y2, z2; ...; xk, yk, zk], must be in centimeter
    :return: [image_x, image_y]
    """
    point_3d = np.array(point_3d)
    point_3d1 = np.ones((point_3d.shape[0], point_3d.shape[1] + 1))
    point_3d1[:, :-1] = point_3d
    point_2d = point_3d1 @ np.array(projection_matrix).T
    image_xy = np.zeros((point_2d.shape[0], 2))
    image_xy[:, 0] = point_2d[:, 0] / point_2d[:, 2]
    image_xy[:, 1] = point_2d[:, 1] / point_2d[:, 2]

    return image_xy


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(root_dir, sequences, extensions):
    images = {}
    image_nums = {}
    for seq in sorted(sequences):
        image_num = 0
        d = os.path.join(root_dir, seq)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (seq, fname)
                    images[item] = path
                    image_num += 1
        image_nums[seq] = image_num

    return images, image_nums


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ACRV(object):
    def __init__(self, data_dir, transform=None):
        # data_dir: the folder that contains sequences of frames
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']
        self.subfolders = os.listdir(data_dir)
        if 'rgb' not in self.subfolders:
            raise(RuntimeError('There is no rgb folder, please check'))
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        if 'rgb_right' in self.subfolders:
            self.rgb_right_dir = os.path.join(data_dir, 'rgb_right')
        if 'depth' in self.subfolders:
            self.depth_dir = os.path.join(data_dir, 'depth')
        if 'time_stamp' in self.subfolders:
            self.time_stamp_dir = os.path.join(data_dir, 'time_stamp')
        if 'calibration' in self.subfolders:
            self.calibration_dir = os.path.join(data_dir, 'calibration')
        if 'wheel_odometry' in self.subfolders:
            self.wheel_odometry_dir = os.path.join(data_dir, 'wheel_odometry')
        self.flag = self.train_test_flag(self.subfolders)
        if self.flag == 'training':
            self.gt_dir = os.path.join(data_dir, 'ground_truth')
        sequences = self._find_sequences(self.rgb_dir)
        samples, sample_nums = make_dataset(self.rgb_dir, sequences, IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.rgb_dir + "\n"
                                "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.data_dir = data_dir
        self.sequences = sequences
        self.samples = samples
        self.sample_keys = sorted(samples.keys())
        self.sample_nums = sample_nums
        self.transform = transform
        self.selected_seq = None  # choose a certain sequence

        if self.flag == 'training':
            self.gt_labels = {}
            for seq in self.sequences:
                self.gt_labels[seq] = self.read_gt_label(seq)

        self.time_stamps = {}
        for seq in self.sequences:
            self.time_stamps[seq] = self.read_time_stamp(seq)

    def __getitem__(self, idx):
        # given an idx, read the image or read the image and ground truth
        if self.flag == 'test':
            return self.get_image(idx)
        if self.flag == 'training':
            return self.get_image(idx), self.get_gt_label(idx), self.get_gt_instance(idx)

    def __len__(self):
        if self.selected_seq:
            return self.sample_nums[self.selected_seq]
        return len(self.samples)

    def __call__(self, seq_name=None):
        # initialize the selected_seq and iterate images just in this sequence
        if isinstance(seq_name, int) or len(str(seq_name)) < 6:
            self.selected_seq = '{0:06d}'.format(int(seq_name))
        else:
            self.selected_seq = seq_name

    def _find_sequences(self, dir):
        # Finds the sequence folders in a dataset.

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            sequences = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            sequences = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        sequences.sort()

        return sequences

    def train_test_flag(self, folders):
        if 'ground_truth' in folders:
            flag = 'training'
            print('This is a training dataset')
        else:
            flag = 'test'
            print('This is a testing dataset')

        return flag

    def idx_to_key(self, idx):
        # key_pair = (sequence_name, image_name) = ('000001', '00000002.png')
        if isinstance(idx, int):
            if self.selected_seq:
                # ends the loop when idx >= the number of images in this sequence
                if idx >= self.sample_nums[self.selected_seq]:
                    raise IndexError()
                image_name = '{0:08d}'.format(idx) + '.png'
                key_pair = (self.selected_seq, image_name)
            else:
                key_pair = self.sample_keys[idx]
        elif len(idx) == 2:
            key_pair = (self.seq_to_key(idx[0]), '{0:08d}'.format(idx[1]) + '.png')
        else:
            raise(RuntimeError('Wrong index'))

        return key_pair

    def seq_to_key(self, seq):
        if isinstance(seq, int):
            seq_key = '{0:06d}'.format(seq)
        elif len(seq) == 6 and isinstance(seq, str):
            seq_key = seq
        else:
            raise(RuntimeError('Wrong sequence name'))

        return seq_key

    def get_image_name(self, idx):
        # get the image name, given the index
        image_key = self.idx_to_key(idx)

        return image_key[1]

    def get_image(self, idx):
        image_key = self.idx_to_key(idx)
        image_file = self.samples[image_key]
        image = pil_loader(image_file)
        image.name = self.get_image_name(idx)
        if self.transform is not None:
            image = self.transform(image)

        # plt.imshow(image)
        # plt.show()
        return image

    def get_right_image(self, idx):
        image_key = self.idx_to_key(idx)
        right_image_file = os.path.join(self.rgb_right_dir, image_key[0], image_key[1])
        right_image = pil_loader(right_image_file)
        if self.transform is not None:
            right_image = self.transform(right_image)

        return right_image

    def get_depth_image(self, idx):
        image_key = self.idx_to_key(idx)
        depth_image_file = os.path.join(self.depth_dir, image_key[0], image_key[1])

        # depth image is stored in uint16
        depth_image = cv2.imread(depth_image_file, -1)
        # depth image should be divided by 255 to get the actual depth in cm
        depth_image = depth_image.astype(np.float32) / 255.0

        if self.transform is not None:
            depth_image = self.transform(depth_image)

        # plt.imshow(depth_image)
        # plt.show()
        return depth_image

    def get_gt_instance(self, idx):
        if self.flag == 'test':
            raise(RuntimeError('This is a test dataset, no ground truth'))
        else:
            image_key = self.idx_to_key(idx)
            instance_image_dir = os.path.join(self.gt_dir, 'instance')
            if not os.path.isdir(instance_image_dir):
                raise(RuntimeError('Instance dir not exist'))
            instance_image_file = os.path.join(instance_image_dir, image_key[0], image_key[1])
            # instance image is stored in uint16
            instance_image = cv2.imread(instance_image_file, -1)
            if self.transform is not None:
                instance_image = self.transform(instance_image)

            # plt.imshow(instance_image)
            # plt.show()
            return instance_image

    def get_gt_segmentation(self, idx):
        if self.flag == 'test':
            raise(RuntimeError('This is a test dataset, no ground truth'))
        else:
            image_key = self.idx_to_key(idx)
            segmentation_image_dir = os.path.join(self.gt_dir, 'segmentation')
            if not os.path.isdir(segmentation_image_dir):
                raise(RuntimeError('Segmentation dir not exist'))
            segmentation_image_file = os.path.join(segmentation_image_dir, image_key[0], image_key[1])
            # segmentation image is stored in uint8
            segmentation_image = cv2.imread(segmentation_image_file, -1)
            if self.transform is not None:
                segmentation_image = self.transform(segmentation_image)

            # plt.imshow(segmentation_image)
            # plt.show()
            return segmentation_image

    def get_label_index(self):
        if self.flag == 'test':
            raise(RuntimeError('This is a test dataset, no ground truth'))
        label_index_file = os.path.join(self.gt_dir, 'label_index.json')
        with open(label_index_file, 'r') as f:
            label_index = json.load(f)

        return label_index

    def read_gt_label(self, seq):
        if self.flag == 'test':
            raise(RuntimeError('This is a test dataset, no ground truth'))
        gt_label_dir = os.path.join(self.gt_dir, 'label')
        if not os.path.isdir(gt_label_dir):
            raise (RuntimeError('Label dir not exist'))
        gt_label_file = os.path.join(gt_label_dir, self.seq_to_key(seq), 'labels.json')
        with open(gt_label_file, 'r') as fp:
            labels = json.load(fp)

        return labels

    def get_gt_label(self, idx):
        if self.flag == 'test':
            raise(RuntimeError('This is a test dataset, no ground truth'))
        else:
            image_key = self.idx_to_key(idx)
        gt_label = self.gt_labels[image_key[0]][image_key[1][:-4]]

        return gt_label

    def get_calibration(self, seq):
        # read projection matrix, pose of the camera in the robot frame and the base line of the stereo cameras, etc.
        calib_dir = os.path.join(self.calibration_dir, self.seq_to_key(seq))
        if not os.path.isdir(calib_dir):
            raise (RuntimeError('calibration dir not exist'))
        calib_file = os.path.join(self.calibration_dir, self.seq_to_key(seq), 'calibration.json')
        with open(calib_file, 'r') as fp:
            calibrations = json.load(fp)

        return calibrations

    def read_time_stamp(self, seq):
        time_stamp_dir = os.path.join(self.time_stamp_dir, self.seq_to_key(seq))
        if not os.path.isdir(time_stamp_dir):
            raise (RuntimeError('time stamp dir not exist'))
        time_stamp_file = os.path.join(time_stamp_dir, 'time_stamp.json')
        with open(time_stamp_file, 'r') as fp:
            time_stamp = json.load(fp)

        return time_stamp

    def get_time_stamp(self, idx):
        image_key = self.idx_to_key(idx)
        time_stamp = self.time_stamps[image_key[0]][image_key[1][:-4]]

        return time_stamp

    def get_wheel_odometry(self, idx):
        image_key = self.idx_to_key(idx)
        wheel_odometry_file = os.path.join(self.wheel_odometry_dir, image_key[0], image_key[1][:-4]+'.json')
        with open(wheel_odometry_file, 'r') as fp:
            wheel_odomemetry = json.load(fp)

        return wheel_odomemetry

    def get_world_T_robot(self, idx):
        wheel_odom = self.get_wheel_odometry(idx)
        world_T_robot = compute_world_T_robot(wheel_odom[0]['pose_rotation'], wheel_odom[0]['pose_translation'])

        return world_T_robot

    def get_robot_T_camera(self, idx):
        image_key = self.idx_to_key(idx)
        calib = self.get_calibration(image_key[0])
        robot_T_camera = compute_robot_T_camera(calib['camera_pose_in_robot']['pose_rotation'],
                                                calib['camera_pose_in_robot']['pose_translation'])

        return robot_T_camera

    def get_camera_T_world(self, idx):
        gt_label = self.get_gt_label(idx)
        camera_T_world = compute_camera_T_world(gt_label['pose_rotation'], gt_label['pose_translation'])

        return camera_T_world


if __name__ == '__main__':
    # replace the data_dir with your own and run the code to learn the usage
    data_dir = '/home/wind/Data/Isaac_sim/AIUE_V01_001/demo/'
    # create the ACRV object with data_dir which should include 'rgb', 'rgb_right', 'depth', 'calibration'
    # 'time_stamp', 'wheel_odometry' and 'ground_truth' (if this is a training dataset) sub-folders
    acrv_data = ACRV(data_dir)

    # get the number of the images
    print(len(acrv_data))

    idx = 0
    # given the idx, get the image name
    print('image name: ', acrv_data.get_image_name(idx))
    # given the idx, get the depth image
    acrv_data.get_depth_image(idx)
    # given the idx, get the rgb image
    acrv_data.get_image(0)
    # given the sequence number and image number, get the rgb image
    idx = ('000000', 2)
    acrv_data.get_image(idx)
    # given the idx, get the right image
    acrv_data.get_right_image(0)

    # given the idx, get the instance ground truth
    acrv_data.get_gt_instance(0)
    # given the idx, get the semantic segmentation truth
    acrv_data.get_gt_segmentation(0)
    # get the mapping from object class names to numbers
    print(acrv_data.get_label_index())
    # get the ground truth annotations, including 2D bbox, 3D bbox and camera pose etc.
    print(acrv_data.get_gt_label(8))

    # given the sequence name, get the calibration information for this sequence
    # including camera parameters, camera and robot pose in the world coordinate system in the first frame
    seq = '000000'  # or 0
    calib = acrv_data.get_calibration(seq)

    # given the idx, get the wheel odometry for this frame
    wheel_odom = acrv_data.get_wheel_odometry(0)

    # given the idx, get the world_T_robot transformation matrix
    # P_w = P_r @ world_T_robot, P_r = [x, y, z, 1] the point in the robot coordinate system
    w_T_r = acrv_data.get_world_T_robot(2)
    # given the idx, get the robot_T_camera transformation matrix
    # P_r = P_c @ robot_T_camera
    r_T_c = acrv_data.get_robot_T_camera(2)
    # compute the world_T_camera tranformation matrix
    w_T_c = r_T_c @ w_T_r
    # given the idx, get the camera_T_world transformation matrix
    c_T_w = acrv_data.get_camera_T_world(2)
    # w_T_c should be very similar to the inverse of c_T_w
    wTc = np.linalg.inv(c_T_w)

    # given one point in the camera coordinate system
    # compute its location in the world coordinate system
    P_c_ori = np.array([189.004486, -86.714561, 213.965942, 1])
    P_w = P_c_ori @ r_T_c @ w_T_r
    print(P_w)
    # using the wTc to compute this location again
    P_w_c = P_c_ori @ wTc
    print(P_w_c)

    # given the idx, use the class name to read images or read images and ground truth (training set)
    img, gt, gt_instance = acrv_data[idx]
    img, _, _ = acrv_data[('000000', 1)]

    # you can also intilize the ACRV object with the specified sequence
    acrv_data('000000')
    img, _, _ = acrv_data[idx]
    print(img.name)
    # for image, _, _ in acrv_data:
    #     print(image.name)
    # print('end')
