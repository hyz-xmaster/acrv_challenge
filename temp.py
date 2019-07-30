import numpy as np
import cv2
import os
import json
from scipy.spatial.transform import Rotation


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
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float)
    transformation_matrix = swap_matrix @ rotation_matrix @ translation_matrix

    return transformation_matrix


def compute_world_T_robot(pose_rotation, pose_translation):
    """
    This function computes the transformation matrix T that transforms one point in the robot coordinate system to
    the world coordinate system. This function needs the scipy package - scipy.spatial.transform.Rotation
    P_w = P_r * T, P_r = [x, y, z, 1] is a row vector in the robot coordinate (x - forward, y - left, z - up)
    MUST NOTICE that after the above computation, P_w.y should be inversed, i.e P_w.y = -1 * P_w.y because the robot
    coordinate is right-handed whereas the Unreal world coordinate system is left-handed

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
    yaw = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                    [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                    [0, 0, 1]])
    pitch = np.array([[np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                      [0, 1, 0],
                      [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]])
    roll = np.array([[1, 0, 0],
                     [0, np.cos(roll_angle), -np.sin(roll_angle)],
                     [0, np.sin(roll_angle), np.cos(roll_angle)]])
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
    :param projection_matrix: 4*4 matrix
    :param point_3d: [x1, y1, z1; x2, y2, z2; ...; xk, yk, zk]
    :return: [image_x, image_y] ?
    """
    point_3d = np.array(np.array(point_3d))
    point_3d1 = np.ones((point_3d.shape[0], point_3d.shape[1] + 1))
    point_3d1[:, :-1] = point_3d
    point_2d = point_3d1 @ np.array(projection_matrix).T
    image_xy = np.zeros((point_2d.shape[0], 2))
    image_xy[:, 0] = point_2d[:, 0] / point_2d[:, 2]
    image_xy[:, 1] = point_2d[:, 1] / point_2d[:, 2]

    return image_xy


# pose_W_R = [-9.318762201994632e-05, 2.577286481612257e-05, 0.528205762557993, 0.8491164013563486, -0.5738143920898438, -8.497393608093262, 0.16249771416187286]
pose_W_R = [0.7533729559846848, -0.005290255290014453, -0.00030417602598601105, 0.6575721328240803,
            -0.6908791065216064, -6.817925930023193, 0.1602310836315155]
quat_W_R = [pose_W_R[1], pose_W_R[2], pose_W_R[3], pose_W_R[0]]
rot_W_R = Rotation.from_quat(quat_W_R)
print(rot_W_R.as_euler('zyx', degrees=True))
translation_W_R = np.array(pose_W_R[4:]).T

pose_R_C = [0, 0, 0, 1, 0.11, 0.06, 0.81]
rot_R_C = Rotation.from_quat(pose_R_C[:4])
print(rot_R_C.as_euler('zyx', degrees=True))
translation_R_C = np.array(pose_R_C[4:]).T

# P_c = [0., 0., 0.]
P_c_ori = [2.32354095, -0.15312108, 2.90937317, 1]
P_r = P_c_ori @ compute_robot_T_camera([1, 0, 0, 0], [0.11, 0.06, 0.81])
print('P_r = ', P_r)

P_c = [2.90937317, -2.32354095, -0.15312108]
P_r = rot_R_C.apply(P_c) + translation_R_C
print('P_r = ', P_r)
print('P_w = ', np.append(P_r, [1]) @ compute_world_T_robot(pose_W_R[:4], pose_W_R[4:]))

P_w = rot_W_R.apply(P_r) + translation_W_R
print('P_w = ', P_w)


# compute camera_T_world
view_rotation = [-82.229767, -0.372368, -0.479645]
view_location = np.array([-74.146164, 669.469727, 96.896843]) / 100.0
T_cw = compute_camera_T_world(view_rotation, view_location)
P_w = P_c_ori @ np.linalg.inv(T_cw)
print('P_w = ', P_w)



projection_matrix = [[480.0,  0.0,   480.0,    0.0],
                     [0.0,   -480.0, 270.0,    0.0],
                     [0.0,    0.0,     1.0,    0.0],
                     [0.0,    0.0,     0.0,    1.0]]
points = [[100, 100, 100], [200, 200, 200]]
image_xy = project_camera_to_image(projection_matrix, points)
print(image_xy)
print('end')