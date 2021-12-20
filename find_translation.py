import argparse
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DEBUG = False
DICTIONARY = cv2.aruco.DICT_5X5_100

if DEBUG:
    IMAGE = "images/test1.png"
    frame = cv2.imread(IMAGE)
else:
    cap = cv2.VideoCapture(4)
    print(cap.isOpened())

aruco_dict = cv2.aruco.Dictionary_get(DICTIONARY)
parameters = cv2.aruco.DetectorParameters_create()

cv_file = cv2.FileStorage("outputs.yml", cv2.FILE_STORAGE_READ)

camera_matrix = cv_file.getNode("K").mat()
dist_matrix = cv_file.getNode("D").mat()

cv_file.release()

x = 0
graph_data_x = []
graph_data_t = []
graph_data_r = []

fig = plt.figure()

translation = plt.subplot(121)
rotation = plt.subplot(122)

real_points = np.array([[[0., 0., 0.], [0., 4.6, 0.], [4.6, 4.6, 0.], [4.6, 0., 0.]],
                        [[0., 15.1, 0.], [0, 19.7, 0.], [4.6, 19.7, 0.], [4.6, 15.1, 0.]],
                        [[16.1, 0., 0.], [16.1, 4.6, 0.], [20.7, 4.6, 0.], [20.7, 0., 0.]],
                        [[16.1, 15.1, 0.], [16.1, 19.7, 0.], [20.7, 19.7, 0.], [20.7, 15.1, 0.]]], dtype=np.float32)

real_points_2d = np.array([[0., 0.], [0., 4.6], [4.6, 4.6], [4.6, 0.],
                           [16.1, 0.], [16.1, 4.6], [20.7, 4.6], [20.7, 0.],
                          [0., 15.1], [0, 19.7], [4.6, 19.7], [4.6, 15.1],
                          [16.1, 15.1], [16.1, 19.7], [20.7, 19.7], [20.7, 15.1]], dtype=np.float32)

board = cv2.aruco.Board_create(real_points, aruco_dict, np.array([24, 42, 69, 48]))

rvecs, tvecs = None, None


def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices


def loop(_i):
    if not DEBUG:
        ret, frame = cap.read()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=camera_matrix,
                                                                distCoeff=dist_matrix)

    corners, ids, rejected_img_points, recovered_ids = cv2.aruco.refineDetectedMarkers(
        image=frame,
        board=board,
        detectedCorners=corners,
        detectedIds=ids,
        rejectedCorners=rejected_img_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_matrix)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    if len(corners) == 4:
        pose, _rvec, _tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_matrix, rvecs, tvecs)

        zipped = zip(ids, corners)
        zipped = sorted(zipped)

        corners = [i[1] for i in zipped]

        these_res_corners = np.concatenate(corners, axis=1)

        ret, mask = cv2.findHomography(real_points_2d, these_res_corners, cv2.RANSAC, 5.0)

        rect = np.array([[[0, 0],
                          [19.8, 0],
                          [19.8, 20.8],
                          [0, 20.8]]], dtype=np.float32)

        new_rect = cv2.perspectiveTransform(rect, ret, (frame.shape[1], frame.shape[0]))
        trans = cv2.getPerspectiveTransform(new_rect, rect)

        frame = cv2.warpPerspective(frame, trans, (1000, 1000), flags=cv2.INTER_LINEAR)
        frame = frame[0:21, 0:20]

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


while True:
    loop(0)
