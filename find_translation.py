import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DEBUG = False
DICTIONARY = cv2.aruco.DICT_5X5_100

if DEBUG:
    IMAGE = "images/test1.png"
    frame = cv2.imread(IMAGE)
else:
    cap = cv2.VideoCapture(4)

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


def loop(_i):
    if not DEBUG:
        ret, frame = cap.read()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict,
                                                              parameters=parameters,
                                                              cameraMatrix=camera_matrix,
                                                              distCoeff=dist_matrix)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera
            # coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix,
                                                                           dist_matrix)

            if ids[i] == 24:
                graph_data_r.append(rvec[0][0])
                graph_data_t.append(tvec[0][0])

                translation.plot(graph_data_t, label="Translation")
                rotation.plot(graph_data_r, label="Rotation")

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec, tvec, 0.01)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


ani = animation.FuncAnimation(fig, loop)
plt.show()
