import argparse
import imutils
import cv2

DEBUG = True
DICTIONARY = cv2.aruco.DICT_5X5_100

if DEBUG:
    IMAGE = "images/test1.png"
    frame = cv2.imread(IMAGE)
else:
    cap = cv2.VideoCapture(0)

aruco_dict = cv2.aruco.Dictionary_get(DICTIONARY)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    if not DEBUG:
        ret, frame = cap.read()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    print("Corners:", corners, "Ids:", ids)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(100, 0, 240))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
