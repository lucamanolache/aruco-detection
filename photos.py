import argparse
import imutils
import cv2

cap = cv2.VideoCapture(4)
x = 0

while True:
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        cv2.imwrite("calibration/calibration{}.jpg".format(x), frame)
        x += 1
