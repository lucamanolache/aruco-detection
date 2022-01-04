# Aruco Detection
A repository to play around with aruco markers.

## Calibration

Run `python3 calibration.py` to calibrate the camera. To use this type c to take pictures of a chess board. Get 20+, the more the merrier.

## Running

Run `python3 find_translation.py` to find the translation and warp the image. The resulting image should be translated as long as the aruco board is detected.
