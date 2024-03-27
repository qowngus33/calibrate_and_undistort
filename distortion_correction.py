import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'data/chessboard.MOV'

K = np.array([[1.91649207e+03, 0.00000000e+00, 9.65930893e+02],
            [0.00000000e+00, 1.90596824e+03, 5.44749588e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) # Derived from `calibrate_camera.py`

dist_coeff = np.array([0.11857126, -0.25188666, -0.00044333, -0.00031169, -0.24261628])

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

width = video.get(cv.CAP_PROP_FRAME_WIDTH)
height = video.get(cv.CAP_PROP_FRAME_HEIGHT)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('data/output.avi', fourcc, 30.0, (int(width), int(height)))

# Run distortion correction
show_rectify = True
map1, map2 = None, None
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Rectify geometric distortion (Alternative: `cv.undistort()`)
    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = "Rectified"
    cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow("Geometric Distortion Correction", img)
    out.write(img)
    key = cv.waitKey(10)
    if key == ord(' '):     # Space: Pause
        key = cv.waitKey()
    if key == 27:           # ESC: Exit
        break
    elif key == ord('\t'):  # Tab: Toggle the mode
        show_rectify = not show_rectify

video.release()
out.release()
cv.destroyAllWindows()
