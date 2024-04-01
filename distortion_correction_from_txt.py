import numpy as np
import cv2 as cv

if __name__ == "__main__":
    # The given video and calibration data
    video_file = 'data/chessboard.MOV'

    # Results saved from camera calibration
    file_name = "result/calibration_results.txt"

    # 데이터 불러오기
    rms = None
    K = []
    dist_coeff = []

    try:
        with open(file_name, "r") as file:
            for line in file:
                if line.startswith("RMS:"):
                    rms = float(line.split(":")[1].strip())
                elif line.startswith("Camera matrix (K):"):
                    for _ in range(3):
                        K.append(list(map(float, file.readline().strip().split())))
                elif line.startswith("Distortion coefficient:"):
                    dist_coeff = list(map(float, file.readline().strip().split()))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("File not found. Check file path or run camera calibration first!")

    print("Loaded RMS:", rms)
    print("Loaded Camera matrix (K):", K)
    print("Loaded Distortion coefficient:", dist_coeff)

    K = np.array(K)
    dist_coeff = np.array(dist_coeff)

    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    width = video.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter('data/output.mov', fourcc, 30.0, (int(width), int(height)))

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
