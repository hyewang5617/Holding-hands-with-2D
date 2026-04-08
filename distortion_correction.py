import cv2 as cv
import numpy as np


INPUT_VIDEO = "chessboard.mp4"


def main():
    data = np.load("calibration_data.npz")
    K = data["mtx"]
    dist = data["dist"]

    video = cv.VideoCapture(INPUT_VIDEO)
    if not video.isOpened():
        print(f"Cannot open input video: {INPUT_VIDEO}")
        raise SystemExit

    print("Running distortion correction... (ESC to quit)")

    map1 = None
    map2 = None

    while True:
        valid, img = video.read()
        if not valid:
            break

        if map1 is None or map2 is None:
            image_size = (img.shape[1], img.shape[0])
            map1, map2 = cv.initUndistortRectifyMap(
                K, dist, None, None, image_size, cv.CV_32FC1
            )

        img_rectified = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)

        cv.putText(img, "Original", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        cv.putText(img_rectified, "Rectified", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        display = np.hstack((img, img_rectified))
        display = cv.resize(display, None, fx=0.5, fy=0.5)
        cv.imshow("Distortion Correction", display)

        if cv.waitKey(30) & 0xFF == 27:
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
