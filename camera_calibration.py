import cv2 as cv
import numpy as np


CHECKERBOARD = (7, 5)
MAX_SAMPLES = 30
MIN_SAMPLES = 10
OUTPUT_VIDEO = "camera_calibration_result.mp4"
INPUT_VIDEO = "chessboard.mp4"


def build_object_points(pattern_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    return objp


def select_evenly_spaced(items, limit):
    if len(items) <= limit:
        return items

    indices = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[i] for i in indices]


def compute_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0.0

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv.projectPoints(objp, rvec, tvec, mtx, dist)
        total_error += cv.norm(imgp, projected, cv.NORM_L2) / len(projected)

    return total_error / len(objpoints)


def main():
    objp = build_object_points(CHECKERBOARD)
    detected_samples = []

    cap = cv.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Cannot open calibration video: {INPUT_VIDEO}")
        raise SystemExit

    print("Collecting chessboard corners... (ESC to quit)")

    image_size = None
    frame_index = 0
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]
            writer = cv.VideoWriter(
                OUTPUT_VIDEO,
                cv.VideoWriter_fourcc(*"mp4v"),
                30.0,
                image_size,
            )

        found, corners = cv.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            corners = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            detected_samples.append((frame_index, corners))
            cv.drawChessboardCorners(frame, CHECKERBOARD, corners, found)

        cv.putText(
            frame,
            f"Detected frames: {len(detected_samples)}",
            (10, 30),
            cv.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 255, 0),
            1,
        )

        if writer is not None:
            writer.write(frame)

        cv.imshow("Calibration", frame)
        if cv.waitKey(30) & 0xFF == 27:
            print("Stopped by ESC.")
            break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()

    print("\n=== Detection Summary ===")
    print("Detected frames:", len(detected_samples))

    if len(detected_samples) < MIN_SAMPLES:
        print(f"Not enough valid samples. Need at least {MIN_SAMPLES}.")
        raise SystemExit

    if image_size is None:
        print("Failed to infer image size.")
        raise SystemExit

    selected_samples = select_evenly_spaced(detected_samples, MAX_SAMPLES)
    selected_indices = [index for index, _ in selected_samples]
    objpoints = [objp.copy() for _ in selected_samples]
    imgpoints = [corners for _, corners in selected_samples]

    print("Selected frame indices:")
    print(selected_indices)
    print("\nRunning camera calibration...")

    rmse, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    mean_error = compute_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    print("Calibration finished.")
    print("\n=== Camera Matrix ===")
    print(mtx)
    print("\n=== Distortion Coefficients ===")
    print(dist)
    print("\n=== RMSE ===")
    print(rmse)
    print("\n=== Mean Reprojection Error ===")
    print(mean_error)

    np.savez("calibration_data.npz", mtx=mtx, dist=dist, image_size=np.array(image_size))
    print("\nSaved calibration_data.npz")
    print(f"Saved {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
