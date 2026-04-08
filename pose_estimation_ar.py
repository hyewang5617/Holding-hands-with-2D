from pathlib import Path

import cv2 as cv
import numpy as np


CHECKERBOARD = (7, 5)
SQUARE_SIZE = 1.0
INPUT_VIDEO = "chessboard.mp4"
OUTPUT_VIDEO = "pose_estimation_result.mp4"
ASSET_PATH = Path("assets/character.png")


def build_object_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    grid = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def load_character_rgba(asset_path):
    if not asset_path.exists():
        print(f"Character asset not found: {asset_path}")
        print("Place your PNG file at assets/character.png")
        raise SystemExit

    character = cv.imread(str(asset_path), cv.IMREAD_UNCHANGED)
    if character is None:
        print(f"Failed to load character asset: {asset_path}")
        raise SystemExit

    if character.ndim != 3 or character.shape[2] != 4:
        print("Character PNG must include an alpha channel.")
        raise SystemExit

    return character


def get_billboard_points(pattern_size, square_size, image_shape):
    board_width = (pattern_size[0] - 1) * square_size
    board_height = (pattern_size[1] - 1) * square_size

    img_h, img_w = image_shape[:2]
    aspect_ratio = img_h / img_w

    panel_width = board_width * 0.55
    panel_height = panel_width * aspect_ratio

    x0 = board_width * 0.2
    x1 = x0 + panel_width
    y_ground = board_height * 0.95
    z_top = -panel_height

    return np.float32(
        [
            [x0, y_ground, 0.0],
            [x1, y_ground, 0.0],
            [x1, y_ground, z_top],
            [x0, y_ground, z_top],
        ]
    )


def alpha_blend_warped(frame, warped_bgr, warped_alpha):
    alpha = warped_alpha.astype(np.float32) / 255.0
    alpha = alpha[..., None]
    frame[:] = (warped_bgr.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def overlay_character(frame, character_rgba, projected_quad):
    h, w = frame.shape[:2]
    src_h, src_w = character_rgba.shape[:2]

    src_quad = np.float32(
        [
            [0, src_h - 1],
            [src_w - 1, src_h - 1],
            [src_w - 1, 0],
            [0, 0],
        ]
    )
    dst_quad = np.float32(projected_quad).reshape(4, 2)

    H = cv.getPerspectiveTransform(src_quad, dst_quad)

    character_bgr = character_rgba[:, :, :3]
    character_alpha = character_rgba[:, :, 3]

    warped_bgr = cv.warpPerspective(
        character_bgr,
        H,
        (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv.warpPerspective(
        character_alpha,
        H,
        (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha_blend_warped(frame, warped_bgr, warped_alpha)


def draw_axes(frame, rvec, tvec, K, dist):
    axis = np.float32(
        [
            [0, 0, 0],
            [2.0, 0, 0],
            [0, 2.0, 0],
            [0, 0, -2.0],
        ]
    )
    axis_img, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
    axis_img = np.int32(axis_img).reshape(-1, 2)
    origin = tuple(axis_img[0])
    cv.line(frame, origin, tuple(axis_img[1]), (0, 0, 255), 3, cv.LINE_AA)
    cv.line(frame, origin, tuple(axis_img[2]), (0, 255, 0), 3, cv.LINE_AA)
    cv.line(frame, origin, tuple(axis_img[3]), (255, 0, 0), 3, cv.LINE_AA)


def main():
    data = np.load("calibration_data.npz")
    K = data["mtx"]
    dist = data["dist"]

    character_rgba = load_character_rgba(ASSET_PATH)
    board_points = build_object_points(CHECKERBOARD, SQUARE_SIZE)
    billboard_points = get_billboard_points(CHECKERBOARD, SQUARE_SIZE, character_rgba.shape)

    video = cv.VideoCapture(INPUT_VIDEO)
    if not video.isOpened():
        print(f"Cannot open input video: {INPUT_VIDEO}")
        raise SystemExit

    writer = None
    print("Running camera pose estimation and PNG AR overlay... (ESC to quit)")

    while True:
        valid, frame = video.read()
        if not valid:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
        )

        status_text = "Chessboard not found"

        if found:
            corners = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )

            success, rvec, tvec = cv.solvePnP(board_points, corners, K, dist)
            if success:
                projected_quad, _ = cv.projectPoints(billboard_points, rvec, tvec, K, dist)
                overlay_character(frame, character_rgba, projected_quad)
                draw_axes(frame, rvec, tvec, K, dist)
                cv.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
                status_text = "Pose estimated + character overlay"

        if writer is None:
            frame_size = (frame.shape[1], frame.shape[0])
            writer = cv.VideoWriter(
                OUTPUT_VIDEO,
                cv.VideoWriter_fourcc(*"mp4v"),
                30.0,
                frame_size,
            )

        cv.putText(frame, status_text, (10, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv.imshow("Camera Pose Estimation and AR", frame)
        writer.write(frame)

        if cv.waitKey(30) & 0xFF == 27:
            break

    video.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()

    print(f"Saved {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
