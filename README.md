# Holding-hands-with-2D

OpenCV-based Homework #4 project for camera pose estimation and AR, built on top of the calibration result from Homework #3.

## Main Submission Target

The main homework script is `pose_estimation_ar.py`.

It:

- detects checkerboard corners from `chessboard.mp4`,
- estimates camera pose with `cv.solvePnP()`,
- projects a PNG or GIF character panel with `cv.projectPoints()`,
- renders the AR result frame by frame.

## Repo Structure

- `pose_estimation_ar.py`
  - main HW4 script
  - camera pose estimation + PNG/GIF-based AR rendering
- `camera_calibration.py`
  - reused HW3 calibration script
  - regenerates `calibration_data.npz` if needed
- `distortion_correction.py`
  - optional helper script
  - used only to inspect or demonstrate distortion correction from HW3
- `calibration_data.npz`
  - saved camera matrix and distortion coefficients from HW3
- `chessboard.mp4`
  - input video for calibration and pose estimation
- `assets/character.gif` or `assets/character.png`
  - your AR character asset
  - GIF is preferred if you want animation
  - PNG also works

## Calibration Result Reused from HW3

Camera matrix:

```text
[[1.20552422e+03 0.00000000e+00 6.44908485e+02]
 [0.00000000e+00 1.21234616e+03 3.57918243e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
```

Distortion coefficients:

```text
[[ 2.74702871e-01 -1.82186265e+00  7.79881090e-04 -3.74130709e-03
   4.29483939e+00]]
```

## Method

1. Load the calibration result from `calibration_data.npz`.
2. Detect checkerboard corners using `cv.findChessboardCorners()`.
3. Refine the corners using `cv.cornerSubPix()`.
4. Estimate pose using `cv.solvePnP()`.
5. Render a PNG or GIF character as a vertical panel on the checkerboard plane.
6. Draw 3D axes for pose visualization.

The AR object is now designed to be your own character asset rather than a simple wireframe object.

## How To Run

Install dependencies:

```bash
python -m pip install opencv-python numpy
```

Run the main homework script:

```bash
python pose_estimation_ar.py
```

Before running it, place one of these files at:

```text
assets/character.gif
assets/character.png
```

Optional:

```bash
python camera_calibration.py
python distortion_correction.py
```

## Notes

- `distortion_correction.py` is not required for the core HW4 task.
- It is kept as an optional helper because the homework extends the previous calibration assignment.
- `character.gif` is loaded first if both files exist.
- For plain RGB PNG files, near-white background is treated as transparent automatically.
- The current repository is prepared for a 2D character-based AR presentation.
