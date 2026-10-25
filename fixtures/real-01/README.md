## real-01 fixture

Place 20-30 real camera images of a ChArUco board in this directory.

### Board parameters

| Parameter     | Value         |
|---------------|---------------|
| Squares X     | 5             |
| Squares Y     | 7             |
| Square length | 30 mm         |
| Marker length | 22 mm         |
| Dictionary    | DICT_4X4_50   |

Images should be PNG or JPEG, all taken with the same camera at the same
resolution without changing zoom or focus between shots.  Vary the board
angle and distance across the frame for best calibration coverage.

The reference calibration script (`tools/reference/calibrate_reference.py`)
will read every image file from this directory, detect ChArUco corners, and
write `reference.json` alongside the images.
