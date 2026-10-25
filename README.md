# Plumbline

**Free camera calibration, in your browser.**

Plumbline is an open-source tool for single-camera calibration using ChArUco boards. All image processing happens locally in your browser — your pictures never leave your machine.

Powered by [OpenCV](https://opencv.org). Made by [Calib.Tools](https://calib.tools).

## P1 Scope

- ChArUco board detection (all predefined ArUco dictionaries)
- Monocular camera calibration (standard, rational, thin prism models)
- Corner coverage heatmap with suggestions
- Board tilt distribution analysis
- Export: OpenCV YAML/JSON, ROS camera_info YAML, Plumbline JSON
- Installable as a PWA (works offline)
- Consistency tests against Python OpenCV

## Local Development

```bash
pnpm install
pnpm dev          # Start dev server at http://localhost:5173
```

## Testing

```bash
pnpm test                    # Run unit tests
pnpm test:consistency        # Run consistency tests against Python reference
```

### Generating Reference Fixtures

```bash
cd tools/reference
pip install -r requirements.txt
python3 render_synthetic.py   # Generate synthetic ChArUco images
python3 calibrate_reference.py  # Run reference calibration
```

## How It Works

1. Configure your ChArUco board parameters (squares, marker size, dictionary)
2. Import images (drag & drop JPG/PNG)
3. Corners are detected using OpenCV's CharucoDetector
4. Review detected corners, exclude bad images
5. Run calibration — get camera matrix, distortion, per-image errors
6. Check the coverage heatmap and tilt distribution for quality assessment
7. Export results in your preferred format

## License

MIT — see [LICENSE](LICENSE).
