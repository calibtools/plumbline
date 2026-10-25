# Plumbline

**Free camera calibration, in your browser.**

Plumbline is an open-source tool for single-camera calibration using ChArUco boards. All image processing happens locally in your browser — your pictures never leave your machine. Detection and calibration run in a Web Worker so the UI stays responsive. Works offline as a PWA.

Powered by [OpenCV](https://opencv.org) (opencv.js 4.9.0, single-file WASM). Made by [Calib.Tools](https://calib.tools).

## P1 Scope

- ChArUco board detection (all predefined ArUco dictionaries)
- Monocular camera calibration (standard, rational, thin prism models)
- Corner coverage heatmap with suggestions
- Board tilt distribution analysis
- Export: OpenCV YAML/JSON, ROS camera_info YAML, Plumbline JSON
- Installable PWA — works offline (Workbox precache)
- Web Worker for detection + calibration (main thread max gap: 20 ms)
- Consistency tests against Python OpenCV (camera matrix relative diff < 3e-9)

## Local Development

```bash
pnpm install
pnpm dev          # Start dev server at http://localhost:5173
```

## Testing

```bash
pnpm test                    # Unit tests
pnpm test:consistency        # Consistency tests (real src/core vs Python reference)
pnpm test:worker             # Worker responsiveness test
pnpm test:offline            # Offline PWA test (requires pnpm build first)
pnpm test:browser            # All browser-based tests
```

### Generating Reference Fixtures

```bash
cd tools/reference
pip install -r requirements.txt
python3 render_synthetic.py     # 25 synthetic ChArUco images (1280×960)
python3 calibrate_reference.py  # Reference calibration → reference.json
```

## License

MIT — see [LICENSE](LICENSE).
