export const en = {
  // App
  appName: 'Plumbline',
  appTagline: 'Free camera calibration, in your browser.',
  privacyNotice: 'All processing happens locally in your browser. Your images are never uploaded.',

  // Board params
  boardConfig: 'Board Configuration',
  squaresX: 'Squares X',
  squaresY: 'Squares Y',
  squareLength: 'Square Length (mm)',
  markerLength: 'Marker Length (mm)',
  dictionary: 'ArUco Dictionary',
  legacyPattern: 'Legacy pattern (pre-OpenCV 4.6)',
  calibtoolsBoard: 'Calib.Tools board preset',
  calibtoolsPlaceholder: 'No presets available yet',
  validationMarkerTooLarge: 'Marker length must be less than square length.',
  validationMinDimension: 'Board must be at least 3×3.',
  validationPositive: 'All dimensions must be positive.',

  // Dictionaries
  dict_4X4_50: 'DICT_4X4_50 (50 markers)',
  dict_4X4_100: 'DICT_4X4_100 (100 markers)',
  dict_4X4_250: 'DICT_4X4_250 (250 markers)',
  dict_4X4_1000: 'DICT_4X4_1000 (1000 markers)',
  dict_5X5_50: 'DICT_5X5_50 (50 markers)',
  dict_5X5_100: 'DICT_5X5_100 (100 markers)',
  dict_5X5_250: 'DICT_5X5_250 (250 markers)',
  dict_5X5_1000: 'DICT_5X5_1000 (1000 markers)',
  dict_6X6_50: 'DICT_6X6_50 (50 markers)',
  dict_6X6_100: 'DICT_6X6_100 (100 markers)',
  dict_6X6_250: 'DICT_6X6_250 (250 markers)',
  dict_6X6_1000: 'DICT_6X6_1000 (1000 markers)',
  dict_7X7_50: 'DICT_7X7_50 (50 markers)',
  dict_7X7_100: 'DICT_7X7_100 (100 markers)',
  dict_7X7_250: 'DICT_7X7_250 (250 markers)',
  dict_7X7_1000: 'DICT_7X7_1000 (1000 markers)',
  dict_ARUCO_ORIGINAL: 'DICT_ARUCO_ORIGINAL',
  dict_APRILTAG_16h5: 'DICT_APRILTAG_16h5',
  dict_APRILTAG_25h9: 'DICT_APRILTAG_25h9',
  dict_APRILTAG_36h10: 'DICT_APRILTAG_36h10',
  dict_APRILTAG_36h11: 'DICT_APRILTAG_36h11',
  dict_ARUCO_MIP_36h12: 'DICT_ARUCO_MIP_36h12',

  // Image import
  importImages: 'Import Images',
  dropOrSelect: 'Drag & drop images here, or click to select',
  supportedFormats: 'JPG and PNG supported. HEIC is not supported.',
  heicNotSupported: 'HEIC format is not supported. Please convert to JPG or PNG first.',
  processing: 'Processing...',
  detecting: 'Detecting corners...',

  // Detection results
  cornersDetected: 'corners detected',
  included: 'Included',
  excluded: 'Excluded',
  excludeReasonFewCorners: 'Too few corners detected (minimum 6 required).',
  excludeReasonNoPnP: 'Insufficient unique IDs to solve PnP.',
  excludeReasonManual: 'Manually excluded.',
  clickToView: 'Click to view with overlay',

  // Calibration
  calibrate: 'Calibrate',
  calibrating: 'Calibrating...',
  calibrationModel: 'Calibration Model',
  modelStandard: 'Standard (k1 k2 p1 p2 k3)',
  fixK3: 'Fix k3 = 0',
  fixTangential: 'Fix tangential (p1 = p2 = 0)',
  rationalModel: 'Rational model (k4 k5 k6)',
  thinPrism: 'Thin prism (s1 s2 s3 s4)',
  recalibrate: 'Recalibrate',

  // Results
  results: 'Calibration Results',
  cameraMatrix: 'Camera Matrix',
  distortionCoeffs: 'Distortion Coefficients',
  overallRMS: 'Overall RMS Reprojection Error',
  perImageRMS: 'Per-Image RMS',
  imageSize: 'Image Size',
  clickToExclude: 'Click a bar to exclude that image and recalibrate.',

  // Coverage heatmap
  coverageHeatmap: 'Corner Coverage Heatmap',
  coverageGridSize: 'Grid size',
  coverageAdvice: 'Coverage advice',
  coverageGood: 'Good coverage across the image.',
  coverageSparse: (region: string) =>
    `${region} has very few corner detections. Try placing the board in that area for a few more shots.`,

  // Tilt distribution
  tiltDistribution: 'Board Tilt Distribution',
  tiltAdviceMostlyFrontal:
    'Most images are nearly frontal (< 15°). Add a few shots at 30–45° tilt for better calibration.',
  tiltAdviceGood: 'Good tilt variety across images.',

  // Export
  exportResults: 'Export',
  exportOpenCVYAML: 'OpenCV YAML',
  exportOpenCVJSON: 'OpenCV JSON',
  exportROSYAML: 'ROS camera_info YAML',
  exportPlumblineJSON: 'Plumbline JSON',

  // Footer
  footer:
    'Plumbline is a free, open-source tool by Calib.Tools · Powered by OpenCV · MIT License',

  // PWA
  install: 'Install',

  // General
  px: 'px',
  mm: 'mm',
  degrees: '°',
} as const;

export type I18nKey = keyof typeof en;
