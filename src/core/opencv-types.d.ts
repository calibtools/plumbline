/** Minimal type declarations for opencv.js 4.9.0 APIs used by Plumbline */

declare namespace cv {
  class Mat {
    constructor();
    constructor(rows: number, cols: number, type: number, scalar?: Scalar);
    rows: number;
    cols: number;
    data: Uint8Array;
    data32F: Float32Array;
    data64F: Float64Array;
    data32S: Int32Array;
    delete(): void;
    type(): number;
    channels(): number;
    clone(): Mat;
    floatAt(row: number, col: number): number;
    doubleAt(row: number, col: number): number;
    intAt(row: number, col: number): number;
    ucharAt(row: number, col: number): number;
  }

  class MatVector {
    constructor();
    size(): number;
    get(index: number): Mat;
    push_back(mat: Mat): void;
    delete(): void;
  }

  class Size {
    constructor(width: number, height: number);
    width: number;
    height: number;
  }

  class Scalar {
    constructor(v0: number, v1?: number, v2?: number, v3?: number);
  }

  class TermCriteria {
    constructor(type: number, maxCount: number, epsilon: number);
  }

  // Constants
  const CV_8UC1: number;
  const CV_8UC3: number;
  const CV_8UC4: number;
  const CV_32FC1: number;
  const CV_32FC2: number;
  const CV_32FC3: number;
  const CV_32SC1: number;
  const CV_64FC1: number;

  const COLOR_RGBA2GRAY: number;
  const COLOR_GRAY2RGBA: number;
  const COLOR_RGBA2BGR: number;
  const COLOR_BGR2RGBA: number;

  const TermCriteria_COUNT: number;
  const TermCriteria_EPS: number;

  // Calibration flags
  const CALIB_FIX_K3: number;
  const CALIB_ZERO_TANGENT_DIST: number;
  const CALIB_RATIONAL_MODEL: number;
  const CALIB_THIN_PRISM_MODEL: number;

  // Dictionary constants
  const DICT_4X4_50: number;
  const DICT_4X4_100: number;
  const DICT_4X4_250: number;
  const DICT_4X4_1000: number;
  const DICT_5X5_50: number;
  const DICT_5X5_100: number;
  const DICT_5X5_250: number;
  const DICT_5X5_1000: number;
  const DICT_6X6_50: number;
  const DICT_6X6_100: number;
  const DICT_6X6_250: number;
  const DICT_6X6_1000: number;
  const DICT_7X7_50: number;
  const DICT_7X7_100: number;
  const DICT_7X7_250: number;
  const DICT_7X7_1000: number;
  const DICT_ARUCO_ORIGINAL: number;
  const DICT_APRILTAG_16h5: number;
  const DICT_APRILTAG_25h9: number;
  const DICT_APRILTAG_36h10: number;
  const DICT_APRILTAG_36h11: number;
  const DICT_ARUCO_MIP_36h12: number;

  // Functions
  function getPredefinedDictionary(dictId: number): Mat;
  function matFromArray(rows: number, cols: number, type: number, data: number[]): Mat;
  function matFromImageData(imageData: ImageData): Mat;
  function cvtColor(src: Mat, dst: Mat, code: number): void;
  function resize(src: Mat, dst: Mat, dsize: Size, fx?: number, fy?: number, interpolation?: number): void;
  function GaussianBlur(src: Mat, dst: Mat, ksize: Size, sigmaX: number): void;

  function calibrateCameraExtended(
    objectPoints: MatVector,
    imagePoints: MatVector,
    imageSize: Size,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    rvecs: MatVector,
    tvecs: MatVector,
    stdDeviationsIntrinsics: Mat,
    stdDeviationsExtrinsics: Mat,
    perViewErrors: Mat,
    flags: number,
    criteria: TermCriteria,
  ): number;

  function solvePnP(
    objectPoints: Mat,
    imagePoints: Mat,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    rvec: Mat,
    tvec: Mat,
  ): boolean;

  function Rodrigues(src: Mat, dst: Mat): void;
  function projectPoints(
    objectPoints: Mat,
    rvec: Mat,
    tvec: Mat,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    imagePoints: Mat,
  ): void;

  function drawDetectedCornersCharuco(
    image: Mat,
    charucoCorners: Mat,
    charucoIds: Mat,
    cornerColor?: Scalar,
  ): void;

  // ArUco classes
  class aruco_Dictionary {
    constructor();
    markerSize: number;
    bytesList: Mat;
    maxCorrectionBits: number;
    delete(): void;
  }

  class aruco_DetectorParameters {
    constructor();
    delete(): void;
  }

  class aruco_CharucoParameters {
    constructor();
    delete(): void;
  }

  class aruco_RefineParameters {
    constructor(minRepDistance: number, errorCorrectionRate: number, checkAllOrders: boolean);
    minRepDistance: number;
    errorCorrectionRate: number;
    checkAllOrders: boolean;
    delete(): void;
  }

  class aruco_CharucoBoard {
    constructor(size: Size, squareLength: number, markerLength: number, dictionary: unknown, ids: Mat);
    generateImage(outSize: Size, img: Mat, marginSize: number, borderBits: number): void;
    setLegacyPattern(legacy: boolean): void;
    getLegacyPattern(): boolean;
    checkCharucoCornersCollinear(charucoIds: Mat): boolean;
    delete(): void;
  }

  class aruco_CharucoDetector {
    constructor(
      board: aruco_CharucoBoard,
      charucoParams: aruco_CharucoParameters,
      detectorParams: aruco_DetectorParameters,
      refineParams: aruco_RefineParameters,
    );
    detectBoard(
      image: Mat,
      charucoCorners: Mat,
      charucoIds: Mat,
      markerCorners: MatVector,
      markerIds: Mat,
    ): void;
    delete(): void;
  }
}
