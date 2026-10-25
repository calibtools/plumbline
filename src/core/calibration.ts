import type {
  BoardConfig,
  CalibrationFlags,
  CalibrationResult,
  ImageDetection,
  PerImageResult,
} from './types';
import { cornerIdToObjectPoint } from './detection';

const VERSION = '0.1.0';

/**
 * Run camera calibration using detected ChArUco corners across multiple images.
 */
export function calibrate(
  detections: ImageDetection[],
  boardConfig: BoardConfig,
  flags: CalibrationFlags,
  imageSize: { width: number; height: number },
): CalibrationResult {
  const included = detections.filter((d) => d.included);
  if (included.length === 0) {
    throw new Error('No images included in calibration.');
  }

  // Build object and image point arrays
  const objPtsVec = new cv.MatVector();
  const imgPtsVec = new cv.MatVector();

  for (const det of included) {
    const objArr: number[] = [];
    const imgArr: number[] = [];
    for (const corner of det.corners) {
      const obj = cornerIdToObjectPoint(corner.id, boardConfig);
      objArr.push(obj.x, obj.y, obj.z);
      imgArr.push(corner.x, corner.y);
    }
    const objMat = cv.matFromArray(det.corners.length, 1, cv.CV_32FC3, objArr);
    const imgMat = cv.matFromArray(det.corners.length, 1, cv.CV_32FC2, imgArr);
    objPtsVec.push_back(objMat);
    imgPtsVec.push_back(imgMat);
    objMat.delete();
    imgMat.delete();
  }

  // Build flags bitmask
  let cvFlags = 0;
  if (flags.fixK3) cvFlags |= cv.CALIB_FIX_K3;
  if (flags.fixTangential) cvFlags |= cv.CALIB_ZERO_TANGENT_DIST;
  if (flags.rationalModel) cvFlags |= cv.CALIB_RATIONAL_MODEL;
  if (flags.thinPrism) cvFlags |= cv.CALIB_THIN_PRISM_MODEL;

  const camMat = new cv.Mat();
  const distCoeffs = new cv.Mat();
  const rvecs = new cv.MatVector();
  const tvecs = new cv.MatVector();
  const stdDevIntr = new cv.Mat();
  const stdDevExtr = new cv.Mat();
  const perViewErr = new cv.Mat();

  const rms = cv.calibrateCameraExtended(
    objPtsVec,
    imgPtsVec,
    new cv.Size(imageSize.width, imageSize.height),
    camMat,
    distCoeffs,
    rvecs,
    tvecs,
    stdDevIntr,
    stdDevExtr,
    perViewErr,
    cvFlags,
    new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 100, 1e-8),
  );

  // Extract results
  const fx = camMat.doubleAt(0, 0);
  const fy = camMat.doubleAt(1, 1);
  const cx = camMat.doubleAt(0, 2);
  const cy = camMat.doubleAt(1, 2);

  const dist: number[] = [];
  for (let i = 0; i < distCoeffs.cols; i++) {
    dist.push(distCoeffs.doubleAt(0, i));
  }

  const perImage: PerImageResult[] = [];
  for (let i = 0; i < included.length; i++) {
    const rv = rvecs.get(i);
    const tv = tvecs.get(i);
    perImage.push({
      filename: included[i].filename,
      rms: perViewErr.doubleAt(i, 0),
      rvec: [rv.doubleAt(0, 0), rv.doubleAt(1, 0), rv.doubleAt(2, 0)],
      tvec: [tv.doubleAt(0, 0), tv.doubleAt(1, 0), tv.doubleAt(2, 0)],
      included: true,
    });
    rv.delete();
    tv.delete();
  }

  // Cleanup
  camMat.delete();
  distCoeffs.delete();
  rvecs.delete();
  tvecs.delete();
  stdDevIntr.delete();
  stdDevExtr.delete();
  perViewErr.delete();
  objPtsVec.delete();
  imgPtsVec.delete();

  return {
    cameraMatrix: { fx, fy, cx, cy },
    distCoeffs: dist,
    rms,
    imageSize,
    perImage,
    flags,
    boardConfig,
    version: VERSION,
  };
}

/**
 * Compute board tilt angle (degrees) from a rotation vector.
 * Tilt = angle between board normal and camera optical axis.
 */
export function computeTiltAngle(rvec: [number, number, number]): number {
  const rvecMat = cv.matFromArray(3, 1, cv.CV_64FC1, rvec);
  const rotMat = new cv.Mat();
  cv.Rodrigues(rvecMat, rotMat);

  // Board normal in board frame is [0, 0, 1].
  // In camera frame: R * [0,0,1] = third column of R.
  const nz = rotMat.doubleAt(2, 2); // z-component of transformed normal

  rvecMat.delete();
  rotMat.delete();

  // Angle between normal and optical axis [0,0,1]
  return (Math.acos(Math.abs(nz)) * 180) / Math.PI;
}
