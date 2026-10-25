import type { CalibrationResult, ImageDetection } from './types';

/**
 * Export calibration results in OpenCV YAML format.
 */
export function toOpenCVYAML(result: CalibrationResult): string {
  const { cameraMatrix: cm, distCoeffs: dc, imageSize: sz } = result;
  return `%YAML:1.0
---
image_width: ${sz.width}
image_height: ${sz.height}
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ ${cm.fx}, 0., ${cm.cx},
           0., ${cm.fy}, ${cm.cy},
           0., 0., 1. ]
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: ${dc.length}
   dt: d
   data: [ ${dc.join(', ')} ]
rms_reprojection_error: ${result.rms}
`;
}

/**
 * Export calibration results in OpenCV JSON format.
 */
export function toOpenCVJSON(result: CalibrationResult): string {
  const { cameraMatrix: cm, distCoeffs: dc, imageSize: sz } = result;
  return JSON.stringify(
    {
      image_width: sz.width,
      image_height: sz.height,
      camera_matrix: {
        rows: 3,
        cols: 3,
        data: [cm.fx, 0, cm.cx, 0, cm.fy, cm.cy, 0, 0, 1],
      },
      distortion_coefficients: {
        rows: 1,
        cols: dc.length,
        data: dc,
      },
      rms_reprojection_error: result.rms,
    },
    null,
    2,
  );
}

/**
 * Export calibration results in ROS camera_info YAML format.
 */
export function toROSCameraInfoYAML(result: CalibrationResult): string {
  const { cameraMatrix: cm, distCoeffs: dc, imageSize: sz } = result;
  // Pad distortion to 5 coefficients if needed
  const d = [...dc];
  while (d.length < 5) d.push(0);

  return `image_width: ${sz.width}
image_height: ${sz.height}
camera_name: plumbline
camera_matrix:
  rows: 3
  cols: 3
  data: [${cm.fx}, 0, ${cm.cx}, 0, ${cm.fy}, ${cm.cy}, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [${d.slice(0, 5).join(', ')}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [${cm.fx}, 0, ${cm.cx}, 0, 0, ${cm.fy}, ${cm.cy}, 0, 0, 0, 1, 0]
`;
}

/**
 * Export full Plumbline JSON (includes all inputs, detections, and results for reproducibility).
 */
export function toPlumblineJSON(
  result: CalibrationResult,
  detections: ImageDetection[],
): string {
  return JSON.stringify(
    {
      plumbline_version: result.version,
      board_config: result.boardConfig,
      calibration_flags: result.flags,
      image_size: result.imageSize,
      camera_matrix: result.cameraMatrix,
      distortion_coefficients: result.distCoeffs,
      rms: result.rms,
      per_image: result.perImage,
      detections: detections.map((d) => ({
        filename: d.filename,
        width: d.width,
        height: d.height,
        corners: d.corners,
        marker_count: d.markerCount,
        included: d.included,
        exclude_reason: d.excludeReason || null,
      })),
    },
    null,
    2,
  );
}

/**
 * Trigger a file download in the browser.
 */
export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
