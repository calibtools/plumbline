import { useState, useCallback } from 'preact/hooks';
import type {
  BoardConfig,
  CalibrationFlags,
  CalibrationResult,
  ImageDetection,
} from '../core/types';
import { en } from '../i18n/en';

interface Props {
  detections: ImageDetection[];
  boardConfig: BoardConfig;
  onResult: (result: CalibrationResult) => void;
  isProcessing: boolean;
  setIsProcessing: (v: boolean) => void;
  setProgressMsg: (v: string) => void;
}

export function CalibrationPanel({
  detections,
  boardConfig,
  onResult,
  isProcessing,
  setIsProcessing,
  setProgressMsg,
}: Props) {
  const [flags, setFlags] = useState<CalibrationFlags>({
    fixK3: false,
    fixTangential: false,
    rationalModel: false,
    thinPrism: false,
  });

  const included = detections.filter((d) => d.included);

  const runCalibration = useCallback(async () => {
    if (included.length === 0) return;

    setIsProcessing(true);
    setProgressMsg(en.calibrating);

    // Allow UI to update
    await new Promise((r) => requestAnimationFrame(r));

    try {
      const { calibrate } = await import('../core/calibration');
      const imageSize = { width: included[0].width, height: included[0].height };
      const result = calibrate(detections, boardConfig, flags, imageSize);
      onResult(result);
    } catch (err) {
      console.error('Calibration failed:', err);
      alert('Calibration failed: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setIsProcessing(false);
      setProgressMsg('');
    }
  }, [included, detections, boardConfig, flags, onResult, setIsProcessing, setProgressMsg]);

  return (
    <section>
      <h2>{en.calibrationModel}</h2>
      <div class="calib-options">
        <div class="checkbox-group">
          <input
            type="checkbox"
            id="fixK3"
            checked={flags.fixK3}
            onChange={(e) =>
              setFlags({ ...flags, fixK3: (e.target as HTMLInputElement).checked })
            }
          />
          <label for="fixK3">{en.fixK3}</label>
        </div>
        <div class="checkbox-group">
          <input
            type="checkbox"
            id="fixTangential"
            checked={flags.fixTangential}
            onChange={(e) =>
              setFlags({ ...flags, fixTangential: (e.target as HTMLInputElement).checked })
            }
          />
          <label for="fixTangential">{en.fixTangential}</label>
        </div>
        <div class="checkbox-group">
          <input
            type="checkbox"
            id="rationalModel"
            checked={flags.rationalModel}
            onChange={(e) =>
              setFlags({ ...flags, rationalModel: (e.target as HTMLInputElement).checked })
            }
          />
          <label for="rationalModel">{en.rationalModel}</label>
        </div>
        <div class="checkbox-group">
          <input
            type="checkbox"
            id="thinPrism"
            checked={flags.thinPrism}
            onChange={(e) =>
              setFlags({ ...flags, thinPrism: (e.target as HTMLInputElement).checked })
            }
          />
          <label for="thinPrism">{en.thinPrism}</label>
        </div>
      </div>
      <div>
        <span style={{ fontSize: '0.8rem', color: '#888', marginRight: '0.75rem' }}>
          {included.length} image{included.length !== 1 ? 's' : ''} included
        </span>
        <button class="btn-primary" onClick={runCalibration} disabled={isProcessing || included.length === 0}>
          {en.calibrate}
        </button>
      </div>
    </section>
  );
}
