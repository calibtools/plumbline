import { useState, useCallback } from 'preact/hooks';
import type {
  BoardConfig,
  CalibrationResult,
  ImageDetection,
} from '../core/types';
import { en } from '../i18n/en';
import { BoardForm } from './BoardForm';
import { ImageImport } from './ImageImport';
import { CalibrationPanel } from './CalibrationPanel';
import { ResultsPanel } from './ResultsPanel';
import { Footer } from './Footer';
import { InstallButton } from './InstallButton';

export function App() {
  const [boardConfig, setBoardConfig] = useState<BoardConfig>({
    squaresX: 5,
    squaresY: 7,
    squareLength: 30,
    markerLength: 22,
    dictionaryId: 0, // DICT_4X4_50
    legacyPattern: false,
  });

  const [detections, setDetections] = useState<ImageDetection[]>([]);
  const [imageDataMap, setImageDataMap] = useState<Map<string, string>>(new Map());
  const [calibResult, setCalibResult] = useState<CalibrationResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressMsg, setProgressMsg] = useState('');

  const handleDetections = useCallback(
    (newDetections: ImageDetection[], thumbnails: Map<string, string>) => {
      setDetections((prev) => [...prev, ...newDetections]);
      setImageDataMap((prev) => {
        const next = new Map(prev);
        thumbnails.forEach((v, k) => next.set(k, v));
        return next;
      });
    },
    [],
  );

  const handleToggleImage = useCallback(
    (filename: string) => {
      setDetections((prev) =>
        prev.map((d) => {
          if (d.filename !== filename) return d;
          return {
            ...d,
            included: !d.included,
            excludeReason: d.included ? 'Manually excluded.' : undefined,
          };
        }),
      );
      setCalibResult(null); // Clear stale result
    },
    [],
  );

  const handleCalibrationResult = useCallback((result: CalibrationResult) => {
    setCalibResult(result);
  }, []);

  return (
    <>
      <header class="header">
        <div class="header-left">
          <h1>{en.appName}</h1>
          <span class="tagline">{en.appTagline}</span>
        </div>
        <InstallButton />
      </header>

      <div class="privacy-notice">{en.privacyNotice}</div>

      <BoardForm config={boardConfig} onChange={setBoardConfig} />

      <ImageImport
        boardConfig={boardConfig}
        onDetections={handleDetections}
        isProcessing={isProcessing}
        setIsProcessing={setIsProcessing}
        progressMsg={progressMsg}
        setProgressMsg={setProgressMsg}
      />

      {detections.length > 0 && (
        <ImageGrid
          detections={detections}
          imageDataMap={imageDataMap}
          onToggle={handleToggleImage}
        />
      )}

      {detections.filter((d) => d.included).length > 0 && (
        <CalibrationPanel
          detections={detections}
          boardConfig={boardConfig}
          onResult={handleCalibrationResult}
          isProcessing={isProcessing}
          setIsProcessing={setIsProcessing}
          setProgressMsg={setProgressMsg}
        />
      )}

      {calibResult && (
        <ResultsPanel
          result={calibResult}
          detections={detections}
          onToggleImage={handleToggleImage}
        />
      )}

      <Footer />
    </>
  );
}

// Image thumbnail grid with overlay modal
function ImageGrid({
  detections,
  imageDataMap,
  onToggle,
}: {
  detections: ImageDetection[];
  imageDataMap: Map<string, string>;
  onToggle: (filename: string) => void;
}) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const selected = detections.find((d) => d.filename === selectedImage);

  return (
    <>
      <h2>{en.importImages} ({detections.length})</h2>
      <div class="image-grid">
        {detections.map((det) => (
          <div
            key={det.filename}
            class={`image-card ${det.included ? '' : 'excluded'}`}
            onClick={() => setSelectedImage(det.filename)}
          >
            {imageDataMap.has(det.filename) && (
              <img src={imageDataMap.get(det.filename)} alt={det.filename} />
            )}
            <div class="image-card-info">
              <div class="corner-count">
                {det.corners.length} {en.cornersDetected}
              </div>
              <div class={`status ${det.included ? '' : 'excluded-status'}`}>
                {det.included ? en.included : en.excluded}
              </div>
              {det.excludeReason && (
                <div class="status excluded-status">{det.excludeReason}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <div class="modal-overlay" onClick={() => setSelectedImage(null)}>
          <div class="modal-content" onClick={(e) => e.stopPropagation()}>
            <CornerOverlayCanvas
              detection={selected}
              imageSrc={imageDataMap.get(selected.filename) || ''}
            />
            <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
              <button
                class={`btn-small ${selected.included ? 'btn-secondary' : 'btn-primary'}`}
                onClick={() => onToggle(selected.filename)}
              >
                {selected.included ? 'Exclude' : 'Include'}
              </button>
              <button
                class="btn-small btn-secondary"
                style={{ marginLeft: '0.5rem' }}
                onClick={() => setSelectedImage(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function CornerOverlayCanvas({
  detection,
  imageSrc,
}: {
  detection: ImageDetection;
  imageSrc: string;
}) {
  const canvasRef = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas) return;
      const img = new Image();
      img.onload = () => {
        // Scale to fit viewport
        const maxW = window.innerWidth * 0.85;
        const maxH = window.innerHeight * 0.8;
        const scale = Math.min(1, maxW / img.width, maxH / img.height);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw corners
        const sx = (canvas.width / detection.width);
        const sy = (canvas.height / detection.height);
        for (const corner of detection.corners) {
          const x = corner.x * sx;
          const y = corner.y * sy;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#E60012';
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1;
          ctx.stroke();

          // ID label
          ctx.font = '10px "Geist Mono", monospace';
          ctx.fillStyle = '#fff';
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 2;
          ctx.strokeText(String(corner.id), x + 6, y - 4);
          ctx.fillText(String(corner.id), x + 6, y - 4);
        }
      };
      img.src = imageSrc;
    },
    [detection, imageSrc],
  );

  return <canvas ref={canvasRef} />;
}
