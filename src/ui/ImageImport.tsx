import { useRef, useCallback } from 'preact/hooks';
import type { BoardConfig, ImageDetection } from '../core/types';
import { en } from '../i18n/en';

interface Props {
  boardConfig: BoardConfig;
  onDetections: (detections: ImageDetection[], thumbnails: Map<string, string>) => void;
  isProcessing: boolean;
  setIsProcessing: (v: boolean) => void;
  progressMsg: string;
  setProgressMsg: (v: string) => void;
}

export function ImageImport({
  boardConfig,
  onDetections,
  isProcessing,
  setIsProcessing,
  progressMsg,
  setProgressMsg,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  const processFiles = useCallback(
    async (files: File[]) => {
      // Filter for supported formats
      const supported = files.filter((f) => {
        const ext = f.name.toLowerCase();
        return ext.endsWith('.jpg') || ext.endsWith('.jpeg') || ext.endsWith('.png');
      });

      const heicFiles = files.filter((f) => {
        const ext = f.name.toLowerCase();
        return ext.endsWith('.heic') || ext.endsWith('.heif');
      });

      if (heicFiles.length > 0) {
        alert(en.heicNotSupported);
      }

      if (supported.length === 0) return;

      setIsProcessing(true);
      setProgressMsg(`Processing 0/${supported.length}...`);

      const newDetections: ImageDetection[] = [];
      const thumbnails = new Map<string, string>();

      // Wait for opencv.js to be ready
      await waitForOpenCV();

      // Process sequentially to manage memory
      for (let i = 0; i < supported.length; i++) {
        const file = supported[i];
        setProgressMsg(`${en.detecting} ${i + 1}/${supported.length}: ${file.name}`);

        try {
          const { imageData, width, height, thumbnailUrl } = await loadImage(file);

          // Create thumbnail
          thumbnails.set(file.name, thumbnailUrl);

          // Run detection in a web worker
          const detection = await detectInWorker(imageData, width, height, file.name, boardConfig);
          newDetections.push(detection);
        } catch (err) {
          console.error(`Failed to process ${file.name}:`, err);
        }
      }

      onDetections(newDetections, thumbnails);
      setIsProcessing(false);
      setProgressMsg('');
    },
    [boardConfig, onDetections, setIsProcessing, setProgressMsg],
  );

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      dropRef.current?.classList.remove('dragging');
      const files = Array.from(e.dataTransfer?.files || []);
      processFiles(files);
    },
    [processFiles],
  );

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    dropRef.current?.classList.add('dragging');
  }, []);

  const handleDragLeave = useCallback(() => {
    dropRef.current?.classList.remove('dragging');
  }, []);

  const handleFileSelect = useCallback(
    (e: Event) => {
      const input = e.target as HTMLInputElement;
      const files = Array.from(input.files || []);
      processFiles(files);
      input.value = ''; // Reset so same files can be selected again
    },
    [processFiles],
  );

  return (
    <section>
      <h2>{en.importImages}</h2>
      <div
        ref={dropRef}
        class="drop-zone"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png"
          multiple
          onChange={handleFileSelect}
        />
        <div>{en.dropOrSelect}</div>
        <div style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>
          {en.supportedFormats}
        </div>
      </div>

      {isProcessing && (
        <div style={{ marginTop: '0.5rem' }}>
          <div class="progress-bar">
            <div class="progress-bar-fill" style={{ width: '100%', animation: 'pulse 1.5s ease-in-out infinite' }} />
          </div>
          <div class="progress-text">{progressMsg}</div>
        </div>
      )}
    </section>
  );
}

function waitForOpenCV(): Promise<void> {
  return new Promise((resolve) => {
    const check = () => {
      if (typeof cv !== 'undefined' && typeof cv.Mat === 'function') {
        resolve();
      } else {
        setTimeout(check, 100);
      }
    };
    check();
  });
}

function loadImage(
  file: File,
): Promise<{ imageData: Uint8ClampedArray; width: number; height: number; thumbnailUrl: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        // Draw to canvas to get pixel data
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height).data;

        // Create thumbnail
        const thumbCanvas = document.createElement('canvas');
        const thumbScale = Math.min(1, 300 / Math.max(img.width, img.height));
        thumbCanvas.width = img.width * thumbScale;
        thumbCanvas.height = img.height * thumbScale;
        const thumbCtx = thumbCanvas.getContext('2d')!;
        thumbCtx.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
        const thumbnailUrl = thumbCanvas.toDataURL('image/jpeg', 0.7);

        resolve({
          imageData: new Uint8ClampedArray(imageData),
          width: img.width,
          height: img.height,
          thumbnailUrl,
        });
      };
      img.onerror = reject;
      img.src = reader.result as string;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Run detection synchronously on main thread (worker approach needs opencv.js in worker context)
// For P1, we detect on main thread with requestAnimationFrame breaks to keep UI responsive
async function detectInWorker(
  imageData: Uint8ClampedArray,
  width: number,
  height: number,
  filename: string,
  boardConfig: BoardConfig,
): Promise<ImageDetection> {
  // Allow UI to update
  await new Promise((r) => requestAnimationFrame(r));

  // Import detection functions dynamically
  const { detectCharucoCorners, buildDetection } = await import('../core/detection');

  const { corners, markerCount } = detectCharucoCorners(imageData, width, height, boardConfig);

  return buildDetection(filename, width, height, corners, markerCount, boardConfig);
}
