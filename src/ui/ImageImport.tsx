import { useRef, useCallback } from 'preact/hooks';
import type { BoardConfig, ImageDetection } from '../core/types';
import { detectInWorker } from '../core/worker-manager';
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

      for (let i = 0; i < supported.length; i++) {
        const file = supported[i];
        setProgressMsg(`${en.detecting} ${i + 1}/${supported.length}: ${file.name}`);

        try {
          const { rgbaBuffer, width, height, thumbnailUrl } = await loadImage(file);
          thumbnails.set(file.name, thumbnailUrl);

          // Run detection in web worker (rgbaBuffer is transferred, zero-copy)
          const detection = await detectInWorker(
            rgbaBuffer,
            file.name,
            width,
            height,
            boardConfig,
            (msg) => setProgressMsg(msg),
          );
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
      input.value = '';
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

/**
 * Load image file into an ArrayBuffer of RGBA pixels (transferable to worker).
 * Also produces a small JPEG thumbnail as a data URL.
 */
function loadImage(
  file: File,
): Promise<{ rgbaBuffer: ArrayBuffer; width: number; height: number; thumbnailUrl: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        // Full-res canvas for pixel data
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        // .buffer gives us the underlying ArrayBuffer (transferable)
        const rgbaBuffer = imageData.data.buffer;

        // Thumbnail
        const thumbCanvas = document.createElement('canvas');
        const thumbScale = Math.min(1, 300 / Math.max(img.width, img.height));
        thumbCanvas.width = img.width * thumbScale;
        thumbCanvas.height = img.height * thumbScale;
        const thumbCtx = thumbCanvas.getContext('2d')!;
        thumbCtx.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
        const thumbnailUrl = thumbCanvas.toDataURL('image/jpeg', 0.7);

        resolve({ rgbaBuffer, width: img.width, height: img.height, thumbnailUrl });
      };
      img.onerror = reject;
      img.src = reader.result as string;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
