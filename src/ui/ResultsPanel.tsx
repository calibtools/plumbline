import { useState, useMemo } from 'preact/hooks';
import type { CalibrationResult, ImageDetection } from '../core/types';
import { computeTiltAngle } from '../core/calibration';
import { buildHeatmap, getCoverageAdvice, getTiltAdvice } from '../core/heatmap';
import {
  toOpenCVYAML,
  toOpenCVJSON,
  toROSCameraInfoYAML,
  toPlumblineJSON,
  downloadFile,
} from '../core/export';
import { en } from '../i18n/en';

interface Props {
  result: CalibrationResult;
  detections: ImageDetection[];
  onToggleImage: (filename: string) => void;
}

export function ResultsPanel({ result, detections, onToggleImage }: Props) {
  const [gridCols, setGridCols] = useState(8);
  const [gridRows, setGridRows] = useState(6);

  const { cameraMatrix: cm, distCoeffs: dc, rms, imageSize: sz, perImage } = result;

  // Heatmap
  const heatmapCells = useMemo(
    () => buildHeatmap(detections, sz.width, sz.height, gridCols, gridRows),
    [detections, sz, gridCols, gridRows],
  );

  const coverageAdvice = useMemo(
    () => getCoverageAdvice(heatmapCells, gridCols, gridRows),
    [heatmapCells, gridCols, gridRows],
  );

  // Tilt distribution
  const tilts = useMemo(
    () => perImage.filter((p) => p.included).map((p) => computeTiltAngle(p.rvec)),
    [perImage],
  );

  const tiltAdvice = useMemo(() => getTiltAdvice(tilts), [tilts]);

  // Max RMS for bar chart scaling
  const maxRms = Math.max(...perImage.map((p) => p.rms));

  // Heatmap max count
  const maxCount = Math.max(...heatmapCells.map((c) => c.count), 1);

  // Tilt histogram: bucket into 0-10, 10-20, ..., 80-90 degrees
  const tiltBuckets = useMemo(() => {
    const buckets = new Array(9).fill(0);
    for (const t of tilts) {
      const idx = Math.min(Math.floor(t / 10), 8);
      buckets[idx]++;
    }
    return buckets;
  }, [tilts]);
  const maxTiltBucket = Math.max(...tiltBuckets, 1);

  return (
    <section>
      <h2>{en.results}</h2>

      <div class="result-grid">
        <div class="result-box">
          <h3>{en.overallRMS}</h3>
          <div class="value large">{rms.toFixed(4)} {en.px}</div>
        </div>
        <div class="result-box">
          <h3>{en.imageSize}</h3>
          <div class="value">{sz.width} × {sz.height} {en.px}</div>
        </div>
      </div>

      <div class="result-grid">
        <div class="result-box">
          <h3>{en.cameraMatrix}</h3>
          <table class="matrix-table">
            <tr>
              <td>{cm.fx.toFixed(2)}</td>
              <td>0</td>
              <td>{cm.cx.toFixed(2)}</td>
            </tr>
            <tr>
              <td>0</td>
              <td>{cm.fy.toFixed(2)}</td>
              <td>{cm.cy.toFixed(2)}</td>
            </tr>
            <tr>
              <td>0</td>
              <td>0</td>
              <td>1</td>
            </tr>
          </table>
        </div>
        <div class="result-box">
          <h3>{en.distortionCoeffs}</h3>
          <div class="value" style={{ fontSize: '0.75rem' }}>
            [{dc.map((v) => v.toExponential(4)).join(', ')}]
          </div>
        </div>
      </div>

      {/* Per-image RMS bar chart */}
      <h3>{en.perImageRMS}</h3>
      <p style={{ fontSize: '0.75rem', color: '#888', marginBottom: '0.25rem' }}>{en.clickToExclude}</p>
      <div class="bar-chart">
        {perImage.map((p, i) => (
          <div
            key={i}
            class="bar"
            style={{ height: `${(p.rms / maxRms) * 100}%` }}
            title={`${p.filename}: ${p.rms.toFixed(4)} px`}
            onClick={() => onToggleImage(p.filename)}
          />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#888' }}>
        <span>0</span>
        <span>{maxRms.toFixed(3)} px</span>
      </div>

      {/* Coverage heatmap */}
      <h3 style={{ marginTop: '1.5rem' }}>{en.coverageHeatmap}</h3>
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '0.5rem', fontSize: '0.8rem' }}>
        <label>{en.coverageGridSize}:</label>
        <input
          type="number" min={2} max={20} value={gridCols} style={{ width: '3rem' }}
          onInput={(e) => setGridCols(Math.max(2, parseInt((e.target as HTMLInputElement).value) || 2))}
        />
        ×
        <input
          type="number" min={2} max={20} value={gridRows} style={{ width: '3rem' }}
          onInput={(e) => setGridRows(Math.max(2, parseInt((e.target as HTMLInputElement).value) || 2))}
        />
      </div>
      <div class="heatmap-container">
        <div
          class="heatmap-grid"
          style={{
            gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
            aspectRatio: `${sz.width}/${sz.height}`,
            width: '100%',
            maxWidth: '500px',
          }}
        >
          {heatmapCells.map((cell) => {
            const intensity = cell.count / maxCount;
            const bg = cell.count === 0
              ? '#f0f0f0'
              : `rgba(230, 0, 18, ${0.15 + intensity * 0.85})`;
            return (
              <div
                key={`${cell.row}-${cell.col}`}
                class={`heatmap-cell ${cell.count === 0 ? 'empty' : ''}`}
                style={{ background: bg }}
              >
                {cell.count}
              </div>
            );
          })}
        </div>
      </div>
      {coverageAdvice && <div class="advice">{coverageAdvice}</div>}

      {/* Tilt distribution */}
      <h3 style={{ marginTop: '1.5rem' }}>{en.tiltDistribution}</h3>
      <div class="histogram">
        {tiltBuckets.map((count, i) => (
          <div
            key={i}
            class="histogram-bar"
            style={{ height: `${(count / maxTiltBucket) * 100}%` }}
            title={`${i * 10}°–${(i + 1) * 10}°: ${count} images`}
          />
        ))}
      </div>
      <div class="histogram-labels">
        <span>0°</span>
        <span>45°</span>
        <span>90°</span>
      </div>
      {tiltAdvice && <div class="advice">{tiltAdvice}</div>}

      {/* Export */}
      <h3 style={{ marginTop: '1.5rem' }}>{en.exportResults}</h3>
      <div class="export-buttons">
        <button class="btn-secondary btn-small" onClick={() => downloadFile(toOpenCVYAML(result), 'calibration.yaml', 'text/yaml')}>
          {en.exportOpenCVYAML}
        </button>
        <button class="btn-secondary btn-small" onClick={() => downloadFile(toOpenCVJSON(result), 'calibration.json', 'application/json')}>
          {en.exportOpenCVJSON}
        </button>
        <button class="btn-secondary btn-small" onClick={() => downloadFile(toROSCameraInfoYAML(result), 'camera_info.yaml', 'text/yaml')}>
          {en.exportROSYAML}
        </button>
        <button class="btn-secondary btn-small" onClick={() => downloadFile(toPlumblineJSON(result, detections), 'plumbline.json', 'application/json')}>
          {en.exportPlumblineJSON}
        </button>
      </div>
    </section>
  );
}
