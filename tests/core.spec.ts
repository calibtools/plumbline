import { describe, it, expect } from 'vitest';
import { cornerIdToObjectPoint } from '../src/core/detection';
import { buildHeatmap, getCoverageAdvice, getTiltAdvice } from '../src/core/heatmap';
import type { BoardConfig, ImageDetection } from '../src/core/types';

const BOARD: BoardConfig = {
  squaresX: 5,
  squaresY: 7,
  squareLength: 30,
  markerLength: 22,
  dictionaryId: 0,
  legacyPattern: false,
};

describe('cornerIdToObjectPoint', () => {
  it('maps corner ID 0 to (1*sq, 1*sq, 0)', () => {
    const pt = cornerIdToObjectPoint(0, BOARD);
    expect(pt.x).toBe(30);
    expect(pt.y).toBe(30);
    expect(pt.z).toBe(0);
  });

  it('maps corner IDs sequentially along rows', () => {
    // innerX = 4, so IDs 0-3 are row 0, 4-7 are row 1, etc.
    const pt3 = cornerIdToObjectPoint(3, BOARD);
    expect(pt3.x).toBe(4 * 30); // col=3 -> (3+1)*30
    expect(pt3.y).toBe(1 * 30); // row=0 -> (0+1)*30

    const pt4 = cornerIdToObjectPoint(4, BOARD);
    expect(pt4.x).toBe(1 * 30); // col=0 -> (0+1)*30
    expect(pt4.y).toBe(2 * 30); // row=1 -> (1+1)*30
  });

  it('handles the last corner ID', () => {
    // For 5x7 board: inner = 4x6 = 24 corners, last ID = 23
    const pt = cornerIdToObjectPoint(23, BOARD);
    const innerX = 4;
    const col = 23 % innerX; // 3
    const row = Math.floor(23 / innerX); // 5
    expect(pt.x).toBe((col + 1) * 30);
    expect(pt.y).toBe((row + 1) * 30);
  });
});

describe('buildHeatmap', () => {
  it('counts corners in correct grid cells', () => {
    const detection: ImageDetection = {
      filename: 'test.jpg',
      width: 800,
      height: 600,
      corners: [
        { id: 0, x: 50, y: 50 },   // top-left cell
        { id: 1, x: 750, y: 550 }, // bottom-right cell
      ],
      markerCount: 2,
      included: true,
    };

    const cells = buildHeatmap([detection], 800, 600, 4, 3);
    // Cell at (0,0) should have 1 corner
    const topLeft = cells.find((c) => c.row === 0 && c.col === 0);
    expect(topLeft?.count).toBe(1);

    // Cell at (2,3) should have 1 corner
    const bottomRight = cells.find((c) => c.row === 2 && c.col === 3);
    expect(bottomRight?.count).toBe(1);
  });

  it('excludes non-included detections', () => {
    const detection: ImageDetection = {
      filename: 'test.jpg',
      width: 800,
      height: 600,
      corners: [{ id: 0, x: 50, y: 50 }],
      markerCount: 1,
      included: false,
    };

    const cells = buildHeatmap([detection], 800, 600, 4, 3);
    const total = cells.reduce((sum, c) => sum + c.count, 0);
    expect(total).toBe(0);
  });
});

describe('getTiltAdvice', () => {
  it('warns when most tilts are frontal', () => {
    const tilts = [5, 8, 3, 10, 12, 7, 4, 6, 9, 11]; // all < 15
    const advice = getTiltAdvice(tilts);
    expect(advice).toContain('frontal');
  });

  it('returns null when tilt variety is good', () => {
    const tilts = [5, 15, 25, 35, 45, 10, 20, 30, 40, 50];
    const advice = getTiltAdvice(tilts);
    expect(advice).toBeNull();
  });
});

describe('getCoverageAdvice', () => {
  it('returns null when coverage is uniform', () => {
    const cells = [
      { row: 0, col: 0, count: 10 },
      { row: 0, col: 1, count: 12 },
      { row: 1, col: 0, count: 11 },
      { row: 1, col: 1, count: 9 },
    ];
    const advice = getCoverageAdvice(cells, 2, 2);
    expect(advice).toBeNull();
  });

  it('identifies sparse regions', () => {
    // Use a 3x3 grid so region names map cleanly
    const cells = [
      { row: 0, col: 0, count: 100 }, { row: 0, col: 1, count: 100 }, { row: 0, col: 2, count: 100 },
      { row: 1, col: 0, count: 100 }, { row: 1, col: 1, count: 100 }, { row: 1, col: 2, count: 100 },
      { row: 2, col: 0, count: 100 }, { row: 2, col: 1, count: 100 }, { row: 2, col: 2, count: 0 },
    ];
    const advice = getCoverageAdvice(cells, 3, 3);
    expect(advice).toContain('bottom-right');
  });
});
