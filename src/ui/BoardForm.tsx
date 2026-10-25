import type { BoardConfig } from '../core/types';
import { en } from '../i18n/en';
import calibtoolsBoards from '../data/calibtools-boards.json';

const DICTIONARIES = [
  { id: 0, label: en.dict_4X4_50 },
  { id: 1, label: en.dict_4X4_100 },
  { id: 2, label: en.dict_4X4_250 },
  { id: 3, label: en.dict_4X4_1000 },
  { id: 4, label: en.dict_5X5_50 },
  { id: 5, label: en.dict_5X5_100 },
  { id: 6, label: en.dict_5X5_250 },
  { id: 7, label: en.dict_5X5_1000 },
  { id: 8, label: en.dict_6X6_50 },
  { id: 9, label: en.dict_6X6_100 },
  { id: 10, label: en.dict_6X6_250 },
  { id: 11, label: en.dict_6X6_1000 },
  { id: 12, label: en.dict_7X7_50 },
  { id: 13, label: en.dict_7X7_100 },
  { id: 14, label: en.dict_7X7_250 },
  { id: 15, label: en.dict_7X7_1000 },
  { id: 16, label: en.dict_ARUCO_ORIGINAL },
  { id: 17, label: en.dict_APRILTAG_16h5 },
  { id: 18, label: en.dict_APRILTAG_25h9 },
  { id: 19, label: en.dict_APRILTAG_36h10 },
  { id: 20, label: en.dict_APRILTAG_36h11 },
  { id: 21, label: en.dict_ARUCO_MIP_36h12 },
];

interface Props {
  config: BoardConfig;
  onChange: (config: BoardConfig) => void;
}

export function BoardForm({ config, onChange }: Props) {
  const errors = validate(config);

  const set = <K extends keyof BoardConfig>(key: K, value: BoardConfig[K]) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <section>
      <h2>{en.boardConfig}</h2>
      <div class="form-grid">
        <div class="form-group">
          <label>{en.squaresX}</label>
          <input
            type="number"
            min={3}
            step={1}
            value={config.squaresX}
            onInput={(e) => set('squaresX', parseInt((e.target as HTMLInputElement).value) || 3)}
          />
        </div>
        <div class="form-group">
          <label>{en.squaresY}</label>
          <input
            type="number"
            min={3}
            step={1}
            value={config.squaresY}
            onInput={(e) => set('squaresY', parseInt((e.target as HTMLInputElement).value) || 3)}
          />
        </div>
        <div class="form-group">
          <label>{en.squareLength}</label>
          <input
            type="number"
            min={0.1}
            step={0.1}
            value={config.squareLength}
            onInput={(e) =>
              set('squareLength', parseFloat((e.target as HTMLInputElement).value) || 0.1)
            }
          />
        </div>
        <div class="form-group">
          <label>{en.markerLength}</label>
          <input
            type="number"
            min={0.1}
            step={0.1}
            value={config.markerLength}
            onInput={(e) =>
              set('markerLength', parseFloat((e.target as HTMLInputElement).value) || 0.1)
            }
          />
        </div>
        <div class="form-group full-width">
          <label>{en.dictionary}</label>
          <select
            value={config.dictionaryId}
            onChange={(e) => set('dictionaryId', parseInt((e.target as HTMLSelectElement).value))}
          >
            {DICTIONARIES.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
        </div>
        <div class="form-group full-width">
          <div class="checkbox-group">
            <input
              type="checkbox"
              id="legacy"
              checked={config.legacyPattern}
              onChange={(e) => set('legacyPattern', (e.target as HTMLInputElement).checked)}
            />
            <label for="legacy">{en.legacyPattern}</label>
          </div>
        </div>
        <div class="form-group full-width">
          <label>{en.calibtoolsBoard}</label>
          <select disabled={calibtoolsBoards.length === 0}>
            <option>{en.calibtoolsPlaceholder}</option>
          </select>
        </div>
      </div>
      {errors.length > 0 && (
        <div>
          {errors.map((err, i) => (
            <div key={i} class="validation-error">
              {err}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function validate(config: BoardConfig): string[] {
  const errors: string[] = [];
  if (config.squaresX < 3 || config.squaresY < 3) {
    errors.push(en.validationMinDimension);
  }
  if (config.squareLength <= 0 || config.markerLength <= 0) {
    errors.push(en.validationPositive);
  }
  if (config.markerLength >= config.squareLength) {
    errors.push(en.validationMarkerTooLarge);
  }
  return errors;
}
