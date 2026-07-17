/**
 * compute_ideal_landmarks.mjs
 *
 * One-time script — run once after training to produce step_ideal_landmarks.json.
 * Reads training_data.json, groups samples by step label, computes the median
 * XYZ position for each of the 33 MediaPipe landmarks per step, and writes the
 * result to step_ideal_landmarks.json.
 *
 * The median (rather than mean) is used because training data contains real
 * outlier frames — median/MAD is more robust for computing a "typical" pose.
 *
 * Usage:
 *   node compute_ideal_landmarks.mjs
 *
 * Output: step_ideal_landmarks.json
 * Schema:
 * {
 *   "Tadasana (Mountain Pose)_Step0": {
 *     "sampleCount": 4210,
 *     "landmarks": [
 *       { "x": 0.512341, "y": 0.234567, "z": -0.001234 },  // landmark 0 (nose)
 *       ...                                                   // landmarks 1–32
 *     ]
 *   },
 *   ...
 * }
 */

import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const TRAINING_DATA_FILE = join(__dirname, 'training_data.json');
const OUTPUT_FILE        = join(__dirname, 'step_ideal_landmarks.json');
const LANDMARK_DIM       = 99; // 33 landmarks × 3 (x, y, z)

// ── Median helper ─────────────────────────────────────────────────────────────
function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

// ── Load training data ────────────────────────────────────────────────────────
console.log('Loading training_data.json…');
if (!fs.existsSync(TRAINING_DATA_FILE)) {
  console.error('training_data.json not found. Run prepare_dataset.py first.');
  process.exit(1);
}

const trainingData = JSON.parse(fs.readFileSync(TRAINING_DATA_FILE, 'utf8'));
console.log(`Loaded ${trainingData.length.toLocaleString()} samples.`);

// ── Group landmark features by step label ─────────────────────────────────────
// Each sample: { label: "PoseName_StepN", features: [117 floats] }
// features[0:99]   = normalized landmark XYZ (99 values = 33 landmarks × 3)
// features[99:111] = joint angles (12)
// features[111:117]= IMU (6, zero-filled)
const byLabel = {};
for (const sample of trainingData) {
  const { label, features } = sample;
  if (!label || !Array.isArray(features)) continue;
  if (!byLabel[label]) byLabel[label] = [];
  byLabel[label].push(features.slice(0, LANDMARK_DIM));
}

const labels = Object.keys(byLabel);
console.log(`Found ${labels.length} unique step labels.`);

// ── Compute median landmark per label ─────────────────────────────────────────
const idealLandmarks = {};
let processed = 0;

for (const label of labels) {
  const samples = byLabel[label]; // array of flat 99-float arrays

  // For each of the 99 dimensions, collect values across all samples → median
  const medianFeatures = Array.from({ length: LANDMARK_DIM }, (_, dim) => {
    const vals = samples.map(s => s[dim]);
    return median(vals);
  });

  // Reshape flat 99 array into 33 landmark objects
  const landmarks = Array.from({ length: 33 }, (_, i) => ({
    x: parseFloat(medianFeatures[i * 3    ].toFixed(6)),
    y: parseFloat(medianFeatures[i * 3 + 1].toFixed(6)),
    z: parseFloat(medianFeatures[i * 3 + 2].toFixed(6)),
  }));

  idealLandmarks[label] = { sampleCount: samples.length, landmarks };

  processed++;
  process.stdout.write(`\r  Processed ${processed} / ${labels.length} labels`);
}

// ── Write output ──────────────────────────────────────────────────────────────
console.log(`\nWriting step_ideal_landmarks.json…`);
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(idealLandmarks, null, 2), 'utf8');
console.log(`Done — ${labels.length} step labels written to step_ideal_landmarks.json`);
