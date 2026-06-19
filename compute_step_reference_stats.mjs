// compute_step_reference_stats.mjs
//
// One-off script: computes per-step-label reference statistics for each
// "posture descriptor" (12 joint angles + 6 distance-based descriptors, see
// pose_features.mjs) over all samples in training_data.json.
//
// Uses MEDIAN + MEDIAN ABSOLUTE DEVIATION (MAD) rather than mean/std. The
// training data combines many participants/sources with real variance and
// occasional outlier frames, so the mean/std can be skewed by a handful of
// odd samples — median/MAD describes the "typical" posture for a step much
// more robustly and is far less sensitive to a few noisy frames.
//
// Output (step_reference_stats.json) is used by server.js's
// /api/analyze-frame to compare a live frame's posture against the typical
// posture for the step the user is supposed to be on, and to generate
// specific corrective feedback (e.g. "widen the stance between your legs")
// instead of a bare "looks more like step N" message.
//
// Run with: node compute_step_reference_stats.mjs

import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { computePostureDescriptors, POSTURE_DESCRIPTOR_NAMES } from './pose_features.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

const TRAINING_DATA_FILE = join(__dirname, 'training_data.json');
const OUTPUT_FILE = join(__dirname, 'step_reference_stats.json');

console.log('Loading training_data.json...');
const data = JSON.parse(fs.readFileSync(TRAINING_DATA_FILE, 'utf8'));
console.log(`Loaded ${data.length} samples.`);

const N = POSTURE_DESCRIPTOR_NAMES.length;

// label -> array of descriptor-vectors
const byLabel = {};

for (const sample of data) {
  const { label, features } = sample;
  if (!label || !Array.isArray(features) || features.length !== 99) continue;

  const desc = computePostureDescriptors(features);

  if (!byLabel[label]) byLabel[label] = [];
  byLabel[label].push(desc);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Median Absolute Deviation, scaled by 1.4826 so it's comparable to a
// standard deviation for roughly-normal data.
function madStd(values, med) {
  const absDevs = values.map(v => Math.abs(v - med));
  return 1.4826 * median(absDevs);
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function std(values, m) {
  const variance = values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

const stats = {};
let labelCount = 0;
for (const [label, samples] of Object.entries(byLabel)) {
  const entry = {};
  for (let i = 0; i < N; i++) {
    const values = samples.map(s => s[i]);
    const med = median(values);
    const robustStd = madStd(values, med);
    const m = mean(values);
    const sd = std(values, m);

    entry[POSTURE_DESCRIPTOR_NAMES[i]] = {
      median: Number(med.toFixed(6)),
      mad: Number(robustStd.toFixed(6)),
      // kept for reference / debugging — not used for live comparisons
      mean: Number(m.toFixed(6)),
      std: Number(sd.toFixed(6)),
    };
  }
  stats[label] = { count: samples.length, descriptors: entry };
  labelCount++;
}

fs.writeFileSync(OUTPUT_FILE, JSON.stringify(stats, null, 2));
console.log(`Wrote reference stats for ${labelCount} labels to ${OUTPUT_FILE}`);
