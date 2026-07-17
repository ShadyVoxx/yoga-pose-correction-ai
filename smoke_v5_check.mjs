// Smoke test for the v5 model (117-dim: 99 landmark + 12 joint-angle + 6 dead
// IMU). Loads the deployed tfjs_model/ and spot-checks predictions against a
// few entries from training_data.json using the same feature pipeline as
// server.js (normalizeLandmarks already applied in training_data.json's
// "features" field, plus computeJointAngles from pose_features.mjs).
//
// Run with: node smoke_v5_check.mjs
import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { computeJointAngles, JOINT_ANGLE_FEATURE_SIZE } from './pose_features.mjs';

const cfg = JSON.parse(fs.readFileSync('./model_config.json', 'utf8'));
console.log('Model config:', JSON.stringify(cfg, null, 2));

const labels = JSON.parse(fs.readFileSync('./labels.json', 'utf8'));
console.log(`Labels: ${labels.length}`);

const model = await tf.loadLayersModel('file://./tfjs_model/model.json');
console.log('Model loaded. Input shape:', JSON.stringify(model.inputs[0].shape));

const data = JSON.parse(fs.readFileSync('./training_data.json', 'utf8'));
const sampleIdxs = [100, 1000, 5000, 20000, 40000, 55000, 60000];

let correct = 0;
for (const idx of sampleIdxs) {
  const sample = data[idx];
  const angles = computeJointAngles(sample.features);
  if (angles.length !== JOINT_ANGLE_FEATURE_SIZE) {
    throw new Error(`Expected ${JOINT_ANGLE_FEATURE_SIZE} angle features, got ${angles.length}`);
  }
  const input = [...sample.features, ...angles, ...new Array(cfg.imuFeatureSize).fill(0)];
  if (input.length !== cfg.totalFeatureSize) {
    throw new Error(`Feature size mismatch: expected ${cfg.totalFeatureSize}, got ${input.length}`);
  }

  const t = tf.tensor2d([input]);
  const pred = model.predict(t);
  const scores = await pred.data();
  t.dispose();
  pred.dispose();

  let maxIdx = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[maxIdx]) maxIdx = i;
  const ok = labels[maxIdx] === sample.label;
  if (ok) correct++;
  console.log(`[${idx}] actual=${sample.label} predicted=${labels[maxIdx]} conf=${(scores[maxIdx] * 100).toFixed(1)}% ${ok ? 'OK' : 'MISS'}`);
}
console.log(`\n${correct}/${sampleIdxs.length} spot-check predictions matched label.`);
