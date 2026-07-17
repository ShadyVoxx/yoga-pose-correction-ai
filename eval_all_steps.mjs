// Empirical validation of the descriptor-based step-completion matching
// (computeDescriptorMatchScore / DESCRIPTOR_MATCH_THRESHOLD) across every
// pose & step in training_data.json — mirrors the exact logic in server.js.
//
// For each "<Pose>_StepN" label:
//   - ownMatchRate: % of that step's own training samples that would score
//     >= DESCRIPTOR_MATCH_THRESHOLD against its OWN reference stats. This is
//     the "would step N actually be detected as complete" rate.
//   - nextMatchRate: % of that step's samples that ALSO score >= threshold
//     against the NEXT step's reference. High values here mean the app could
//     advance too fast — step N's posture already looks like step N+1.
//
// Run: node eval_all_steps.mjs

import fs from 'fs';
import { computePostureDescriptors, POSTURE_DESCRIPTOR_NAMES } from './pose_features.mjs';

const trainingData = JSON.parse(fs.readFileSync(new URL('./training_data.json', import.meta.url)));
const stepRefStats = JSON.parse(fs.readFileSync(new URL('./step_reference_stats.json', import.meta.url)));

const SKIP_DESCRIPTORS = new Set([
  'leftElbowAngle', 'rightElbowAngle',
  'leftShoulderAngle', 'rightShoulderAngle',
  'leftWristHeight', 'rightWristHeight',
  'leftWristShoulderDist', 'rightWristShoulderDist',
]);

const FEEDBACK_Z_THRESHOLD = 1.6;
const DESCRIPTOR_MATCH_THRESHOLD = 0.7;

const DESCRIPTOR_STD_FLOOR = {
  leftElbowAngle: 0.40, rightElbowAngle: 0.40,
  leftKneeAngle: 0.40, rightKneeAngle: 0.40,
  leftHipAngle: 0.35, rightHipAngle: 0.35,
  leftShoulderAngle: 0.45, rightShoulderAngle: 0.45,
  leftAnkleAngle: 0.35, rightAnkleAngle: 0.35,
  spineTiltAngle: 0.18, shoulderLineTiltAngle: 0.18,
  footSeparation: 0.05, kneeSeparation: 0.05,
  leftWristHeight: 0.15, rightWristHeight: 0.15,
  leftWristShoulderDist: 0.15, rightWristShoulderDist: 0.15,
};
function descriptorStdFloor(name) {
  return DESCRIPTOR_STD_FLOOR[name] ?? (name.toLowerCase().includes('angle') ? 0.2 : 0.05);
}

function computeDescriptorMatchScore(liveDescriptors, refEntry) {
  if (!refEntry || !refEntry.descriptors) return null;
  let total = 0;
  let withinTolerance = 0;
  for (let i = 0; i < POSTURE_DESCRIPTOR_NAMES.length; i++) {
    const name = POSTURE_DESCRIPTOR_NAMES[i];
    if (SKIP_DESCRIPTORS.has(name)) continue;
    const ref = refEntry.descriptors[name];
    if (!ref) continue;
    total++;
    const spread = Math.max(ref.mad, descriptorStdFloor(name));
    const z = (liveDescriptors[i] - ref.median) / spread;
    if (Math.abs(z) < FEEDBACK_Z_THRESHOLD) withinTolerance++;
  }
  return total > 0 ? withinTolerance / total : null;
}

// Group samples by label, precompute descriptors
const byLabel = new Map();
for (const sample of trainingData) {
  const desc = computePostureDescriptors(sample.features);
  if (!byLabel.has(sample.label)) byLabel.set(sample.label, []);
  byLabel.get(sample.label).push(desc);
}

// Group labels by pose, sorted by stepIndex
const byPose = new Map();
for (const sample of trainingData) {
  if (!byPose.has(sample.poseName)) byPose.set(sample.poseName, new Set());
  byPose.get(sample.poseName).add(sample.stepIndex);
}

const results = [];
for (const [poseName, stepSet] of byPose) {
  const steps = [...stepSet].sort((a, b) => a - b);
  for (const stepIndex of steps) {
    const label = `${poseName}_Step${stepIndex}`;
    const samples = byLabel.get(label);
    if (!samples) continue;
    const ownRef = stepRefStats[label];

    const nextLabel = `${poseName}_Step${stepIndex + 1}`;
    const nextRef = stepRefStats[nextLabel];

    let ownMatches = 0, nextMatches = 0, nextTotal = 0;
    for (const desc of samples) {
      const ownScore = computeDescriptorMatchScore(desc, ownRef);
      if (ownScore !== null && ownScore >= DESCRIPTOR_MATCH_THRESHOLD) ownMatches++;

      if (nextRef) {
        const nextScore = computeDescriptorMatchScore(desc, nextRef);
        if (nextScore !== null) {
          nextTotal++;
          if (nextScore >= DESCRIPTOR_MATCH_THRESHOLD) nextMatches++;
        }
      }
    }

    results.push({
      poseName, stepIndex, label,
      n: samples.length,
      ownMatchRate: ownMatches / samples.length,
      nextMatchRate: nextTotal > 0 ? nextMatches / nextTotal : null,
    });
  }
}

// Print report
let flagged = [];
for (const r of results) {
  const own = (r.ownMatchRate * 100).toFixed(0);
  const next = r.nextMatchRate !== null ? (r.nextMatchRate * 100).toFixed(0) + '%' : 'n/a';
  console.log(`${r.label.padEnd(45)} n=${String(r.n).padEnd(4)} own=${own}%  matchesNextStepRef=${next}`);

  if (r.ownMatchRate < 0.5) {
    flagged.push(`${r.label}: only ${own}% of its own samples pass (descriptor-only) — may rely heavily on classifier or get stuck.`);
  }
  if (r.nextMatchRate !== null && r.nextMatchRate > 0.5) {
    flagged.push(`${r.label}: ${next} of its samples ALSO match the next step's reference — risk of advancing too fast through this step.`);
  }
}

console.log('\n=== FLAGGED (possible issues) ===');
if (flagged.length === 0) console.log('None — all steps look reasonable.');
else flagged.forEach(f => console.log('- ' + f));
