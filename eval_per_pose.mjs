import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

const model = await tf.loadLayersModel('file:///sessions/confident-epic-edison/mnt/yoga-pose-correction-ai/tfjs_model_new/model.json');
const raw = JSON.parse(readFileSync('./training_data.json'));
const labels = JSON.parse(readFileSync('./labels.json'));

const labelIndex = {};
labels.forEach((l, i) => labelIndex[l] = i);

// Stratified 20% val split (last 20% per class)
const byClass = {};
raw.forEach(s => { if (!byClass[s.label]) byClass[s.label] = []; byClass[s.label].push(s); });

const valSamples = [];
Object.values(byClass).forEach(arr => {
  const n = Math.max(1, Math.floor(arr.length * 0.2));
  valSamples.push(...arr.slice(-n));
});
console.log(`Val samples: ${valSamples.length}`);

const perPose = {};
const BATCH = 2000;
for (let i = 0; i < valSamples.length; i += BATCH) {
  const batch = valSamples.slice(i, i + BATCH);
  const xs = tf.tensor2d(batch.map(s => s.features));
  const predArr = Array.from((await model.predict(xs).argMax(1).data()));
  xs.dispose();
  batch.forEach((s, j) => {
    const trueIdx = labelIndex[s.label];
    const poseName = s.label.replace(/_Step\d+$/, '');
    if (!perPose[poseName]) perPose[poseName] = { correct: 0, total: 0 };
    perPose[poseName].total++;
    if (predArr[j] === trueIdx) perPose[poseName].correct++;
  });
}

let totalC = 0, totalT = 0;
console.log('\nPose | Val Samples | Correct | Accuracy');
Object.entries(perPose).sort((a,b) => a[0].localeCompare(b[0])).forEach(([pose, {correct, total}]) => {
  console.log(`${pose} | ${total} | ${correct} | ${(correct/total*100).toFixed(2)}%`);
  totalC += correct; totalT += total;
});
console.log(`\nOVERALL | ${totalT} | ${totalC} | ${(totalC/totalT*100).toFixed(2)}%`);
