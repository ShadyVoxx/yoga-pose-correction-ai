const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function main() {
  // Load model
  const model = await tf.loadLayersModel('file:///sessions/confident-epic-edison/mnt/yoga-pose-correction-ai/tfjs_model_new/model.json');

  // Load data
  const raw = JSON.parse(fs.readFileSync('/sessions/confident-epic-edison/mnt/yoga-pose-correction-ai/training_data.json'));
  const labels = JSON.parse(fs.readFileSync('/sessions/confident-epic-edison/mnt/yoga-pose-correction-ai/labels.json'));

  // Build label→index map
  const labelIndex = {};
  labels.forEach((l, i) => labelIndex[l] = i);

  // Shuffle with fixed seed then take last 20% as val (matches training split)
  // Use same seed approach: sort by label for stratified split
  const byClass = {};
  raw.forEach(s => {
    if (!byClass[s.label]) byClass[s.label] = [];
    byClass[s.label].push(s);
  });

  const valSamples = [];
  Object.values(byClass).forEach(arr => {
    const n = Math.floor(arr.length * 0.2);
    // Take last 20% (consistent with how train_model.js splits)
    valSamples.push(...arr.slice(-n));
  });

  console.log(`Val samples: ${valSamples.length}`);

  // Predict in batches
  const BATCH = 1000;
  const perPose = {}; // pose_name -> { correct, total }

  for (let i = 0; i < valSamples.length; i += BATCH) {
    const batch = valSamples.slice(i, i + BATCH);
    const xs = tf.tensor2d(batch.map(s => s.features));
    const preds = model.predict(xs);
    const predIdx = Array.from(await preds.argMax(1).data());
    xs.dispose(); preds.dispose();

    batch.forEach((s, j) => {
      const trueIdx = labelIndex[s.label];
      const poseName = s.label.replace(/_Step\d+$/, '');
      if (!perPose[poseName]) perPose[poseName] = { correct: 0, total: 0 };
      perPose[poseName].total++;
      if (predIdx[j] === trueIdx) perPose[poseName].correct++;
    });
  }

  // Print results
  console.log('\nPer-pose accuracy on validation set (20% stratified hold-out):');
  console.log('Pose | Correct | Total | Accuracy');
  let totalCorrect = 0, totalAll = 0;
  Object.entries(perPose).sort((a,b)=>a[0].localeCompare(b[0])).forEach(([pose, {correct, total}]) => {
    const acc = (correct/total*100).toFixed(2);
    console.log(`${pose} | ${correct} | ${total} | ${acc}%`);
    totalCorrect += correct; totalAll += total;
  });
  console.log(`\nOverall | ${totalCorrect} | ${totalAll} | ${(totalCorrect/totalAll*100).toFixed(2)}%`);
}

main().catch(console.error);
