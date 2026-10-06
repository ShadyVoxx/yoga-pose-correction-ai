// Continuation training run on top of the v2 checkpoint, using:
//  - class-balanced minibatch sampling (uniform over classes, then random sample
//    within class) to address the ~33x class imbalance in the merged dataset
//  - learning-rate decay over the course of the run
// Resumable across many short calls via progress_v3.json, mirroring v2's design.
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import fs from 'fs';
import path from 'path';

await tf.setBackend('cpu');

const BATCH_SIZE = 64;
const TOTAL_EPOCHS = parseInt(process.env.TOTAL_EPOCHS || '18', 10);
const TIME_BUDGET_MS = parseInt(process.env.TIME_BUDGET_MS || '37000', 10);
const MODEL_DIR = './tfjs_model';
const PROGRESS_FILE = './progress_v3.json';

const meta = JSON.parse(fs.readFileSync('./preprocess_meta.json', 'utf8'));
const uniqueLabels = JSON.parse(fs.readFileSync('./labels.json', 'utf8'));
const { trainSamples, valSamples, numClasses, totalFeatureSize } = meta;

console.log(`Train samples: ${trainSamples}, Val: ${valSamples}, classes: ${numClasses}, features: ${totalFeatureSize}`);

const trainX = new Float32Array(fs.readFileSync('./train_X.bin').buffer);
const trainY = new Float32Array(fs.readFileSync('./train_Y.bin').buffer);
const valXArr = new Float32Array(fs.readFileSync('./val_X.bin').buffer);
const valYArr = new Float32Array(fs.readFileSync('./val_Y.bin').buffer);
const trainLabels = new Int32Array(fs.readFileSync('./train_labels.bin').buffer);

// Build per-class index lists for balanced sampling
const classIndices = Array.from({ length: numClasses }, () => []);
for (let i = 0; i < trainSamples; i++) classIndices[trainLabels[i]].push(i);

const batchesPerEpoch = Math.ceil(trainSamples / BATCH_SIZE);
const TOTAL_BATCHES = TOTAL_EPOCHS * batchesPerEpoch;

function lrForProgress(frac) {
    if (frac < 0.6) return 0.001;
    if (frac < 0.85) return 0.0005;
    return 0.0002;
}

let progress = { batchesDone: 0, valAcc: 0, valLoss: 0, lr: 0.001 };
if (fs.existsSync(PROGRESS_FILE)) {
    progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
}

// Always load from saved checkpoint (the v2 result on first run, then our own
// updated checkpoint on subsequent runs).
console.log(`Loading model from ${MODEL_DIR} (v3 batch ${progress.batchesDone}/${TOTAL_BATCHES})...`);
const modelJson = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model.json'), 'utf8'));
const weightData = fs.readFileSync(path.join(MODEL_DIR, 'weights.bin'));
let model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs: modelJson.weightsManifest[0].weights,
    weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength),
}));

let currentLr = lrForProgress(progress.batchesDone / TOTAL_BATCHES);
model.compile({ optimizer: tf.train.adam(currentLr), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
console.log(`Using learning rate ${currentLr}`);

const t0 = Date.now();
const xBatch = new Float32Array(BATCH_SIZE * totalFeatureSize);
const yBatch = new Float32Array(BATCH_SIZE * numClasses);
let batchesThisRun = 0;

while (progress.batchesDone < TOTAL_BATCHES && (Date.now() - t0) < TIME_BUDGET_MS) {
    // Check if LR needs to change at this point
    const frac = progress.batchesDone / TOTAL_BATCHES;
    const targetLr = lrForProgress(frac);
    if (targetLr !== currentLr) {
        currentLr = targetLr;
        model.compile({ optimizer: tf.train.adam(currentLr), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        console.log(`  -> LR changed to ${currentLr} at batch ${progress.batchesDone}`);
    }

    for (let b = 0; b < BATCH_SIZE; b++) {
        const c = Math.floor(Math.random() * numClasses);
        const pool = classIndices[c];
        const idx = pool[Math.floor(Math.random() * pool.length)];
        xBatch.set(trainX.subarray(idx * totalFeatureSize, (idx + 1) * totalFeatureSize), b * totalFeatureSize);
        yBatch.set(trainY.subarray(idx * numClasses, (idx + 1) * numClasses), b * numClasses);
    }
    const xb = tf.tensor2d(xBatch, [BATCH_SIZE, totalFeatureSize]);
    const yb = tf.tensor2d(yBatch, [BATCH_SIZE, numClasses]);
    await model.trainOnBatch(xb, yb);
    xb.dispose();
    yb.dispose();
    progress.batchesDone++;
    batchesThisRun++;
}

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`Processed ${batchesThisRun} batches in ${elapsed}s (total ${progress.batchesDone}/${TOTAL_BATCHES}, ~epoch ${(progress.batchesDone / batchesPerEpoch).toFixed(2)})`);

// Validation pass
const valXT = tf.tensor2d(valXArr, [valSamples, totalFeatureSize]);
const valYT = tf.tensor2d(valYArr, [valSamples, numClasses]);
const evalResult = model.evaluate(valXT, valYT, { batchSize: 256 });
const valLoss = (await evalResult[0].data())[0];
const valAcc = (await evalResult[1].data())[0];
valXT.dispose();
valYT.dispose();
evalResult[0].dispose();
evalResult[1].dispose();
progress.valAcc = valAcc;
progress.valLoss = valLoss;
progress.lr = currentLr;
console.log(`Val acc: ${(valAcc * 100).toFixed(2)}%, val loss: ${valLoss.toFixed(4)}`);

// Save checkpoint
fs.mkdirSync(MODEL_DIR, { recursive: true });
await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const weightDataOut = Buffer.from(artifacts.weightData);
    fs.writeFileSync(path.join(MODEL_DIR, 'weights.bin'), weightDataOut);
    const modelJsonOut = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }],
    };
    fs.writeFileSync(path.join(MODEL_DIR, 'model.json'), JSON.stringify(modelJsonOut));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
}));

fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress));

if (progress.batchesDone >= TOTAL_BATCHES) {
    fs.writeFileSync('./model_config.json', JSON.stringify({
        imuEnabled: false,
        landmarkFeatureSize: 99,
        imuFeatureSize: 6,
        totalFeatureSize,
        numClasses,
        trainedAt: new Date().toISOString(),
        trainSamples,
        valSamples,
        valAccuracy: valAcc,
        notes: 'v3: continued from v2 checkpoint with class-balanced minibatch sampling + LR decay (0.001 -> 0.0005 -> 0.0002)',
    }, null, 2));
    console.log(`\n✅ DONE. Final val_acc=${(valAcc * 100).toFixed(2)}% val_loss=${valLoss.toFixed(4)}`);
} else {
    console.log(`⏸ Checkpoint saved at batch ${progress.batchesDone}/${TOTAL_BATCHES}. Run again to continue.`);
}
