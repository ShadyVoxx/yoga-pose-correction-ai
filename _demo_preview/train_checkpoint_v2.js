// Batch-based checkpointed training using preprocessed binary tensors.
// Designed to be re-invoked many times (each call processes batches for up to
// TIME_BUDGET_MS, then checkpoints model + progress and exits).
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import fs from 'fs';
import path from 'path';

await tf.setBackend('cpu');

const BATCH_SIZE = 64;
const TOTAL_EPOCHS = parseInt(process.env.TOTAL_EPOCHS || '40', 10);
const TIME_BUDGET_MS = parseInt(process.env.TIME_BUDGET_MS || '35000', 10);
const MODEL_DIR = './tfjs_model';
const PROGRESS_FILE = './progress.json';

const meta = JSON.parse(fs.readFileSync('./preprocess_meta.json', 'utf8'));
const uniqueLabels = JSON.parse(fs.readFileSync('./labels.json', 'utf8'));
const { trainSamples, valSamples, numClasses, totalFeatureSize } = meta;

console.log(`Train samples: ${trainSamples}, Val: ${valSamples}, classes: ${numClasses}, features: ${totalFeatureSize}`);

const trainX = new Float32Array(fs.readFileSync('./train_X.bin').buffer);
const trainY = new Float32Array(fs.readFileSync('./train_Y.bin').buffer);
const valXArr = new Float32Array(fs.readFileSync('./val_X.bin').buffer);
const valYArr = new Float32Array(fs.readFileSync('./val_Y.bin').buffer);

const batchesPerEpoch = Math.ceil(trainSamples / BATCH_SIZE);
const TOTAL_BATCHES = TOTAL_EPOCHS * batchesPerEpoch;

let progress = { batchesDone: 0, valAcc: 0, valLoss: 0 };
if (fs.existsSync(PROGRESS_FILE)) {
    progress = JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
}

let model;
if (progress.batchesDone > 0 && fs.existsSync(path.join(MODEL_DIR, 'model.json'))) {
    console.log(`Resuming from checkpoint (batch ${progress.batchesDone}/${TOTAL_BATCHES})...`);
    const modelJson = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model.json'), 'utf8'));
    const weightData = fs.readFileSync(path.join(MODEL_DIR, 'weights.bin'));
    model = await tf.loadLayersModel(tf.io.fromMemory({
        modelTopology: modelJson.modelTopology,
        weightSpecs: modelJson.weightsManifest[0].weights,
        weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength),
    }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
} else {
    console.log('Building new model...');
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [totalFeatureSize] }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    model.summary();
}

const t0 = Date.now();
const xBatch = new Float32Array(BATCH_SIZE * totalFeatureSize);
const yBatch = new Float32Array(BATCH_SIZE * numClasses);
let batchesThisRun = 0;

while (progress.batchesDone < TOTAL_BATCHES && (Date.now() - t0) < TIME_BUDGET_MS) {
    for (let b = 0; b < BATCH_SIZE; b++) {
        const idx = Math.floor(Math.random() * trainSamples);
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

// Quick validation pass
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
console.log(`Val acc: ${(valAcc * 100).toFixed(2)}%, val loss: ${valLoss.toFixed(4)}`);

// Save checkpoint
fs.mkdirSync(MODEL_DIR, { recursive: true });
await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const weightData = Buffer.from(artifacts.weightData);
    fs.writeFileSync(path.join(MODEL_DIR, 'weights.bin'), weightData);
    const modelJson = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }],
    };
    fs.writeFileSync(path.join(MODEL_DIR, 'model.json'), JSON.stringify(modelJson));
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
    }, null, 2));
    console.log(`\n✅ DONE. Final val_acc=${(valAcc * 100).toFixed(2)}% val_loss=${valLoss.toFixed(4)}`);
} else {
    console.log(`⏸ Checkpoint saved at batch ${progress.batchesDone}/${TOTAL_BATCHES}. Run again to continue.`);
}
