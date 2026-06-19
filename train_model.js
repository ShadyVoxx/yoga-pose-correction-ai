import tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_FILE = join(__dirname, 'training_data.json');
const MODEL_DIR = join(__dirname, 'tfjs_model');
const CONFIG_FILE = join(__dirname, 'model_config.json');

// --------------------------------------------------------------------------
// IMU feature toggle.
//
// Every training sample stores BOTH the 99 MediaPipe landmark features AND
// a 6-value IMU feature vector ([ax, ay, az, gx, gy, gz], real values pulled
// from the recorded sensor stream). The IMU extraction/alignment logic is
// fully implemented in prepare_dataset.py and server.js.
//
// However, IMU_ENABLED is kept `false` until there's enough sensor data for
// the model to learn a reliable signal from it. While disabled, the IMU
// slots in the feature vector are filled with zeros for every sample (both
// during training here and during live inference in server.js), so those
// input weights contribute nothing to the prediction — i.e. "weight 0".
//
// To turn IMU features on later: flip this flag here AND in server.js,
// then re-run this script. No data re-collection/reprocessing is needed —
// the real imuFeatures are already stored in training_data.json.
// --------------------------------------------------------------------------
export const IMU_ENABLED = false;
export const LANDMARK_FEATURE_SIZE = 99; // 33 landmarks * (x, y, z)
export const IMU_FEATURE_SIZE = 6;       // [ax, ay, az, gx, gy, gz]
export const TOTAL_FEATURE_SIZE = LANDMARK_FEATURE_SIZE + IMU_FEATURE_SIZE;

function buildFeatureVector(item) {
    const imu = IMU_ENABLED && Array.isArray(item.imuFeatures) && item.imuFeatures.length === IMU_FEATURE_SIZE
        ? item.imuFeatures
        : new Array(IMU_FEATURE_SIZE).fill(0);
    return [...item.features, ...imu];
}

async function train() {
    if (!fs.existsSync(DATA_FILE)) {
        console.error('❌ No training_data.json found. Run prepare_dataset.py or the collector first!');
        process.exit(1);
    }

    console.log('Loading dataset...');
    const data = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));

    if (data.length === 0) {
        console.error('❌ Dataset is empty.');
        process.exit(1);
    }

    console.log(`IMU features: ${IMU_ENABLED ? 'ENABLED' : 'disabled (zero-filled, weight = 0)'}`);
    console.log(`Feature vector size: ${TOTAL_FEATURE_SIZE} (${LANDMARK_FEATURE_SIZE} landmark + ${IMU_FEATURE_SIZE} IMU)`);

    // Extract unique labels to create categorical one-hot encoding
    const uniqueLabels = [...new Set(data.map(d => d.label))].sort();
    console.log(`Found ${uniqueLabels.length} unique pose steps across ${new Set(data.map(d => d.poseName)).size} poses`);

    fs.writeFileSync(join(__dirname, 'labels.json'), JSON.stringify(uniqueLabels));

    // Shuffle then split into train/validation sets
    const shuffled = [...data];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const splitIdx = Math.floor(shuffled.length * 0.85);
    const trainData = shuffled.slice(0, splitIdx);
    const valData = shuffled.slice(splitIdx);

    const buildXY = (rows) => {
        const xs = [];
        const ys = [];
        rows.forEach(item => {
            xs.push(buildFeatureVector(item));
            const labelIdx = uniqueLabels.indexOf(item.label);
            const oneHot = Array(uniqueLabels.length).fill(0);
            oneHot[labelIdx] = 1;
            ys.push(oneHot);
        });
        return { xs: tf.tensor2d(xs), ys: tf.tensor2d(ys) };
    };

    const { xs: trainX, ys: trainY } = buildXY(trainData);
    const { xs: valX, ys: valY } = buildXY(valData);

    console.log(`Train samples: ${trainData.length}, Validation samples: ${valData.length}`);
    console.log(`Input Shape: ${trainX.shape}`);
    console.log(`Label Shape: ${trainY.shape}`);

    // Build the MLP model
    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 256,
        activation: 'relu',
        inputShape: [TOTAL_FEATURE_SIZE]
    }));

    model.add(tf.layers.dropout({ rate: 0.3 }));

    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));

    model.add(tf.layers.dropout({ rate: 0.2 }));

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: uniqueLabels.length,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    model.summary();

    console.log('\nTraining model...');

    await model.fit(trainX, trainY, {
        epochs: 40,
        batchSize: 64,
        shuffle: true,
        validationData: [valX, valY],
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `Epoch ${epoch + 1}/40 - loss: ${logs.loss.toFixed(4)} - acc: ${logs.acc.toFixed(4)} ` +
                    `- val_loss: ${logs.val_loss.toFixed(4)} - val_acc: ${logs.val_acc.toFixed(4)}`
                );
            }
        }
    });

    const evalResult = model.evaluate(valX, valY);
    const valLoss = (await evalResult[0].data())[0];
    const valAcc = (await evalResult[1].data())[0];

    console.log('\n✅ Training complete!');
    console.log(`📊 Final validation accuracy: ${(valAcc * 100).toFixed(2)}% (loss: ${valLoss.toFixed(4)})`);

    // Save model
    await model.save(`file://${MODEL_DIR}`);
    console.log(`📦 Model saved to ${MODEL_DIR}`);

    // Save a small config describing how this model was built, so server.js
    // (and future retrains) stay in sync about feature layout.
    fs.writeFileSync(CONFIG_FILE, JSON.stringify({
        imuEnabled: IMU_ENABLED,
        landmarkFeatureSize: LANDMARK_FEATURE_SIZE,
        imuFeatureSize: IMU_FEATURE_SIZE,
        totalFeatureSize: TOTAL_FEATURE_SIZE,
        numClasses: uniqueLabels.length,
        trainedAt: new Date().toISOString(),
        trainSamples: trainData.length,
        valSamples: valData.length,
        valAccuracy: valAcc,
    }, null, 2));

    trainX.dispose();
    trainY.dispose();
    valX.dispose();
    valY.dispose();
}

train().catch(console.error);
