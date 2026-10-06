// One-time preprocessing: convert training_data.json into binary Float32Array
// files (train_X.bin, train_Y.bin, val_X.bin, val_Y.bin) for fast loading during
// the checkpointed training loop (avoids re-parsing an 80MB+ JSON file on every
// resumed training call).
import fs from 'fs';
import { computeJointAngles, JOINT_ANGLE_FEATURE_SIZE } from '../pose_features.mjs';

const IMU_FEATURE_SIZE = 6;
const LANDMARK_FEATURE_SIZE = 99;
const ANGLE_FEATURE_SIZE = JOINT_ANGLE_FEATURE_SIZE;
const TOTAL_FEATURE_SIZE = LANDMARK_FEATURE_SIZE + ANGLE_FEATURE_SIZE + IMU_FEATURE_SIZE;

console.log('Loading training_data.json...');
const data = JSON.parse(fs.readFileSync('./training_data.json', 'utf8'));
console.log(`Loaded ${data.length} samples`);

const uniqueLabels = [...new Set(data.map(d => d.label))].sort();
fs.writeFileSync('./labels.json', JSON.stringify(uniqueLabels));
console.log(`${uniqueLabels.length} unique labels`);

function mulberry32(a) {
    return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

const rand = mulberry32(42);
const shuffled = [...data];
for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
}
const splitIdx = Math.floor(shuffled.length * 0.85);
const trainData = shuffled.slice(0, splitIdx);
const valData = shuffled.slice(splitIdx);
console.log(`Train: ${trainData.length}, Val: ${valData.length}`);

function buildXY(rows, prefix) {
    const xs = new Float32Array(rows.length * TOTAL_FEATURE_SIZE);
    const ys = new Float32Array(rows.length * uniqueLabels.length);
    rows.forEach((item, i) => {
        const angles = computeJointAngles(item.features);
        const fv = [...item.features, ...angles, ...new Array(IMU_FEATURE_SIZE).fill(0)];
        xs.set(fv, i * TOTAL_FEATURE_SIZE);
        const labelIdx = uniqueLabels.indexOf(item.label);
        ys[i * uniqueLabels.length + labelIdx] = 1;
    });
    fs.writeFileSync(`./${prefix}_X.bin`, Buffer.from(xs.buffer));
    fs.writeFileSync(`./${prefix}_Y.bin`, Buffer.from(ys.buffer));
}

buildXY(trainData, 'train');
buildXY(valData, 'val');

fs.writeFileSync('./preprocess_meta.json', JSON.stringify({
    trainSamples: trainData.length,
    valSamples: valData.length,
    numClasses: uniqueLabels.length,
    landmarkFeatureSize: LANDMARK_FEATURE_SIZE,
    angleFeatureSize: ANGLE_FEATURE_SIZE,
    imuFeatureSize: IMU_FEATURE_SIZE,
    totalFeatureSize: TOTAL_FEATURE_SIZE,
}, null, 2));

console.log('Done. Wrote train_X.bin, train_Y.bin, val_X.bin, val_Y.bin, preprocess_meta.json, labels.json');
