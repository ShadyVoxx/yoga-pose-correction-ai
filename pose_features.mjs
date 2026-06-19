// pose_features.mjs — shared joint-angle feature extraction.
//
// Computes 12 derived "joint angle" / posture-orientation features from a
// normalized 99-dim landmark vector (33 MediaPipe landmarks x (x,y,z),
// hip-centered and torso-scale-normalized — see normalize_landmarks() in
// prepare_dataset_v2.py / normalizeLandmarks() in server.js).
//
// These features are highly discriminative for yoga "step" classification
// (e.g. arm extension, leg bend, torso/spine tilt, lateral lean) and are
// computed deterministically from landmarks already in the pipeline — no new
// sensor data required.
//
// IMPORTANT: this module MUST be used identically at training-set
// preprocessing time (preprocess_for_training.mjs) and at live inference
// time (server.js) so the model sees the same feature representation in
// both places.

// MediaPipe Pose landmark indices used below.
const NOSE = 0;
const L_SHOULDER = 11, R_SHOULDER = 12;
const L_ELBOW = 13, R_ELBOW = 14;
const L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24;
const L_KNEE = 25, R_KNEE = 26;
const L_ANKLE = 27, R_ANKLE = 28;
const L_FOOT_INDEX = 31, R_FOOT_INDEX = 32;

export const JOINT_ANGLE_FEATURE_SIZE = 12;

export const JOINT_ANGLE_NAMES = [
    'leftElbowAngle', 'rightElbowAngle',
    'leftKneeAngle', 'rightKneeAngle',
    'leftHipAngle', 'rightHipAngle',
    'leftShoulderAngle', 'rightShoulderAngle',
    'leftAnkleAngle', 'rightAnkleAngle',
    'spineTiltAngle', 'shoulderLineTiltAngle',
];

function pt(flat99, idx) {
    return [flat99[idx * 3], flat99[idx * 3 + 1], flat99[idx * 3 + 2]];
}

function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Angle (radians, 0..pi) at vertex `b` formed by points a-b-c.
function angleAt(a, b, c) {
    const ba = sub(a, b);
    const bc = sub(c, b);
    const nba = norm(ba), nbc = norm(bc);
    if (nba < 1e-9 || nbc < 1e-9) return 0;
    let cos = dot(ba, bc) / (nba * nbc);
    cos = Math.max(-1, Math.min(1, cos));
    return Math.acos(cos);
}

// Angle (radians, 0..pi) between vector `v` and a reference direction `ref`.
function angleBetween(v, ref) {
    const nv = norm(v), nr = norm(ref);
    if (nv < 1e-9 || nr < 1e-9) return 0;
    let cos = dot(v, ref) / (nv * nr);
    cos = Math.max(-1, Math.min(1, cos));
    return Math.acos(cos);
}

/**
 * Compute the 12 joint-angle features from a normalized 99-dim landmark
 * vector (flat array of 33 x (x,y,z), hip-centered + torso-scale-normalized).
 * Returns an array of 12 floats (radians, range 0..pi).
 */
export function computeJointAngles(flat99) {
    const lShoulder = pt(flat99, L_SHOULDER), rShoulder = pt(flat99, R_SHOULDER);
    const lElbow = pt(flat99, L_ELBOW), rElbow = pt(flat99, R_ELBOW);
    const lWrist = pt(flat99, L_WRIST), rWrist = pt(flat99, R_WRIST);
    const lHip = pt(flat99, L_HIP), rHip = pt(flat99, R_HIP);
    const lKnee = pt(flat99, L_KNEE), rKnee = pt(flat99, R_KNEE);
    const lAnkle = pt(flat99, L_ANKLE), rAnkle = pt(flat99, R_ANKLE);
    const lFoot = pt(flat99, L_FOOT_INDEX), rFoot = pt(flat99, R_FOOT_INDEX);

    const leftElbowAngle = angleAt(lShoulder, lElbow, lWrist);
    const rightElbowAngle = angleAt(rShoulder, rElbow, rWrist);

    const leftKneeAngle = angleAt(lHip, lKnee, lAnkle);
    const rightKneeAngle = angleAt(rHip, rKnee, rAnkle);

    const leftHipAngle = angleAt(lShoulder, lHip, lKnee);
    const rightHipAngle = angleAt(rShoulder, rHip, rKnee);

    const leftShoulderAngle = angleAt(lElbow, lShoulder, lHip);
    const rightShoulderAngle = angleAt(rElbow, rShoulder, rHip);

    const leftAnkleAngle = angleAt(lKnee, lAnkle, lFoot);
    const rightAnkleAngle = angleAt(rKnee, rAnkle, rFoot);

    // Spine tilt: angle between the torso vector (hip-mid -> shoulder-mid,
    // which after hip-centering normalization is just the shoulder-mid
    // point itself) and the vertical axis. In normalized landmark space,
    // "up" is the -y direction.
    const shoulderMid = [
        (lShoulder[0] + rShoulder[0]) / 2,
        (lShoulder[1] + rShoulder[1]) / 2,
        (lShoulder[2] + rShoulder[2]) / 2,
    ];
    const spineTiltAngle = angleBetween(shoulderMid, [0, -1, 0]);

    // Shoulder-line tilt: angle between the left->right shoulder vector and
    // the horizontal axis — captures lateral lean / twist.
    const shoulderLine = sub(rShoulder, lShoulder);
    const shoulderLineTiltAngle = angleBetween(shoulderLine, [1, 0, 0]);

    return [
        leftElbowAngle, rightElbowAngle,
        leftKneeAngle, rightKneeAngle,
        leftHipAngle, rightHipAngle,
        leftShoulderAngle, rightShoulderAngle,
        leftAnkleAngle, rightAnkleAngle,
        spineTiltAngle, shoulderLineTiltAngle,
    ];
}

// --------------------------------------------------------------------------
// Posture "descriptors" — the 12 joint angles above, plus a handful of extra
// distance-based measurements that capture things joint angles alone don't
// (most importantly, stance width and how high the arms are raised). These
// are used to generate specific, body-part-level corrective feedback by
// comparing a live frame's descriptors against the mean/std of the same
// descriptors computed over training samples for the expected step (see
// compute_step_reference_stats.mjs / step_reference_stats.json).
//
// NOT used as model input — purely for human-readable feedback generation.
// --------------------------------------------------------------------------

export const POSTURE_DESCRIPTOR_NAMES = [
    ...JOINT_ANGLE_NAMES,
    'footSeparation',
    'kneeSeparation',
    'leftWristHeight', 'rightWristHeight',
    'leftWristShoulderDist', 'rightWristShoulderDist',
];

export const POSTURE_DESCRIPTOR_SIZE = POSTURE_DESCRIPTOR_NAMES.length;

/**
 * Compute the posture descriptor vector from a normalized 99-dim landmark
 * vector. Returns an array with POSTURE_DESCRIPTOR_SIZE floats.
 */
export function computePostureDescriptors(flat99) {
    const angles = computeJointAngles(flat99);

    const lAnkle = pt(flat99, L_ANKLE), rAnkle = pt(flat99, R_ANKLE);
    const lKnee = pt(flat99, L_KNEE), rKnee = pt(flat99, R_KNEE);
    const lWrist = pt(flat99, L_WRIST), rWrist = pt(flat99, R_WRIST);
    const lShoulder = pt(flat99, L_SHOULDER), rShoulder = pt(flat99, R_SHOULDER);

    // Horizontal distance between feet / knees — low values mean a narrow
    // stance, high values mean a wide stance.
    const footSeparation = Math.abs(lAnkle[0] - rAnkle[0]);
    const kneeSeparation = Math.abs(lKnee[0] - rKnee[0]);

    // How far each wrist is above (positive) or below (negative) its
    // shoulder, in normalized image-space. In normalized landmark space "up"
    // is -y, so shoulder.y - wrist.y > 0 means the wrist is above the
    // shoulder (arm raised).
    const leftWristHeight = lShoulder[1] - lWrist[1];
    const rightWristHeight = rShoulder[1] - rWrist[1];

    // How far each wrist is extended away from its shoulder overall
    // (captures arm extension independent of direction).
    const leftWristShoulderDist = norm(sub(lWrist, lShoulder));
    const rightWristShoulderDist = norm(sub(rWrist, rShoulder));

    return [
        ...angles,
        footSeparation,
        kneeSeparation,
        leftWristHeight, rightWristHeight,
        leftWristShoulderDist, rightWristShoulderDist,
    ];
}
