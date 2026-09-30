/**
 * yogatwin_angles.mjs
 *
 * Joint-angle analysis pipeline ported from YogaTwin-AI
 * (github.com/Amitha-13/YogaTwin-AI / app1/src/utils/practicePoseAnalysis.js
 *  and server/main6.py compare_pose logic).
 *
 * Uses RAW MediaPipe landmark objects (with .x, .y) — call these functions
 * BEFORE normalizeLandmarks(), passing req.body.landmarks directly.
 *
 * Angle indices (8 joints):
 *   [0] right elbow   [1] left elbow
 *   [2] right shoulder [3] left shoulder
 *   [4] right hip     [5] left hip
 *   [6] right knee    [7] left knee
 */

// ── Landmark indices ─────────────────────────────────────────────────────────
const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,    RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,    RIGHT_WRIST: 16,
  LEFT_HIP: 23,      RIGHT_HIP: 24,
  LEFT_KNEE: 25,     RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,    RIGHT_ANKLE: 28,
};

/**
 * Calculate the 2-D angle (degrees, 0–180) at vertex b formed by a–b–c.
 * Mirrors calculateAngle() from YogaTwin practicePoseAnalysis.js / main6.py.
 */
export function calculateAngle(a, b, c) {
  let radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return Math.round(angle);
}

/**
 * Compute the 8 canonical joint angles from raw MediaPipe landmark array.
 * Returns [rightElbow, leftElbow, rightShoulder, leftShoulder,
 *          rightHip, leftHip, rightKnee, leftKnee] in degrees.
 */
export function computeEightAngles(landmarks) {
  const p = i => landmarks[i];
  return [
    calculateAngle(p(LM.RIGHT_SHOULDER),  p(LM.RIGHT_ELBOW),   p(LM.RIGHT_WRIST)),   // 0 right elbow
    calculateAngle(p(LM.LEFT_SHOULDER),   p(LM.LEFT_ELBOW),    p(LM.LEFT_WRIST)),    // 1 left elbow
    calculateAngle(p(LM.RIGHT_ELBOW),     p(LM.RIGHT_SHOULDER), p(LM.RIGHT_HIP)),    // 2 right shoulder
    calculateAngle(p(LM.LEFT_ELBOW),      p(LM.LEFT_SHOULDER),  p(LM.LEFT_HIP)),     // 3 left shoulder
    calculateAngle(p(LM.RIGHT_SHOULDER),  p(LM.RIGHT_HIP),     p(LM.RIGHT_KNEE)),    // 4 right hip
    calculateAngle(p(LM.LEFT_SHOULDER),   p(LM.LEFT_HIP),      p(LM.LEFT_KNEE)),     // 5 left hip
    calculateAngle(p(LM.RIGHT_HIP),       p(LM.RIGHT_KNEE),    p(LM.RIGHT_ANKLE)),   // 6 right knee
    calculateAngle(p(LM.LEFT_HIP),        p(LM.LEFT_KNEE),     p(LM.LEFT_ANKLE)),    // 7 left knee
  ];
}

/**
 * Similarity score [0, 1] between user angles and target angles.
 * Mirrors diffCompareAngle() from YogaTwin.
 */
export function diffCompareAngle(user, target) {
  const diffs = user.map((u, i) => {
    const mid = (u + target[i]) / 2;
    return mid === 0 ? 0 : Math.abs(u - target[i]) / mid;
  });
  const avg = diffs.reduce((s, x) => s + x, 0) / diffs.length;
  return Math.max(0, Math.min(1, 1 - avg));
}

/**
 * Generate specific corrective instructions by comparing user angles to target.
 * Tolerance is 15° (mirrors YogaTwin ANGLE_TOL).
 * Returns array of short instruction strings (may be empty if pose is correct).
 */
const TOL = 15;
export function getCorrections(user, target) {
  const u = user, t = target;
  const out = [];
  if (u[0] < t[0] - TOL) out.push("Extend your right arm at the elbow");
  if (u[0] > t[0] + TOL) out.push("Bend your right arm slightly at the elbow");
  if (u[1] < t[1] - TOL) out.push("Extend your left arm at the elbow");
  if (u[1] > t[1] + TOL) out.push("Bend your left arm slightly at the elbow");
  if (u[2] < t[2] - TOL) out.push("Lift your right arm higher");
  if (u[2] > t[2] + TOL) out.push("Lower your right arm slightly");
  if (u[3] < t[3] - TOL) out.push("Lift your left arm higher");
  if (u[3] > t[3] + TOL) out.push("Lower your left arm slightly");
  if (u[4] < t[4] - TOL) out.push("Open your right hip — extend that side more");
  if (u[4] > t[4] + TOL) out.push("Reduce the angle at your right hip");
  if (u[5] < t[5] - TOL) out.push("Open your left hip — extend that side more");
  if (u[5] > t[5] + TOL) out.push("Reduce the angle at your left hip");
  if (u[6] < t[6] - TOL) out.push("Straighten your right knee");
  if (u[6] > t[6] + TOL) out.push("Bend your right knee more");
  if (u[7] < t[7] - TOL) out.push("Straighten your left knee");
  if (u[7] > t[7] + TOL) out.push("Bend your left knee more");
  return out;
}

// ── Target angles per pose ────────────────────────────────────────────────────
// Format: [rightElbow, leftElbow, rightShoulder, leftShoulder,
//          rightHip, leftHip, rightKnee, leftKnee] degrees
// Sources: YogaTwin-AI practicePoseTargets.js (Mountain, Tree, Triangle, Cobra)
//          + derived targets for the remaining Common Yoga Protocol poses.

export const POSE_TARGET_ANGLES = {
  // ── Standing ──────────────────────────────────────────────────────────────
  'Tadasana (Mountain Pose)':
    [175, 175, 175, 175, 175, 175, 175, 175],

  'Vrkasana (Tree Pose)':
    [165, 170, 160, 175, 110, 175, 165, 175],

  // Pada-hastasana: arms extended overhead on the way up, deep forward fold
  'Pada-hastasana (Hand-to-Foot Pose)':
    [175, 175, 165, 165, 80, 80, 170, 170],

  // Ardha Cakrasana: backbend — arms overhead, spine arched, knees straight
  'Ardha Cakrasana (Half-Wheel Pose)':
    [160, 160, 60, 60, 165, 165, 175, 175],

  // Ardha Katichakrasana: lateral side-bend, one arm overhead
  'Ardha Katichakrasana (Half Waist-Wheel Pose)':
    [175, 50, 165, 90, 175, 175, 175, 175],

  // Trikonasana: arms to shoulder level, body bent to one side, knees straight
  'Trikonasana (Triangle Pose)':
    [170, 100, 120, 120, 100, 170, 165, 165],

  // Parivritta Trikonasana: revolved — similar geometry, opposite arm down
  'Parivritta Trikonasana (Revolved Triangle Pose)':
    [100, 170, 120, 120, 170, 100, 165, 165],

  // ── Sitting ───────────────────────────────────────────────────────────────
  // Ardha Ustrasana: kneeling backbend — hands on hips, chest lifted
  'Ardha Ustrasana (Half Camel Pose)':
    [160, 160, 55, 55, 145, 145, 90, 90],

  // Vakrasana: seated twist — one knee bent, torso rotated
  'Vakrasana (Twisted Pose)':
    [160, 160, 130, 90, 90, 170, 130, 150],

  // ── Prone ─────────────────────────────────────────────────────────────────
  // Makarasana: prone rest — arms folded, legs relaxed
  'Makarasana (Crocodile Pose)':
    [110, 110, 90, 90, 175, 175, 175, 175],

  // Bhujangasana: cobra — arms pressing up, hips down, legs straight
  'Bhujangasana (Cobra Pose)':
    [165, 165, 55, 55, 140, 140, 165, 165],

  // ── Supine ────────────────────────────────────────────────────────────────
  // Ardha Halasana: legs raised to 90°, arms flat, knees straight
  'Ardha Halasana (Half Plough Pose)':
    [175, 175, 135, 135, 90, 90, 175, 175],

  // Savasana: completely flat and relaxed
  'Savasana (Corpse Pose)':
    [175, 175, 175, 175, 175, 175, 175, 175],
};

/**
 * Returns the 8-angle target for a pose name, or null if not in the table.
 */
export function getTargetAngles(poseName) {
  return POSE_TARGET_ANGLES[poseName] ?? null;
}

/**
 * Minimum visibility of the 8 key joints used in angle computation.
 * Returns a value in [0, 1]; landmarks below ~0.35 should be skipped.
 */
export function keyJointVisibility(landmarks) {
  const indices = [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW,    LM.RIGHT_ELBOW,
    LM.LEFT_HIP,      LM.RIGHT_HIP,
    LM.LEFT_KNEE,     LM.RIGHT_KNEE,
  ];
  return Math.min(...indices.map(i => landmarks[i]?.visibility ?? 0));
}
