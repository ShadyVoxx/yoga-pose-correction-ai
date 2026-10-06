const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageBreak,
} = require("docx");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840;
const MARGIN = 1440;
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN; // 9360

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60 },
    children: typeof text === "string" ? [new TextRun(text)] : text,
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, ...opts, children: [new TextRun(text)] });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 120 },
    shading: { fill: "F4F2FA", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "A78BFA", space: 4 } },
    children: [new TextRun({ text, font: "Consolas", size: 19 })],
  });
}

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "7C3AED", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF" })] })],
  });
}
function cell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: Array.isArray(text)
      ? text
      : [new Paragraph({ children: [new TextRun({ text, ...opts })] })],
  });
}

// ---- Technology Stack table ----
const techCols = [3120, 6240];
const techTable = new Table({
  width: { size: CONTENT_WIDTH, type: WidthType.DXA },
  columnWidths: techCols,
  rows: [
    new TableRow({ children: [headerCell("Layer", techCols[0]), headerCell("Technology", techCols[1])] }),
    new TableRow({ children: [cell("Pose Estimation", techCols[0]), cell("Google MediaPipe Pose (33-landmark body model), loaded client-side via CDN", techCols[1])] }),
    new TableRow({ children: [cell("ML Inference (server)", techCols[0]), cell("TensorFlow.js (Node, @tensorflow/tfjs-node) — feedforward MLP classifier", techCols[1])] }),
    new TableRow({ children: [cell("Backend / API", techCols[0]), cell("Node.js + Express 4, ES modules. Endpoints: GET /api/poses, POST /api/analyze-frame, POST /api/ollama-feedback", techCols[1])] }),
    new TableRow({ children: [cell("Local LLM Coaching", techCols[0]), cell("Ollama (default model: llama3.2), served locally at OLLAMA_HOST", techCols[1])] }),
    new TableRow({ children: [cell("Frontend", techCols[0]), cell("Vanilla HTML/CSS/JavaScript (public/index.html, app.js, index.css) — dark glassmorphism UI", techCols[1])] }),
    new TableRow({ children: [cell("Camera Capture", techCols[0]), cell("Browser MediaDevices.getUserMedia (webcam, mirrored display)", techCols[1])] }),
    new TableRow({ children: [cell("Data Pipeline (offline)", techCols[0]), cell("Python 3 + NumPy (prepare_dataset_v2.py) — normalization, mirror augmentation, from-scratch k-means clustering", techCols[1])] }),
    new TableRow({ children: [cell("File Uploads", techCols[0]), cell("Multer (multipart/form-data) for video-upload analysis pathway", techCols[1])] }),
    new TableRow({ children: [cell("Configuration", techCols[0]), cell("dotenv (.env file)", techCols[1])] }),
  ],
});

// ---- Settings table ----
const settingsCols = [2400, 2400, 4560];
const settingsTable = new Table({
  width: { size: CONTENT_WIDTH, type: WidthType.DXA },
  columnWidths: settingsCols,
  rows: [
    new TableRow({ children: [headerCell("Setting (.env)", settingsCols[0]), headerCell("Default Value", settingsCols[1]), headerCell("Description", settingsCols[2])] }),
    new TableRow({ children: [
      cell("PORT", settingsCols[0]),
      cell("3000", settingsCols[1]),
      cell("Port the Express server listens on. Change to run the app on a different port (e.g. if 3000 is in use).", settingsCols[2]),
    ] }),
    new TableRow({ children: [
      cell("OLLAMA_HOST", settingsCols[0]),
      cell("http://localhost:11434", settingsCols[1]),
      cell("URL of the local Ollama server used for AI coaching messages. Must point to a running Ollama instance.", settingsCols[2]),
    ] }),
    new TableRow({ children: [
      cell("OLLAMA_MODEL", settingsCols[0]),
      cell("llama3.2", settingsCols[1]),
      cell("Name of the Ollama model used to generate coaching feedback messages (e.g. \"Coach Tip\" text shown to the user). Can be swapped for any locally-pulled Ollama model.", settingsCols[2]),
    ] }),
    new TableRow({ children: [
      cell("IMU_ENABLED", settingsCols[0]),
      cell("false", settingsCols[1]),
      cell("Toggles whether the 6 IMU sensor feature slots are populated from real sensor data. Currently zero-filled; the model and pipeline already support 105 total input features (99 landmark + 6 IMU).", settingsCols[2]),
    ] }),
  ],
});

// ---- Coaching message trigger settings (logic-level, not env) ----
const msgCols = [3300, 6060];
const msgTable = new Table({
  width: { size: CONTENT_WIDTH, type: WidthType.DXA },
  columnWidths: msgCols,
  rows: [
    new TableRow({ children: [headerCell("Parameter", msgCols[0]), headerCell("Value / Behavior", msgCols[1])] }),
    new TableRow({ children: [cell("Fast ML check interval", msgCols[0]), cell("Every 500 ms — sends current landmarks to /api/analyze-frame to check step completion or violation", msgCols[1])] }),
    new TableRow({ children: [cell("Step-complete advance delay", msgCols[0]), cell("2 seconds after a step is marked complete before moving to the next step", msgCols[1])] }),
    new TableRow({ children: [cell("Violation → LLM trigger threshold", msgCols[0]), cell("3+ seconds of continuous \"violation\" (step not satisfied) triggers a call to /api/ollama-feedback for a detailed \"Coach Tip\"", msgCols[1])] }),
    new TableRow({ children: [cell("LLM coaching cooldown", msgCols[0]), cell("10 seconds between Ollama coaching calls, to avoid repeated/overlapping requests", msgCols[1])] }),
  ],
});

// ---- POSES summary table ----
const poseCols = [1400, 2400, 5560];
const poseRows = [
  ["STA-01", "Tadasana (Mountain Pose)", "6 steps"],
  ["STA-02", "Vrkasana (Tree Pose)", "5 steps"],
  ["STA-03", "Pada-hastasana (Hand-to-Foot Pose)", "6 steps"],
  ["STA-04-I", "Ardha Cakrasana (Half-Wheel Pose)", "6 steps"],
  ["STA-04-II", "Ardha Katichakrasana (Half Waist-Wheel Pose)", "5 steps"],
  ["STA-05-I", "Trikonasana (Triangle Pose)", "7 steps"],
  ["STA-05-II", "Parivritta Trikonasana (Revolved Triangle Pose)", "6 steps"],
  ["SIA-01", "Ardha Ustrasana (Half Camel Pose)", "6 steps"],
  ["SIA-02", "Vakrasana (Twisted Pose)", "6 steps"],
  ["PR-01", "Makarasana (Crocodile Pose)", "5 steps"],
  ["PR-02", "Bhujangasana (Cobra Pose)", "6 steps"],
  ["SU-01", "Ardha Halasana (Half Plough Pose)", "5 steps"],
  ["SU-02", "Savasana (Corpse Pose)", "5 steps"],
];
const poseTable = new Table({
  width: { size: CONTENT_WIDTH, type: WidthType.DXA },
  columnWidths: poseCols,
  rows: [
    new TableRow({ children: [headerCell("ID", poseCols[0]), headerCell("Pose", poseCols[1]), headerCell("Steps Tracked", poseCols[2])] }),
    ...poseRows.map(([id, name, steps]) => new TableRow({ children: [cell(id, poseCols[0]), cell(name, poseCols[1]), cell(steps, poseCols[2])] })),
  ],
});

// ---- IMU placement table (10-sensor design, future target spec) ----
const imuCols = [620, 1600, 1600, 2400, 3140];
const imuPlacementRows = [
  ["1", "Right Arm (Mid-Upper)", "Between shoulder & elbow", "Tracks upper arm elevation, shoulder flexion/abduction, arm alignment"],
  ["2", "Right Arm (Mid-Lower)", "Between elbow & wrist", "Detects elbow flexion/extension, forearm rotation (pronation/supination)"],
  ["3", "Left Arm (Mid-Upper)", "Between shoulder & elbow", "Monitors symmetry with right arm, shoulder movement"],
  ["4", "Left Arm (Mid-Lower)", "Between elbow & wrist", "Captures fine forearm motion, wrist alignment"],
  ["5", "Spine (Upper-Back)", "Thoracic region", "Tracks posture (upright/slouch), spinal extension, upper body tilt"],
  ["6", "Spine (Lower-Back)", "Lumbar/Pelvic region", "Measures pelvic tilt, lower back bending, core stability"],
  ["7", "Right Leg (Mid-Thigh)", "Between hip & knee", "Detects hip flexion/extension, leg lifting, balance"],
  ["8", "Left Leg (Mid-Thigh)", "Between hip & knee", "Ensures bilateral symmetry, weight distribution"],
  ["9", "Left Leg (Mid-Shin)", "Between knee & ankle", "Captures knee angle indirectly, lower leg alignment"],
  ["10", "Right Leg (Mid-Shin)", "Between knee & ankle", "Tracks stability, stance, and foot-ground interaction dynamics"],
];
const imuTable = new Table({
  width: { size: CONTENT_WIDTH, type: WidthType.DXA },
  columnWidths: imuCols,
  rows: [
    new TableRow({ children: [
      headerCell("Sensor #", imuCols[0]), headerCell("Body Placement", imuCols[1]), headerCell("Exact Position", imuCols[2]),
      headerCell("Parameters Captured", imuCols[3]), headerCell("Purpose", imuCols[4]),
    ] }),
    ...imuPlacementRows.map(([num, placement, position, purpose]) => new TableRow({ children: [
      cell(num, imuCols[0]),
      cell(placement, imuCols[1]),
      cell(position, imuCols[2]),
      cell("Accelerometer (Ax, Ay, Az), Gyroscope (Gx, Gy, Gz), Orientation (Quaternion/Euler)", imuCols[3]),
      cell(purpose, imuCols[4]),
    ] })),
  ],
});

// ---- Image helpers ----
const setupImg = fs.readFileSync("/sessions/confident-epic-edison/mnt/yoga-pose-correction-ai/_demo_preview/mockup_setup.png");
const liveImg = fs.readFileSync("/sessions/confident-epic-edison/mnt/yoga-pose-correction-ai/_demo_preview/mockup_live.png");

function imageParagraph(buf, w, h, title, desc, captionText) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [new ImageRun({
        type: "png",
        data: buf,
        transformation: { width: w, height: h },
        altText: { title, description: desc, name: title },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: captionText, italics: true, size: 18, color: "666666" })],
    }),
  ];
}

const doc = new Document({
  creator: "YogaAlign Team",
  title: "YogaAlign – User Manual",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "1F1147" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "7C3AED", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "5B21B6" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: "111122" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "○", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1260, hanging: 360 } } } },
      ] },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } },
    },
    children: [
      // ---- Title block ----
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: "YogaAlign — AI Pose Correction Coach", bold: true, size: 40, color: "1F1147" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: "User Manual", bold: true, size: 30, color: "7C3AED" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 8 } },
        children: [new TextRun({ text: "Date: June 13, 2026", size: 20, color: "666666" })],
      }),

      // ================= 1. OBJECTIVES =================
      h1("1. Objectives"),
      p("YogaAlign is a locally-run, AI-powered yoga practice coach. Its objectives are:"),
      bullet("Provide real-time, step-by-step guidance through 13 yoga poses, verifying each step of the pose using camera-based body pose estimation."),
      bullet("Detect when a user's posture deviates from the expected step and explain what to adjust, using a combination of a fast ML classifier and a local LLM for richer feedback."),
      bullet("Run entirely on local infrastructure — no cloud ML APIs or paid LLM services — for privacy and offline use."),
      bullet("Be extensible to optional wearable IMU sensor input for higher-accuracy pose verification in future iterations."),
      bullet("Offer a simple, distraction-free web interface usable on a laptop with a standard webcam."),

      // ================= 2. TECHNOLOGY STACK =================
      h1("2. Technology Stack"),
      p("The application is composed of the following layers and technologies:"),
      techTable,
      new Paragraph({ spacing: { before: 120, after: 120 }, children: [new TextRun(
        "All components — the Express server, the TensorFlow.js model, and the Ollama LLM — run on the same machine as the user, " +
        "so no pose data, video, or sensor data leaves the device."
      )] }),

      // ================= 3. METHODOLOGY / IMPLEMENTATION =================
      h1("3. Methodology & Implementation Details"),

      h2("3.1 Data Collection"),
      p("The training set combines two recording batches, now merged into a single dataset spanning 15 participants across 18 recording sessions. Each recording captured:"),
      bullet("MediaPipe 33-point body landmarks (x, y, z coordinates per frame)"),
      bullet("Synchronized 6-axis IMU sensor packets (accelerometer + gyroscope) for the original batch, see Section 5.3"),
      bullet("Per-session metadata (pose name, participant, recording duration)"),
      bullet("Batch 1 — 7 participants, 9 sessions, 40 individual pose recordings, covering all 13 supported poses."),
      bullet("Batch 2 — 8 additional participants, 9 further sessions, covering the 7 “Standing” poses (Tadasana through Parivritta Trikonasana). These recordings were processed with merge_new_participants.py, which applies the same normalization, k-means step-labeling, median smoothing, and mirror-augmentation pipeline as Section 3.2, with the 6 IMU feature slots zero-filled for consistency with Batch 1's feature format, and merged into training_data.json."),

      h2("3.2 Dataset Preparation (prepare_dataset_v2.py)"),
      p("Raw recordings are converted into a labeled training set through three steps:"),
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Landmark Normalization")] }),
      p("Every frame's 33 landmarks are re-expressed relative to the body itself, removing dependence on camera distance and framing:"),
      bullet("Translate so the hip-midpoint — the average of left hip (landmark 23) and right hip (landmark 24) — becomes the origin (0,0,0)."),
      bullet("Scale all coordinates by the distance between the shoulder-midpoint (average of landmarks 11 and 12) and the hip-midpoint (an approximation of torso length)."),
      bullet("An epsilon guard (scale < 1e-6 → scale = 1) prevents division-by-zero for degenerate frames."),
      code("hipMid = avg(landmark[23], landmark[24])\nshoulderMid = avg(landmark[11], landmark[12])\nscale = distance(shoulderMid, hipMid)  // guarded: if < 1e-6, scale = 1\nnormalized[i] = (landmark[i] - hipMid) / scale"),
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Mirror Augmentation")] }),
      p("Every normalized sample is duplicated with the x-axis negated and left/right landmark pairs swapped (e.g. left shoulder ↔ right shoulder), doubling the sample count and teaching the model to recognize poses performed facing either direction."),
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Clustering-Based Step Labeling")] }),
      p("Each pose is broken into a fixed number of \"steps\" (matching the step list shown to the user). Instead of dividing each recording into equal time chunks, frames are grouped using k-means clustering implemented from scratch (k-means++ initialization, K = number of steps for that pose). Clusters are then ordered by their temporal centroid (mean frame index within the recording) and the resulting per-frame labels are smoothed with a median filter (window = 5) to remove noise/jitter at step boundaries."),
      p("Applied to Batch 1 alone, this produced a dataset of 29,116 samples across 74 distinct step-level classes spanning the 13 poses listed in Section 5.1. After merging Batch 2 (Section 3.1) using the same pipeline, the combined dataset spans 66,410 samples across the same 74 step-level classes — split 56,448 for training and 9,962 for validation (an 85/15 split)."),

      h2("3.3 Model Architecture & Training"),
      p("The pose-step classifier is a feedforward neural network (multilayer perceptron) built with TensorFlow.js:"),
      code("Input (117 features)\n  → Dense(256, activation='relu') → Dropout(0.3)\n  → Dense(128, activation='relu') → Dropout(0.2)\n  → Dense(64,  activation='relu')\n  → Dense(74,  activation='softmax')\n\nLoss: categoricalCrossentropy   Optimizer: Adam (lr = 0.001)   Batch size: 64"),
      p("The 117-dimensional input feature vector is composed of:"),
      bullet("99 features — the 33 normalized landmarks × (x, y, z)"),
      bullet("12 features — derived joint-angle features (see below)"),
      bullet("6 features — IMU sensor channels (zero-filled while IMU_ENABLED = false; see Section 5.3)"),
      p("Training on the v2 dataset (normalized + mirror-augmented + cluster-labeled, Batch 1 only — 29,116 samples) raised validation accuracy from a 57.78% baseline (pre-normalization / naive labeling) to 87.25%. After merging Batch 2 (Section 3.1) and retraining on the full 66,410-sample dataset, validation accuracy was initially 74.38%. This is a more conservative but more representative figure: the merged dataset is over twice the size and covers 8 additional participants, so the model now has to generalize across substantially more body types and movement styles than the 7-participant set the 87.25% figure was measured on."),
      p("A follow-up (v3) training pass addressed the largest source of remaining error: the 74 step classes are unevenly represented (102–3,233 training samples per class, a ~33x range). Continuing training with class-balanced minibatch sampling — each batch drawn by picking a class uniformly at random and then a sample uniformly within that class, rather than sampling in proportion to raw class frequency — combined with a learning-rate decay schedule (0.001 for the first 60% of training, 0.0005 for the next 25%, 0.0002 for the final 15%) over roughly 21 epochs, raised validation accuracy from 74.38% to 79.27% (val loss 0.50), with no change to the model architecture or the 105-feature input format."),
      p("A further (v5) iteration added 12 joint-angle features, computed deterministically from the 33 landmarks (pose_features.mjs): left/right elbow, knee, hip, shoulder and ankle angles, a spine-tilt angle, and a shoulder-line-tilt angle. With the input vector expanded from 105 to 117 features, the model was retrained from scratch using a two-phase recipe — phase 1: 18 epochs with uniform random sampling (mirroring the original v2 baseline, reaching 75.03%), then phase 2: 21 epochs with the same class-balanced sampling + learning-rate decay schedule used for v3. The result was 81.68% validation accuracy (val loss 0.5214) — a +2.4 point improvement over v3, and the first version to clear the 80% target. v5 is the currently deployed model. Remaining headroom is expected to come from a larger network and/or real IMU accelerometer/gyroscope data (see Section 5.3)."),

      h2("3.4 Real-Time Inference Pipeline"),
      p("The /api/analyze-frame endpoint in server.js receives the current pose name, step index, and the 33 raw landmarks from the browser for each analyzed frame. It then:"),
      bullet("Applies the exact same normalizeLandmarks() transformation used during training (hip-centering + torso-scale normalization), so the model sees data in the same representation it was trained on."),
      bullet("Computes the same 12 joint-angle features used during training (computeJointAngles() in pose_features.mjs), from the normalized landmarks."),
      bullet("Concatenates the 99 normalized landmark features, 12 joint-angle features, and 6 IMU features (currently zeros)."),
      bullet("Runs the 117-feature vector through the TensorFlow.js model to get a probability distribution over the 74 step classes."),
      bullet("Compares the predicted step against the expected current step for the selected pose, returning either stepComplete (advance to next step) or hasViolation (step requirements not yet met, with a confidence score)."),

      h2("3.5 Hybrid Feedback & Local LLM Coaching"),
      p("Two feedback loops work together:"),
      bullet("Fast loop (every 500 ms): the ML classifier checks step completion/violation and updates the on-screen step card immediately."),
      bullet("Deep loop (LLM): if a violation persists for 3+ seconds, the frontend calls /api/ollama-feedback, which sends the pose, step, and landmark context to a local Ollama model (default llama3.2). Ollama returns structured JSON coaching feedback, displayed as a \"Coach Tip\" in the AI feedback panel. A 10-second cooldown prevents repeated calls while the user keeps adjusting."),
      p("This design keeps the experience responsive (sub-second ML feedback) while reserving the slower, more detailed LLM explanations for moments where the user genuinely needs more guidance — and it replaces the earlier cloud-based Gemini integration entirely, so all coaching runs offline."),

      // ================= 4. COMPLETED WORK WITH SNAPSHOTS =================
      new Paragraph({ children: [new PageBreak()] }),
      h1("4. Completed Work — Snapshots of Results"),
      p("The web application provides two main screens: a pose-selection screen, and a live-coaching screen with step-by-step guidance and AI feedback."),

      h2("4.1 Pose Selection Screen"),
      ...imageParagraph(setupImg, 470, 305, "Pose Selection Screen", "YogaAlign pose selection screen showing the dropdown of 13 supported poses and Start Live Coaching button",
        "Figure 1 — Pose selection screen. Users pick from all 13 supported poses (e.g. Trikonasana / Triangle Pose) before starting a session."),

      h2("4.2 Live Coaching Screen"),
      ...imageParagraph(liveImg, 470, 363, "Live Coaching Screen", "YogaAlign live coaching screen showing camera feed placeholder, current step instruction card, and AI feedback panel",
        "Figure 2 — Live coaching screen. The camera feed appears at top (with MediaPipe landmark overlay), the current step instruction is shown in the step card, and AI feedback (\"Coach Tip\") appears below when the user needs adjustment."),

      p("Model results: the 74-class step classifier, retrained on the merged 66,410-sample / 15-participant dataset with class-balanced sampling and learning-rate decay (v3), achieves 79.27% validation accuracy (up from 74.38% for the initial merged-dataset training run, and from a 57.78% baseline before normalization, mirror augmentation, and clustering-based labeling were introduced; see Section 3.3 for full discussion, including the change from the earlier 87.25% figure measured on the smaller 7-participant dataset)."),

      h2("4.3 Supported Poses & Step Breakdown"),
      p("The application currently supports the following 13 poses, each broken down into a sequence of verifiable steps:"),
      poseTable,

      // ================= 5. SETTINGS, CONNECTIVITY =================
      new Paragraph({ children: [new PageBreak()] }),
      h1("5. Settings & Connectivity"),

      h2("5.1 Application Settings"),
      p("All runtime configuration is stored in a .env file at the project root (see .env.example for a template):"),
      settingsTable,
      new Paragraph({ spacing: { before: 160, after: 80 }, children: [new TextRun({ text: "Coaching Message & Timing Behavior", bold: true })] }),
      p("In addition to the .env settings above, the following timing parameters control the coaching message logic (currently set in app.js / server.js):"),
      msgTable,

      h2("5.2 Camera Feed Connectivity"),
      bullet("The browser requests camera access via navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } }) — the front-facing/webcam camera."),
      bullet("The live video is rendered mirrored (scaleX(-1)) so the user sees themselves as in a mirror, matching natural movement expectations."),
      bullet("Pose detection runs via the MediaPipe Pose model, loaded from the jsDelivr CDN (@mediapipe/pose/pose.js and @mediapipe/drawing_utils/drawing_utils.js) directly in the browser — no video frames are sent to any external server for pose detection."),
      bullet("Only the resulting 33 landmark coordinates (not raw video) are sent from the browser to the local Express server for ML classification, via POST /api/analyze-frame."),
      bullet("No camera setup beyond browser permission is required; any standard built-in or USB webcam supported by the browser will work."),

      h2("5.3 IMU Sensor Design (Future Target Spec)"),
      p("Today, IMU_ENABLED = false: the model's 105-feature input vector reserves 6 slots for IMU data, but these are zero-filled at inference time. The feature pipeline and server.js logic for incorporating IMU data are implemented end-to-end, but no wearable hardware has been deployed yet."),
      p("A revised hardware design — a 10-sensor IMU array, one BNO085 module per major limb segment plus upper and lower spine — has been specified as the target for the next hardware iteration. Each sensor reports an Accelerometer reading (Ax, Ay, Az), a Gyroscope reading (Gx, Gy, Gz), and a fused Orientation estimate (Quaternion/Euler). The 10 placements and their purpose in pose analysis are:"),
      imuTable,
      new Paragraph({ spacing: { before: 120, after: 120 }, children: [new TextRun(
        "These 10 placements cover all major kinematic chains (arms, spine, legs), enabling indirect joint-angle estimation (e.g. elbow angle " +
        "from the difference between the upper- and lower-arm sensors), left/right symmetry analysis, and posture/balance metrics such as " +
        "pelvic tilt and sway — all directly useful for pose correction feedback."
      )] }),
      new Paragraph({ spacing: { before: 0, after: 120 }, children: [new TextRun({ text:
        "This is a significant expansion over the current 6-feature IMU placeholder (60 raw channels across 10 sensors vs. 6). " +
        "Implementing it will require: (1) deploying the 10-sensor array during a future recording session, (2) extending " +
        "merge_new_participants.py / prepare_dataset_v2.py to ingest the new per-sensor packet format, (3) redesigning the model's " +
        "input feature vector to accommodate the larger IMU channel set, and (4) retraining with non-zero IMU features and setting " +
        "IMU_ENABLED = true. None of this is implemented yet — see Progress Report, Section 3 (\"IMU sensor fusion\").",
        italics: true, color: "666666", size: 20,
      })] }),
      new Paragraph({ spacing: { before: 0, after: 120 }, children: [new TextRun({ text:
        "Status check (this iteration): the original 7-participant batch's training samples already carry real, non-zero " +
        "6-axis accelerometer/gyroscope readings from the earlier single-IMU setup — those were never zero-weighted. The raw " +
        "sensor logs recorded alongside the 8 new participants' sessions (imu_data.json / imu.json / imu.jsonl, one entry per " +
        "sensor of the new 10-sensor array) were inspected directly: every packet's accel and gyro fields are [0, 0, 0], with " +
        "only the orientation/quaternion field populated. There is therefore no real accelerometer/gyroscope data currently " +
        "available to merge in for the new participants — the zero-filled IMU features used for that batch (Section 3.1) " +
        "remain the correct choice until the sensor firmware streams non-zero accel/gyro values.",
        italics: true, color: "666666", size: 20,
      })] }),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("/tmp/docgen/User_Manual.docx", buffer);
  console.log("written");
});
