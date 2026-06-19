document.addEventListener('DOMContentLoaded', () => {
  const els = {
    poseSelect: document.getElementById('pose-select'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    setupSection: document.getElementById('setup-section'),
    liveSection: document.getElementById('live-section'),
    video: document.getElementById('webcam-video'),
    canvas: document.getElementById('capture-canvas'),
    stepNumber: document.getElementById('step-number'),
    stepInstruction: document.getElementById('step-instruction'),
    aiFeedback: document.getElementById('ai-feedback'),
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text')
  };

  let posesData = [];
  let currentPose = null;
  let currentStepIndex = 0;
  let stream = null;
  let isAnalyzing = false;

  // Hybrid Timer Logic
  let violationStartTime = null;
  let lastOllamaRequestTime = 0;
  const OLLAMA_COOLDOWN = 10000; // 10s cooldown for rich feedback to avoid spam
  let isRequestingOllama = false;

  // --------------------------------------------------------------------
  // Live MediaPipe Pose integration (temporary demo wiring).
  // Runs MediaPipe Pose on the webcam feed in-browser and feeds the 33
  // resulting (x, y, z) landmarks to /api/analyze-frame, exactly the
  // format server.js expects (it handles normalization + joint-angle
  // feature computation server-side). This is a stand-in for the future
  // dedicated frontend that will own the MediaPipe integration.
  // --------------------------------------------------------------------
  let latestLandmarks = null;
  let poseModel = null;
  let poseDetectionActive = false;

  // MediaPipe occasionally drops a single frame (motion blur, brief
  // occlusion during a transition like lifting an arm/heel) without the
  // person actually having left the frame. If we treated every dropped
  // frame as "they just reappeared", settleUntil below would keep getting
  // pushed forward forever and the UI would get stuck on "Get into
  // position…" indefinitely. Require a short run of consecutive missed
  // detections before treating the person as actually gone.
  let missingFrameStreak = 0;
  const MISSING_FRAMES_THRESHOLD = 6;

  function initPoseModel() {
    if (poseModel) return poseModel;
    poseModel = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    poseModel.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    poseModel.onResults(onPoseResults);
    return poseModel;
  }

  function onPoseResults(results) {
    if (results.poseLandmarks && results.poseLandmarks.length === 33) {
      missingFrameStreak = 0;
      // If the person just (re)appeared in frame, give them a moment to
      // settle into position before we start commenting on their posture —
      // otherwise feedback can fire mid-walk/recentre.
      if (!latestLandmarks) {
        settleUntil = Date.now() + SETTLE_GRACE_MS;
      }
      // Keep MediaPipe's per-landmark `visibility` score — used below to
      // detect when the camera isn't actually showing the user's whole body
      // (e.g. laptop held up close to the face), which otherwise produces
      // garbage/extrapolated landmarks that the model can still confidently
      // (and wrongly) match against a step.
      latestLandmarks = results.poseLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility }));
    } else {
      // Don't immediately treat one dropped frame as "person left" — only
      // clear latestLandmarks (and thus trigger the reappear/settle logic
      // above) after several consecutive misses in a row.
      missingFrameStreak++;
      if (missingFrameStreak >= MISSING_FRAMES_THRESHOLD) {
        latestLandmarks = null;
      }
    }
  }

  // MediaPipe landmark indices for shoulders, hips, knees, and ankles — the
  // body points the normalization + joint-angle features depend on most.
  // If these aren't reasonably visible, the computed pose is meaningless.
  const BODY_VISIBILITY_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28];
  const MIN_LANDMARK_VISIBILITY = 0.5;
  // Small tolerance around the [0,1] normalized frame so landmarks right at
  // the edge aren't unfairly flagged, but anything clearly outside the frame
  // (e.g. heels off the bottom edge when the camera is too zoomed-in) is.
  const FRAME_BOUNDS_TOLERANCE = 0.02;

  function isFullBodyVisible(landmarks) {
    return BODY_VISIBILITY_LANDMARKS.every(idx => {
      const lm = landmarks[idx];
      if (!lm) return false;
      // MediaPipe can report a confident `visibility` score for a landmark
      // it has extrapolated *outside* the actual camera frame (e.g. heels
      // lifted up and off the bottom edge of a tight shot) — so visibility
      // alone isn't enough. Also require the landmark's (x, y) position to
      // actually fall within the visible frame.
      const inFrame = lm.x >= -FRAME_BOUNDS_TOLERANCE && lm.x <= 1 + FRAME_BOUNDS_TOLERANCE
        && lm.y >= -FRAME_BOUNDS_TOLERANCE && lm.y <= 1 + FRAME_BOUNDS_TOLERANCE;
      return (lm.visibility ?? 0) >= MIN_LANDMARK_VISIBILITY && inFrame;
    });
  }

  async function poseDetectionLoop() {
    if (!poseDetectionActive) return;
    if (els.video.videoWidth > 0 && els.video.readyState >= 2) {
      try {
        await poseModel.send({ image: els.video });
      } catch (e) {
        console.error('MediaPipe Pose error:', e);
      }
    }
    if (poseDetectionActive) requestAnimationFrame(poseDetectionLoop);
  }

  // 1. Fetch poses
  fetch('/api/poses')
    .then(res => res.json())
    .then(poses => {
      posesData = poses;
      els.poseSelect.innerHTML = '<option value="" disabled selected>Select a pose...</option>';
      poses.forEach(pose => {
        if (pose.steps && pose.steps.length > 0) {
          const opt = document.createElement('option');
          opt.value = pose.name;
          opt.textContent = pose.name;
          els.poseSelect.appendChild(opt);
        }
      });
    });

  // 2. Start Practice
  els.btnStart.addEventListener('click', async () => {
    const selectedName = els.poseSelect.value;
    if (!selectedName) {
      alert('Please select a pose first.');
      return;
    }

    currentPose = posesData.find(p => p.name === selectedName);
    currentStepIndex = 0;

    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      els.video.srcObject = stream;

      els.setupSection.hidden = true;
      els.liveSection.hidden = false;

      updateStepUI();

      // Start live MediaPipe Pose detection on the camera feed (runs in the
      // background — no skeleton overlay, just feeds landmarks to the model)
      latestLandmarks = null;
      initPoseModel();
      poseDetectionActive = true;
      requestAnimationFrame(poseDetectionLoop);

      // Start the Hybrid High-Speed Loop (TF.js Polling)
      // We poll every 500ms for "real-time" feedback without overwhelming the network
      requestAnimationFrame(mainLoop);

      els.statusDot.className = 'status-dot recording';
      els.statusText.textContent = 'Watching...';

    } catch (err) {
      alert('Could not access webcam. Please allow camera permissions. ' + err.message);
    }
  });

  // 3. Stop Practice
  els.btnStop.addEventListener('click', stopPractice);

  function stopPractice() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }
    poseDetectionActive = false;
    latestLandmarks = null;
    els.setupSection.hidden = false;
    els.liveSection.hidden = true;
    currentPose = null;
  }

  function updateStepUI() {
    // Reset per-step smoothing/timers whenever the step changes
    predictionWindow = [];
    violationStartTime = null;
    feedbackHoldUntil = 0;
    // Give the user a moment to get into position for the new step before
    // we start giving corrective feedback.
    settleUntil = Date.now() + SETTLE_GRACE_MS;

    if (currentStepIndex < currentPose.steps.length) {
      els.stepNumber.textContent = `Step ${currentStepIndex + 1} of ${currentPose.steps.length}`;
      els.stepInstruction.textContent = currentPose.steps[currentStepIndex];
      els.aiFeedback.textContent = 'Awaiting correct position...';
      els.aiFeedback.className = 'ai-feedback';
    } else {
      els.stepNumber.textContent = 'Practice Complete!';
      els.stepInstruction.textContent = 'Great job finishing the pose sequence.';
      els.aiFeedback.textContent = '';
      els.aiFeedback.className = 'ai-feedback success';
      els.statusDot.className = 'status-dot';
      els.statusText.textContent = 'Finished';
    }
  }

  let lastPollTime = 0;
  const POLL_INTERVAL = 500; // 500ms for TF.js classification

  // Temporal smoothing for step-completion detection. A single frame from a
  // live camera is noisy (jitter, brief misclassifications), so we require a
  // majority of the recent frames to match the expected step before
  // advancing, instead of a single high-confidence frame.
  const WINDOW_SIZE = 6;          // ~3s of polling at 500ms
  const REQUIRED_MATCHES = 2;     // out of the window must match (was 3 — too strict for some steps)
  const MATCH_CONF_THRESHOLD = 0.25; // per-frame confidence to count as a "match" (was 0.35)
  let predictionWindow = [];

  // Keep a recent Ollama coaching tip on screen for a bit instead of letting
  // the next 500ms ML poll immediately overwrite it.
  let feedbackHoldUntil = 0;
  const FEEDBACK_HOLD_MS = 4000;

  // Grace period after (a) a person first appears/reappears in frame, or
  // (b) a new step begins. During this window we show a neutral "get into
  // position" message instead of corrective feedback, and don't feed these
  // transitional frames into the prediction window or violation timer — so
  // walking back to recentre yourself doesn't immediately trigger feedback.
  const SETTLE_GRACE_MS = 1500;
  let settleUntil = 0;

  async function mainLoop(now) {
    if (!currentPose) return; // Stopped

    // 1. Every 500ms, run the Fast ML Classification
    if (now - lastPollTime > POLL_INTERVAL) {
      lastPollTime = now;
      await runFastMLCheck();
    }

    requestAnimationFrame(mainLoop);
  }

  async function runFastMLCheck() {
    // Landmarks come from the live MediaPipe Pose detector (see
    // onPoseResults/poseDetectionLoop above). Skip analysis until a person
    // is actually detected in frame.
    if (!latestLandmarks) {
      if (Date.now() >= feedbackHoldUntil) {
        els.aiFeedback.className = 'ai-feedback';
        els.aiFeedback.textContent = 'Step into the camera frame so MediaPipe can detect your pose…';
      }
      return;
    }

    // Settling period: just got into frame / started a new step. Skip
    // analysis entirely so transitional movement doesn't generate feedback
    // or pollute the prediction window / violation timer.
    if (Date.now() < settleUntil) {
      if (Date.now() >= feedbackHoldUntil) {
        els.aiFeedback.className = 'ai-feedback';
        els.aiFeedback.textContent = 'Get into position…';
      }
      return;
    }

    // Whole-body visibility gate. If the camera is too close / cropped (e.g.
    // brought up near the face), MediaPipe still returns 33 landmarks but
    // the hips/knees/ankles are extrapolated guesses with low `visibility`.
    // Feeding that into the model can produce a confident-but-meaningless
    // "match", so skip analysis entirely until the whole body is in frame.
    if (!isFullBodyVisible(latestLandmarks)) {
      if (Date.now() >= feedbackHoldUntil) {
        els.aiFeedback.className = 'ai-feedback';
        els.aiFeedback.textContent = 'Step back so your whole body — shoulders to ankles — is visible in frame.';
      }
      // Treat re-entering full view like reappearing in frame: give a brief
      // settle period before resuming feedback.
      settleUntil = Date.now() + SETTLE_GRACE_MS;
      return;
    }

    try {
      const res = await fetch('/api/analyze-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          poseName: currentPose.name,
          currentStepIndex: currentStepIndex,
          landmarks: latestLandmarks
        })
      });

      const data = await res.json();
      if (!data.success) return;

      const { isMatch, matchConfidence, feedback } = data.analysis;

      // Record this frame's result in the rolling window
      predictionWindow.push({ isMatch: !!isMatch, confidence: matchConfidence || 0 });
      if (predictionWindow.length > WINDOW_SIZE) predictionWindow.shift();

      const matchCount = predictionWindow.filter(
        p => p.isMatch && p.confidence > MATCH_CONF_THRESHOLD
      ).length;
      const stepComplete = predictionWindow.length === WINDOW_SIZE && matchCount >= REQUIRED_MATCHES;

      if (stepComplete) {
        predictionWindow = [];
        violationStartTime = null;
        feedbackHoldUntil = 0;
        els.aiFeedback.className = 'ai-feedback success';
        els.aiFeedback.textContent = '✅ Looking good — hold the position!';

        // Wait 2s and move to next step
        setTimeout(() => {
          if (currentPose) {
            currentStepIndex++;
            updateStepUI();
          }
        }, 2000);

      } else {
        // START/TRACK VIOLATION TIMER
        if (!violationStartTime) {
          violationStartTime = Date.now();
        } else {
          const elapsed = Date.now() - violationStartTime;
          if (elapsed > 3000) {
            // BEEN IN VIOLATION FOR 3+ SECONDS -> TRIGGER LOCAL LLM (Ollama)
            triggerOllamaFallback();
          }
        }

        // Don't stomp on a recently-shown Ollama coaching tip
        if (Date.now() >= feedbackHoldUntil) {
          els.aiFeedback.className = 'ai-feedback';
          els.aiFeedback.textContent = '⚠️ ' + feedback;
        }
      }
    } catch (e) {
      console.error("Fast ML Error:", e);
    }
  }

  // Capture the current webcam frame as a base64 JPEG (no data: prefix) so
  // the server can hand it to a local vision model (e.g. llava/moondream via
  // Ollama) for Gemini-Vision-style feedback — no internet/API key needed.
  function captureFrameBase64() {
    try {
      const video = els.video;
      if (!video.videoWidth || !video.videoHeight) return null;
      const canvas = els.canvas;
      // Downscale to keep the payload small and inference fast.
      const maxDim = 480;
      const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      return dataUrl.split(',')[1];
    } catch (e) {
      console.error('Frame capture error:', e);
      return null;
    }
  }

  async function triggerOllamaFallback() {
    const now = Date.now();
    if (isRequestingOllama || (now - lastOllamaRequestTime < OLLAMA_COOLDOWN)) return;

    isRequestingOllama = true;
    els.statusDot.className = 'status-dot analyzing';
    els.statusText.textContent = 'AI Coach Thinking...';

    try {
      const imageBase64 = captureFrameBase64();
      const res = await fetch('/api/ollama-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          poseName: currentPose.name,
          currentStepIndex: currentStepIndex,
          landmarks: latestLandmarks,
          imageBase64
        })
      });

      const data = await res.json();
      if (data.success) {
        // DISPLAY RICH LLM FEEDBACK, and hold it on screen for a few seconds
        // so the next 500ms ML poll doesn't immediately overwrite it.
        els.aiFeedback.className = 'ai-feedback';
        els.aiFeedback.innerHTML = `<strong>Coach Tip:</strong> ${data.analysis.feedback}`;
        feedbackHoldUntil = Date.now() + FEEDBACK_HOLD_MS;
        lastOllamaRequestTime = Date.now();
      } else if (data.error) {
        console.warn("Ollama Fallback:", data.error);
      }
    } catch (e) {
      console.error("Ollama Fallback Error:", e);
    } finally {
      isRequestingOllama = false;
      els.statusDot.className = 'status-dot recording';
      els.statusText.textContent = 'Watching...';
    }
  }
});
