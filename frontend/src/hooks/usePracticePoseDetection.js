import { useEffect, useRef, useState } from "react";
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import {
  computeEightAngles,
  diffCompareAngle,
  getCorrections,
  minJointVisibility,
  POSE_LM,
} from "../utils/practicePoseAnalysis";
import { getTargetAnglesForPoseName } from "../data/practicePoseTargets";

const WASM_BASE = "/assets/mediapipe/wasm";

const MODEL_PATH =
  "/assets/mediapipe/models/pose_landmarker_lite.task";

// How often to hit the ML backend (ms). 1500ms = ~once every 1.5s —
// enough to feel live without hammering local compute.
const ML_UPDATE_MS = 1500;

// Ollama is called only after this many ms of continuous violation —
// avoids firing on momentary wobbles.
const OLLAMA_VIOLATION_MS = 3000;
// Minimum gap between Ollama calls so we don't spam a slow local model.
const OLLAMA_COOLDOWN_MS = 15000;

const CORE_INDICES = [
  POSE_LM.LEFT_SHOULDER,
  POSE_LM.RIGHT_SHOULDER,
  POSE_LM.LEFT_HIP,
  POSE_LM.RIGHT_HIP,
];

const UI_UPDATE_MS = 120;

function syncCanvasToVideo(canvas, video) {
  if (!canvas || !video || !video.videoWidth) return;
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const rw = video.getBoundingClientRect().width;
  const rh = video.getBoundingClientRect().height;
  canvas.style.width = `${rw}px`;
  canvas.style.height = `${rh}px`;
}

/**
 * Loads MediaPipe PoseLandmarker, draws skeleton on canvas, updates practice context (throttled).
 * Every ML_UPDATE_MS it also POSTs landmarks to /api/analyze-frame so the server-side
 * TF.js model + posture descriptors can produce richer corrective feedback.
 */
export function usePracticePoseDetection({
  videoRef,
  canvasRef,
  enabled,
  practicePoseName,
  currentStepIndex = 0,
  userName,
  age,
  setDetectedPose,
  setConfidence,
  setCorrections,
  setMlFeedback,
  setOllamaFeedback,
}) {
  const [mediapipeError, setMediapipeError] = useState(null);
  const lastVideoTimeRef = useRef(-1);
  const lastUiUpdateRef = useRef(0);
  const lastMlUpdateRef = useRef(0);
  const mlPendingRef = useRef(false);
  const violationStartRef = useRef(null);   // when continuous violation began
  const lastOllamaRef = useRef(0);          // last Ollama call timestamp
  const ollamaPendingRef = useRef(false);
  const practicePoseNameRef = useRef(practicePoseName);
  practicePoseNameRef.current = practicePoseName;
  const currentStepIndexRef = useRef(currentStepIndex);
  currentStepIndexRef.current = currentStepIndex;

  useEffect(() => {
    if (!enabled) {
      setMediapipeError(null);
      return;
    }

    let cancelled = false;
    let landmarker = null;
    let rafId = 0;
    let drawingUtils = null;

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
        landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_PATH,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch (e) {
        if (!cancelled) {
          setMediapipeError(
            e?.message ||
              "Could not load pose model. Check your network and try again."
          );
        }
        return;
      }

      if (cancelled) {
        landmarker?.close();
        return;
      }

      setMediapipeError(null);
      const targetAngles = () =>
        getTargetAnglesForPoseName(practicePoseNameRef.current);

      const loop = () => {
        if (cancelled) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!landmarker || !video || !canvas) {
          rafId = requestAnimationFrame(loop);
          return;
        }

        if (video.readyState < 2) {
          rafId = requestAnimationFrame(loop);
          return;
        }

        syncCanvasToVideo(canvas, video);

        const ctx = canvas.getContext("2d");
        if (!ctx) {
          rafId = requestAnimationFrame(loop);
          return;
        }

        if (!drawingUtils) {
          drawingUtils = new DrawingUtils(ctx);
        }

        if (video.currentTime === lastVideoTimeRef.current) {
          rafId = requestAnimationFrame(loop);
          return;
        }
        lastVideoTimeRef.current = video.currentTime;

        let result;
        try {
          result = landmarker.detectForVideo(video, performance.now());
          console.log("Pose Result:", result);
        } catch {
          rafId = requestAnimationFrame(loop);
          return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const landmarks = result?.landmarks?.[0];
        const now = performance.now();

        if (
          landmarks &&
          minJointVisibility(landmarks, CORE_INDICES) >= 0.35
        ) {
          drawingUtils.drawConnectors(
            landmarks,
            PoseLandmarker.POSE_CONNECTIONS,
            { color: "#00c853", lineWidth: 4 }
          );
          drawingUtils.drawLandmarks(landmarks, {
            color: "#ff1744",
            lineWidth: 2,
            radius: 5,
          });

          const angles = computeEightAngles(landmarks);
          const tAngles = targetAngles();
          const corrections = getCorrections(angles, tAngles);
          const aScore = diffCompareAngle(angles, tAngles);
          const conf = Math.round(
            Math.max(0, Math.min(100, (1 - aScore) * 100))
          );

          if (now - lastUiUpdateRef.current >= UI_UPDATE_MS) {
            lastUiUpdateRef.current = now;
            setCorrections(corrections);
            setConfidence(conf);
            const label = practicePoseNameRef.current;
            setDetectedPose(conf >= 48 ? label : "—");
          }

          // ── ML backend call (throttled) ──────────────────────────────────
          if (
            now - lastMlUpdateRef.current >= ML_UPDATE_MS &&
            !mlPendingRef.current &&
            practicePoseNameRef.current
          ) {
            lastMlUpdateRef.current = now;
            mlPendingRef.current = true;
            const poseName = practicePoseNameRef.current;
            const stepIndex = currentStepIndexRef.current;
            fetch("/api/analyze-frame", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                poseName,
                currentStepIndex: stepIndex,
                landmarks,
                userName: userName ?? null,
                age: age ?? null,
              }),
            })
              .then((r) => (r.ok ? r.json() : null))
              .then((data) => {
                if (!data) return;
                if (data.analysis?.feedback && typeof setMlFeedback === "function") {
                  setMlFeedback(data.analysis.feedback);
                }
                // Track violation window for Ollama trigger
                const isMatch = data.analysis?.isMatch ?? true;
                if (isMatch) {
                  violationStartRef.current = null; // reset on match
                } else {
                  if (violationStartRef.current === null) {
                    violationStartRef.current = performance.now();
                  }
                }
              })
              .catch(() => {})
              .finally(() => {
                mlPendingRef.current = false;
              });
          }

          // ── Ollama LLM fallback (after 3s continuous violation) ──────────
          // Only fires when the ML model has consistently reported a mismatch
          // for OLLAMA_VIOLATION_MS, and we're past the cooldown window.
          // Captures the live canvas frame as JPEG so the vision model can
          // actually look at the user's body, not just coordinate numbers.
          const violationDuration = violationStartRef.current !== null
            ? now - violationStartRef.current : 0;

          if (
            violationDuration >= OLLAMA_VIOLATION_MS &&
            !ollamaPendingRef.current &&
            now - lastOllamaRef.current >= OLLAMA_COOLDOWN_MS &&
            practicePoseNameRef.current
          ) {
            lastOllamaRef.current = now;
            ollamaPendingRef.current = true;
            const poseName = practicePoseNameRef.current;
            const stepIndex = currentStepIndexRef.current;

            // Snapshot the canvas (has skeleton overlay drawn on it)
            let imageBase64 = null;
            try {
              const canvas = canvasRef.current;
              if (canvas) imageBase64 = canvas.toDataURL("image/jpeg", 0.6);
            } catch { /* cross-origin canvas taint — skip image */ }

            fetch("/api/ollama-feedback", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                poseName,
                currentStepIndex: stepIndex,
                landmarks,
                imageBase64,
                userName: userName ?? null,
                age: age ?? null,
              }),
            })
              .then((r) => (r.ok ? r.json() : null))
              .then((data) => {
                if (data?.analysis?.feedback && typeof setOllamaFeedback === "function") {
                  setOllamaFeedback(data.analysis.feedback);
                }
              })
              .catch(() => {})
              .finally(() => {
                ollamaPendingRef.current = false;
              });
          }
          // ────────────────────────────────────────────────────────────────
        } else {
          if (now - lastUiUpdateRef.current >= UI_UPDATE_MS) {
            lastUiUpdateRef.current = now;
            setDetectedPose("—");
            setConfidence(0);
            setCorrections([
              "Step into frame so your upper body and hips are visible",
            ]);
          }
        }

        rafId = requestAnimationFrame(loop);
      };

      rafId = requestAnimationFrame(loop);
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      lastVideoTimeRef.current = -1;
      landmarker?.close();
      landmarker = null;
      drawingUtils = null;
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
    };
  }, [enabled, videoRef, canvasRef, setDetectedPose, setConfidence, setCorrections]);

  return { mediapipeError };
}
