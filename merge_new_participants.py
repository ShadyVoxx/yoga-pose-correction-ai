#!/usr/bin/env python3
"""
merge_new_participants.py — Merge the 8 newly-collected participants' pose/landmark
recordings (from SensorData/Sessions/) into training_data.json.

Scope (landmarks only, per project decision):
  - Only the 33-point MediaPipe landmark data from the new sessions is used.
  - imuFeatures for every new entry is zero-filled ([0,0,0,0,0,0]), matching the
    IMU_ENABLED=false / zero-filled convention already used by train_model.js.
  - The new sessions only cover the 7 "Standing" poses (STA-01 .. STA-05-II), so
    no new pose/step labels are introduced — only more samples for existing
    74 step classes.

The new data comes in two on-disk formats:
  Format A — per-pose subdirectories, each with its own landmarks.json
             (a JSON list of frame dicts: {timestamp, frame_id, landmarks, pose_id}).
             e.g. session_2026-05-21/<participant>/STA-01_Mountain_Pose/landmarks.json
  Format B — a single combined-session landmarks.json or landmarks.jsonl per
             participant, where each frame carries its own pose_id.
             e.g. Anisha_K_20260518_144115/landmarks.json

Both formats share the same per-frame schema:
  {"timestamp": ..., "frame_id": ..., "pose_id": "STA-01",
   "landmarks": [{"x":.., "y":.., "z":.., "visibility":..}, ... x33]}

Processing pipeline mirrors prepare_dataset_v2.py's process_zip(): downsample,
normalize landmarks (hip-centered/torso-scale), k-means cluster into the pose's
number of steps, order clusters chronologically, median-smooth, clip, then emit
each frame plus a mirrored duplicate.
"""

import json
import os
import sys
from glob import glob

import numpy as np

# --------------------------------------------------------------------------
# Shared with prepare_dataset_v2.py — keep in sync.
# --------------------------------------------------------------------------
POSE_MAP = {
    "STA-01":    ("Tadasana (Mountain Pose)", 6),
    "STA-02":    ("Vrkasana (Tree Pose)", 5),
    "STA-03":    ("Pada-hastasana (Hand-to-Foot Pose)", 6),
    "STA-04-I":  ("Ardha Cakrasana (Half-Wheel Pose)", 6),
    "STA-04-II": ("Ardha Katichakrasana (Half Waist-Wheel Pose)", 5),
    "STA-05-I":  ("Trikonasana (Triangle Pose)", 7),
    "STA-05-II": ("Parivritta Trikonasana (Revolved Triangle Pose)", 6),
    "SIA-01":    ("Ardha Ustrasana (Half Camel Pose)", 6),
    "SIA-02":    ("Vakrasana (Twisted Pose)", 6),
    "PR-01":     ("Makarasana (Crocodile Pose)", 5),
    "PR-02":     ("Bhujangasana (Cobra Pose)", 6),
    "SU-01":     ("Ardha Halasana (Half Plough Pose)", 5),
    "SU-02":     ("Savasana (Corpse Pose)", 5),
}

DOWNSAMPLE_EVERY = 2

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def normalize_landmarks(flat99):
    pts = np.asarray(flat99, dtype=np.float64).reshape(33, 3)
    hip_mid = (pts[LEFT_HIP] + pts[RIGHT_HIP]) / 2.0
    shoulder_mid = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    if scale < 1e-6:
        scale = 1.0
    norm = (pts - hip_mid) / scale
    return norm.flatten().tolist()


def mirror_landmarks(flat99):
    pts = np.asarray(flat99, dtype=np.float64).reshape(33, 3).copy()
    pts[:, 0] = -pts[:, 0]
    for a, b in MIRROR_PAIRS:
        pts[[a, b]] = pts[[b, a]]
    return pts.flatten().tolist()


def kmeans(X, k, iters=50, seed=0):
    n = X.shape[0]
    if n <= k:
        return np.arange(n) % k
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, 1)]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        idx = rng.choice(n, 1, p=probs)
        centers = np.vstack([centers, X[idx]])

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])
        centers = new_centers
    return labels


def median_smooth(arr, window=5):
    n = len(arr)
    out = np.array(arr, dtype=int)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals, counts = np.unique(out[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


ZERO_IMU = [0.0] * 6


def frames_to_entries(frames, pose_id, source_label):
    if pose_id not in POSE_MAP:
        print(f"    ⚠️  skipping unmapped pose_id '{pose_id}' ({source_label})")
        return []

    pose_name, num_steps = POSE_MAP[pose_id]

    frames = sorted(frames, key=lambda f: f.get("frame_id", 0))
    sampled = [
        f for idx, f in enumerate(frames)
        if idx % DOWNSAMPLE_EVERY == 0 and len(f.get("landmarks", [])) == 33
    ]
    if not sampled:
        return []

    norm_feats = []
    for frame in sampled:
        flat = []
        for lm in frame["landmarks"]:
            flat.extend([lm["x"], lm["y"], lm["z"]])
        norm_feats.append(normalize_landmarks(flat))
    X = np.array(norm_feats, dtype=np.float64)

    k = min(num_steps, len(sampled))
    cluster_ids = kmeans(X, k)

    centroids_time = {}
    for c in range(k):
        idxs = np.where(cluster_ids == c)[0]
        centroids_time[c] = idxs.mean() if len(idxs) else 1e18
    order = sorted(range(k), key=lambda c: centroids_time[c])
    cluster_to_step = {c: step for step, c in enumerate(order)}
    step_seq = np.array([cluster_to_step[c] for c in cluster_ids], dtype=int)

    step_seq = median_smooth(step_seq, window=5)
    step_seq = np.clip(step_seq, 0, num_steps - 1)

    entries = []
    for norm_flat, step_index in zip(norm_feats, step_seq):
        step_index = int(step_index)
        entries.append({
            "poseName": pose_name,
            "stepIndex": step_index,
            "label": f"{pose_name}_Step{step_index}",
            "features": [round(f, 6) for f in norm_flat],
            "imuFeatures": list(ZERO_IMU),
        })
        mirrored = mirror_landmarks(norm_flat)
        entries.append({
            "poseName": pose_name,
            "stepIndex": step_index,
            "label": f"{pose_name}_Step{step_index}",
            "features": [round(f, 6) for f in mirrored],
            "imuFeatures": list(ZERO_IMU),
        })

    print(f"    ✅ {source_label} [{pose_id}] -> {len(entries)} samples ({pose_name})")
    return entries


def load_frames_from_file(path):
    if path.endswith(".jsonl"):
        frames = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return frames
    with open(path) as f:
        return json.load(f)


def process_per_pose_session(session_dir):
    """Format A: directory containing per-pose subdirectories, each with landmarks.json."""
    all_entries = []
    for pd in sorted(glob(os.path.join(session_dir, "*"))):
        if not os.path.isdir(pd):
            continue
        lm_path = os.path.join(pd, "landmarks.json")
        if not os.path.exists(lm_path):
            lm_path = os.path.join(pd, "landmarks.jsonl")
        if not os.path.exists(lm_path):
            continue

        pose_id = None
        meta_path = os.path.join(pd, "metadata.json")
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
                pose_id = meta.get("poseId")
            except (json.JSONDecodeError, OSError):
                pass

        frames = load_frames_from_file(lm_path)

        groups = {}
        for fr in frames:
            pid = fr.get("pose_id") or pose_id
            groups.setdefault(pid, []).append(fr)

        for pid, frs in groups.items():
            all_entries.extend(frames_to_entries(frs, pid, os.path.basename(pd)))

    return all_entries


def process_combined_session(session_dir):
    """Format B: a single landmarks.json/.jsonl with per-frame pose_id covering all poses."""
    lm_path = None
    for cand in ("landmarks.json", "landmarks.jsonl"):
        p = os.path.join(session_dir, cand)
        if os.path.exists(p):
            lm_path = p
            break
    if lm_path is None:
        return []

    frames = load_frames_from_file(lm_path)
    groups = {}
    for fr in frames:
        pid = fr.get("pose_id")
        groups.setdefault(pid, []).append(fr)

    all_entries = []
    for pid, frs in groups.items():
        all_entries.extend(frames_to_entries(frs, pid, os.path.basename(session_dir)))
    return all_entries


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 merge_new_participants.py <sensordata_root> <existing_training_data.json> [--out OUT.json]")
        sys.exit(1)

    sensordata_root = sys.argv[1]
    existing_path = sys.argv[2]
    out_path = existing_path
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out_path = sys.argv[i + 1]

    with open(existing_path) as f:
        existing = json.load(f)
    print(f"📦 Existing entries: {len(existing)}")

    new_entries = []
    for d in sorted(glob(os.path.join(sensordata_root, "*"))):
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "landmarks.json")) or os.path.exists(os.path.join(d, "landmarks.jsonl")):
            print(f"📂 {d} (combined-session format)")
            new_entries.extend(process_combined_session(d))
        else:
            # Parent directory containing per-session subdirectories
            for sub in sorted(glob(os.path.join(d, "*"))):
                if os.path.isdir(sub):
                    print(f"📂 {sub} (per-pose-subdirectory format)")
                    new_entries.extend(process_per_pose_session(sub))

    print(f"\n🆕 New entries: {len(new_entries)}")
    merged = existing + new_entries
    print(f"🔗 Merged total: {len(merged)}")

    with open(out_path, "w") as f:
        json.dump(merged, f)

    labels = sorted(set(e["label"] for e in merged))
    print(f"💾 Wrote {len(merged)} entries to {out_path}")
    print(f"📊 {len(labels)} unique step labels across {len(set(e['poseName'] for e in merged))} poses")


if __name__ == "__main__":
    main()
